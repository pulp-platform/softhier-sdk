import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TRACE_PERFETTO_DIR = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace_parser = load_module(
    'soft_hier_trace_parser', TRACE_PERFETTO_DIR / 'parse.py'
)
trace_visualizer = load_module(
    'soft_hier_trace_visualizer', TRACE_PERFETTO_DIR / 'visualize.py'
)


def trace_line(cluster, component, message, time=1000, cycles=10):
    return (
        f'{time}: {cycles}: '
        f'[\x1b[34m/chip/cluster_{cluster}/{component}/trace\x1b[0m] '
        f'{message}\n'
    )


class TraceParserTest(unittest.TestCase):

    def parse(self, lines):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / 'trace.txt'
            output_path = Path(temp_dir) / 'roi.json'
            input_path.write_text(''.join(lines))
            trace_parser.parse_trace(input_path, output_path)
            return json.loads(output_path.read_text())

    def test_colored_annotation_line_and_maximum_id(self):
        line = trace_line(
            3,
            'cluster_registers',
            'Load weights: 120 ns -> 450 ns | period = 330 ns | '
            'annotation_id = 4294967295'
        )

        roi = self.parse([line])
        region = roi['cluster_3'][0]

        self.assertEqual(region['label'], 'Load weights')
        self.assertEqual(region['category'], 'annotation')
        self.assertEqual(region['track'], 'annotation_4294967295')
        self.assertEqual(region['tstart'], 120)
        self.assertEqual(region['tend'], 450)
        self.assertEqual(region['attrs']['annotation_id'], 4294967295)
        self.assertEqual(region['attrs']['period_ns'], 330)
        self.assertEqual(region['attrs']['info'], line.rstrip('\n'))

    def test_same_label_different_ids_and_rename_keep_stable_tracks(self):
        roi = self.parse([
            trace_line(
                0,
                'cluster_registers',
                'compute: 10 ns -> 40 ns | period = 30 ns | '
                'annotation_id = 1'
            ),
            trace_line(
                0,
                'cluster_registers',
                'compute: 20 ns -> 50 ns | period = 30 ns | '
                'annotation_id = 2'
            ),
            trace_line(
                0,
                'cluster_registers',
                'renamed compute: 60 ns -> 90 ns | period = 30 ns | '
                'annotation_id = 1'
            )
        ])

        regions = roi['cluster_0']
        self.assertEqual(
            [region['track'] for region in regions],
            ['annotation_1', 'annotation_2', 'annotation_1']
        )
        self.assertEqual(
            [region['label'] for region in regions],
            ['compute', 'compute', 'renamed compute']
        )

    def test_special_characters_round_trip(self):
        words = 'Load: "A\\\\B" 100% | x -> y'
        roi = self.parse([
            trace_line(
                1,
                'cluster_registers',
                f'{words}: 1001 ns -> 2003 ns | period = 1002 ns | '
                'annotation_id = 7'
            )
        ])

        self.assertEqual(roi['cluster_1'][0]['label'], words)

    def test_annotations_coexist_with_legacy_events(self):
        roi = self.parse([
            trace_line(
                2,
                'redmule',
                '[LightRedmule] Finished : 10 ns ---> 20 ns | work | '
                'uti = 0.875 |'
            ),
            trace_line(
                2,
                'idma/fe',
                '[iDMA] Finished : 21 ns ---> 30 ns'
            ),
            trace_line(
                2,
                'cluster_registers',
                'Cluster Sync: 31 ns -> 40 ns | period = 9 ns | Type = 0'
            ),
            trace_line(
                2,
                'cluster_registers',
                'user phase: 41 ns -> 55 ns | period = 14 ns | '
                'annotation_id = 9'
            )
        ])

        regions = roi['cluster_2']
        self.assertEqual(
            [region['label'] for region in regions],
            ['redmule', 'idma', 'cluster sync', 'user phase']
        )
        self.assertNotIn('track', regions[0])
        self.assertEqual(regions[-1]['track'], 'annotation_9')


class TraceVisualizerTest(unittest.TestCase):

    def visualize(self, roi):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / 'roi.json'
            output_path = Path(temp_dir) / 'perfetto.json'
            input_path.write_text(json.dumps(roi))
            trace_visualizer.roi_to_perfetto(input_path, output_path)
            return json.loads(output_path.read_text())

    def test_annotation_track_category_and_timing_conversion(self):
        perfetto = self.visualize({
            'cluster_2': [
                {
                    'label': 'Load: "A\\\\B" 100%',
                    'category': 'annotation',
                    'track': 'annotation_7',
                    'tstart': 1001,
                    'tend': 2003,
                    'attrs': {
                        'annotation_id': 7,
                        'period_ns': 1002
                    }
                },
                {
                    'label': 'redmule',
                    'tstart': 3000,
                    'tend': 4500,
                    'attrs': {'util': 0.875}
                }
            ]
        })

        annotation = perfetto['traceEvents'][1]
        legacy = perfetto['traceEvents'][2]

        self.assertEqual(annotation['name'], 'Load: "A\\\\B" 100%')
        self.assertEqual(annotation['pid'], 'cluster_2')
        self.assertEqual(annotation['tid'], 'annotation_7')
        self.assertEqual(annotation['cat'], 'annotation')
        self.assertAlmostEqual(annotation['ts'], 1.001)
        self.assertAlmostEqual(annotation['dur'], 1.002)

        self.assertEqual(legacy['tid'], 'redmule')
        self.assertNotIn('cat', legacy)
        self.assertEqual(legacy['ts'], 3.0)
        self.assertEqual(legacy['dur'], 1.5)


if __name__ == '__main__':
    unittest.main()
