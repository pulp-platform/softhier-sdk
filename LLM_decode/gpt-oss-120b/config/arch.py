"""SDK implementation architecture with enough address space for FP16 experts."""
from pathlib import Path
import runpy

_sdk = Path(__file__).resolve().parents[3]
_Base = runpy.run_path(str(_sdk / 'implementation/config/arch/arch.py'))['FlexClusterArch']


class FlexClusterArch(_Base):
    def __init__(self):
        super().__init__()
        # Expose a 4 GiB logical region on each populated edge. Aliases share storage.
        # The run's DRAM address mapping covers 1 GiB per physical channel.
        self.hbm_node_addr_space = 0x100000000
