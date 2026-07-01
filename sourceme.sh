#!/usr/bin/env bash

#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Chi Zhang <chizhang@ethz.ch>

if [ -n "${ZSH_VERSION:-}" ]; then
    _softhier_sdk_dir="$(dirname "$(readlink -f -- "${(%):-%x}")")"
else
    _softhier_sdk_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
fi

export PYTHONPATH="${_softhier_sdk_dir}/utilities:${PYTHONPATH:-}"

_softhier_toolchain_dir="${_softhier_sdk_dir}/toolchain"
_softhier_toolchain_install="${_softhier_toolchain_dir}/install"
_softhier_toolchain_bin="${_softhier_toolchain_install}/bin"

_softhier_install_toolchain() {
    if [ -x "${_softhier_toolchain_bin}/riscv32-unknown-elf-gcc" ]; then
        return 0
    fi

    make sh-toolchain
    return 0
}

_softhier_install_toolchain || true

export SOFTHIER_RISCV_TOOLCHAIN="${_softhier_toolchain_install}"
if [ -d "${_softhier_toolchain_bin}" ]; then
    export PATH="${_softhier_toolchain_bin}:${PATH}"
fi

unset _softhier_sdk_dir
unset _softhier_toolchain_dir
unset _softhier_toolchain_install
unset _softhier_toolchain_bin
unset -f _softhier_install_toolchain
