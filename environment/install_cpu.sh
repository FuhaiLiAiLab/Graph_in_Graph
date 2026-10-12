#!/usr/bin/env bash
set -euo pipefail
: "${CONDA_PREFIX:?Activate llfs-gig-revision before running this script}"
python -c 'import platform, sys; assert platform.system() == "Linux" and platform.machine() == "x86_64", "Requires Linux x86_64"; assert sys.version_info[:2] == (3, 10), "Requires Python 3.10"'
python -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-index --only-binary=:all: --no-deps \
  'torch_scatter==2.1.2+pt25cpu' 'torch_sparse==0.6.18+pt25cpu' \
  --find-links 'https://data.pyg.org/whl/torch-2.5.0+cpu.html'
python -m pip install 'torch_geometric==2.6.1'
python -m pip check
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$script_dir/check_environment.py" --device cpu
