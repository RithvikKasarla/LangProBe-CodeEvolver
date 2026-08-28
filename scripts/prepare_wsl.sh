#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="$repo_root/.runtime"
appworld_venv="$runtime/appworld-venv"
appworld_root="$runtime/appworld-root"
alfworld_data="$runtime/alfworld-data"
alfworld_venv="$runtime/alfworld-venv"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"

mkdir -p "$runtime"

"$uv_bin" venv --python 3.11 "$appworld_venv"
"$uv_bin" pip install --python "$appworld_venv/bin/python" \
  'appworld==0.1.3' 'click==8.1.7'
APPWORLD_ROOT="$appworld_root" "$appworld_venv/bin/appworld" install
APPWORLD_ROOT="$appworld_root" "$appworld_venv/bin/appworld" download data --root "$appworld_root"
"$appworld_venv/bin/appworld" verify tests --root "$appworld_root"
"$appworld_venv/bin/appworld" verify tasks --root "$appworld_root"
APPWORLD_ROOT="$appworld_root" "$appworld_venv/bin/python" "$repo_root/scripts/export_appworld_data.py"

"$uv_bin" venv --python 3.11 "$repo_root/env"
"$uv_bin" pip install --python "$repo_root/env/bin/python" -r "$repo_root/requirements.txt"

"$uv_bin" venv --python 3.9 "$alfworld_venv"
"$uv_bin" pip install --python "$alfworld_venv/bin/python" \
  'setuptools==80.9.0' 'wheel==0.47.0' 'Cython==3.2.8'
# The Python 3.9 wheels for this legacy chain are incomplete. Install compiled
# dependencies in dependency order so each package can find the previous
# package's Cython headers.
for package in \
  'blis==1.3.3' \
  'thinc==8.3.9' \
  'spacy==3.8.11' \
  'textworld[pddl]==1.6.1'; do
  "$uv_bin" pip install --python "$alfworld_venv/bin/python" \
    --no-build-isolation "$package"
done
"$uv_bin" pip install --python "$alfworld_venv/bin/python" \
  --no-build-isolation -r "$repo_root/alfworld-runtime-requirements.txt"
"$uv_bin" pip check --python "$alfworld_venv/bin/python"
if [[ ! -f "$alfworld_data/.complete" ]]; then
  ALFWORLD_DATA="$alfworld_data" "$alfworld_venv/bin/alfworld-download"
  touch "$alfworld_data/.complete"
fi
ALFWORLD_DATA="$alfworld_data" "$repo_root/env/bin/python" "$repo_root/scripts/export_alfworld_data.py"

echo "Benchmark runtimes and datasets are ready."
