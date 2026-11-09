# source this file before running anything:  source env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"; [ -d "$VENV" ] || VENV="$ROOT/../.venv"          # venv inside the repo, or next to it
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$VENV/lib/python3.10/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
export PYOPENGL_PLATFORM=egl
# data locations (override by exporting before sourcing)
export SFMT_DATA="${SFMT_DATA:-$ROOT/../data}"
export SFMT_MIXAMO_PATH="${SFMT_MIXAMO_PATH:-$SFMT_DATA/Mixamo}"
export SFMT_MIXAMO_SIMPLIFY_PATH="${SFMT_MIXAMO_SIMPLIFY_PATH:-$SFMT_MIXAMO_PATH/Mixamo_simplify_new}"
export SFMT_SMPLH_PATH="${SFMT_SMPLH_PATH:-$SFMT_DATA/smplh}"
export SFMT_LOG_DIR="${SFMT_LOG_DIR:-$ROOT/../runs}"
