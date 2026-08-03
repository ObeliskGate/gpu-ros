#!/usr/bin/env bash
set -euo pipefail
echo "DEPRECATED: use docker/phase2-amd.sh; forwarding to the unified AMD environment entrypoint" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/phase2-amd.sh" "$@"
