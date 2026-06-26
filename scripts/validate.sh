#!/usr/bin/env bash
# Run the organizers' validator on a submission CSV before uploading.
#   bash scripts/validate.sh ./submission.csv
set -euo pipefail

CSV="${1:-./submission.csv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${HERE}/validate_submission.py" "${CSV}"
