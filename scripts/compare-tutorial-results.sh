#!/usr/bin/env bash
set -euo pipefail

status=0
if ! diff -u examples/form8949.csv form8949_example.csv; then
  status=1
fi

if [[ $status -eq 0 ]]; then
  echo "Tutorial example matches expected outputs."
else
  echo "Tutorial results differ from expected outputs."
fi

exit "$status"