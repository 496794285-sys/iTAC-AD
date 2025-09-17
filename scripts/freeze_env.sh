#!/usr/bin/env bash
# 导出可复现环境（conda itacad）
set -e
conda env export --name itacad --no-builds | sed -e '/^prefix:/d' > environment-lock.yml
python - <<'PY'
import pkgutil, json, sys
import pkg_resources as pr
print(json.dumps({d.project_name: d.version for d in pr.working_set}, indent=2))
PY