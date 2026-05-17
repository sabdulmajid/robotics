#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_aware_skill_planning.openpi_libero import load_smoke_config, run_openpi_libero_smoke
from risk_aware_skill_planning.openpi_libero.smoke import write_smoke_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenPI/LIBERO setup without running a full benchmark")
    parser.add_argument("--config", default="configs/openpi_libero_smoke.yaml")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config = load_smoke_config(args.config)
    status = run_openpi_libero_smoke(config)
    config.status_path.parent.mkdir(parents=True, exist_ok=True)
    config.status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    write_smoke_report(status, config.report_path)
    print(json.dumps({"ok": status["ok"], "status_path": str(config.status_path)}, indent=2, sort_keys=True))
    return 0 if status["ok"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
