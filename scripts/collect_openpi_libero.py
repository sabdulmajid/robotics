#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess

from risk_aware_skill_planning.backends.openpi.config import load_openpi_experiment_config
from risk_aware_skill_planning.backends.openpi.libero_runner import sbatch_command, sbatch_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit or print an OpenPI/LIBERO rollout collection job")
    parser.add_argument("--config", required=True)
    parser.add_argument("--submit", action="store_true", help="Submit with sbatch. Default prints the exact command.")
    args = parser.parse_args()

    config = load_openpi_experiment_config(args.config)
    command = sbatch_command(config)
    payload = {"ok": True, "environment": sbatch_environment(config), "command": command}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.submit:
        return 0
    env = dict(os.environ)
    env.update(sbatch_environment(config))
    completed = subprocess.run(["sbatch", "slurm/openpi_libero_rollouts.sbatch"], check=False, text=True, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
