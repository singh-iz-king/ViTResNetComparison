import torch

import argparse
import os
import subprocess
import sys
import tempfile
import yaml


def set_by_dotted_path(cfg: dict, dotted_key: str, value):
    """
    Example dotted_key: "model.init_encoder_from"
    """
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        set_by_dotted_path(cfg, k, v)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, default="configs/pipeline.yaml")
    args = parser.parse_args()

    pipe = yaml.safe_load(open(args.pipeline, "r"))["pipeline"]

    for stage in pipe:
        name = stage["name"]
        base_cfg_path = stage["config"]
        overrides = stage.get("overrides", {})

        print(f"\n=== Stage: {name} ===")
        cfg = yaml.safe_load(open(base_cfg_path, "r"))
        cfg = apply_overrides(cfg, overrides)

        # Write merged config to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
            merged_cfg_path = f.name

        # Run training
        try:
            subprocess.check_call(
                [sys.executable, "-m", "scripts.train", "--config", merged_cfg_path]
            )
        finally:
            os.remove(merged_cfg_path)

        # Optional: verify outputs exist
        outputs = stage.get("outputs", {})
        for _, path in outputs.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Stage '{name}' expected output file not found: {path}"
                )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
