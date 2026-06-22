# run_inference.py

import argparse
import os
import sys

import toml

from trainer.trainer_base import get_class

# Add the project root to Python path so local modules can be imported reliably.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> None:
    """
    Entry point for running inference using a TOML configuration file.

    This script:
      1) parses CLI args,
      2) loads the TOML config,
      3) dynamically instantiates an inference runner from `inference.<inference_name>`,
      4) executes the inference pipeline via `runner.run()`.
    """
    # -------------------- 1) Parse arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run inference using a TOML configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the TOML configuration file for inference.",
    )
    args = parser.parse_args()

    # -------------------- 2) Load config --------------------
    try:
        config = toml.load(args.config)
        print(f"Loaded config from: {args.config}")
    except Exception as e:
        print(f"[Error] Failed to load or parse TOML config: {e}")
        sys.exit(1)

    # -------------------- 3) Instantiate inference runner --------------------
    try:
        # Get the inference class name from the [Inference] section.
        inference_name = config["Inference"]["inference_name"]
        # The class must exist under the `inference` package.
        inference_class = get_class(f"inference.{inference_name}")
        print(f"Instantiating inference runner: '{inference_name}'")
    except KeyError:
        print("[Error] Missing 'inference_name' in the [Inference] section of the config.")
        sys.exit(1)

    inference_runner = inference_class(config=config)

    # -------------------- 4) Run inference --------------------
    inference_runner.run()


if __name__ == "__main__":
    main()