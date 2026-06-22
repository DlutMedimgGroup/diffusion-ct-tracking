# run_training.py

import argparse
import os
import sys
import toml

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Add the project root to the Python path. This allows the script to be run
# from anywhere and still find the custom modules (e.g., 'engine', 'models').
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can safely import from our custom packages.
from trainer.trainer_base import get_class

def main():
    """
    The main entry point for starting a training process.

    This script parses a command-line argument for the configuration file path,
    loads the configuration, dynamically instantiates the specified trainer,
    and starts the training process by calling its .run() method.
    """
    # 1. --- Argument Parsing ---
    # Set up an argument parser to accept the path to the config file.
    parser = argparse.ArgumentParser(
        description="Run a training session using a TOML configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the TOML configuration file for the training run.",
    )
    args = parser.parse_args()

    # 2. --- Configuration Loading ---
    # Load the specified TOML configuration file.
    try:
        config = toml.load(args.config)
        print(f"Successfully loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing TOML file: {e}")
        sys.exit(1)

    # 3. --- Trainer Instantiation ---
    # Dynamically get the trainer class specified in the config.
    try:
        trainer_name = config['Task']['trainer_name']
        # The class must exist in the 'trainer' package.
        trainer_class = get_class(f"trainer.{trainer_name}")
        print(f"Instantiating trainer: '{trainer_name}'")
    except KeyError:
        print("Error: 'trainer_name' not found in the [Task] section of the config file.")
        sys.exit(1)

    # Create an instance of the trainer, passing the full configuration.
    # The trainer and its components will handle parsing this config internally.
    trainer = trainer_class(config=config)

    # 4. --- Start Training ---
    # The .run() method encapsulates the entire training and validation loop.
    trainer.run()
        

if __name__ == "__main__":
    main()

