# run_training_ddp.py

import argparse
import os
import sys
import toml
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trainer.trainer_base import get_class

def main():
    parser = argparse.ArgumentParser(description="Run a DDP training session.")
    parser.add_argument("--config", type=str, required=True, help="Path to the TOML configuration file.")
    args = parser.parse_args()

    try:
        config = toml.load(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

    # Check if DDP environment variables are set
    if 'RANK' not in os.environ:
        print("DDP environment variables not found. This script must be launched with 'torchrun'.")
        sys.exit(1)

    try:
        trainer_name = config['Task']['trainer_name']
        trainer_class = get_class(f"trainer.{trainer_name}")
        
        # Instantiate the trainer, telling it to run in DDP mode
        trainer = trainer_class(config=config, is_ddp=True)
        
        trainer.run()
        
    except Exception as e:
        # Use torch.distributed.get_rank() if initialized, otherwise use RANK env var
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print(f"\nAn error occurred during DDP training: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()