# inference/inference_base.py

import glob
import os
from abc import ABC, abstractmethod

import torch

# Reuse the dynamic class loader.
from trainer.trainer_base import get_class


class InferenceBase(ABC):
    """
    Abstract base class for all inference modules.

    Responsibilities:
      - Parse config sections needed for inference
      - Set device
      - Build dataloader and model
      - Load a trained checkpoint
      - Provide a standard `run()` loop
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full configuration dict parsed from TOML.
        """
        self.config = config
        self._load_configs(config)

        self.device = torch.device(self.inference_config.get("device", "cuda:0"))

        # Output directory:
        # <PROJECT_ROOT>/inference_results/<Task.task_name>/<Inference.output_dir_name>/
        self.output_dir = os.path.join(
            self.task_config["project_path"],
            "inference_results",
            self.task_config["task_name"],
            self.inference_config.get("output_dir_name", "default_output"),
        )
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Inference outputs will be saved to: {self.output_dir}")

        self.dataloader = None
        self.dataloader_manager = None
        self.model = None

    def _load_configs(self, config: dict):
        """Parse config and populate inference-related attributes."""
        self.inference_config = config.get("Inference")
        if self.inference_config is None:
            raise KeyError("Config must contain an [Inference] section.")

        self.task_config = config["Task"]

        inference_dataloader_name = config["Inference"].get("inference_dataloader_name")
        self.data_config = {
            "dataloader_name": inference_dataloader_name,
            inference_dataloader_name: config["Inference"].get(inference_dataloader_name),
        }

        self.model_config = config["Model"]

    def _setup_components(self):
        """Instantiate dataloader, model, and load checkpoint."""
        print("--- Setting up inference components ---")

        dataloader_name = self.data_config["dataloader_name"]
        dataloader_class = get_class(f"data_loader.{dataloader_name}")
        self.dataloader_manager = dataloader_class(config=self.config, inference=True)
        self.dataloader = self.dataloader_manager.get_test_loader()
        print(f"Loaded dataloader '{dataloader_name}' in inference mode.")

        model_name = self.model_config["model_name"]
        model_class = get_class(f"model.{model_name}")
        self.model = model_class(config=self.config).to(self.device)
        self.model.eval()
        print(f"Instantiated model '{model_name}'.")

        self._load_checkpoint()

    def _load_checkpoint(self):
        """Resolve and load the checkpoint specified by [Inference].checkpoint_name."""
        trained_model_dir = os.path.join(
            self.task_config["project_path"],
            "trained_models",
            self.task_config["task_name"],
        )

        checkpoint_name = self.inference_config.get("checkpoint_name", "best_tre")

        if checkpoint_name.endswith(".pth"):
            checkpoint_path = os.path.join(trained_model_dir, checkpoint_name)
        elif checkpoint_name == "latest":
            checkpoint_path = os.path.join(trained_model_dir, "checkpoint_latest.pth")
        else:
            pattern = os.path.join(trained_model_dir, f"checkpoint_{checkpoint_name}_*.pth")
            found_files = glob.glob(pattern)
            if not found_files:
                raise FileNotFoundError(
                    f"No checkpoint matched '{checkpoint_name}' under {trained_model_dir} "
                    f"(pattern: {pattern})."
                )
            checkpoint_path = sorted(found_files)[-1]

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = checkpoint["model_state_dict"]

        # If diffusion teacher is disabled, allow extra keys in the checkpoint.
        strict = True
        if hasattr(self.model, "enable_diffusion_teacher") and (not getattr(self.model, "enable_diffusion_teacher")):
            strict = False
            print("[Info] enable_diffusion_teacher=False -> strict=False when loading checkpoint.")

        missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)

        if (len(missing) > 0) or (len(unexpected) > 0):
            print(f"[Checkpoint] strict={strict} | missing={len(missing)} | unexpected={len(unexpected)}")
            # Uncomment for debugging:
            # print("  missing (first 10):", missing[:10])
            # print("  unexpected (first 10):", unexpected[:10])

    @abstractmethod
    def predict(self, batch_data: dict, index: int):
        """Run prediction for one batch/sample."""
        raise NotImplementedError

    def run(self):
        """Run the full inference loop."""
        self._setup_components()
        print(f"\n--- Running inference for task '{self.task_config['task_name']}' ---")

        with torch.no_grad():
            for i, batch_data in enumerate(self.dataloader):
                self.predict(batch_data, i)

        print("\n--- Inference finished ---")