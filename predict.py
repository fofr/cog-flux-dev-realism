# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(workflow)

    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        aspect_ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1088),
            "5:4": (1088, 896),
            "9:16": (768, 1344),
            "9:21": (640, 1536),
        }
        return aspect_ratios.get(aspect_ratio)

    def update_workflow(self, workflow, **kwargs):
        empty_latent_image = workflow["5"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]
        empty_latent_image["batch_size"] = kwargs["num_outputs"]

        shift = workflow["61"]["inputs"]
        shift["width"] = kwargs["width"]
        shift["height"] = kwargs["height"]

        lora = workflow["72"]["inputs"]
        lora["strength_model"] = kwargs["lora_strength"]

        prompt = workflow["6"]["inputs"]
        prompt["text"] = kwargs["prompt"]

        noise = workflow["25"]["inputs"]
        noise["noise_seed"] = kwargs["seed"]

        guidance = workflow["60"]["inputs"]
        guidance["guidance"] = kwargs["guidance"]

        sampler = workflow["17"]["inputs"]
        sampler["steps"] = kwargs["num_inference_steps"]

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1",
        ),
        num_outputs: int = Input(
            description="Number of outputs to generate", default=1, le=4, ge=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. Recommended range is 28-50",
            ge=1,
            le=50,
            default=30,
        ),
        guidance: float = Input(
            description="Guidance for generated image",
            ge=0,
            le=10,
            default=3.5,
        ),
        lora_strength: float = Input(
            description="Strength of flux-realism lora, 0 is disabled",
            ge=0,
            le=2,
            default=0.8,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)

        self.update_workflow(
            workflow,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            lora_strength=lora_strength,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            num_outputs=num_outputs,
        )

        self.comfyUI.connect()
        self.comfyUI.run_workflow(workflow)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
