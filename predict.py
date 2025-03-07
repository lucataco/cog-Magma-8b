# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/microsoft/Magma-8B/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Use bfloat16 precision as specified in the model card
        self.dtype = torch.bfloat16
        # download weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
            
        # Load model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE, 
            trust_remote_code=True, 
            torch_dtype=self.dtype
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_CACHE, 
            trust_remote_code=True
        )
        
        # Move model to GPU
        self.model.to("cuda")
        print("Model loaded successfully")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Text prompt to guide the model's response", default="What is in this image?"),
        system_prompt: str = Input(description="System prompt to set the context", default="You are agent that can see, talk and act."),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate", default=128, ge=1, le=1024),
        temperature: float = Input(description="Sampling temperature", default=0.0, ge=0.0, le=2.0),
        do_sample: bool = Input(description="Whether to use sampling or greedy decoding", default=False),
        num_beams: int = Input(description="Number of beams for beam search", default=1, ge=1, le=5),
    ) -> str:
        """Run a single prediction on the model"""
        # Load and convert image
        image_pil = Image.open(image)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        # Create conversation format
        convs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<image_start><image><image_end>\n{prompt}"},
        ]
        
        # Apply chat template
        prompt_text = self.processor.tokenizer.apply_chat_template(
            convs, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            images=[image_pil], 
            texts=prompt_text, 
            return_tensors="pt"
        )
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        inputs = inputs.to("cuda").to(self.dtype)
        
        # Set generation arguments
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "use_cache": True,
            "num_beams": num_beams,
        }
        
        # Generate response
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, **generation_args)
        
        # Decode response
        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        response = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
        
        return response
