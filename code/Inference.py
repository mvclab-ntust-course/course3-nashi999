from diffusers import DiffusionPipeline
import torch

model_path = "./pytorch_lora_weights.safetensors"
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_lora_weights(model_path)

prompt = "A flying hamster"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save(prompt + ".png")