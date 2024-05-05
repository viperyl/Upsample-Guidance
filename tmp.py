import os
from sd15_upsample import StableDiffusionUpsamplingGuidancePipeline

os.environ["http_proxy"] = "127.0.0.1:15777"
os.environ["https_proxy"] = "127.0.0.1:15777"


pipe = StableDiffusionUpsamplingGuidancePipeline.from_single_file("/data_ssd/comfyui_models/checkpoints/15/realisticVisionV60B1_v51VAE.safetensors")
pipe.to("cuda:1")

prompt = "a cute cat"



img = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    time_factor=1.2,
    scale_factor=2,
    us_eta=0.6
).images[0]

img.save("/home/114514/upsample_guidance/res/us_3.png")