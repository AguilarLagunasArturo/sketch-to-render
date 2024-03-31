import os
import torch
import random
from diffusers \
import \
    DiffusionPipeline, \
    StableDiffusionControlNetPipeline, \
    ControlNetModel, \
    UniPCMultistepScheduler, \
    DPMSolverMultistepScheduler, \
    EulerAncestralDiscreteScheduler
from PIL import Image
from diffusers.utils import load_image
from AuxFn \
import \
    process_image, \
    file_chooser

HOME = os.environ.get("HOME")
PATH_TO_HF_MODELS = os.path.join(HOME, "Development/AI/stable-diffusion-webui/models/HuggingFace")
PATH_TO_CN_MODELS = os.path.join(HOME, "Development/AI/stable-diffusion-webui/models/ControlNet")

# MODEL_BASE = os.path.join(PATH_TO_HF_MODELS, "astranime_V6")
MODEL_BASE = file_chooser(PATH_TO_HF_MODELS)
MODEL_NAME = MODEL_BASE.split("/")[-1]

ouput_path_for_processing = "./src"

# character = input('Prompt: ')
# character = "((portrait)),bbw,shiny skin,eyeliner,eyeshadow,eyelashes,black eyes,straight hair,long hair,head band,black hair,((mature woman,milf,wide neck)):1.3" if character == "" else character
character = "1girl,portrait,sad,eye bags,white uniform,tie,pony tail,black hair"

p = (
    f"(masterpiece,best quality,hires,high resolution:1.2),(extremely detailed,realistic,intricate details),(nigh light,ominous,dark ambient),3d,cg," # cinematic lighting,sunlight,volumetric
    # f"BREAK,"
    f"{character},looking at viewer,"
    # f"BREAK,"
    f"1960s \(style\),vintage fantasy,film grain" # (soviet poster:1.4),
)
n = (
    "(worst quality:2), (low quality:2),(normal quality:2), lowres, bad anatomy,((extra limbs)),((extra legs)),((fused legs)),((extra arms)),((fused arms)),"
    "normal quality, ((monochrome)), ((grayscale)), ((watermark)), uneven eyes, lazy eye,(((mutated hand))),multiple navel,ornaments,((lowres:1.8)),((blurry:1.8)),noise,"
)

print(f"[+] Prompt:\n\t{p}")
print(f"[+] Negative:\n\t{n}")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_BASE,
    custom_pipeline="lpw_stable_diffusion",
    revision='0.15.1', # 0.15.1
    custom_revision='0.15.1', # 0.15.1
    torch_dtype=torch.float32, # 32
    cache_dir="./.cache",
)
# pipe=pipe.to("cpu")
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True) # use_karras=True,
pipe.safety_checker = None

seed = 78471
# seed = int(random.random()*9999)
generator = torch.Generator(device="cpu").manual_seed(seed)
image = pipe( # .text2img
    prompt=p,
    negative_prompt=n,
    num_inference_steps=32,
    guidance_scale=8,
    generator=generator,
    width=512, height=768,
    max_embeddings_multiples=6
).images[0]
image.save(os.path.join(ouput_path_for_processing, f"{MODEL_NAME}_{seed}.png"))
