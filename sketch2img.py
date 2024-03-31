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

# MODEL_SCRIBBLE = os.path.join(PATH_TO_CN_MODELS, "sd-controlnet-scribble")
MODEL_SCRIBBLE = file_chooser(PATH_TO_CN_MODELS)

sketches_path = "./src/sks"
output_path = "./src"
ouput_path_for_processing = "./src"

# Load controlnet
controlnet = ControlNetModel.from_pretrained(
    MODEL_SCRIBBLE, # "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True,
    low_cpu_mem_usage=False,
)

# Load base model
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    MODEL_BASE,
    controlnet=controlnet,
    torch_dtype=torch.float32,
    # safety_checker=None,
    low_cpu_mem_usage=False,
)

# pipe=pipe.to("cpu")
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True) # use_karras=True,
pipe.safety_checker = None

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
    "normal quality, ((monochrome)), ((grayscale)), ((watermark)), uneven eyes, lazy eye,(((mutated hand))),multiple navel,ornaments,"
)

print(f"[+] Prompt:\n\t{p}")
print(f"[+] Negative:\n\t{n}")

# sketch chooser
sketch_path = file_chooser(sketches_path)

processed_image = process_image(sketch_path, ouput_path_for_processing, thumbnail_size = (512, 512)) # (512, 512), ouput_path_for_processing,
for i in range(100):
    seed = int(random.random()*9999)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        p , processed_image # Image.open(image_path)
        , negative_prompt = n
        , num_inference_steps=32
        , generator = generator
    ).images[0]
    image.save(os.path.join(output_path, f"{MODEL_NAME}_{i:04d}_{seed}.png"))
