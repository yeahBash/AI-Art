import torch
# for parameters ui
from ipywidgets import interactive_output, fixed, VBox, HBox
import ipywidgets as widgets
from IPython.display import display
# for config
import json
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
from diffusers import EulerDiscreteScheduler, DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler
from PIL import Image

BASE_PIPELINES = {"StableDiffusionXLPipeline":StableDiffusionXLPipeline,
                  "StableDiffusionXLImg2ImgPipeline":StableDiffusionXLImg2ImgPipeline,
                  "StableDiffusionXLInpaintPipeline":StableDiffusionXLInpaintPipeline,}
REFINER_PIPELINES = {"StableDiffusionXLImg2ImgPipeline":StableDiffusionXLImg2ImgPipeline}
SCHEDULERS = {"EulerDiscreteScheduler":EulerDiscreteScheduler, 
              "DDIMScheduler":DDIMScheduler, 
              "LMSDiscreteScheduler":LMSDiscreteScheduler,
              "DPMSolverMultistepScheduler":DPMSolverMultistepScheduler}
PRECISION = {"torch.float16":torch.float16}

class SDXLConfig:
    CONFIG_PATH = "sdxl_config.json"
    
    def __init__(self,
                 base_pipe_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_pipe_model: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 torch_dtype_str: str = "torch.float16",
                 base_pipeline_type_str: str = "StableDiffusionXLPipeline",
                 refiner_pipeline_type_str: str = "StableDiffusionXLImg2ImgPipeline",
                 scheduler_type_str: str = "LMSDiscreteScheduler",
                 variant: str = "fp16",
                 use_safetensors: bool = True,
                 #safety_checker = None
                 prompt: str = None,
                 prompt_2: str = None,
                 negative_prompt: str = None,
                 negative_prompt_2: str = None,
                 use_compel: bool = False,
                 num_inference_steps: int = 40,
                 width: int = 768,
                 height: int = 768,
                 guidance_scale: float = 7.5,
                 high_noise_frac: float = 0.8,
                 seed: int = 12345,
                 use_refiner: bool = False,
                 strength: float = 0.3,
                 image_path: str = "ref_image.jpg",
                 mask_path: str = "mask.jpg",
                 ):
        self.base_pipe_model = base_pipe_model
        self.refiner_pipe_model = refiner_pipe_model
        self.torch_dtype_str = torch_dtype_str
        self.base_pipeline_type_str = base_pipeline_type_str
        self.refiner_pipeline_type_str = refiner_pipeline_type_str
        self.scheduler_type_str = scheduler_type_str
        self.variant = variant
        self.use_safetensors = use_safetensors
        self.prompt = prompt
        self.prompt_2 = prompt_2
        self.negative_prompt = negative_prompt
        self.negative_prompt_2 = negative_prompt_2
        self.use_compel = use_compel
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height
        self.guidance_scale = guidance_scale
        self.high_noise_frac = high_noise_frac
        self.seed = seed
        self.use_refiner = use_refiner
        self.strength = strength
        self.image_path = image_path
        self.mask_path = mask_path

    @property
    def torch_dtype(self):return PRECISION[self.torch_dtype_str]
    @property
    def base_pipeline_type(self):return BASE_PIPELINES[self.base_pipeline_type_str]
    @property
    def refiner_pipeline_type(self):return REFINER_PIPELINES[self.refiner_pipeline_type_str]
    @property
    def scheduler_type(self):return SCHEDULERS[self.scheduler_type_str]
    @property
    def image(self):return Image.open(self.image_path)
    @property
    def mask(self):return Image.open(self.mask_path)
    @property
    def kwargs(self): # choose additional arguments
        kwargs = {}
        match self.base_pipeline_type_str:
            case "StableDiffusionXLPipeline":
                kwargs['width'] = self.width
                kwargs['height'] = self.height
            case "StableDiffusionXLImg2ImgPipeline":
                kwargs['image'] = self.image
                kwargs['strength'] = self.strength
            case "StableDiffusionXLInpaintPipeline":
                kwargs['width'] = self.width
                kwargs['height'] = self.height
                kwargs['image'] = self.image
                kwargs['strength'] = self.strength
                kwargs['mask'] = self.mask
        return kwargs
    
    def to_json(obj):
        if isinstance(obj, SDXLConfig):
            return obj.__dict__
    def from_json(dict: dict):
            return SDXLConfig(**dict)
    def load_config():
         with open(SDXLConfig.CONFIG_PATH, "r") as read_file:
            return json.load(read_file, object_hook=SDXLConfig.from_json)
    def save_config(self):
         with open(SDXLConfig.CONFIG_PATH, "w") as write_file:
            json.dump(self, write_file, skipkeys=True, indent=1, default=SDXLConfig.to_json)

    def set_ui(self):
        # TODO: not best workaround to get variable name
        def f(x, f_name): setattr(self, f_name.split('=')[0].split('.')[1], x)

        style = {'description_width': 'initial'}
        left_box = {
            # prompts
            f'{self.prompt=}':widgets.Textarea(value=self.prompt, placeholder='Type positive1...', description='Prompt1:', style=style),
            f'{self.prompt_2=}':widgets.Textarea(value=self.prompt_2, placeholder='Type positive2...', description='Prompt2:', style=style),
            f'{self.negative_prompt=}':widgets.Textarea(value=self.negative_prompt, placeholder='Type negative1...', description='Negative Prompt1:', style = style),
            f'{self.negative_prompt_2=}':widgets.Textarea(value=self.negative_prompt_2, placeholder='Type negative2...', description='Negative Prompt2:', style = style),
            f'{self.use_compel=}':widgets.Checkbox(value=self.use_compel, description="Use Compel", indent=False, style=style)
        }
        right_box = {
            # models, precisions, schedulers
            f'{self.base_pipe_model=}':widgets.Text(value=self.base_pipe_model, placeholder='', description='Base model:', style=style),
            f'{self.refiner_pipe_model=}':widgets.Text(value=self.refiner_pipe_model, placeholder='', description='Refiner model:', style=style),
            f'{self.torch_dtype_str=}':widgets.Dropdown(value=self.torch_dtype_str, options=PRECISION.keys(), description='dtype:', style=style),
            f'{self.variant=}':widgets.Text(value=self.variant, placeholder='', description='Variant:', style=style),
            f'{self.use_safetensors=}':widgets.Checkbox(value=self.use_safetensors, description="Use safetensors", indent=False, style=style),
            f'{self.base_pipeline_type_str=}':widgets.Dropdown(value=self.base_pipeline_type_str, options=BASE_PIPELINES.keys(), description='Base type:', style=style),
            f'{self.refiner_pipeline_type_str=}':widgets.Dropdown(value=self.refiner_pipeline_type_str, options=REFINER_PIPELINES.keys(), description='Refiner type:', style=style),
            f'{self.scheduler_type_str=}':widgets.Dropdown(value=self.scheduler_type_str, options=SCHEDULERS.keys(), description='Scheduler type:', style=style),
            
            # inference properties
            f'{self.num_inference_steps=}':widgets.IntSlider(value=self.num_inference_steps, min=10, max=100, step=5, description="Num inference steps:", continuous_update=False, style=style),
            f'{self.width=}':widgets.IntSlider(value=self.width, min=512, max=1024, step=64, description="Width:", continuous_update=False, style=style),
            f'{self.height=}':widgets.IntSlider(value=self.height, min=512, max=1024, step=64, description="Height:", continuous_update=False, style=style),
            f'{self.guidance_scale=}':widgets.FloatSlider(value=self.guidance_scale, min=0, max=10, step=0.25, description="Guidance scale:", continuous_update=False, style=style),
            f'{self.seed=}':widgets.IntSlider(value=self.seed, min=0, max=1000000, step=1, description="Seed:", continuous_update=False, style=style),
            f'{self.high_noise_frac=}':widgets.FloatSlider(value=self.high_noise_frac, min=0, max=1, step=0.05, description="High noise frac:", continuous_update=False, style=style),

            # refiner
            f'{self.use_refiner=}':widgets.Checkbox(value=self.use_refiner, description="Use refiner", indent=False, style=style),

            # img2img and inpaint
            f'{self.strength=}':widgets.FloatSlider(value=self.strength, min=0, max=1, step=0.05, description="Strength:", continuous_update=False, style=style),
            f'{self.image_path=}':widgets.Text(value=self.image_path, placeholder='', description='Image path:', style=style),
            f'{self.mask_path=}':widgets.Text(value=self.mask_path, placeholder='', description='Mask path:', style=style)
        }
        
        ui = HBox([VBox(list(left_box.values())), VBox(list(right_box.values()))])
        boxes = left_box | right_box
        [interactive_output(f, {'x':x, 'f_name':fixed(f_name)}) for f_name,x in boxes.items()]
        display(ui)