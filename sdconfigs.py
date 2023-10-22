import torch
# for parameters ui
from ipywidgets import interactive_output, fixed, VBox, HBox, Layout
import ipywidgets as widgets
from IPython.display import display
# for config
import json
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
from diffusers import EulerDiscreteScheduler, DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from PIL import Image

BASE_PIPELINES = {"StableDiffusionXLPipeline":StableDiffusionXLPipeline,
                  "StableDiffusionXLImg2ImgPipeline":StableDiffusionXLImg2ImgPipeline,
                  "StableDiffusionXLInpaintPipeline":StableDiffusionXLInpaintPipeline,}
REFINER_PIPELINES = {"StableDiffusionXLImg2ImgPipeline":StableDiffusionXLImg2ImgPipeline}
SCHEDULERS = {"EulerDiscreteScheduler":EulerDiscreteScheduler, 
              "DDIMScheduler":DDIMScheduler, 
              "LMSDiscreteScheduler":LMSDiscreteScheduler,
              "DPMSolverMultistepScheduler":DPMSolverMultistepScheduler,
              "DPMSolverSDEScheduler":DPMSolverSDEScheduler}
PRECISION = {"torch.float16":torch.float16}

class SDXLConfig:    
    def __init__(self,
                 base_pipe_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_pipe_model: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 torch_dtype_str: str = "torch.float16",
                 base_pipeline_type_str: str = "StableDiffusionXLPipeline",
                 refiner_pipeline_type_str: str = "StableDiffusionXLImg2ImgPipeline",
                 scheduler_type_str: str = "LMSDiscreteScheduler",
                 use_karras_sigmas: bool = False,
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
                 image_path: str = "ref_image.png",
                 mask_path: str = "mask.png",
                 ):
        self.base_pipe_model = base_pipe_model
        self.refiner_pipe_model = refiner_pipe_model
        self.torch_dtype_str = torch_dtype_str
        self.base_pipeline_type_str = base_pipeline_type_str
        self.refiner_pipeline_type_str = refiner_pipeline_type_str
        self.scheduler_type_str = scheduler_type_str
        self.use_karras_sigmas = use_karras_sigmas
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
    def image(self):return Image.open(self.image_path) if self.base_pipeline_type == StableDiffusionXLImg2ImgPipeline or self.base_pipeline_type == StableDiffusionXLInpaintPipeline else None
    @property
    def mask(self):return Image.open(self.mask_path) if self.base_pipeline_type == StableDiffusionXLInpaintPipeline else None
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
                kwargs['mask_image'] = self.mask
        return kwargs
    
    @staticmethod
    def to_json(obj):
        if isinstance(obj, SDXLConfig):
            return obj.__dict__
    @classmethod
    def from_json(cls, dict: dict):
            return cls(**dict)
    @staticmethod
    def load_config(configPath: str):
         with open(configPath, "r") as read_file:
            return json.load(read_file, object_hook=SDXLConfig.from_json)
    
    def save_config(self, configPath: str):
         with open(configPath, "w") as write_file:
            json.dump(self, write_file, skipkeys=True, indent=1, default=SDXLConfig.to_json)

    def set_ui(self):
        # TODO: not best workaround to get variable name
        def f(x, name): setattr(self, name, x)
        def g(f_name): return f_name.split('=')[0].split('.')[1]

        def on_base_pipe_dropdown_changed(change): on_base_pipe_changed(change.new)
        def on_base_pipe_changed(value): 
            right_box[g(f'{self.width=}')].disabled = False if value != "StableDiffusionXLImg2ImgPipeline" else True
            right_box[g(f'{self.height=}')].disabled = False if value != "StableDiffusionXLImg2ImgPipeline" else True
            right_box[g(f'{self.strength=}')].disabled = False if value != "StableDiffusionXLPipeline" else True
            right_box[g(f'{self.image_path=}')].disabled = False if value != "StableDiffusionXLPipeline" else True
            right_box[g(f'{self.mask_path=}')].disabled = False if value == "StableDiffusionXLInpaintPipeline" else True
            right_box[g(f'{self.use_refiner=}')].disabled = False if value == "StableDiffusionXLPipeline" else True
            right_box[g(f'{self.high_noise_frac=}')].disabled = False if value == "StableDiffusionXLPipeline" else True
            
            if value != "StableDiffusionXLPipeline":
                right_box[g(f'{self.use_refiner=}')].value = False
                right_box[g(f'{self.high_noise_frac=}')].value = 1.0
            
        base_pipeline_str_key = g(f'{self.base_pipeline_type_str=}')
        prompts_layout = Layout( width='auto', height='100%')
        items_style = {'description_width': 'initial'}
        items_layout = Layout( width='auto')
        left_box = {
            # prompts
            g(f'{self.prompt=}'):widgets.Textarea(value=self.prompt, placeholder='Type positive1...', description='Prompt1:', style=items_style, layout=prompts_layout),
            g(f'{self.prompt_2=}'):widgets.Textarea(value=self.prompt_2, placeholder='Type positive2...', description='Prompt2:', style=items_style, layout=prompts_layout),
            g(f'{self.negative_prompt=}'):widgets.Textarea(value=self.negative_prompt, placeholder='Type negative1...', description='Negative Prompt1:', style=items_style, layout=prompts_layout),
            g(f'{self.negative_prompt_2=}'):widgets.Textarea(value=self.negative_prompt_2, placeholder='Type negative2...', description='Negative Prompt2:', style=items_style, layout=prompts_layout),
            g(f'{self.use_compel=}'):widgets.Checkbox(value=self.use_compel, description="Use Compel", indent=False, style=items_style, layout=items_layout)
        }
        right_box = {
            # models, precisions, schedulers
            g(f'{self.base_pipe_model=}'):widgets.Text(value=self.base_pipe_model, placeholder='', description='Base model:', style=items_style, layout=items_layout),
            g(f'{self.refiner_pipe_model=}'):widgets.Text(value=self.refiner_pipe_model, placeholder='', description='Refiner model:', style=items_style, layout=items_layout),
            g(f'{self.torch_dtype_str=}'):widgets.Dropdown(value=self.torch_dtype_str, options=PRECISION.keys(), description='dtype:', style=items_style, layout=items_layout),
            g(f'{self.variant=}'):widgets.Text(value=self.variant, placeholder='', description='Variant:', style=items_style, layout=items_layout),
            g(f'{self.use_safetensors=}'):widgets.Checkbox(value=self.use_safetensors, description="Use safetensors", indent=False, style=items_style, layout=items_layout),
            base_pipeline_str_key:widgets.Dropdown(value=self.base_pipeline_type_str, options=BASE_PIPELINES.keys(), description='Base type:', style=items_style, layout=items_layout),
            g(f'{self.refiner_pipeline_type_str=}'):widgets.Dropdown(value=self.refiner_pipeline_type_str, options=REFINER_PIPELINES.keys(), description='Refiner type:', style=items_style, layout=items_layout),
            g(f'{self.scheduler_type_str=}'):widgets.Dropdown(value=self.scheduler_type_str, options=SCHEDULERS.keys(), description='Scheduler type:', style=items_style, layout=items_layout),
            g(f'{self.use_karras_sigmas=}'):widgets.Checkbox(value=self.use_karras_sigmas, description="Use karras sigmas", indent=False, style=items_style, layout=items_layout),
            
            # inference properties
            g(f'{self.num_inference_steps=}'):widgets.IntSlider(value=self.num_inference_steps, min=10, max=100, step=5, description="Num inference steps:", continuous_update=False, style=items_style, layout=items_layout),
            g(f'{self.width=}'):widgets.IntSlider(value=self.width, min=512, max=1024, step=64, description="Width:", continuous_update=False, style=items_style, layout=items_layout),
            g(f'{self.height=}'):widgets.IntSlider(value=self.height, min=512, max=1024, step=64, description="Height:", continuous_update=False, style=items_style, layout=items_layout),
            g(f'{self.guidance_scale=}'):widgets.FloatSlider(value=self.guidance_scale, min=0, max=10, step=0.25, description="Guidance scale:", continuous_update=False, style=items_style, layout=items_layout),
            g(f'{self.seed=}'):widgets.IntSlider(value=self.seed, min=0, max=1000000, step=1, description="Seed:", continuous_update=False, style=items_style, layout=items_layout),
            g(f'{self.high_noise_frac=}'):widgets.FloatSlider(value=self.high_noise_frac, min=0, max=1, step=0.05, description="High noise frac:", continuous_update=False, style=items_style, layout=items_layout),

            # refiner
            g(f'{self.use_refiner=}'):widgets.Checkbox(value=self.use_refiner, description="Use refiner", indent=False, style=items_style, layout=items_layout),

            # img2img and inpaint
            g(f'{self.strength=}'):widgets.FloatSlider(value=self.strength, min=0, max=1, step=0.05, description="Strength:", continuous_update=False, style=items_style, layout=items_layout),
            g(f'{self.image_path=}'):widgets.Text(value=self.image_path, placeholder='', description='Image path:', style=items_style, layout=items_layout),
            g(f'{self.mask_path=}'):widgets.Text(value=self.mask_path, placeholder='', description='Mask path:', style=items_style, layout=items_layout)
        }
        
        # on base pipeline type changed
        on_base_pipe_changed(self.base_pipeline_type_str)
        right_box[base_pipeline_str_key].observe(on_base_pipe_dropdown_changed, names='value')
        
        # layout
        ui = HBox([VBox(list(left_box.values()), layout=Layout(width='50%', margin='10px 20px 10px 0')),
                   VBox(list(right_box.values()), layout=Layout(width='50%', margin='10px 0 10px 0'))])
        boxes = left_box | right_box
        [interactive_output(f, {'x':x, 'name':fixed(name)}) for name,x in boxes.items()]
        display(ui)