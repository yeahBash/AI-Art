import torch
# for parameters ui
from ipywidgets import interactive_output, fixed, Layout
import ipywidgets as widgets
# for config
import json
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, KandinskyV22CombinedPipeline
from diffusers import EulerDiscreteScheduler, DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from PIL import Image
from dataclasses import dataclass, field
from typing import Union

@dataclass
class Params:
    kwargs: list[str] = field(default_factory=list)
    exclude_ui: list[str] = field(default_factory=list)
    change_ui_values: dict[str, Union[int, bool, float, str]] = field(default_factory=dict)

@dataclass
class PipelineWrapper:
    type: DiffusionPipeline
    params: Params

BASE_MODELS = {"stabilityai/stable-diffusion-xl-base-1.0":
                                        Params(kwargs=['variant'],
                                               exclude_ui=[],
                                               change_ui_values={"guidance_scale":7.5,"timestep_spacing":"linspace",
                                                                 "num_inference_steps":40,"width":768,"height":768}), 
               "stabilityai/sdxl-turbo":
                                        Params(kwargs=['variant'], 
                                               exclude_ui=["negative_prompt","negative_prompt_2","high_noise_frac",
                                                           "guidance_scale","timestep_spacing"],
                                               change_ui_values={"guidance_scale":0.0,"timestep_spacing":"trailing","num_inference_steps":1,
                                                                 "high_noise_frac":1.0,"width":512,"height":512,"use_refiner":False}), 
               "kandinsky-community/kandinsky-2-2-decoder":
                                        Params(kwargs=[],
                                               exclude_ui=["prompt_2","negative_prompt_2","high_noise_frac","use_compel"],
                                               change_ui_values={"guidance_scale":7.5,"timestep_spacing":"linspace","num_inference_steps":40,
                                                                 "width":512,"height":512, "use_compel":False})}
TXT2IMG_PIPELINES = {"StableDiffusionXLPipeline":
                        PipelineWrapper(StableDiffusionXLPipeline, 
                                        Params(kwargs=['prompt_2','negative_prompt_2','denoising_end',
                                                       'width','height','image','strength','mask_image'],
                                               exclude_ui=["image_path","strength","mask_path"])),
                     "KandinskyV22CombinedPipeline":
                        PipelineWrapper(KandinskyV22CombinedPipeline, 
                                        Params(kwargs=['width','height'],
                                               exclude_ui=["image_path","strength","mask_path"]))}
IMG2IMG_PIPELINES = {"StableDiffusionXLImg2ImgPipeline":
                        PipelineWrapper(StableDiffusionXLImg2ImgPipeline, 
                                        Params(kwargs=['prompt_2','negative_prompt_2','denoising_end',
                                                       'image','strength'],
                                               exclude_ui=["width","height","mask_path"]))}
INPAINTING_PIPELINES = {"StableDiffusionXLInpaintPipeline":
                        PipelineWrapper(StableDiffusionXLInpaintPipeline, 
                                        Params(kwargs=['prompt_2','negative_prompt_2','denoising_end',
                                                       'width','height','image','strength','mask_image']))}

BASE_PIPELINES = TXT2IMG_PIPELINES | IMG2IMG_PIPELINES | INPAINTING_PIPELINES
REFINER_MODELS = ["stabilityai/stable-diffusion-xl-refiner-1.0"]
REFINER_PIPELINES = {"StableDiffusionXLImg2ImgPipeline":StableDiffusionXLImg2ImgPipeline}
SCHEDULERS = {"EulerDiscreteScheduler":EulerDiscreteScheduler, 
              "DDIMScheduler":DDIMScheduler, 
              "LMSDiscreteScheduler":LMSDiscreteScheduler,
              "DPMSolverMultistepScheduler":DPMSolverMultistepScheduler,
              "DPMSolverSDEScheduler":DPMSolverSDEScheduler}
SCHEDULER_TIMESTEP_SPACINGS = ["linspace", "leading", "trailing"]
PRECISION = {"torch.float16":torch.float16}
VARIANTS = ["fp16"]

@dataclass
class UIData:
    prompt: widgets.widgets.Textarea = None
    prompt_box: list[widgets.widget_core.CoreWidget] = None
    params_box: list[widgets.widget_core.CoreWidget] = None

@dataclass
class SDXLConfig:
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_model: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    torch_dtype_str: str = "torch.float16"
    base_pipeline_type_str: str = "StableDiffusionXLPipeline"
    refiner_pipeline_type_str: str = "StableDiffusionXLImg2ImgPipeline"
    scheduler_type_str: str = "LMSDiscreteScheduler"
    use_karras_sigmas: bool = True
    timestep_spacing: str = "linspace"
    variant: str = "fp16"
    use_safetensors: bool = True
    #safety_checker = None
    prompt: str = None
    prompt_2: str = None
    negative_prompt: str = None
    negative_prompt_2: str = None
    use_compel: bool = False
    num_inference_steps: int = 40
    width: int = 768
    height: int = 768
    guidance_scale: float = 7.5
    high_noise_frac: float = 1.0
    seed: int = 12345
    use_refiner: bool = False
    strength: float = 0.3
    image_path: str = "ref_image.png"
    mask_path: str = "mask.png"

    @property
    def torch_dtype(self):return PRECISION[self.torch_dtype_str]
    @property
    def base_pipeline_type(self):return BASE_PIPELINES[self.base_pipeline_type_str].type
    @property
    def refiner_pipeline_type(self):return REFINER_PIPELINES[self.refiner_pipeline_type_str]
    @property
    def prompt_arg(self) -> str:return self.prompt if self.prompt != "" and not self.use_compel else None
    @property
    def prompt_2_arg(self) -> str:return self.prompt_2 if self.prompt_2 != "" and not self.use_compel else None
    @property
    def negative_prompt_arg(self) -> str:return self.negative_prompt if self.negative_prompt != "" and not self.use_compel else None
    @property
    def negative_prompt_2_arg(self) -> str:return self.negative_prompt_2 if self.negative_prompt_2 != "" and not self.use_compel else None
    @property
    def scheduler_type(self):return SCHEDULERS[self.scheduler_type_str]
    @property
    def image(self):return Image.open(self.image_path) if self.base_pipeline_type in IMG2IMG_PIPELINES or self.base_pipeline_type in INPAINTING_PIPELINES else None
    @property
    def mask_image(self):return Image.open(self.mask_path) if self.base_pipeline_type in INPAINTING_PIPELINES else None
    @property
    def model_kwargs(self): # choose additional arguments for models
        kwargs = {}
        for arg in BASE_MODELS[self.base_model].kwargs:
            kwargs[arg] = getattr(self, arg)
        return kwargs
    @property
    def pipe_kwargs(self): # choose additional arguments for pipelines
        kwargs = {}
        for arg in BASE_PIPELINES[self.base_pipeline_type_str].params.kwargs:
            match arg:
                case 'prompt_2':
                    config_attr = 'prompt_2_arg'
                case 'negative_prompt_2':
                    config_attr = 'negative_prompt_2_arg'
                case 'denoising_end':
                    config_attr = 'high_noise_frac'
                case _:
                    config_attr = arg
            kwargs[arg] = getattr(self, config_attr)
        return kwargs
    
    @property
    def is_turbo(self):return self.base_model == "stabilityai/sdxl-turbo"
    @property
    def is_base_to_refiner(self):return self.use_refiner and self.high_noise_frac == 1.0

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
    def model_params_equals(self, obj) -> bool: # TODO: refactor
        if isinstance(obj, SDXLConfig):
            return (self.base_model == obj.base_model and 
                    self.refiner_model == obj.refiner_model and 
                    self.torch_dtype_str == obj.torch_dtype_str and 
                    self.base_pipeline_type_str == obj.base_pipeline_type_str and 
                    self.refiner_pipeline_type_str == obj.refiner_pipeline_type_str and 
                    self.scheduler_type_str == obj.scheduler_type_str and 
                    self.use_karras_sigmas == obj.use_karras_sigmas and 
                    self.timestep_spacing == obj.timestep_spacing and 
                    self.variant == obj.variant and 
                    self.use_safetensors == obj.use_safetensors)
        
    def get_ui(self) -> UIData:
        def set_config_attr(x, name): setattr(self, name, x)
        def should_disable_ui_element(ui_key, model, pipe) -> bool: return ui_key in BASE_PIPELINES[pipe].params.exclude_ui or ui_key in BASE_MODELS[model].exclude_ui
        def on_base_model_dropdown_changed(change): 
            disable_based_on_model(change.new)
            change_values_based_on_model(change.new)
        def disable_based_on_model(value): 
            for ui_key, widget in boxes.items():
                widget.disabled = should_disable_ui_element(ui_key, model=value, pipe=self.base_pipeline_type_str)
        def change_values_based_on_model(value): 
            for ui_key,ui_value in BASE_MODELS[value].change_ui_values.items():
                boxes[ui_key].value = ui_value
        def on_base_pipe_dropdown_changed(change): 
            disable_based_on_pipe(change.new)
            change_values_based_on_pipe(change.new)
        def disable_based_on_pipe(value): 
            for ui_key, widget in boxes.items():
                widget.disabled = should_disable_ui_element(ui_key, model=self.base_model, pipe=value)
        def change_values_based_on_pipe(value): 
            for ui_key,ui_value in BASE_PIPELINES[value].params.change_ui_values.items():
                boxes[ui_key].value = ui_value
        
        prompts_layout = Layout( width='auto', height='100%')
        items_style = {'description_width': 'initial'}
        items_layout = Layout( width='auto')
        prompt_box = {
            # prompts
            "prompt":widgets.Textarea(value=self.prompt, placeholder='Type positive1...', description='Prompt1:', style=items_style, layout=prompts_layout),
            "prompt_2":widgets.Textarea(value=self.prompt_2, placeholder='Type positive2...', description='Prompt2:', style=items_style, layout=prompts_layout),
            "negative_prompt":widgets.Textarea(value=self.negative_prompt, placeholder='Type negative1...', description='Negative Prompt1:', style=items_style, layout=prompts_layout),
            "negative_prompt_2":widgets.Textarea(value=self.negative_prompt_2, placeholder='Type negative2...', description='Negative Prompt2:', style=items_style, layout=prompts_layout),
            "use_compel":widgets.Checkbox(value=self.use_compel, description="Use Compel", indent=False, style=items_style, layout=items_layout)
        }
        params_box = {
            # models, precisions, schedulers
            "base_model":widgets.Dropdown(value=self.base_model, options=BASE_MODELS.keys(), description='Base model:', style=items_style, layout=items_layout),
            "refiner_model":widgets.Dropdown(value=self.refiner_model, options=REFINER_MODELS, description='Refiner model:', style=items_style, layout=items_layout),
            "torch_dtype_str":widgets.Dropdown(value=self.torch_dtype_str, options=PRECISION.keys(), description='dtype:', style=items_style, layout=items_layout),
            "variant":widgets.Dropdown(value=self.variant, options=VARIANTS, description='Variant:', style=items_style, layout=items_layout),
            "use_safetensors":widgets.Checkbox(value=self.use_safetensors, description="Use safetensors", indent=False, style=items_style, layout=items_layout),
            "base_pipeline_type_str":widgets.Dropdown(value=self.base_pipeline_type_str, options=BASE_PIPELINES.keys(), description='Base type:', style=items_style, layout=items_layout),
            "refiner_pipeline_type_str":widgets.Dropdown(value=self.refiner_pipeline_type_str, options=REFINER_PIPELINES.keys(), description='Refiner type:', style=items_style, layout=items_layout),
            "scheduler_type_str":widgets.Dropdown(value=self.scheduler_type_str, options=SCHEDULERS.keys(), description='Scheduler type:', style=items_style, layout=items_layout),
            "timestep_spacing":widgets.Dropdown(value=self.timestep_spacing, options=SCHEDULER_TIMESTEP_SPACINGS, description='Scheduler timestep spacing:', style=items_style, layout=items_layout),
            "use_karras_sigmas":widgets.Checkbox(value=self.use_karras_sigmas, description="Use karras sigmas", indent=False, style=items_style, layout=items_layout),
            
            # inference properties
            "num_inference_steps":widgets.IntSlider(value=self.num_inference_steps, min=1, max=100, step=1, description="Num inference steps:", continuous_update=False, style=items_style, layout=items_layout),
            "width":widgets.IntSlider(value=self.width, min=512, max=1024, step=64, description="Width:", continuous_update=False, style=items_style, layout=items_layout),
            "height":widgets.IntSlider(value=self.height, min=512, max=1024, step=64, description="Height:", continuous_update=False, style=items_style, layout=items_layout),
            "guidance_scale":widgets.FloatSlider(value=self.guidance_scale, min=0, max=10, step=0.25, description="Guidance scale:", continuous_update=False, style=items_style, layout=items_layout),
            "seed":widgets.IntSlider(value=self.seed, min=0, max=1000000, step=1, description="Seed:", continuous_update=False, style=items_style, layout=items_layout),
            "high_noise_frac":widgets.FloatSlider(value=self.high_noise_frac, min=0, max=1, step=0.05, description="High noise frac:", continuous_update=False, style=items_style, layout=items_layout),

            # refiner
            "use_refiner":widgets.Checkbox(value=self.use_refiner, description="Use refiner", indent=False, style=items_style, layout=items_layout),

            # img2img and inpaint
            "strength":widgets.FloatSlider(value=self.strength, min=0, max=1, step=0.05, description="Strength:", continuous_update=False, style=items_style, layout=items_layout),
            "image_path":widgets.Text(value=self.image_path, placeholder='', description='Image path:', style=items_style, layout=items_layout),
            "mask_path":widgets.Text(value=self.mask_path, placeholder='', description='Mask path:', style=items_style, layout=items_layout)
        }

        boxes = prompt_box | params_box
        # on base model changed
        disable_based_on_model(self.base_model)
        params_box["base_model"].observe(on_base_model_dropdown_changed, names='value')
        # on base pipeline type changed
        disable_based_on_pipe(self.base_pipeline_type_str)
        params_box["base_pipeline_type_str"].observe(on_base_pipe_dropdown_changed, names='value')

        # layout
        [interactive_output(set_config_attr, {'x':x, 'name':fixed(name)}) for name,x in boxes.items()]
        return UIData(prompt_box["prompt"], list(prompt_box.values()), list(params_box.values()))