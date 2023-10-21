import torch
# for saving results
import shutil
from datetime import datetime
# for image_grid
from PIL import Image

# utilities methods
@staticmethod
def to_cuda(pipe, start_mess, end_mess):
    if(torch.cuda.is_available()):
        print(start_mess)
        pipe = pipe.to("cuda")
    else:
        print("CUDA IS NOT AVAILABLE")
    print(end_mess)

@staticmethod
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

@staticmethod
def postprocess_latent(pipe, latent):
    vae_output = pipe.vae.decode(
        latent.images / pipe.vae.config.scaling_factor, return_dict=False
    )[0].detach()
    return pipe.image_processor.postprocess(vae_output, output_type="pil")[0]

@staticmethod
def save_results(image:Image, config_path:str, ref_image:Image = None, mask_image:Image = None):
    now_date = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    shutil.copyfile(config_path, f"results\{now_date}.json")
    image.save(f"results\{now_date}.png")
    if ref_image is not None:
        ref_image.save(f"results\\ref_{now_date}.png")
    if mask_image is not None:
        mask_image.save(f"results\mask_{now_date}.png")