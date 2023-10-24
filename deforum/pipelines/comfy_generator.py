
import os, sys
import secrets
import subprocess

import numpy as np
import torch
import torchsde
from PIL import Image

from deforum import default_cache_folder, fetch_and_download_model
from deforum.cmd import root_path, comfy_path
from deforum.pipelines.cond_tools import blend_tensors
from deforum.rng.rng import ImageRNG

# 1. Check if the "src" directory exists
# if not os.path.exists(os.path.join(root_path, "src")):
#     os.makedirs(os.path.join(root_path, 'src'))
# # 2. Check if "ComfyUI" exists
# if not os.path.exists(comfy_path):
#     # Clone the repository if it doesn't exist
#     subprocess.run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI", comfy_path])
# else:
#     # 3. If "ComfyUI" does exist, check its commit hash
#     current_folder = os.getcwd()
#     os.chdir(comfy_path)
#     current_commit = subprocess.getoutput("git rev-parse HEAD")
#
#     # 4. Reset to the desired commit if necessary
#     if current_commit != "4185324":  # replace with the full commit hash if needed
#         subprocess.run(["git", "fetch", "origin"])
#         subprocess.run(["git", "reset", "--hard", "b935bea3a0201221eca7b0337bc60a329871300a"])  # replace with the full commit hash if needed
#         subprocess.run(["git", "pull", "origin", "master"])
#     os.chdir(current_folder)

comfy_path = os.path.join(root_path, "src/ComfyUI")
sys.path.append(comfy_path)

from collections import namedtuple


# Define the namedtuple structure based on the properties identified
CLIArgs = namedtuple('CLIArgs', [
    'cpu',
    'normalvram',
    'lowvram',
    'novram',
    'highvram',
    'gpu_only',
    'disable_xformers',
    'use_pytorch_cross_attention',
    'use_split_cross_attention',
    'use_quad_cross_attention',
    'fp16_vae',
    'bf16_vae',
    'fp32_vae',
    'force_fp32',
    'force_fp16',
    'disable_smart_memory',
    'disable_ipex_optimize',
    'listen',
    'port',
    'enable_cors_header',
    'extra_model_paths_config',
    'output_directory',
    'temp_directory',
    'input_directory',
    'auto_launch',
    'disable_auto_launch',
    'cuda_device',
    'cuda_malloc',
    'disable_cuda_malloc',
    'dont_upcast_attention',
    'bf16_unet',
    'directml',
    'preview_method',
    'dont_print_server',
    'quick_test_for_ci',
    'windows_standalone_build',
    'disable_metadata'
])

# Update the mock args object with default values for the new properties
mock_args = CLIArgs(
    cpu=False,
    normalvram=False,
    lowvram=False,
    novram=False,
    highvram=True,
    gpu_only=True,
    disable_xformers=True,
    use_pytorch_cross_attention=True,
    use_split_cross_attention=False,
    use_quad_cross_attention=False,
    fp16_vae=True,
    bf16_vae=False,
    fp32_vae=False,
    force_fp32=False,
    force_fp16=False,
    disable_smart_memory=False,
    disable_ipex_optimize=True,
    listen="127.0.0.1",
    port=8188,
    enable_cors_header=None,
    extra_model_paths_config=None,
    output_directory=None,
    temp_directory=None,
    input_directory=None,
    auto_launch=False,
    disable_auto_launch=True,
    cuda_device=0,
    cuda_malloc=False,
    disable_cuda_malloc=True,
    dont_upcast_attention=True,
    bf16_unet=True,
    directml=None,
    preview_method="none",
    dont_print_server=True,
    quick_test_for_ci=False,
    windows_standalone_build=False,
    disable_metadata=False
)
from comfy.cli_args import LatentPreviewMethod as lp
class MockCLIArgsModule:
    args = mock_args
    LatentPreviewMethod = lp

# Add the mock module to sys.modules under the name 'comfy.cli_args'
sys.modules['comfy.cli_args'] = MockCLIArgsModule()

import comfy.sd
import comfy.diffusers_load

class DeforumBatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if torch.abs(t0 - t1) < 1e-6:  # or some other small value
            # Handle this case, e.g., return a zero tensor of appropriate shape
            return torch.zeros_like(t0)

        if self.cpu_tree:
            w = torch.stack(
                [tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (
                            self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]


import comfy.k_diffusion.sampling

comfy.k_diffusion.sampling.BatchedBrownianTree = DeforumBatchedBrownianTree

class ComfyDeforumGenerator:

    def __init__(self, model_path:str=None, lcm=False):
        #from comfy import model_management, controlnet

        #model_management.vram_state = model_management.vram_state.HIGH_VRAM
        self.clip_skip = -2
        self.device = "cuda"

        if not lcm:
            if model_path == None:
                models_dir = os.path.join(default_cache_folder)
                fetch_and_download_model(125703, default_cache_folder)
                model_path = os.path.join(models_dir, "protovisionXLHighFidelity3D_release0620Bakedvae.safetensors")

            self.load_model(model_path)

            self.pipeline_type = "comfy"

        else:
            self.load_lcm()

            self.pipeline_type = "diffusers_lcm"

        # self.controlnet = controlnet.load_controlnet(model_name)

        self.rng = None
    def encode_latent(self, latent, subseed, subseed_strength):
        with torch.inference_mode():
            latent = latent.to(torch.float32)
            latent = self.vae.encode_tiled(latent[:,:,:,:3])
            latent = latent.to("cuda")

        return {"samples":latent}

    def generate_latent(self, width, height, seed, subseed, subseed_strength, seed_resize_from_h=None, seed_resize_from_w=None):
        shape = [4, height // 8, width // 8]
        if self.rng == None:
            self.rng = ImageRNG(shape=shape, seeds=[seed], subseeds=[subseed], subseed_strength=subseed_strength, seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
        noise = self.rng.next()
        # noise = torch.zeros([1, 4, width // 8, height // 8])
        return {"samples":noise}

    def get_conds(self, prompt):
        with torch.inference_mode():
            clip_skip = -2
            if self.clip_skip != clip_skip or self.clip.layer_idx != clip_skip:
                self.clip.layer_idx = clip_skip
                self.clip.clip_layer(clip_skip)
                self.clip_skip = clip_skip

            tokens = self.clip.tokenize(prompt)
            cond, pooled = self.clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]
    def load_model(self, model_path:str):


        # comfy.sd.load_checkpoint_guess_config

        self.model, self.clip, self.vae, clipvision = comfy.sd.load_checkpoint_guess_config(model_path, output_vae=True,
                                                                             output_clip=True,
                                                                             embedding_directory="models/embeddings")
        # model_path = os.path.join(root_path, "models/checkpoints/SSD-1B")
        # self.model, self.clip, self.vae = comfy.diffusers_load.load_diffusers(model_path, output_vae=True, output_clip=True,
        #                                     embedding_directory="models/embeddings")
    def load_lcm(self):
        from deforum.lcm.lcm_pipeline import LatentConsistencyModelPipeline

        from deforum.lcm.lcm_scheduler import LCMScheduler
        self.scheduler = LCMScheduler.from_pretrained(
            os.path.join(root_path, "configs/lcm_scheduler.json"))

        self.pipe = LatentConsistencyModelPipeline.from_pretrained(
            pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
            scheduler=self.scheduler
        ).to("cuda")
        from deforum.lcm.lcm_i2i_pipeline import LatentConsistencyModelImg2ImgPipeline
        # self.img2img_pipe = LatentConsistencyModelImg2ImgPipeline(
        #     unet=self.pipe.unet,
        #     vae=self.pipe.vae,
        #     text_encoder=self.pipe.text_encoder,
        #     tokenizer=self.pipe.tokenizer,
        #     scheduler=self.pipe.scheduler,
        #     feature_extractor=self.pipe.feature_extractor,
        #     safety_checker=None,
        # )
        self.img2img_pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
            safety_checker=None,
        ).to("cuda")



    def __call__(self,
                 prompt=None,
                 next_prompt=None,
                 prompt_blend=None,
                 negative_prompt="",
                 steps=25,
                 scale=7.5,
                 sampler_name="dpmpp_2m_sde",
                 scheduler="karras",
                 width=None,
                 height=None,
                 seed=-1,
                 strength=0.65,
                 init_image=None,
                 subseed=-1,
                 subseed_strength=0.6,
                 cnet_image=None,
                 cond=None,
                 n_cond=None,
                 return_latent=None,
                 latent=None,
                 last_step=None,
                 seed_resize_from_h=1024,
                 seed_resize_from_w=1024,
                 *args,
                 **kwargs):

        SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
        if self.pipeline_type == "comfy":
            if seed == -1:
                seed = secrets.randbelow(18446744073709551615)

            print("I wanna use", strength)

            if strength > 1:
                strength = 1.0
                init_image = None
            if strength == 0.0:
                strength = 1.0
            if subseed == -1:
                subseed = secrets.randbelow(18446744073709551615)

            if cnet_image is not None:
                cnet_image = torch.from_numpy(np.array(cnet_image).astype(np.float32) / 255.0).unsqueeze(0)

            if init_image is None:
                if width == None:
                    width = 1024
                if height == None:
                    height = 960
                latent = self.generate_latent(width, height, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w)

            else:
                latent = torch.from_numpy(np.array(init_image).astype(np.float32) / 255.0).unsqueeze(0)

                latent = self.encode_latent(latent, subseed, subseed_strength)


                #Implement Img2Img
            if prompt is not None:
                cond = self.get_conds(prompt)
                n_cond = self.get_conds(negative_prompt)

            if next_prompt is not None:
                if next_prompt != prompt and next_prompt != "":
                    if 0.0 < prompt_blend < 1.0:
                        next_cond = self.get_conds(next_prompt)
                        cond = blend_tensors(cond[0], next_cond[0], blend_value=prompt_blend)



            if cnet_image is not None:
                cond = apply_controlnet(cond, self.controlnet, cnet_image, 1.0)


            # from nodes import common_ksampler as ksampler

            last_step = int((1-strength) * steps) + 1 if strength != 1.0 else steps
            last_step = steps if last_step == None else last_step
            sample = common_ksampler_with_custom_noise(model=self.model,
                                                       seed=seed,
                                                       steps=steps,
                                                       cfg=scale,
                                                       sampler_name=sampler_name,
                                                       scheduler=scheduler,
                                                       positive=cond,
                                                       negative=n_cond,
                                                       latent=latent,
                                                       denoise=strength,
                                                       disable_noise=False,
                                                       start_step=0,
                                                       last_step=last_step,
                                                       force_full_denoise=True,
                                                       noise=self.rng)



            decoded = self.decode_sample(sample[0]["samples"])

            np_array = np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0]
            image = Image.fromarray(np_array)
            #image = Image.fromarray(np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0])
            image = image.convert("RGB")
            if return_latent:
                return sample[0]["samples"], image
            else:
                return image
        elif self.pipeline_type == "diffusers_lcm":
            if init_image is None:
                image = self.pipe(
                        prompt=prompt,
                        width=width,
                        height=height,
                        guidance_scale=scale,
                        num_inference_steps=int(steps/5),
                        num_images_per_prompt=1,
                        lcm_origin_steps=50,
                        output_type="pil",
                    ).images[0]
            else:
                # init_image = np.array(init_image)
                # init_image = Image.fromarray(init_image)
                image = self.img2img_pipe(
                        prompt=prompt,
                        strength=strength,
                        image=init_image,
                        width=width,
                        height=height,
                        guidance_scale=scale,
                        num_inference_steps=int(steps/5),
                        num_images_per_prompt=1,
                        lcm_origin_steps=50,
                        output_type="pil",
                    ).images[0]

            return image


    def decode_sample(self, sample):
        with torch.inference_mode():
            sample = sample.to(torch.float32)
            self.vae.first_stage_model.cuda()
            decoded = self.vae.decode_tiled(sample).detach()

        return decoded

def common_ksampler_with_custom_noise(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                                      denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                                      force_full_denoise=False, noise=None):
    latent_image = latent["samples"]
    if noise is not None:
        rng_noise = noise.next().detach().cpu()
        noise = rng_noise.clone()
    else:
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            from comfy.sample import prepare_noise
            noise = prepare_noise(latent_image, seed, batch_inds)


    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    # callback = latent_preview.prepare_callback(model, steps)
    # disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    from comfy.sample import sample as sample_k

    samples = sample_k(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                                  last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=None,
                                  disable_pbar=False, seed=seed)
    out = latent.copy()
    out["samples"] = samples

    return (out,)

def apply_controlnet(conditioning, control_net, image, strength):
    with torch.inference_mode():
        if strength == 0:
            return (conditioning, )

        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
    return c




