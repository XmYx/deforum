
import os, sys
import secrets
import subprocess

import numpy as np
import torch
from PIL import Image

from deforum.cmd import root_path, comfy_path
from deforum.pipelines.cond_tools import blend_tensors
from deforum.rng.rng import ImageRNG

# 1. Check if the "src" directory exists
if not os.path.exists(os.path.join(root_path, "src")):
    os.makedirs(os.path.join(root_path, 'src'))
# 2. Check if "ComfyUI" exists
if not os.path.exists(comfy_path):
    # Clone the repository if it doesn't exist
    subprocess.run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI", comfy_path])
else:
    # 3. If "ComfyUI" does exist, check its commit hash
    current_folder = os.getcwd()
    os.chdir(comfy_path)
    current_commit = subprocess.getoutput("git rev-parse HEAD")

    # 4. Reset to the desired commit if necessary
    if current_commit != "4185324":  # replace with the full commit hash if needed
        subprocess.run(["git", "fetch", "origin"])
        subprocess.run(["git", "reset", "--hard", "4185324"])  # replace with the full commit hash if needed
        subprocess.run(["git", "pull", "origin", "master"])
    os.chdir(current_folder)
comfy_path = os.path.join(root_path, "src/ComfyUI")
sys.path.append(comfy_path)


from comfy.sample import sample as sample_k
from comfy import model_management, controlnet
from comfy.sd import load_checkpoint_guess_config


class ComfyDeforumGenerator:

    def __init__(self):

        model_management.vram_state = model_management.vram_state.HIGH_VRAM
        self.clip_skip = -2
        self.device = "cuda"
        self.load_model()

        model_name = os.path.join(root_path, "models/controlnet/diffusers_xl_canny_mid.safetensors")

        self.controlnet = controlnet.load_controlnet(model_name)

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
    def load_model(self):
        ckpt_path = os.path.join(root_path, "models/checkpoints/protovisionXLHighFidelity3D_release0620Bakedvae.safetensors")
        self.model, self.clip, self.vae, clipvision = load_checkpoint_guess_config(ckpt_path, output_vae=True,
                                                                             output_clip=True,
                                                                             embedding_directory="models/embeddings")
    def generate(self,
                 prompt=None,
                 next_prompt=None,
                 prompt_blend=None,
                 negative_prompt="",
                 steps=25,
                 scale=7.5,
                 sampler_name="dpmpp_2m_sde_gpu",
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
                 seed_resize_from_w=1024):

        SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]

        if seed == -1:
            seed = secrets.randbelow(18446744073709551615)


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


        #from nodes import common_ksampler as ksampler

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

    rng_noise = noise.next().detach().cpu()

    noise = rng_noise.clone()


    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    # callback = latent_preview.prepare_callback(model, steps)
    # disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED


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



