import os, sys
import subprocess
import secrets
import time
import argparse

from types import SimpleNamespace
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, WebSocket, Depends

from deforum.FILM.interpolator import Interpolator
from deforum.general_utils import substitute_placeholders
from deforum.main import Deforum
from deforum.animation.new_args import DeforumArgs, DeforumAnimArgs, ParseqArgs, LoopArgs, RootArgs, DeforumOutputArgs
from pydantic import BaseModel
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
comfy_path = os.path.join(root_path, "src/ComfyUI")
sys.path.extend([os.path.join(os.getcwd(), "deforum", "exttools")])

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



class ComfyDeforumGenerator:

    def __init__(self):
        from comfy import model_management, controlnet

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
        from comfy.sd import load_checkpoint_guess_config
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

frames = []
cadence_frames = []

class Settings(BaseModel):
    file_content: str

async def get_deforum():
    deforum = setup_deforum()
    return deforum

def get_canny_image(image):
    import cv2
    import numpy as np
    image = cv2.Canny(np.array(image), 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def keyframeExamples():
    return '''{
    "0": "https://deforum.github.io/a1/Gi1.png",
    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",
    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",
    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",
    "max_f-20": "https://deforum.github.io/a1/Gi1.png"
}'''

def DeforumAnimPrompts():
    return r"""{
    "0": "abstract art of the sky, stars, galaxy, planets",
    "120": "fluid abstract painting, the galaxies twirling",
    "200": "abstract overview of the planet earth, highly detailed artwork",
    "280": "cinematic motion in an abstract twirl"
}
    """

import json

def merge_dicts_from_txt(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
        data_dict = json.loads(content)

    return data_dict
def extract_values(args):
    return {key: value['value'] for key, value in args.items()}


def save_as_gif(frames, filename):
    # Convert frames to gif
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # You can adjust this duration as needed
        loop=0,
    )

def save_as_h264(frames, filename, audio_path=None, fps=12):
    from tqdm import tqdm
    import numpy as np
    import subprocess
    import time

    if len(frames) > 0:
        width = frames[0].size[0]
        height = frames[0].size[1]

        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
               '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-',
               '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0',
               '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23', '-an', filename]

        video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for frame in tqdm(frames, desc="Saving MP4 (ffmpeg)"):
            frame_np = np.array(frame)  # Convert the PIL image to numpy array
            video_writer.stdin.write(frame_np.tobytes())

        _, stderr = video_writer.communicate()

        if video_writer.returncode != 0:
            print(f"FFmpeg encountered an error: {stderr.decode('utf-8')}")
            return

        # if audio path is provided, merge the audio and the video
        if audio_path is not None:
            output_filename = f"output/mp4s/{time.strftime('%Y%m%d%H%M%S')}_with_audio.mp4"
            cmd = ['ffmpeg', '-y', '-i', filename, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', output_filename]

            result = subprocess.run(cmd, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Audio file merge failed from path {audio_path}\n{result.stderr.decode('utf-8')}")
    else:
        print("The buffer is empty, cannot save.")

def pil_to_tensor(image):
    """
    Convert a PIL Image to a PyTorch tensor.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # If you wish to normalize, you can use the following line:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def datacallback(data=None):

    if data:
        image = data.get("image")
        cadence_frame = data.get("cadence_frame")

    if image:
        frames.append(image)
    elif cadence_frame:
        cadence_frames.append(cadence_frame)


def setup_deforum(img_gen=None):
    args = SimpleNamespace(**extract_values(DeforumArgs()))
    anim_args = SimpleNamespace(**extract_values(DeforumAnimArgs()))
    parseg_args = SimpleNamespace(**extract_values(ParseqArgs()))
    loop_args = SimpleNamespace(**extract_values(LoopArgs()))
    root = SimpleNamespace(**RootArgs())

    output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}

    video_args = SimpleNamespace(**output_args_dict)
    controlnet_args = None
    deforum = Deforum(args, anim_args, video_args, parseg_args, loop_args, controlnet_args, root)
    setattr(deforum.loop_args, "init_images", "")
    animation_prompts = DeforumAnimPrompts()
    deforum.root.animation_prompts = json.loads(animation_prompts)
    deforum.animation_prompts = deforum.root.animation_prompts
    deforum.args.timestring = time.strftime('%Y%m%d%H%M%S')
    current_arg_list = [deforum.args, deforum.anim_args, deforum.video_args, deforum.parseq_args]
    full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")
    deforum.root.raw_batch_name = deforum.args.batch_name
    deforum.args.batch_name = substitute_placeholders(deforum.args.batch_name, current_arg_list, full_base_folder_path)
    deforum.args.outdir = os.path.join(full_base_folder_path, str(deforum.args.batch_name))
    if deforum.args.seed == -1 or deforum.args.seed == "-1":
        setattr(deforum.args, "seed", secrets.randbelow(999999999999999999))
        setattr(deforum.root, "raw_seed", int(deforum.args.seed))
        setattr(deforum.args, "seed_internal", 0)
    else:
        deforum.args.seed = int(deforum.args.seed)

    deforum.args.W = 1024
    deforum.args.H = 576
    deforum.args.steps = 20
    deforum.anim_args.animation_mode = "3D"
    deforum.anim_args.zoom = "0: (1.0)"
    deforum.anim_args.translate_z = "0: (4)"
    deforum.anim_args.strength_schedule = "0: (0.52)"
    deforum.anim_args.diffusion_cadence = 1
    deforum.anim_args.max_frames = 375
    deforum.video_args.store_frames_in_ram = True
    deforum.datacallback = datacallback

    return deforum


def reset_deforum(deforum):
    args = SimpleNamespace(**extract_values(DeforumArgs()))
    anim_args = SimpleNamespace(**extract_values(DeforumAnimArgs()))
    parseg_args = SimpleNamespace(**extract_values(ParseqArgs()))
    loop_args = SimpleNamespace(**extract_values(LoopArgs()))
    root = SimpleNamespace(**RootArgs())

    output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}

    video_args = SimpleNamespace(**output_args_dict)
    controlnet_args = None
    deforum.args = args
    deforum.anim_args = anim_args
    deforum.parseq_args = parseg_args
    deforum.loop_args = loop_args
    deforum.root = root
    deforum.video_args = video_args
    deforum.controlnet_args = None
    setattr(deforum.loop_args, "init_images", "")
    animation_prompts = DeforumAnimPrompts()
    deforum.root.animation_prompts = json.loads(animation_prompts)
    deforum.animation_prompts = deforum.root.animation_prompts
    deforum.args.timestring = time.strftime('%Y%m%d%H%M%S')
    current_arg_list = [deforum.args, deforum.anim_args, deforum.video_args, deforum.parseq_args]
    full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")
    deforum.root.raw_batch_name = deforum.args.batch_name
    deforum.args.batch_name = substitute_placeholders(deforum.args.batch_name, current_arg_list, full_base_folder_path)
    deforum.args.outdir = os.path.join(full_base_folder_path, str(deforum.args.batch_name))
    if deforum.args.seed == -1 or deforum.args.seed == "-1":
        setattr(deforum.args, "seed", secrets.randbelow(999999999999999999))
        setattr(deforum.root, "raw_seed", int(deforum.args.seed))
        setattr(deforum.args, "seed_internal", 0)
    else:
        deforum.args.seed = int(deforum.args.seed)

    deforum.args.W = 1024
    deforum.args.H = 576
    deforum.args.steps = 20
    deforum.anim_args.animation_mode = "3D"
    deforum.anim_args.zoom = "0: (1.0)"
    deforum.anim_args.translate_z = "0: (4)"
    deforum.anim_args.strength_schedule = "0: (0.52)"
    deforum.anim_args.diffusion_cadence = 1
    deforum.anim_args.max_frames = 375
    deforum.video_args.store_frames_in_ram = True
    # deforum.datacallback = datacallback
    # deforum.generate_txt2img = generate_txt2img_comfy


def generate_txt2img_comfy(prompt, next_prompt, blend_value, negative_prompt, args, anim_args, root, frame,
                           init_image=None):
    args.strength = 1.0 if init_image is None else args.strength
    from deforum.avfunctions.video_audio_utilities import get_frame_name

    cnet_image = None
    input_file = os.path.join(args.outdir, 'inputframes',
                              get_frame_name(anim_args.video_init_path) + f"{frame:09}.jpg")

    # if os.path.isfile(input_file):
    #     input_frame = Image.open(input_file)
    #     cnet_image = get_canny_image(input_frame)
    #     cnet_image = ImageOps.invert(cnet_image)

    if prompt == "!reset!":
        init_image = None
        args.strength = 1.0
        prompt = next_prompt
    gen_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": args.steps,
        "seed": args.seed,
        "scale": args.scale,
        "strength": args.strength,
        "init_image": init_image,
        "width": args.W,
        "height": args.H,
        "cnet_image": cnet_image,
        "next_prompt": next_prompt,
        "prompt_blend": blend_value
    }

    if anim_args.enable_subseed_scheduling:
        gen_args["subseed"] = root.subseed
        gen_args["subseed_strength"] = root.subseed_strength
        gen_args["seed_resize_from_h"] = args.seed_resize_from_h
        gen_args["seed_resize_from_w"] = args.seed_resize_from_w
    #image = img_gen.generate(**gen_args)
    return img_gen.generate(**gen_args)

# app = FastAPI()
#
#
# @app.post("/start_deforum")
# async def start_deforum(settings: Settings, deforum=Depends(get_deforum)):
#     merged_data = json.loads(settings.file_content)
#
#     # Update the SimpleNamespace objects as you did in the main() function
#     for key, value in merged_data.items():
#         if key == "prompts": deforum.root.animation_prompts = value
#
#         if hasattr(deforum.args, key):
#             setattr(deforum.args, key, value)
#         if hasattr(deforum.anim_args, key):
#             setattr(deforum.anim_args, key, value)
#         if hasattr(deforum.parseq_args, key):
#             setattr(deforum.parseq_args, key, value)
#         if hasattr(deforum.loop_args, key):
#             setattr(deforum.loop_args, key, value)
#         if hasattr(deforum.video_args, key):
#             setattr(deforum.video_args, key, value)
#
#     return {"status": "success"}
#
#
# @app.websocket("/ws/datacallback")
# async def websocket_endpoint(websocket: WebSocket, deforum=Depends(get_deforum)):
#     global ws
#     ws = websocket
#     await websocket.accept()
#     deforum.datacallback = ws_datacallback
#
#     success = deforum()
#
#     return {"status": "done"}
#
# async def ws_datacallback(data=None):
#     if data:
#         image = data.get("image")
#         # Send image via WebSocket
#         if image:
#             await ws.send_bytes(image)




def main():
    process = None
    parser = argparse.ArgumentParser(description="Load settings from a txt file and run the deforum process.")
    parser.add_argument("--file", type=str, help="Path to the txt file containing dictionaries to merge.")
    parser.add_argument("--pipeline", type=str, default="deforum", help="Path to the txt file containing dictionaries to merge.")
    args_main = parser.parse_args()


    #deforum.enable_internal_controlnet()
    try:
        if args_main.pipeline == "deforum":

            global img_gen
            img_gen = ComfyDeforumGenerator()



            deforum = setup_deforum()
            deforum.generate_txt2img = generate_txt2img_comfy
            if args_main.file:
                merged_data = merge_dicts_from_txt(args_main.file)
                # 3. Update the SimpleNamespace objects
                for key, value in merged_data.items():

                    if key == "prompts": deforum.root.animation_prompts = value

                    if hasattr(deforum.args, key):
                        setattr(deforum.args, key, value)
                    if hasattr(deforum.anim_args, key):
                        setattr(deforum.anim_args, key, value)
                    if hasattr(deforum.parseq_args, key):
                        setattr(deforum.parseq_args, key, value)
                    if hasattr(deforum.loop_args, key):
                        setattr(deforum.loop_args, key, value)
                    if hasattr(deforum.video_args, key):
                        setattr(deforum.video_args, key, value)
            if deforum.args.seed == -1:
                deforum.args.seed = secrets.randbelow(18446744073709551615)

            success = deforum()
            output_filename_base = os.path.join(deforum.args.timestring)

            interpolator = Interpolator()

            interpolated = interpolator(frames, 1)

            save_as_h264(frames, output_filename_base + ".mp4", fps=15)
            save_as_h264(interpolated, output_filename_base + "_FILM.mp4", fps=30)
            if len(cadence_frames) > 0:
                save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")

        elif args_main.pipeline == "real2real":
            from deforum.pipelines.r2r_pipeline import Real2RealPipeline
            real2real = Real2RealPipeline()

            prompts = [
                "Starry night, Abstract painting by picasso",
                "PLanets and stars on the night sky, Abstract painting by picasso",
                "Galaxy, Abstract painting by picasso",
            ]

            keys = [30,30,30,30]

            real2real(fixed_seed=True,
                      mirror_conds=False,
                      use_feedback_loop=True,
                      prompts=prompts,
                      keys=keys,
                      strength=0.45)

        elif args_main.pipeline == "webui":
            # from deforum import streamlit_ui
            # cmd = ["streamlit", "run", f"{root_path}/deforum/streamlit_ui.py"]
            # process = subprocess.Popen(cmd)
            import streamlit.web.cli as stcli
            stcli.main(["run", f"{root_path}/deforum/streamlit_ui.py"])
        elif args_main.pipeline == "api":
            import uvicorn


            uvicorn.run(app, host="0.0.0.0", port=8000)

    except KeyboardInterrupt:
        if process:  # Check if there's a process reference
            process.terminate()  # Terminate the process
        print("\nKeyboardInterrupt detected. Interpolating and saving frames before exiting...")
        try:
            interpolator = Interpolator()
            interpolated = interpolator(frames, 1)
            output_filename_base = os.path.join(deforum.args.timestring)
            save_as_h264(frames, output_filename_base + ".mp4", fps=15)
            save_as_h264(interpolated, output_filename_base + "_FILM.mp4", fps=30)
            if len(cadence_frames) > 0:
                save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")
        except:
            pass

if __name__ == "__main__":
    main()
