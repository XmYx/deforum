import argparse
import json
import os, sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
import subprocess
from deforum.animation.new_args import DeforumArgs, DeforumAnimArgs, ParseqArgs, LoopArgs, RootArgs, DeforumOutputArgs

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

comfy_path = os.path.join(root_path, "src/ComfyUI")

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

img_gen = None

sys.path.extend([os.path.join(os.getcwd(), "deforum", "exttools")])

import secrets
import time

from deforum.general_utils import substitute_placeholders
from deforum.main import Deforum

import subprocess
import torch

frames = []
cadence_frames = []
from deforum import rng_test

# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


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


class ComfyDeforumGenerator:

    def __init__(self):
        from comfy import controlnet
        self.clip_skip = -2
        self.device = "cuda"
        self.load_model()
        self.controlnet = controlnet.load_controlnet(os.path.join(root_path, "models/controlnet/diffusers_xl_canny_mid.safetensors"))

    def encode_latent(self, latent, subseed, subseed_strength):
        with torch.inference_mode():
            latent = self.vae.encode_tiled(latent)
            latent = latent.to("cuda")
            if subseed is not None:
                nv_rng = rng_test.Generator(subseed)
                subnoise = torch.asarray(nv_rng.randn(latent.shape), device="cuda")
                latent = slerp(subseed_strength, latent, subnoise)
        return {"samples":latent}

    def generate_latent(self, width, height, seed, subseed, subseed_strength):

        #target_shape = (4, height // 8, width // 8)

        #torch.manual_seed(seed)

        #noise = torch.randn(target_shape, device=self.device).unsqueeze(0)
        nv_rng = rng_test.Generator(seed)
        noise = torch.asarray(nv_rng.randn([1, 4, height // 8, width // 8]), device="cuda")
        if subseed is not None:
            nv_rng = rng_test.Generator(subseed)
            subnoise = torch.asarray(nv_rng.randn(noise.shape), device="cuda")
            noise = slerp(subseed_strength, noise, subnoise)
        #noise = torch.zeros([1, 4, height // 8, width // 8])

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
                 prompt="Yellow submarine",
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
                 subseed=None,
                 subseed_strength=None,
                 cnet_image=None):
        if seed == -1:
            seed = secrets.randbelow(999999999999)

        if cnet_image is not None:
            cnet_image = torch.from_numpy(np.array(cnet_image).astype(np.float32) / 255.0).unsqueeze(0)

        if init_image is None:
            if width == None:
                width = 1024
            if height == None:
                height = 960
            latent = self.generate_latent(width, height, seed, subseed, subseed_strength)

        else:
            latent = torch.from_numpy(np.array(init_image).astype(np.float32) / 255.0).unsqueeze(0)

            latent = self.encode_latent(latent, subseed, subseed_strength)
            #Implement Img2Img

        cond = self.get_conds(prompt)
        n_cond = self.get_conds(negative_prompt)


        if cnet_image is not None:
            cond = apply_controlnet(cond, self.controlnet, cnet_image, 1.0)


        from nodes import common_ksampler as ksampler

        last_step = int((1-strength) * steps) if strength != 1.0 else steps
        last_step = steps
        print(f"[Running With STRENGTH: [{strength} LAST STEP: {last_step}] ]")
        print(f"[CFG: [{scale} LAST STEP: {last_step}] ]")

        sample = ksampler(model=self.model,
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
                          force_full_denoise=True)

        decoded = self.decode_sample(sample[0]["samples"])


        image = Image.fromarray(np.clip(255. * decoded.cpu().numpy(), 0, 255).astype(np.uint8)[0])
        image = image.convert("RGB")

        return image

    def decode_sample(self, sample):
        self.vae.first_stage_model.cuda()
        decoded = self.vae.decode_tiled(sample).detach()
        return decoded


def get_canny_image(image):
    import cv2
    import numpy as np
    image = cv2.Canny(np.array(image), 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image
def generate_txt2img_comfy(prompt, next_prompt, blend_value, negative_prompt, args, anim_args, root, frame,
                                       init_image=None):
    global img_gen
    if img_gen == None:
        img_gen = ComfyDeforumGenerator()
    args.strength = 1.0 if init_image is None else args.strength
    from deforum.avfunctions.video_audio_utilities import get_frame_name

    cnet_image = None
    input_file = os.path.join(args.outdir, 'inputframes', get_frame_name(anim_args.video_init_path) + f"{frame:09}.jpg")
    # if os.path.isfile(input_file):
    #     input_frame = Image.open(input_file)
    #
    #     cnet_image = get_canny_image(input_frame)
    #
    #     cnet_image = ImageOps.invert(cnet_image)

    gen_args = {
        "prompt":prompt,
        "negative_prompt":negative_prompt,
        "steps":args.steps,
        "seed":args.seed,
        "scale":args.scale,
        "strength":args.strength,
        "init_image":init_image,
        "width":args.W,
        "height":args.H,
        "cnet_image":cnet_image
    }

    if deforum.anim_args.enable_subseed_scheduling:
        gen_args["subseed"] = root.subseed
        gen_args["subseed_strength"] = root.subseed_strength

    image = img_gen.generate(**gen_args)

    return image


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
    if len(frames) > 0:
        # Assuming frames are PIL images, convert the first one to numpy array to get its shape
        #frame_np = np.array(frames[0])
        width = frames[0].size[0]
        height = frames[0].size[1]

        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt',
               'rgb24', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-an',
               filename]
        video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        for frame in tqdm(frames, desc="Saving MP4 (ffmpeg)"):
            frame_np = np.array(frame)  # Convert the PIL image to numpy array
            video_writer.stdin.write(frame_np.tobytes())
        video_writer.communicate()

        # if audio path is provided, merge the audio and the video
        if audio_path is not None:
            try:
                output_filename = f"output/mp4s/{time.strftime('%Y%m%d%H%M%S')}_with_audio.mp4"
                cmd = ['ffmpeg', '-y', '-i', filename, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict',
                       'experimental', output_filename]
                subprocess.run(cmd)
            except Exception as e:
                print(f"Audio file merge failed from path {audio_path}\n{repr(e)}")
    else:
        print("The buffer is empty, cannot save.")
def datacallback(data=None):

    if data:
        image = data.get("image")
        cadence_frame = data.get("cadence_frame")

    if image:
        frames.append(image)
    elif cadence_frame:
        cadence_frames.append(cadence_frame)


def main():
    global deforum
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

    # deforum.args.W = 1024
    # deforum.args.H = 576
    # deforum.args.steps = 20
    # deforum.anim_args.animation_mode = "3D"
    # deforum.anim_args.zoom = "0: (1.0)"
    # deforum.anim_args.translate_z = "0: (4)"
    # deforum.anim_args.strength_schedule = "0: (0.52)"
    # deforum.anim_args.diffusion_cadence = 1
    # deforum.anim_args.max_frames = 375
    # deforum.video_args.store_frames_in_ram = True

    deforum.datacallback = datacallback



    parser = argparse.ArgumentParser(description="Load settings from a txt file and run the deforum process.")
    parser.add_argument("--file", type=str, help="Path to the txt file containing dictionaries to merge.")
    args_main = parser.parse_args()

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
    #deforum.enable_internal_controlnet()
    deforum.generate_txt2img = generate_txt2img_comfy

    print(deforum.root.animation_prompts)
    success = deforum()
    #
    #
    output_filename_base = os.path.join(deforum.args.timestring)
    save_as_h264(frames, output_filename_base + ".mp4")
    if len(cadence_frames) > 0:
        save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")

if __name__ == "__main__":
    main()
