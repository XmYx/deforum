import argparse
import math
import os, sys
from types import SimpleNamespace
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageOps
import secrets
import time

from deforum.general_utils import substitute_placeholders
from deforum.main import Deforum

import subprocess
from deforum.animation.new_args import DeforumArgs, DeforumAnimArgs, ParseqArgs, LoopArgs, RootArgs, DeforumOutputArgs

from deforum.FILM.inference import FilmModel

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
            latent = latent.to(torch.float32)
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
            if subseed == -1:
                subseed = secrets.randbelow(999999999999)
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
                 prompt=None,
                 negative_prompt="",
                 steps=25,
                 scale=7.5,
                 sampler_name="dpmpp_3m_sde_gpu",
                 scheduler="karras",
                 width=None,
                 height=None,
                 seed=-1,
                 strength=0.65,
                 init_image=None,
                 subseed=None,
                 subseed_strength=None,
                 cnet_image=None,
                 cond=None,
                 n_cond=None,
                 return_latent=None,
                 latent=None,
                 last_step=None):
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
        if prompt is not None:
            cond = self.get_conds(prompt)
            n_cond = self.get_conds(negative_prompt)


        if cnet_image is not None:
            cond = apply_controlnet(cond, self.controlnet, cnet_image, 1.0)


        from nodes import common_ksampler as ksampler

        last_step = int((1-strength) * steps) if strength != 1.0 else steps
        last_step = steps if last_step == None else last_step
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

from clip_interrogator import Config, Interrogator
import torch.nn.functional as F
import torch


def calculate_global_average(conds):
    """
    Calculate the global average for both tensor and pooled_tensor across all conditions.

    Args:
    - conds (list): List of conditionings.

    Returns:
    - Tuple containing global averages for tensor and pooled_tensor respectively.
    """
    total_tensors = len(conds)
    sum_tensor = torch.zeros_like(conds[0][0][0])
    sum_pooled_tensor = torch.zeros_like(conds[0][0][1]['pooled_output'])

    for cond in conds:
        sum_tensor += cond[0][0]
        sum_pooled_tensor += cond[0][1]['pooled_output']

    avg_tensor = sum_tensor / total_tensors
    avg_pooled_tensor = sum_pooled_tensor / total_tensors

    return avg_tensor, avg_pooled_tensor

def pad_tensor_to_match_size(tensor1, tensor2):
    """
    Pad tensor1 or tensor2 (whichever is smaller in the second dimension) to match the size of the other.
    Fills the newly created empty area with the other tensor's data.
    """
    d1, d2 = tensor1.size(1), tensor2.size(1)
    diff = d2 - d1

    if diff > 0:  # tensor1 is smaller, pad it
        # Get a slice from tensor2 and append to tensor1
        slice_from_tensor2 = tensor2[:, :diff]
        tensor1 = torch.cat((tensor1, slice_from_tensor2), dim=1)
    elif diff < 0:  # tensor2 is smaller, pad it
        # Get a slice from tensor1 and append to tensor2
        slice_from_tensor1 = tensor1[:, :abs(diff)]
        tensor2 = torch.cat((tensor2, slice_from_tensor1), dim=1)

    return tensor1, tensor2
def pyramid_blend(tensor1, tensor2, blend_value):
    # For simplicity, we'll use two levels of blending
    downsampled1 = F.avg_pool2d(tensor1, 2)
    downsampled2 = F.avg_pool2d(tensor2, 2)

    blended_low = (1 - blend_value) * downsampled1 + blend_value * downsampled2
    blended_high = tensor1 + tensor2 - F.interpolate(blended_low, scale_factor=2)

    return blended_high


def gaussian_blend(tensor1, tensor2, blend_value):
    sigma = 0.5  # Adjust for desired smoothness
    weight = torch.exp(-((blend_value - 0.5) ** 2) / (2 * sigma ** 2))

    return (1 - weight) * tensor1 + weight * tensor2
def sigmoidal_blend(tensor1, tensor2, blend_value):
    # Convert blend_value into a tensor with the same shape as tensor1 and tensor2
    blend_tensor = torch.full_like(tensor1, blend_value)
    weight = 1 / (1 + torch.exp(-10 * (blend_tensor - 0.5)))  # Sigmoid function centered at 0.5
    return (1 - weight) * tensor1 + weight * tensor2

def blend_tensors(obj1, obj2, blend_value, blend_method="pyramid"):
    """
    Blends tensors in two given objects based on a blend value using various blending strategies.
    """

    tensor1, tensor2 = pad_tensor_to_match_size(obj1[0], obj2[0])
    pooled_tensor1, pooled_tensor2 = pad_tensor_to_match_size(obj1[1]['pooled_output'], obj2[1]['pooled_output'])

    if blend_method == "linear":
        weight = blend_value
        blended_cond = (1 - weight) * tensor1 + weight * tensor2
        blended_pooled = (1 - weight) * pooled_tensor1 + weight * pooled_tensor2

    elif blend_method == "sigmoidal":
        blended_cond = sigmoidal_blend(tensor1, tensor2, blend_value)
        blended_pooled = sigmoidal_blend(pooled_tensor1, pooled_tensor2, blend_value)

    elif blend_method == "gaussian":
        blended_cond = gaussian_blend(tensor1, tensor2, blend_value)
        blended_pooled = gaussian_blend(pooled_tensor1, pooled_tensor2, blend_value)

    elif blend_method == "pyramid":
        blended_cond = pyramid_blend(tensor1, tensor2, blend_value)
        blended_pooled = pyramid_blend(pooled_tensor1, pooled_tensor2, blend_value)

    return [[blended_cond, {"pooled_output": blended_pooled}]]
from torchvision import transforms
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

class Real2RealPipeline:
    def __init__(self):
        self.generator = ComfyDeforumGenerator()

        config = Config(clip_model_name="ViT-L-14/openai")
        self.interrogator = Interrogator(config)

        self.interrogator.config.blip_num_beams = 64
        self.interrogator.config.chunk_size = 64
        self.interrogator.config.flavor_intermediate_count = 512  # if clip_model_name == MODELS[0] else 1024

        self.film = FilmModel()

    def image_to_prompt(self, image, mode='negative'):
        with torch.no_grad():
            image = image.convert('RGB').resize((512,512), resample=Image.Resampling.LANCZOS)
            if mode == 'best':
                prompt = self.interrogator.interrogate(image)
            elif mode == 'classic':
                prompt = self.interrogator.interrogate_classic(image)
            elif mode == 'fast':
                prompt = self.interrogator.interrogate_fast(image)
            elif mode == 'negative':
                prompt = self.interrogator.interrogate_negative(image)

        return prompt

    def blend_conditionings(self, blend_value=15, overlap=5, keys=None):
        """
        Generate a set number of blended frames from the input conditionings.

        Parameters:
        - conds: List of conditionings in the form [[blended_cond, {"pooled_output": blended_pooled}]].
        - num_blends: Number of blended frames to produce between each pair.
        - overlap: Amount of the next conditioning to introduce into the current blend.

        Returns:
        - List of blended conditionings.
        """
        blended_conds = []
        alpha = overlap / 100  # Convert overlap from percentage to fraction

        for i in range(len(self.conds) - 1):
            curr_cond, curr_pooled_dict = self.conds[i][0]
            curr_pooled = curr_pooled_dict['pooled_output']
            next_cond, next_pooled_dict = self.conds[i + 1][0]
            next_pooled = next_pooled_dict['pooled_output']

            # Add the current conditioning
            blended_conds.append([[curr_cond, {"pooled_output": curr_pooled}]])

            # Generate blended frames between the current and next conditioning
            for j in range(1, blend_value + 1):
                blend_ratio = j / (blend_value + 1)
                blended_cond = (1 - blend_ratio) * curr_cond + blend_ratio * next_cond
                blended_pooled = (1 - blend_ratio) * curr_pooled + blend_ratio * next_pooled

                # Add overlap from the next conditioning
                if j >= blend_value - 1 and i < len(
                        self.conds) - 2:  # Check if we are at the last frames and not at the last pair
                    subsequent_cond, subsequent_pooled_dict = self.conds[i + 2][0]
                    subsequent_pooled = subsequent_pooled_dict['pooled_output']

                    blended_cond += alpha * subsequent_cond
                    blended_pooled += alpha * subsequent_pooled

                blended_conds.append([[blended_cond, {"pooled_output": blended_pooled}]])

        # Add the last conditioning
        blended_conds.append(self.conds[-1])

        return blended_conds

    def blend_conditionings_sinusoidal(self, blend_value=15, overlap=5, keys=None, add_average=True):
        """
        Generate a set number of blended frames from the input conditionings using a sinusoidal transition.

        Parameters:
        - conds: List of conditionings in the form [[blended_cond, {"pooled_output": blended_pooled}]].
        - blend_value: Number of blended frames to produce between each pair.
        - overlap: Amount of the next conditioning to introduce into the current blend.

        Returns:
        - List of blended conditionings.
        """
        blended_conds = []
        alpha = overlap / 100  # Convert overlap from percentage to fraction
        avg_tensor, avg_pooled_tensor = calculate_global_average(self.conds)
        for i in range(len(self.conds) - 1):

            if keys is not None:
                if len(keys) >= len(self.conds):
                    blend_value = keys[i]
            curr_cond_value, curr_pooled_dict = self.conds[i][0]
            next_cond_value, next_pooled_dict = self.conds[i + 1][0]

            # Add the current conditioning
            blended_conds.append([[curr_cond_value, curr_pooled_dict]])

            # Generate blended frames between the current and next conditioning using a sinusoidal transition
            for j in range(1, blend_value + 1):
                # Calculate the sine weight based on the current frame position
                sine_weight = 0.5 * (1 - math.cos(math.pi * j / blend_value))
                blended_cond = (1 - sine_weight) * curr_cond_value + sine_weight * next_cond_value
                blended_pooled = (1 - sine_weight) * curr_pooled_dict['pooled_output'] + sine_weight * next_pooled_dict[
                    'pooled_output']

                # Add overlap from the next conditioning
                if j >= blend_value - 1 and i < len(
                        self.conds) - 2:  # Check if we are at the last frames and not at the last pair
                    subsequent_cond_value, subsequent_pooled_dict = self.conds[i + 2][0]

                    # Calculate overlap alpha using a sinusoidal function
                    overlap_alpha = 0.5 * (1 - math.cos(math.pi * (j - (blend_value - overlap)) / overlap))
                    blended_cond = (1 - overlap_alpha) * blended_cond + overlap_alpha * subsequent_cond_value
                    blended_pooled = (1 - overlap_alpha) * blended_pooled + overlap_alpha * subsequent_pooled_dict[
                        'pooled_output']

                    #blended_cond += alpha * subsequent_cond_value
                    #blended_pooled += alpha * subsequent_pooled_dict['pooled_output']

                blended_conds.append([[blended_cond, {"pooled_output": blended_pooled}]])

        # Add the last conditioning
        blended_conds.append(self.conds[-1])
        if add_average:
            # Apply the global average to each conditioning tensor
            for i in range(len(blended_conds)):
                blended_conds[i][0][0] = blended_conds[i][0][0] + 0.25 * avg_tensor
                blended_conds[i][0][1]['pooled_output'] = blended_conds[i][0][1][
                                                               'pooled_output'] + 0.25 * avg_pooled_tensor
        return blended_conds
    def blend_all_conds(self, blend_frames, keys=None, blend_method="linear", add_average=False,
                        blend_type="exponential", overlap=5):
        """
        Create blended conditionings between consecutive pairs of conds.

        Args:
        - blend_frames (int): Number of interpolated conditions between each original pair.
        - blend_method (str): The blending strategy to use. Options: "linear", "sigmoidal", "gaussian", "pyramid".
        - overlap (int): Number of frames to overlap between consecutive scenes.

        Returns:
        - list of blended conds
        """
        blended_conds = []
        blend_values = []
        blended_conds.extend([self.conds[0]] * blend_frames)
        avg_tensor, avg_pooled_tensor = calculate_global_average(self.conds)

        # Iterate over all consecutive pairs of conds
        for i in range(len(self.conds) - 1):
            if keys is not None:
                blend_frames = keys[i]

            # Main blending for the current condition
            for frame in range(blend_frames):

                if blend_type == "linear":
                    blend_value = frame / float(blend_frames)
                elif blend_type == "exponential":
                    lambda_val = -np.log(0.01) / blend_frames
                    blend_value = 1 - np.exp(-lambda_val * frame)
                else:
                    raise ValueError(f"Unsupported blend type: {blend_type}")

                blended_cond = blend_tensors(self.conds[i][0], self.conds[i + 1][0], blend_value, blend_method)
                blended_conds.append(blended_cond)
                blend_values.append(blend_value)

                # Introduce overlap towards the end of the blending
                if frame >= blend_frames - overlap:
                    # Check if self.conds[i + 2] exists
                    next_cond = self.conds[i + 2] if i + 2 < len(self.conds) else None
                    if next_cond is not None:
                        overlap_blend_value = (frame - (blend_frames - overlap)) / float(overlap)
                        blended_cond = blend_tensors(blended_cond[0], next_cond[0], overlap_blend_value, blend_method)
                        blended_conds.append(blended_cond)
                        blend_values.append(overlap_blend_value)

        print("Real2Real created blended conds:", blend_values)
        print(len(blended_conds))

        if add_average:
            # Apply the global average to each conditioning tensor
            for i in range(len(blended_conds)):
                blended_conds[i][0][0] = blended_conds[i][0][0] + 0.25 * avg_tensor
                blended_conds[i][0][1]['pooled_output'] = blended_conds[i][0][1][
                                                               'pooled_output'] + 0.25 * avg_pooled_tensor

        return blended_conds
    def __call__(self,
                 images=None,
                 prompts=None,
                 keys=None,
                 blend_frames=6,
                 fixed_seed = False,
                 mirror_conds=False,
                 mirror_frames=False,
                 use_feedback_loop=True,
                 steps=25,
                 *args, **kwargs):

        conds = []
        image = None
        with torch.inference_mode():
            seed = secrets.randbelow(18446744073709551615)
            if images == None and prompts == None:
                images = ["/home/mix/Downloads/test_3.jpg", "/home/mix/Downloads/test_4.jpg", "/home/mix/Downloads/test_5.jpg", "/home/mix/Downloads/test_6.jpg"]
                images = [Image.open(image) for image in images]
                prompts = []
            if len(prompts) == 0:
                for image in images:

                    prompt = self.image_to_prompt(image)
                    print("[ Created Prompt", prompt, "]")
                    prompts.append(prompt)
            n_cond = self.generator.get_conds("blurry, nsfw, nude, text, porn, sex, xxx")

            for prompt in prompts:
                conds.append(self.generator.get_conds(prompt))

            self.conds = conds

            #blended_conds = self.blend_all_conds(blend_frames, keys)
            blended_conds = self.blend_conditionings_sinusoidal(keys=keys)

            if mirror_conds:
                blended_conds_copy = blended_conds.copy()
                blended_conds_copy = blended_conds_copy[::-1]  # Use slicing to reverse the list
                blended_conds = blended_conds + blended_conds_copy

            images = []
            self.interrogator.blip_model.to('cpu')
            self.interrogator.clip_model.to('cpu')
            progress_bar = tqdm(total=len(blended_conds), desc="Processing frames", position=0,
                                leave=True)
            image = None
            prev_image = None
            x = 0
            for cond in blended_conds:
                if not fixed_seed:
                    seed = secrets.randbelow(18446744073709551615)
                if not use_feedback_loop:
                    image = None
                strength = 1.0 if image == None else 0.45


                last_step = steps
                image = self.generator.generate(cond=cond,
                                                n_cond=n_cond,
                                                seed=seed,
                                                return_latent=False,
                                                strength=strength,
                                                init_image=image,
                                                steps=steps,
                                                last_step=last_step,
                                                width=768,
                                                height=768,
                                                scale=5.0)
                if x == 0:
                    image.save("frame_1.png", "PNG")

                x+=1

                prev_image = image

                images.append(image)
                progress_bar.update(1)


        images = [np.array(image) for image in images]

        interpolated = []
        interpolated.append(images[0])
        for i in range(len(images) - 1):  # We subtract 1 to avoid out-of-index errors
            image1 = images[i]
            image2 = images[i + 1]

            # Assuming self.film returns a list of interpolated frames
            interpolated_frames = self.film.inference(image1, image2, 4)
            interpolated_frames.pop(0)
            interpolated_frames.pop(-1)
            # Append the interpolated frames to the interpolated list
            interpolated.extend(interpolated_frames)

        interpolated = [Image.fromarray(image) for image in interpolated]
        images = [Image.fromarray(image) for image in images]

        progress_bar.close()
        save_as_h264(images, "Real2Real_" + time.strftime('%Y%m%d%H%M%S') + ".mp4")
        save_as_h264(interpolated, "Real2Real_" + time.strftime('%Y%m%d%H%M%S') + "_FILM" + ".mp4")


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
    parser.add_argument("--pipeline", type=str, default="deforum", help="Path to the txt file containing dictionaries to merge.")
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
    if args_main.pipeline == "deforum":
        success = deforum()
        output_filename_base = os.path.join(deforum.args.timestring)
        save_as_h264(frames, output_filename_base + ".mp4")
        if len(cadence_frames) > 0:
            save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")

    else:
        real2real = Real2RealPipeline()

        prompts = [
            "Starry night, Abstract painting by picasso",
            "PLanets and stars on the night sky, Abstract painting by picasso",
            "Galaxy, Abstract painting by picasso",
            "Starry night, Abstract painting by picasso"
        ]

        keys = [25,25,25,25]

        real2real(fixed_seed=True,
                  mirror_conds=False,
                  use_feedback_loop=False,
                  prompts=prompts,
                  keys=keys)


if __name__ == "__main__":
    main()
