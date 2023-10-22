from deforum.cmd import ComfyDeforumGenerator, save_as_h264
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
from deforum.pipelines.cond_tools import calculate_global_average, slerp, blend_tensors
from deforum.rng.rng import ImageRNG
from pydantic import BaseModel

from clip_interrogator import Config, Interrogator
import torch.nn.functional as F
import torch


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

    def blend_conditionings_sinusoidal(self, blend_value=15, overlap=5, keys=None, add_average=True, slerp_conds=True):
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
        last_alpha = 0.0
        for i in range(len(self.conds) - 1):

            if keys is not None:
                if len(keys) >= len(self.conds):
                    blend_value = keys[i]
            curr_cond_value, curr_pooled_dict = self.conds[i][0]
            next_cond_value, next_pooled_dict = self.conds[i + 1][0]

            # Add the current conditioning
            #blended_conds.append([[curr_cond_value, curr_pooled_dict]])

            # Generate blended frames between the current and next conditioning using a sinusoidal transition
            for j in range(1, blend_value + 1):
                # Calculate the sine weight based on the current frame position
                sine_weight = 0.5 * (1 - math.cos(math.pi * j / blend_value))
                if last_alpha >= sine_weight:
                    sine_weight = last_alpha
                print("sine_weight", sine_weight)
                if slerp_conds:
                    blended_cond = slerp(sine_weight, curr_cond_value, next_cond_value)
                    blended_pooled = slerp(sine_weight, curr_pooled_dict['pooled_output'], next_pooled_dict[
                            'pooled_output'])
                else:
                    blended_cond = (1 - sine_weight) * curr_cond_value + sine_weight * next_cond_value
                    blended_pooled = (1 - sine_weight) * curr_pooled_dict['pooled_output'] + sine_weight * next_pooled_dict[
                        'pooled_output']

                # Add overlap from the next conditioning
                if j >= blend_value - overlap and i < len(self.conds) - 2:  # Check if we are in the overlap frames and not at the last pair
                    subsequent_cond_value, subsequent_pooled_dict = self.conds[i + 2][0]

                    # Calculate overlap alpha using a sinusoidal function
                    overlap_alpha = 0.05 * (1 - math.cos(math.pi * (j - (blend_value - overlap)) / overlap))

                    print("overlap_alpha", overlap_alpha)
                    if slerp_conds:
                        blended_cond = slerp(overlap_alpha, blended_cond, subsequent_cond_value)
                        blended_pooled = slerp(overlap_alpha, blended_pooled, subsequent_pooled_dict)
                    else:
                        blended_cond = (1 - overlap_alpha) * blended_cond + overlap_alpha * subsequent_cond_value
                        blended_pooled = (1 - overlap_alpha) * blended_pooled + overlap_alpha * subsequent_pooled_dict[
                            'pooled_output']
                    last_alpha = overlap_alpha

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

    def blend_conditionings_sinusoidal_v2(self, blend_value=15, overlap=5, keys=None, add_average=True):
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
        last_alpha = 0.0

        for i in range(len(self.conds) - 1):
            if keys is not None and len(keys) >= len(self.conds):
                blend_value = keys[i]

            curr_cond_value, curr_pooled_dict = self.conds[i][0]
            next_cond_value, next_pooled_dict = self.conds[i + 1][0]

            # Generate blended frames between the current and next conditioning using a sinusoidal transition
            for j in range(blend_value):
                # Calculate the sine weight based on the current frame position
                sine_weight = 0.5 * (1 - math.cos(math.pi * j / blend_value))
                if last_alpha >= sine_weight:
                    sine_weight = last_alpha

                blended_cond = (1 - sine_weight) * curr_cond_value + sine_weight * next_cond_value
                blended_pooled = (1 - sine_weight) * curr_pooled_dict['pooled_output'] + sine_weight * next_pooled_dict[
                    'pooled_output']

                # Introduce overlap from the subsequent conditioning early
                if j < overlap and i < len(self.conds) - 2:  # Check if we are in the overlap frames and not at the last pair
                    subsequent_cond_value, subsequent_pooled_dict = self.conds[i + 2][0]

                    # Calculate overlap alpha using a sinusoidal function
                    overlap_alpha = 0.05 * (1 - math.cos(math.pi * j / overlap))
                    blended_cond = (1 - overlap_alpha) * blended_cond + overlap_alpha * subsequent_cond_value
                    blended_pooled = (1 - overlap_alpha) * blended_pooled + overlap_alpha * subsequent_pooled_dict[
                        'pooled_output']
                    last_alpha = overlap_alpha

                blended_conds.append([[blended_cond, {"pooled_output": blended_pooled}]])

        # Add the last conditioning
        blended_conds.append(self.conds[-1])

        if add_average:
            # Apply the global average to each conditioning tensor
            for i in range(len(blended_conds)):
                blended_conds[i][0][0] = blended_conds[i][0][0] + 0.25 * avg_tensor
                blended_conds[i][0][1]['pooled_output'] = blended_conds[i][0][1]['pooled_output'] + 0.25 * avg_pooled_tensor

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
                 strength=0.75,
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
            blended_conds = self.blend_conditionings_sinusoidal_v2(keys=keys)

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
                strength = 1.0 if image == None else strength


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