import gc
import importlib
import json
import math
import os
import random
import secrets
import textwrap
import time
from datetime import datetime
from typing import Optional, Callable

import PIL
import cv2
import numexpr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageChops, ImageEnhance

# from deforum import (root_path,
#                      fetch_and_download_model,
#                      available_engines,
#                      ComfyDeforumGenerator,
#                      available_engine_classes,
#                      available_pipelines)
from deforum.animation.animation import get_flip_perspective_matrix, flip_3d_perspective, transform_image_3d_new, \
    anim_frame_warp
from deforum.animation.animation_key_frames import DeformAnimKeys, LooperAnimKeys
from deforum.animation.base_args import DeforumAnimPrompts
from deforum.animation.new_args import DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, RootArgs, ParseqArgs, LoopArgs
from deforum.avfunctions.colors.colors import maintain_colors
from deforum.avfunctions.hybridvideo.hybrid_video import autocontrast_grayscale, get_matrix_for_hybrid_motion_prev, \
    get_matrix_for_hybrid_motion, image_transform_ransac, get_flow_for_hybrid_motion_prev, get_flow_for_hybrid_motion, \
    image_transform_optical_flow, get_flow_from_images, hybrid_composite, abs_flow_to_rel_flow, rel_flow_to_abs_flow, \
    hybrid_generation
from deforum.avfunctions.image.image_sharpening import unsharp_mask
from deforum.avfunctions.image.load_images import load_image, get_mask_from_file, load_img, prepare_mask, \
    check_mask_for_errors
from deforum.avfunctions.image.save_images import save_image
from deforum.avfunctions.interpolation.RAFT import RAFT
from deforum.avfunctions.masks.composable_masks import compose_mask_with_check
from deforum.avfunctions.masks.masks import do_overlay_mask
from deforum.avfunctions.noise.noise import add_noise
from deforum.avfunctions.video_audio_utilities import get_frame_name, get_next_frame
from deforum.cmd import extract_values
from deforum.datafunctions.prompt import prepare_prompt, check_is_number, split_weighted_subprompts
from deforum.datafunctions.seed import next_seed
from deforum.exttools.depth import DepthModel
from deforum.general_utils import pairwise_repl

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
default_cache_folder = os.path.join(root_path, "models/checkpoints")

script_start_time = time.time()

from deforum.exttools import py3d_tools as p3d

class DeforumBase:

    @classmethod
    def from_civitai(cls,
                     modelid:str=None,
                     generator:str="comfy",
                     pipeline:str="DeforumAnimationPipeline",
                     cache_dir:str=default_cache_folder,
                     lcm=False):

        from deforum import available_engines
        assert generator in available_engines, f"Make sure to use one of the available engines: {available_engines}"
        from deforum import available_pipelines
        assert pipeline in available_pipelines, f"Make sure to use one of the available pipelines: {available_pipelines}"

        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        assert os.path.isdir(cache_dir), "Could not create the requested cache dir, make sure the application has permissions"
        # Download model from CivitAi if not in default or given cache folder
        if modelid != None:
            # try:
            from deforum import fetch_and_download_model
            model_file_name = fetch_and_download_model(modelId=modelid)
            model_path = os.path.join(cache_dir, model_file_name)
            # except Exception as e:
            #     model_path = None
            #     print(e)
        else:
            model_path = None
        from deforum import ComfyDeforumGenerator

        generator = ComfyDeforumGenerator(model_path=model_path, lcm=lcm)

        deforum_module = importlib.import_module(cls.__module__.split(".")[0])

        # print(cls.__name__)

        #pipeline_class = getattr(deforum_module, pipeline)
        pipeline_class = getattr(deforum_module, cls.__name__)

        pipe = pipeline_class(generator)

        return pipe


class DeforumGenerationObject:

    def __init__(self, *args, **kwargs):


        #placeholder to set defaults:

        base_args = extract_values(DeforumArgs())
        anim_args = extract_values(DeforumAnimArgs())
        parseg_args = extract_values(ParseqArgs())
        loop_args = extract_values(LoopArgs())
        root = RootArgs()
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}

        merged_args = {**base_args, **anim_args, **parseg_args, **loop_args, **output_args_dict, **root}
        for key, value in merged_args.items():
            setattr(self, key, value)

        self.parseq_manifest = None
        animation_prompts = DeforumAnimPrompts()
        self.animation_prompts = json.loads(animation_prompts)
        self.timestring = time.strftime('%Y%m%d%H%M%S')
        #current_arg_list = [deforum.args, deforum.anim_args, deforum.video_args, deforum.parseq_args]
        full_base_folder_path = os.path.join(root_path, "output/deforum")


        self.raw_batch_name = self.batch_name
        #self.batch_name = substitute_placeholders(deforum.args.batch_name, current_arg_list,
        #                                                  full_base_folder_path)
        self.batch_name = f"Deforum_{self.timestring}"
        self.outdir = os.path.join(full_base_folder_path, str(self.batch_name))

        os.makedirs(self.outdir, exist_ok=True)

        if self.seed == -1 or self.seed == "-1":
            setattr(self, "seed", secrets.randbelow(999999999999999999))
            setattr(self, "raw_seed", int(self.seed))
            setattr(self, "seed_internal", 0)
        else:
            self.seed = int(self.seed)

        self.prompts = None


        # initialize vars
        self.prev_img = None
        self.color_match_sample = None
        self.start_frame = 0
        self.frame_idx = 0
        self.flow = None
        self.prev_flow = None
        self.image = None
        self.store_frames_in_ram = None
        self.turbo_prev_image, self.turbo_prev_frame_idx = None, 0
        self.turbo_next_image, self.turbo_next_frame_idx = None, 0
        self.contrast = 1.0
        self.hybrid_use_full_video = True
        self.turbo_steps = self.diffusion_cadence
        # Setting all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, attribute, default=None):
            return getattr(self, attribute, default)

    def to_dict(self):
        """Returns all instance attributes as a dictionary."""
        return self.__dict__.copy()

    def update_from_kwargs(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_settings_file(cls, settings_file_path: str = None):
        instance = cls()

        if settings_file_path and os.path.isfile(settings_file_path):
            file_ext = os.path.splitext(settings_file_path)[1]

            # Load data based on file type
            if file_ext == '.json':
                with open(settings_file_path, 'r') as f:
                    data = json.load(f)
            elif file_ext == '.txt':
                with open(settings_file_path, 'r') as f:
                    content = f.read()
                    data = json.loads(content)
            else:
                raise ValueError("Unsupported file type")

            # Set attributes based on loaded data
            for key, value in data.items():
                setattr(instance, key, value)

        if hasattr(instance, "diffusion_cadence"):
            instance.turbo_steps =  int(instance.diffusion_cadence)

        if hasattr(instance, "using_video_init"):
            if instance.using_video_init:
                instance.turbo_steps = 1
        if instance.prompts != None:
            instance.animation_prompts = instance.prompts
        return instance


class DeforumKeyFrame:

    def get(self, attribute, default=None):
            return getattr(self, attribute, default)
    @classmethod
    def from_keys(cls, keys, frame_idx):
        instance = cls()
        instance.noise = keys.noise_schedule_series[frame_idx]
        instance.strength = keys.strength_schedule_series[frame_idx]
        instance.scale = keys.cfg_scale_schedule_series[frame_idx]
        instance.contrast = keys.contrast_schedule_series[frame_idx]
        instance.kernel = int(keys.kernel_schedule_series[frame_idx])
        instance.sigma = keys.sigma_schedule_series[frame_idx]
        instance.amount = keys.amount_schedule_series[frame_idx]
        instance.threshold = keys.threshold_schedule_series[frame_idx]
        instance.cadence_flow_factor = keys.cadence_flow_factor_schedule_series[frame_idx]
        instance.redo_flow_factor = keys.redo_flow_factor_schedule_series[frame_idx]
        instance.hybrid_comp_schedules = {
            "alpha": keys.hybrid_comp_alpha_schedule_series[frame_idx],
            "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[frame_idx],
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
            "mask_auto_contrast_cutoff_low": int(
                keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx]),
            "mask_auto_contrast_cutoff_high": int(
                keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx]),
            "flow_factor": keys.hybrid_flow_factor_schedule_series[frame_idx]
        }
        instance.scheduled_sampler_name = None
        instance.scheduled_clipskip = None
        instance.scheduled_noise_multiplier = None
        instance.scheduled_ddim_eta = None
        instance.scheduled_ancestral_eta = None

        return instance


class Logger:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.log_file = None
        self.current_datetime = datetime.now()
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.terminal_width = self.get_terminal_width()

    def get_terminal_width(self):
        """Get the width of the terminal."""
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except (ImportError, AttributeError):
            # Default width
            return 80

    def start_session(self):
        year, month, day = self.current_datetime.strftime('%Y'), self.current_datetime.strftime(
            '%m'), self.current_datetime.strftime('%d')
        log_path = os.path.join(self.root_path, 'logs', year, month, day)
        os.makedirs(log_path, exist_ok=True)

        self.log_file = open(os.path.join(log_path, f"metrics_{self.timestamp}.log"), "a")

        self.log_file.write("=" * self.terminal_width + "\n")
        self.log_file.write("Log Session Started: " + self.timestamp.center(self.terminal_width - 20) + "\n")
        self.log_file.write("=" * self.terminal_width + "\n")

    def log(self, message: str, timestamped: bool = True):
        if timestamped:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f"[{timestamp}] {message}"

        # Wrap the message to the terminal width
        wrapped_text = "\n".join(textwrap.wrap(message, width=self.terminal_width))

        self.log_file.write(f"{wrapped_text}\n")

    def __call__(self, message: str, timestamped: bool = True, *args, **kwargs):
        self.log(message, timestamped)

    def close_session(self):
        if self.log_file:
            self.log_file.write("\n" + "=" * self.terminal_width + "\n")
            self.log_file.write("Log Session Ended".center(self.terminal_width) + "\n")
            self.log_file.write("=" * self.terminal_width + "\n")
            self.log_file.close()

class DeforumPipeline(DeforumBase):

    def __init__(self,
                 generator:Callable,
                 logger:Optional[Callable]=None):

        super().__init__()

        # assert generator in available_engine_classes, f"Make sure to use one of the available engines: {available_engine_classes}"

        self.generator = generator
        self.logger = logger

        self.prep_fns = []
        self.shoot_fns = []
        self.post_fns = []



class DeforumAnimationPipeline(DeforumPipeline):

    def __init__(self,
                 generator: Callable,
                 logger: Optional[Callable]=None
                 ):
        super().__init__(generator, logger or Logger(root_path))
        self.prep_fns = []
        self.shoot_fns = []
        self.post_fns = []
        self.images = []

    def __call__(self,
                 settings_file:str=None,
                 *args,
                 **kwargs) -> DeforumGenerationObject:

        # Function to log metrics to a timestamped file

        self.logger.start_session()

        start_total_time = time.time()

        duration = (start_total_time - script_start_time) * 1000
        self.logger.log(f"Script startup / model loading took {duration:.2f} ms")

        if settings_file:
            # try:
            self.gen = DeforumGenerationObject.from_settings_file(settings_file)
            # except Exception as e:
            #     print(e)
            #     self.gen = DeforumGenerationObject(**kwargs)
        else:
            self.gen = DeforumGenerationObject(**kwargs)

        self.gen.update_from_kwargs(**kwargs)

        setup_start = time.time()
        self.pre_setup()
        setup_end = time.time()
        duration = (setup_end - setup_start) * 1000
        self.logger.log(f"pre_setup took {duration:.2f} ms")

        setup_start = time.time()
        self.setup()
        setup_end = time.time()
        duration = (setup_end - setup_start) * 1000
        self.logger.log(f"loop took {duration:.2f} ms")

        # Log names of functions in each list if they have functions
        if self.prep_fns:
            self.logger.log("Functions in prep_fns:", timestamped=False)
            for fn in self.prep_fns:
                self.logger.log(fn.__name__, timestamped=False)

        if self.shoot_fns:
            self.logger.log("Functions in shoot_fns:", timestamped=False)
            for fn in self.shoot_fns:
                self.logger.log(fn.__name__, timestamped=False)

        if self.post_fns:
            self.logger.log("Functions in post_fns:", timestamped=False)
            for fn in self.post_fns:
                self.logger.log(fn.__name__, timestamped=False)

        self.logger.log(str(self.gen.to_dict()), timestamped=False)

        # PREP LOOP
        for fn in self.prep_fns:
            start_time = time.time()
            fn(self)
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            self.logger.log(f"{fn.__name__} took {duration:.2f} ms")

        while self.gen.frame_idx  < self.gen.max_frames:
            # MAIN LOOP
            frame_start = time.time()
            for fn in self.shoot_fns:
                start_time = time.time()
                with torch.inference_mode():
                    with torch.no_grad():
                        fn(self)
                end_time = time.time()
                duration = (end_time - start_time) * 1000
                self.logger.log(f"{fn.__name__} took {duration:.2f} ms")
            duration = (time.time() - frame_start) * 1000
            self.logger.log(f"----------------------------- Frame {self.gen.frame_idx + 1} took {duration:.2f} ms")

        # POST LOOP
        for fn in self.post_fns:
            start_time = time.time()
            fn(self)
            duration = (time.time() - start_time) * 1000
            self.logger.log(f"{fn.__name__} took {duration:.2f} ms")

        total_duration = (time.time() - start_total_time) * 1000
        average_time_per_frame = total_duration / self.gen.max_frames

        self.logger.log(f"Total time taken: {total_duration:.2f} ms")
        self.logger.log(f"Average time per frame: {average_time_per_frame:.2f} ms")

        self.logger.close_session()
        print("[ DEFORUM RENDER COMPLETE ]")
        return self.gen
    def pre_setup(self):
        frame_warp_modes = ['2D', '3D']
        hybrid_motion_modes = ['Affine', 'Perspective', 'Optical Flow']


        if self.gen.animation_mode in frame_warp_modes:
            # handle hybrid video generation
            if self.gen.hybrid_composite != 'None' or self.gen.hybrid_motion in hybrid_motion_modes:
                _, _, self.gen.inputfiles = hybrid_generation(self.gen, self.gen, self.gen)
                self.gen.hybrid_frame_path = os.path.join(self.gen.outdir, 'hybridframes')

        if int(self.gen.seed) == -1:
            self.gen.seed = secrets.randbelow(18446744073709551615)
        self.gen.max_frames += 1
        self.gen.keys = DeformAnimKeys(self.gen, self.gen.seed)
        self.gen.loopSchedulesAndData = LooperAnimKeys(self.gen, self.gen, self.gen.seed)
        prompt_series = pd.Series([np.nan for a in range(self.gen.max_frames)])
        for i, prompt in self.gen.animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        prompt_series = prompt_series.ffill().bfill()
        self.gen.prompt_series = prompt_series
        self.gen.max_frames -= 1

        # check for video inits
        self.gen.using_vid_init = self.gen.animation_mode == 'Video Input'

        # load depth model for 3D
        self.gen.predict_depths = (
                                     self.gen.animation_mode == '3D' and self.gen.use_depth_warping) or self.gen.save_depth_maps
        self.gen.predict_depths = self.gen.predict_depths or (
                self.gen.hybrid_composite and self.gen.hybrid_comp_mask_type in ['Depth', 'Video Depth'])
        if self.gen.predict_depths:
            # if self.opts is not None:
            #     self.keep_in_vram = self.opts.data.get("deforum_keep_3d_models_in_vram")
            # else:
            self.gen.keep_in_vram = True
            # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
            # TODO Set device in root in webui
            device = "cuda"
            self.depth_model = DepthModel(self.gen.models_path, device, self.gen.half_precision,
                                     keep_in_vram=self.gen.keep_in_vram,
                                     depth_algorithm=self.gen.depth_algorithm, Width=self.gen.W,
                                     Height=self.gen.H,
                                     midas_weight=self.gen.midas_weight)
            print(f"[ Loaded Depth model ]")
            # depth-based hybrid composite mask requires saved depth maps
            if self.gen.hybrid_composite != 'None' and self.gen.hybrid_comp_mask_type == 'Depth':
                self.gen.save_depth_maps = True
        else:
            self.depth_model = None
            self.gen.save_depth_maps = False

        self.raft_model = None
        load_raft = (self.gen.optical_flow_cadence == "RAFT" and int(self.gen.diffusion_cadence) > 0) or \
                    (self.gen.hybrid_motion == "Optical Flow" and self.gen.hybrid_flow_method == "RAFT") or \
                    (self.gen.optical_flow_redo_generation == "RAFT")
        if load_raft:
            print("[ Loading RAFT model ]")
            self.raft_model = RAFT()



    def setup(self, *args, **kwargs) -> None:

        hybrid_available = self.gen.hybrid_composite != 'None' or self.gen.hybrid_motion in ['Optical Flow', 'Affine', 'Perspective']

        turbo_steps = self.gen.get('turbo_steps', 1)
        if turbo_steps > 1:
            self.shoot_fns.append(make_cadence_frames)
        if self.gen.color_coherence == 'Video Input' and hybrid_available:
            self.shoot_fns.append(color_match_video_input)
        if self.gen.animation_mode in ['2D', '3D']:
            self.shoot_fns.append(anim_frame_warp_cls)

        if self.gen.hybrid_composite == 'Before Motion':
            self.shoot_fns.append(hybrid_composite_cls)

        if self.gen.hybrid_motion in ['Affine', 'Perspective']:
            self.shoot_fns.append(affine_persp_motion)

        if self.gen.hybrid_motion in ['Optical Flow']:
            self.shoot_fns.append(optical_flow_motion)

        if self.gen.hybrid_composite == 'Normal':
            self.shoot_fns.append(hybrid_composite_cls)

        if self.gen.color_coherence != 'None':
            self.shoot_fns.append(color_match_cls)

        self.shoot_fns.append(set_contrast_image)

        if self.gen.use_mask or self.gen.use_noise_mask:
            self.shoot_fns.append(handle_noise_mask)

        self.shoot_fns.append(add_noise_cls)

        self.shoot_fns.append(get_generation_params)

        if self.gen.optical_flow_redo_generation != 'None':
            self.shoot_fns.append(optical_flow_redo)

        if int(self.gen.diffusion_redo) > 0:
            self.shoot_fns.append(diffusion_redo)

        self.shoot_fns.append(main_generate_with_cls)

        if self.gen.hybrid_composite == 'After Generation':
            self.shoot_fns.append(post_hybrid_composite_cls)

        if self.gen.color_coherence != 'None':
            self.shoot_fns.append(post_color_match_with_cls)

        if self.gen.overlay_mask:
            self.shoot_fns.append(overlay_mask_cls)

        self.shoot_fns.append(post_gen_cls)

    def reset(self, *args, **kwargs) -> None:
        self.prep_fns = []
        self.shoot_fns = []
        self.post_fns = []

    def datacallback(self, data):
        pass

    def generate(self):

        assert self.gen.prompt is not None

        # Setup the pipeline
        # p = get_webui_sd_pipeline(args, root, frame)
        prompt, negative_prompt = split_weighted_subprompts(self.gen.prompt, self.gen.frame_idx, self.gen.max_frames)

        # print("DEFORUM CONDITIONING INTERPOLATION")

        def generate_blend_values(distance_to_next_prompt, blend_type="linear"):
            if blend_type == "linear":
                return [i / distance_to_next_prompt for i in range(distance_to_next_prompt + 1)]
            elif blend_type == "exponential":
                base = 2
                return [1 / (1 + math.exp(-8 * (i / distance_to_next_prompt - 0.5))) for i in
                        range(distance_to_next_prompt + 1)]
            else:
                raise ValueError(f"Unknown blend type: {blend_type}")

        def get_next_prompt_and_blend(current_index, prompt_series, blend_type="exponential"):
            # Find where the current prompt ends
            next_prompt_start = current_index + 1
            while next_prompt_start < len(prompt_series) and prompt_series.iloc[next_prompt_start] == \
                    prompt_series.iloc[
                        current_index]:
                next_prompt_start += 1

            if next_prompt_start >= len(prompt_series):
                return "", 1.0
                # raise ValueError("Already at the last prompt, no next prompt available.")

            # Calculate blend value
            distance_to_next = next_prompt_start - current_index
            blend_values = generate_blend_values(distance_to_next, blend_type)
            blend_value = blend_values[1]  # Blend value for the next frame after the current index

            return prompt_series.iloc[next_prompt_start], blend_value

        next_prompt, blend_value = get_next_prompt_and_blend(self.gen.frame_idx, self.gen.prompt_series)
        # print("DEBUG", next_prompt, blend_value)

        # blend_value = 1.0
        # next_prompt = ""
        if not self.gen.use_init and self.gen.strength > 0 and self.gen.strength_0_no_init:
            self.gen.strength = 0
        processed = None
        mask_image = None
        init_image = None
        image_init0 = None

        if self.gen.use_looper and self.gen.animation_mode in ['2D', '3D']:
            self.gen.strength = self.gen.imageStrength
            tweeningFrames = self.gen.tweeningFrameSchedule
            blendFactor = .07
            colorCorrectionFactor = self.gen.colorCorrectionFactor
            jsonImages = json.loads(self.gen.imagesToKeyframe)
            # find which image to show
            parsedImages = {}
            frameToChoose = 0
            max_f = self.gen.max_frames - 1

            for key, value in jsonImages.items():
                if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
                    parsedImages[key] = value
                else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                    parsedImages[int(numexpr.evaluate(key))] = value

            framesToImageSwapOn = list(map(int, list(parsedImages.keys())))

            for swappingFrame in framesToImageSwapOn[1:]:
                frameToChoose += (self.gen.frame_idx >= int(swappingFrame))

            # find which frame to do our swapping on for tweening
            skipFrame = 25
            for fs, fe in pairwise_repl(framesToImageSwapOn):
                if fs <= self.gen.frame_idx <= fe:
                    skipFrame = fe - fs
            if skipFrame > 0:
                # print("frame % skipFrame", frame % skipFrame)

                if self.gen.frame_idx % skipFrame <= tweeningFrames:  # number of tweening frames
                    blendFactor = self.gen.blendFactorMax - self.gen.blendFactorSlope * math.cos(
                        (self.gen.frame_idx % tweeningFrames) / (tweeningFrames / 2))
            else:
                print("LOOPER ERROR, AVOIDING DIVISION BY 0")
            init_image2, _ = load_img(list(jsonImages.values())[frameToChoose],
                                      shape=(self.gen.W, self.gen.H),
                                      use_alpha_as_mask=self.gen.use_alpha_as_mask)
            image_init0 = list(jsonImages.values())[0]
            # print(" TYPE", type(image_init0))


        else:  # they passed in a single init image
            image_init0 = self.gen.init_image

        available_samplers = {
            'euler a': 'Euler a',
            'euler': 'Euler',
            'lms': 'LMS',
            'heun': 'Heun',
            'dpm2': 'DPM2',
            'dpm2 a': 'DPM2 a',
            'dpm++ 2s a': 'DPM++ 2S a',
            'dpm++ 2m': 'DPM++ 2M',
            'dpm++ sde': 'DPM++ SDE',
            'dpm fast': 'DPM fast',
            'dpm adaptive': 'DPM adaptive',
            'lms karras': 'LMS Karras',
            'dpm2 karras': 'DPM2 Karras',
            'dpm2 a karras': 'DPM2 a Karras',
            'dpm++ 2s a karras': 'DPM++ 2S a Karras',
            'dpm++ 2m karras': 'DPM++ 2M Karras',
            'dpm++ sde karras': 'DPM++ SDE Karras'
        }
        """if sampler_name is not None:
            if sampler_name in available_samplers.keys():
                p.sampler_name = available_samplers[sampler_name]
            else:
                raise RuntimeError(
                    f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")"""

        # if self.gen.checkpoint is not None:
        #    info = sd_models.get_closet_checkpoint_match(self.gen.checkpoint)
        #    if info is None:
        #        raise RuntimeError(f"Unknown checkpoint: {self.gen.checkpoint}")
        #    sd_models.reload_model_weights(info=info)

        if self.gen.init_sample is not None:
            # TODO: cleanup init_sample remains later
            img = self.gen.init_sample
            init_image = img
            image_init0 = img
            if self.gen.use_looper and isJson(self.gen.imagesToKeyframe) and self.gen.animation_mode in ['2D', '3D']:
                init_image = Image.blend(init_image, init_image2, blendFactor)
                correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
                color_corrections = [correction_colors]

        # this is the first pass
        elif (self.gen.use_looper and self.gen.animation_mode in ['2D', '3D']) or (
                self.gen.use_init and ((self.gen.init_image != None and self.gen.init_image != ''))):
            init_image, mask_image = load_img(image_init0,  # initial init image
                                              shape=(self.gen.W, self.gen.H),
                                              use_alpha_as_mask=self.gen.use_alpha_as_mask)

        else:

            # if self.gen.animation_mode != 'Interpolation':
            #    print(f"Not using an init image (doing pure txt2img)")
            """p_txt = StableDiffusionProcessingTxt2Img( 
                sd_model=sd_model,
                outpath_samples=self.gen.tmp_deforum_run_duplicated_folder,
                outpath_grids=self.gen.tmp_deforum_run_duplicated_folder,
                prompt=p.prompt,
                styles=p.styles,
                negative_prompt=p.negative_prompt,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=p.sampler_name,
                batch_size=p.batch_size,
                n_iter=p.n_iter,
                steps=p.steps,
                cfg_scale=p.cfg_scale,
                width=p.width,
                height=p.height,
                restore_faces=p.restore_faces,
                tiling=p.tiling,
                enable_hr=None,
                denoising_strength=None,
            )"""

            # print_combined_table(args, anim_args, p_txt, keys, frame)  # print dynamic table to cli

            # if is_controlnet_enabled(controlnet_args):
            #    process_with_controlnet(p_txt, args, anim_args, loop_args, controlnet_args, root, is_img2img=False,
            #                            self.gen.frame_idx=frame)

            # processed = self.generate_txt2img(prompt, next_prompt, blend_value, negative_prompt, args, anim_args, root, self.gen.frame_idx,
            #                                init_image)

            self.genstrength = 1.0 if init_image is None else self.gen.strength
            from deforum.avfunctions.video_audio_utilities import get_frame_name

            cnet_image = None
            input_file = os.path.join(self.gen.outdir, 'inputframes',
                                      get_frame_name(self.gen.video_init_path) + f"{self.gen.frame_idx:09}.jpg")

            # if os.path.isfile(input_file):
            #     input_frame = Image.open(input_file)
            #     cnet_image = get_canny_image(input_frame)
            #     cnet_image = ImageOps.invert(cnet_image)

            if prompt == "!reset!":
                self.gen.init_image = None
                self.genstrength = 1.0
                prompt = next_prompt

            if negative_prompt == "":
                negative_prompt = self.gen.animation_prompts_negative

            gen_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": self.gen.steps,
                "seed": self.gen.seed,
                "scale": self.gen.scale,
                # Comfy uses inverted strength compared to auto1111
                "strength": self.genstrength,
                "init_image": init_image,
                "width": self.gen.W,
                "height": self.gen.H,
                "cnet_image": cnet_image,
                "next_prompt": next_prompt,
                "prompt_blend": blend_value
            }

            # print(f"DEFORUM GEN ARGS: [{gen_args}] ")

            if self.gen.enable_subseed_scheduling:
                gen_args["subseed"] = self.gen.subseed
                gen_args["subseed_strength"] = self.gen.subseed_strength
                gen_args["seed_resize_from_h"] = self.gen.seed_resize_from_h
                gen_args["seed_resize_from_w"] = self.gen.seed_resize_from_w

            processed = self.generator(**gen_args)

            torch.cuda.empty_cache()



        if processed is None:
            # Mask functions
            if self.gen.use_mask:
                mask_image = self.gen.mask_image
                mask = prepare_mask(self.gen.mask_file if mask_image is None else mask_image,
                                    (self.gen.W, self.gen.H),
                                    self.gen.mask_contrast_adjust,
                                    self.gen.mask_brightness_adjust)
                inpainting_mask_invert = self.gen.invert_mask
                inpainting_fill = self.gen.fill
                inpaint_full_res = self.gen.full_res_mask
                inpaint_full_res_padding = self.gen.full_res_mask_padding
                # prevent loaded mask from throwing errors in Image operations if completely black and crop and resize in webui pipeline
                # doing this after contrast and brightness adjustments to ensure that mask is not passed as black or blank
                mask = check_mask_for_errors(mask, self.gen.invert_mask)
                self.gen.noise_mask = mask

            else:
                mask = None

            assert not ((mask is not None and self.gen.use_mask and self.gen.overlay_mask) and (
                    self.gen.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

            image_mask = mask
            image_cfg_scale = self.gen.pix2pix_img_cfg_scale

            # print_combined_table(args, anim_args, p, keys, frame)  # print dynamic table to cli

            # if is_controlnet_enabled(controlnet_args):
            #    process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img=True,
            #                            self.gen.frame_idx=frame)
            self.gen.strength = 1.0 if init_image is None else self.gen.strength
            from deforum.avfunctions.video_audio_utilities import get_frame_name

            cnet_image = None
            input_file = os.path.join(self.gen.outdir, 'inputframes',
                                      get_frame_name(self.gen.video_init_path) + f"{self.gen.frame_idx:09}.jpg")

            # if os.path.isfile(input_file):
            #     input_frame = Image.open(input_file)
            #     cnet_image = get_canny_image(input_frame)
            #     cnet_image = ImageOps.invert(cnet_image)

            if prompt == "!reset!":
                init_image = None
                self.gen.strength = 1.0
                prompt = next_prompt

            if negative_prompt == "":
                negative_prompt = self.gen.animation_prompts_negative

            gen_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": self.gen.steps,
                "seed": self.gen.seed,
                "scale": self.gen.scale,
                "strength": self.gen.strength,
                "init_image": init_image,
                "width": self.gen.W,
                "height": self.gen.H,
                "cnet_image": cnet_image,
                "next_prompt": next_prompt,
                "prompt_blend": blend_value
            }

            #print(f"DEFORUM GEN ARGS: [{gen_args}] ")

            if self.gen.enable_subseed_scheduling:
                gen_args["subseed"] = self.gen.subseed
                gen_args["subseed_strength"] = self.gen.subseed_strength
                gen_args["seed_resize_from_h"] = self.gen.seed_resize_from_h
                gen_args["seed_resize_from_w"] = self.gen.seed_resize_from_w


            processed = self.generator(**gen_args)

        if self.gen.first_frame == None:
            self.gen.first_frame = processed

        return processed

def anim_frame_warp_cls(cls):
    if cls.gen.prev_img is not None:
        mask = None
        if cls.gen.use_depth_warping:
            if cls.gen.depth is None and cls.depth_model is not None:
                cls.gen.depth = cls.depth_model.predict(cls.gen.opencv_image, cls.gen.midas_weight, cls.gen.half_precision)
        else:
            depth = None

        if cls.gen.animation_mode == '2D':
            cls.gen.prev_img = anim_frame_warp_2d_cls(cls, cls.gen.prev_img)
        else:  # '3D'
            cls.gen.prev_img, cls.gen.mask = anim_frame_warp_3d_cls(cls, cls.gen.prev_img)
    return
def anim_frame_warp_cls_image(cls, image):
    if image is not None:
        mask = None
        if cls.gen.use_depth_warping:
            if cls.gen.depth is None and cls.depth_model is not None:
                cls.gen.depth = cls.depth_model.predict(image, cls.gen.midas_weight, cls.gen.half_precision)
        else:
            depth = None

        if cls.gen.animation_mode == '2D':
            image = anim_frame_warp_2d_cls(cls, image)
        else:  # '3D'
            image, mask = anim_frame_warp_3d_cls(cls, image)
    return image, mask

def anim_frame_warp_2d_cls(cls, image):
    angle = cls.gen.keys.angle_series[cls.gen.frame_idx]
    zoom = cls.gen.keys.zoom_series[cls.gen.frame_idx]
    translation_x = cls.gen.keys.translation_x_series[cls.gen.frame_idx]
    translation_y = cls.gen.keys.translation_y_series[cls.gen.frame_idx]
    transform_center_x = cls.gen.keys.transform_center_x_series[cls.gen.frame_idx]
    transform_center_y = cls.gen.keys.transform_center_y_series[cls.gen.frame_idx]
    center_point = (cls.gen.W * transform_center_x, cls.gen.H * transform_center_y)
    rot_mat = cv2.getRotationMatrix2D(center_point, angle, zoom)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    if cls.gen.enable_perspective_flip:
        bM = get_flip_perspective_matrix(cls.gen.W, cls.gen.H, cls.gen.keys, cls.gen.frame_idx)
        rot_mat = np.matmul(bM, rot_mat, trans_mat)
    else:
        rot_mat = np.matmul(rot_mat, trans_mat)
    return cv2.warpPerspective(
        image,
        rot_mat,
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_WRAP if cls.gen.border == 'wrap' else cv2.BORDER_REPLICATE
    )

def anim_frame_warp_3d_cls(cls, image):
    TRANSLATION_SCALE = 1.0 / 200.0  # matches Disco
    translate_xyz = [
        -cls.gen.keys.translation_x_series[cls.gen.frame_idx] * TRANSLATION_SCALE,
        cls.gen.keys.translation_y_series[cls.gen.frame_idx] * TRANSLATION_SCALE,
        -cls.gen.keys.translation_z_series[cls.gen.frame_idx] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(cls.gen.keys.rotation_3d_x_series[cls.gen.frame_idx]),
        math.radians(cls.gen.keys.rotation_3d_y_series[cls.gen.frame_idx]),
        math.radians(cls.gen.keys.rotation_3d_z_series[cls.gen.frame_idx])
    ]
    if cls.gen.enable_perspective_flip:
        image = flip_3d_perspective(cls.gen, image, cls.gen.keys, cls.gen.frame_idx)
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device="cuda"), "XYZ").unsqueeze(0)
    result, mask = transform_image_3d_new(torch.device('cuda'), image, cls.gen.depth, rot_mat, translate_xyz,
                                               cls.gen, cls.gen.keys, cls.gen.frame_idx)
    torch.cuda.empty_cache()
    return result, mask


def hybrid_composite_cls(cls):
    if cls.gen.prev_img is not None:
        video_frame = os.path.join(cls.gen.outdir, 'inputframes',
                                   get_frame_name(cls.gen.video_init_path) + f"{cls.gen.frame_idx:09}.jpg")
        video_depth_frame = os.path.join(cls.gen.outdir, 'hybridframes',
                                         get_frame_name(cls.gen.video_init_path) + f"_vid_depth{cls.gen.frame_idx:09}.jpg")
        depth_frame = os.path.join(cls.gen.outdir, f"{cls.gen.timestring}_depth_{cls.gen.frame_idx - 1:09}.png")
        mask_frame = os.path.join(cls.gen.outdir, 'hybridframes',
                                  get_frame_name(cls.gen.video_init_path) + f"_mask{cls.gen.frame_idx:09}.jpg")
        comp_frame = os.path.join(cls.gen.outdir, 'hybridframes',
                                  get_frame_name(cls.gen.video_init_path) + f"_comp{cls.gen.frame_idx:09}.jpg")
        prev_frame = os.path.join(cls.gen.outdir, 'hybridframes',
                                  get_frame_name(cls.gen.video_init_path) + f"_prev{cls.gen.frame_idx:09}.jpg")
        cls.gen.prev_img = cv2.cvtColor(cls.gen.prev_img, cv2.COLOR_BGR2RGB)
        prev_img_hybrid = Image.fromarray(cls.gen.prev_img)
        if cls.gen.hybrid_use_init_image:
            video_image = load_image(cls.gen.init_image, cls.gen.init_image_box)
        else:
            video_image = Image.open(video_frame)
        video_image = video_image.resize((cls.gen.W, cls.gen.H), PIL.Image.LANCZOS)
        hybrid_mask = None

        # composite mask types
        if cls.gen.hybrid_comp_mask_type == 'Depth':  # get depth from last generation
            hybrid_mask = Image.open(depth_frame)
        elif cls.gen.hybrid_comp_mask_type == 'Video Depth':  # get video depth
            video_depth = cls.depth_model.predict(np.array(video_image), cls.gen.midas_weight, cls.gen.half_precision)
            cls.depth_model.save(video_depth_frame, video_depth)
            hybrid_mask = Image.open(video_depth_frame)
        elif cls.gen.hybrid_comp_mask_type == 'Blend':  # create blend mask image
            hybrid_mask = Image.blend(ImageOps.grayscale(prev_img_hybrid), ImageOps.grayscale(video_image),
                                      cls.gen.hybrid_comp_schedules['mask_blend_alpha'])
        elif cls.gen.hybrid_comp_mask_type == 'Difference':  # create difference mask image
            hybrid_mask = ImageChops.difference(ImageOps.grayscale(prev_img_hybrid), ImageOps.grayscale(video_image))

        # optionally invert mask, if mask type is defined
        if cls.gen.hybrid_comp_mask_inverse and cls.gen.hybrid_comp_mask_type != "None":
            hybrid_mask = ImageOps.invert(hybrid_mask)

        # if a mask type is selected, make composition
        if hybrid_mask is None:
            hybrid_comp = video_image
        else:
            # ensure grayscale
            hybrid_mask = ImageOps.grayscale(hybrid_mask)
            # equalization before
            if cls.gen.hybrid_comp_mask_equalize in ['Before', 'Both']:
                hybrid_mask = ImageOps.equalize(hybrid_mask)
                # contrast
            hybrid_mask = ImageEnhance.Contrast(hybrid_mask).enhance(cls.gen.hybrid_comp_schedules['mask_contrast'])
            # auto contrast with cutoffs lo/hi
            if cls.gen.hybrid_comp_mask_auto_contrast:
                hybrid_mask = autocontrast_grayscale(np.array(hybrid_mask),
                                                     cls.gen.hybrid_comp_schedules['mask_auto_contrast_cutoff_low'],
                                                     cls.gen.hybrid_comp_schedules['mask_auto_contrast_cutoff_high'])
                hybrid_mask = Image.fromarray(hybrid_mask)
                hybrid_mask = ImageOps.grayscale(hybrid_mask)
            if cls.gen.hybrid_comp_save_extra_frames:
                hybrid_mask.save(mask_frame)
                # equalization after
            if cls.gen.hybrid_comp_mask_equalize in ['After', 'Both']:
                hybrid_mask = ImageOps.equalize(hybrid_mask)
                # do compositing and save
            hybrid_comp = Image.composite(prev_img_hybrid, video_image, hybrid_mask)
            if cls.gen.hybrid_comp_save_extra_frames:
                hybrid_comp.save(comp_frame)

        # final blend of composite with prev_img, or just a blend if no composite is selected
        hybrid_blend = Image.blend(prev_img_hybrid, hybrid_comp, cls.gen.hybrid_comp_schedules['alpha'])
        if cls.gen.hybrid_comp_save_extra_frames:
            hybrid_blend.save(prev_frame)

        cls.gen.prev_img = cv2.cvtColor(np.array(hybrid_blend), cv2.COLOR_RGB2BGR)

    # restore to np array and return
    return

def affine_persp_motion(cls):
    if cls.gen.hybrid_motion_use_prev_img:
        matrix = get_matrix_for_hybrid_motion_prev(cls.gen.frame_idx - 1, (cls.gen.W, cls.gen.H), cls.gen.inputfiles, cls.gen.prev_img,
                                                   cls.gen.hybrid_motion)
    else:
        matrix = get_matrix_for_hybrid_motion(cls.gen.frame_idx - 1, (cls.gen.W, cls.gen.H), cls.gen.inputfiles,
                                              cls.gen.hybrid_motion)
    cls.gen.prev_img = image_transform_ransac(cls.gen.prev_img, matrix, cls.gen.hybrid_motion)
    return
def optical_flow_motion(cls):
    if cls.gen.prev_img is not None:
        if cls.gen.hybrid_motion_use_prev_img:
            cls.gen.flow = get_flow_for_hybrid_motion_prev(cls.gen.frame_idx - 1, (cls.gen.W, cls.gen.H), cls.gen.inputfiles,
                                                   cls.gen.hybrid_frame_path, cls.gen.prev_flow, cls.gen.prev_img,
                                                   cls.gen.hybrid_flow_method, cls.raft_model,
                                                   cls.gen.hybrid_flow_consistency,
                                                   cls.gen.hybrid_consistency_blur,
                                                   cls.gen.hybrid_comp_save_extra_frames)


        else:
            cls.gen.flow = get_flow_for_hybrid_motion(cls.gen.frame_idx - 1, (cls.gen.W, cls.gen.H), cls.gen.inputfiles, cls.gen.hybrid_frame_path,
                                              cls.gen.prev_flow, cls.gen.hybrid_flow_method, cls.raft_model,
                                              cls.gen.hybrid_flow_consistency,
                                              cls.gen.hybrid_consistency_blur,
                                              cls.gen.hybrid_comp_save_extra_frames)
        cls.gen.prev_img = image_transform_optical_flow(cls.gen.prev_img, cls.gen.flow, cls.gen.hybrid_comp_schedules['flow_factor'])
        cls.gen.prev_flow = cls.gen.flow

    return
def color_match_cls(cls):
    if cls.gen.color_match_sample is None and cls.gen.prev_img is not None:
            cls.gen.color_match_sample = cls.gen.prev_img.copy()
    elif cls.gen.prev_img is not None:
        cls.gen.prev_img = maintain_colors(cls.gen.prev_img, cls.gen.color_match_sample, cls.gen.color_coherence)
    return
def set_contrast_image(cls):
    if cls.gen.prev_img is not None:
        # intercept and override to grayscale
        if cls.gen.color_force_grayscale:
            cls.gen.prev_img = cv2.cvtColor(cls.gen.prev_img, cv2.COLOR_BGR2GRAY)
            cls.gen.prev_img = cv2.cvtColor(cls.gen.prev_img, cv2.COLOR_GRAY2BGR)

        # apply scaling
        cls.gen.contrast_image = (cls.gen.prev_img * cls.gen.contrast).round().astype(np.uint8)
        # anti-blur
        if cls.gen.amount > 0:
            cls.gen.contrast_image = unsharp_mask(cls.gen.contrast_image, (cls.gen.kernel, cls.gen.kernel), cls.gen.sigma, cls.gen.amount, cls.gen.threshold,
                                          cls.gen.mask_image if cls.gen.use_mask else None)
    return
def handle_noise_mask(cls):
    cls.gen.noise_mask = compose_mask_with_check(cls.gen, cls.gen, cls.gen.noise_mask_seq, cls.gen.noise_mask_vals, Image.fromarray(
        cv2.cvtColor(cls.gen.contrast_image, cv2.COLOR_BGR2RGB)))
    return
def add_noise_cls(cls):
    if cls.gen.prev_img is not None:
        noised_image = add_noise(cls.gen.contrast_image, cls.gen.noise, cls.gen.seed, cls.gen.noise_type,
                                 (cls.gen.perlin_w, cls.gen.perlin_h, cls.gen.perlin_octaves,
                                  cls.gen.perlin_persistence),
                                 cls.gen.noise_mask, cls.gen.invert_mask)

        # use transformed previous frame as init for current
        cls.gen.use_init = True
        cls.gen.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
        cls.gen.strength = max(0.0, min(1.0, cls.gen.strength))
    return
def get_generation_params(cls):

    frame_idx = cls.gen.frame_idx
    keys = cls.gen.keys

    # print(f"\033[36mAnimation frame: \033[0m{frame_idx}/{cls.gen.max_frames}  ")

    cls.gen.noise = keys.noise_schedule_series[frame_idx]
    cls.gen.strength = keys.strength_schedule_series[frame_idx]
    cls.gen.scale = keys.cfg_scale_schedule_series[frame_idx]
    cls.gen.contrast = keys.contrast_schedule_series[frame_idx]
    cls.gen.kernel = int(keys.kernel_schedule_series[frame_idx])
    cls.gen.sigma = keys.sigma_schedule_series[frame_idx]
    cls.gen.amount = keys.amount_schedule_series[frame_idx]
    cls.gen.threshold = keys.threshold_schedule_series[frame_idx]
    cls.gen.cadence_flow_factor = keys.cadence_flow_factor_schedule_series[frame_idx]
    cls.gen.redo_flow_factor = keys.redo_flow_factor_schedule_series[frame_idx]
    cls.gen.hybrid_comp_schedules = {
        "alpha": keys.hybrid_comp_alpha_schedule_series[frame_idx],
        "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[frame_idx],
        "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
        "mask_auto_contrast_cutoff_low": int(
            keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx]),
        "mask_auto_contrast_cutoff_high": int(
            keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx]),
        "flow_factor": keys.hybrid_flow_factor_schedule_series[frame_idx]
    }
    cls.gen.scheduled_sampler_name = None
    cls.gen.scheduled_clipskip = None
    cls.gen.scheduled_noise_multiplier = None
    cls.gen.scheduled_ddim_eta = None
    cls.gen.scheduled_ancestral_eta = None

    cls.gen.mask_seq = None
    cls.gen.noise_mask_seq = None
    if cls.gen.enable_steps_scheduling and keys.steps_schedule_series[frame_idx] is not None:
        cls.gen.steps = int(keys.steps_schedule_series[frame_idx])
    if cls.gen.enable_sampler_scheduling and keys.sampler_schedule_series[frame_idx] is not None:
        cls.gen.scheduled_sampler_name = keys.sampler_schedule_series[frame_idx].casefold()
    if cls.gen.enable_clipskip_scheduling and keys.clipskip_schedule_series[frame_idx] is not None:
        cls.gen.scheduled_clipskip = int(keys.clipskip_schedule_series[frame_idx])
    if cls.gen.enable_noise_multiplier_scheduling and keys.noise_multiplier_schedule_series[
        frame_idx] is not None:
        cls.gen.scheduled_noise_multiplier = float(keys.noise_multiplier_schedule_series[frame_idx])
    if cls.gen.enable_ddim_eta_scheduling and keys.ddim_eta_schedule_series[frame_idx] is not None:
        cls.gen.scheduled_ddim_eta = float(keys.ddim_eta_schedule_series[frame_idx])
    if cls.gen.enable_ancestral_eta_scheduling and keys.ancestral_eta_schedule_series[frame_idx] is not None:
        cls.gen.scheduled_ancestral_eta = float(keys.ancestral_eta_schedule_series[frame_idx])
    if cls.gen.use_mask and keys.mask_schedule_series[frame_idx] is not None:
        cls.gen.mask_seq = keys.mask_schedule_series[frame_idx]
    if cls.gen.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
        cls.gen.noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]

    if cls.gen.use_mask and not cls.gen.use_noise_mask:
        cls.gen.noise_mask_seq = cls.gen.mask_seq

    cls.gen.depth = None

    # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
    cls.gen.pix2pix_img_cfg_scale = float(cls.gen.keys.pix2pix_img_cfg_scale_series[cls.gen.frame_idx])

    # grab prompt for current frame
    cls.gen.prompt = cls.gen.prompt_series[cls.gen.frame_idx]

    # if cls.gen.seed_behavior == 'schedule' or parseq_adapter.manages_seed():
    #     cls.gen.seed = int(keys.seed_schedule_series[frame_idx])

    if cls.gen.enable_checkpoint_scheduling:
        cls.gen.checkpoint = cls.gen.keys.checkpoint_schedule_series[cls.gen.frame_idx]
    else:
        cls.gen.checkpoint = None

    # SubSeed scheduling
    if cls.gen.enable_subseed_scheduling:
        cls.gen.subseed = int(cls.gen.keys.subseed_schedule_series[cls.gen.frame_idx])
        cls.gen.subseed_strength = float(cls.gen.keys.subseed_strength_schedule_series[cls.gen.frame_idx])

    # if parseq_adapter.manages_seed():
    #     cls.gen.enable_subseed_scheduling = True
    #     cls.gen.subseed = int(keys.subseed_schedule_series[frame_idx])
    #     cls.gen.subseed_strength = keys.subseed_strength_schedule_series[frame_idx]

    # set value back into the prompt - prepare and report prompt and seed
    cls.gen.prompt = prepare_prompt(cls.gen.prompt, cls.gen.max_frames, cls.gen.seed, cls.gen.frame_idx)

    # grab init image for current frame
    if cls.gen.using_vid_init:
        init_frame = get_next_frame(cls.gen.outdir, cls.gen.video_init_path, cls.gen.frame_idx, False)
        # print(f"Using video init frame {init_frame}")
        cls.gen.init_image = init_frame
        cls.gen.init_image_box = None  # init_image_box not used in this case
        cls.gen.strength = max(0.0, min(1.0, cls.gen.strength))
    if cls.gen.use_mask_video:
        cls.gen.mask_file = get_mask_from_file(get_next_frame(cls.gen.outdir, cls.gen.video_mask_path, cls.gen.frame_idx, True),
                                            cls.gen)
        cls.gen.noise_mask = get_mask_from_file(
            get_next_frame(cls.gen.outdir, cls.gen.video_mask_path, cls.gen.frame_idx, True), cls.gen)

        cls.gen.mask_vals['video_mask'] = get_mask_from_file(
            get_next_frame(cls.gen.outdir, cls.gen.video_mask_path, cls.gen.frame_idx, True), cls.gen)

    if cls.gen.use_mask:
        cls.gen.mask_image = compose_mask_with_check(cls.gen, cls.gen, cls.gen.mask_seq, cls.gen.mask_vals,
                                                  cls.gen.init_sample) if cls.gen.init_sample is not None else None  # we need it only after the first frame anyway

    # setting up some arguments for the looper
    cls.gen.imageStrength = cls.gen.loopSchedulesAndData.image_strength_schedule_series[cls.gen.frame_idx]
    cls.gen.blendFactorMax = cls.gen.loopSchedulesAndData.blendFactorMax_series[cls.gen.frame_idx]
    cls.gen.blendFactorSlope = cls.gen.loopSchedulesAndData.blendFactorSlope_series[cls.gen.frame_idx]
    cls.gen.tweeningFrameSchedule = cls.gen.loopSchedulesAndData.tweening_frames_schedule_series[cls.gen.frame_idx]
    cls.gen.colorCorrectionFactor = cls.gen.loopSchedulesAndData.color_correction_factor_series[cls.gen.frame_idx]
    cls.gen.use_looper = cls.gen.loopSchedulesAndData.use_looper
    cls.gen.imagesToKeyframe = cls.gen.loopSchedulesAndData.imagesToKeyframe

    # if 'img2img_fix_steps' in opts.data and opts.data[
    #     "img2img_fix_steps"]:  # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
    #     opts.data["img2img_fix_steps"] = False
    # if scheduled_clipskip is not None:
    #     opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip
    # if scheduled_noise_multiplier is not None:
    #     opts.data["initial_noise_multiplier"] = scheduled_noise_multiplier
    # if scheduled_ddim_eta is not None:
    #     opts.data["eta_ddim"] = scheduled_ddim_eta
    # if scheduled_ancestral_eta is not None:
    #     opts.data["eta_ancestral"] = scheduled_ancestral_eta

    # if cls.gen.animation_mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram):
    #     if predict_depths: depth_model.to('cpu')
    #     devices.torch_gc()
    #     lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
    #     sd_hijack.model_hijack.hijack(sd_model)
    return
def optical_flow_redo(cls):
    optical_flow_redo_generation = cls.gen.optical_flow_redo_generation  # if not cls.gen.motion_preview_mode else 'None'

    # optical flow redo before generation
    if optical_flow_redo_generation != 'None' and cls.gen.prev_img is not None and cls.gen.strength > 0:
        stored_seed = cls.gen.seed
        cls.gen.seed = random.randint(0, 2 ** 32 - 1)
        # print(
        #     f"Optical flow redo is diffusing and warping using {optical_flow_redo_generation} and seed {cls.gen.seed} optical flow before generation.")

        disposable_image = cls.generate()

        disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
        disposable_flow = get_flow_from_images(cls.gen.prev_img, disposable_image, optical_flow_redo_generation, cls.raft_model)
        disposable_image = cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB)
        disposable_image = image_transform_optical_flow(disposable_image, disposable_flow, cls.gen.redo_flow_factor)
        cls.gen.seed = stored_seed
        cls.gen.init_sample = Image.fromarray(disposable_image)
        del (disposable_image, disposable_flow, stored_seed)
        gc.collect()

    return
def diffusion_redo(cls):
    if int(cls.gen.diffusion_redo) > 0 and cls.gen.prev_img is not None and cls.gen.strength > 0 and not cls.gen.motion_preview_mode:
        stored_seed = cls.gen.seed
        for n in range(0, int(cls.gen.diffusion_redo)):
            # print(f"Redo generation {n + 1} of {int(cls.gen.diffusion_redo)} before final generation")
            cls.gen.seed = random.randint(0, 2 ** 32 - 1)
            disposable_image = cls.generate()
            disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
            # color match on last one only
            if n == int(cls.gen.diffusion_redo):
                disposable_image = maintain_colors(cls.gen.prev_img, cls.gen.color_match_sample, cls.gen.color_coherence)
            cls.gen.seed = stored_seed
            cls.gen.init_sample = Image.fromarray(cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB))
        del (disposable_image, stored_seed)
        gc.collect()

    return
def main_generate_with_cls(cls):
    cls.gen.image = cls.generate()

    return
def post_hybrid_composite_cls(cls):
    # do hybrid video after generation
    if cls.gen.frame_idx > 0 and cls.gen.hybrid_composite == 'After Generation':
        image = cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR)
        cls.gen, image = hybrid_composite(cls.gen, cls.gen, cls.gen.frame_idx, image, cls.depth_model, cls.gen.hybrid_comp_schedules, cls.gen)
        cls.gen.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return
def post_color_match_with_cls(cls):
    # color matching on first frame is after generation, color match was collected earlier, so we do an extra generation to avoid the corruption introduced by the color match of first output
    if cls.gen.frame_idx == 0 and (cls.gen.color_coherence == 'Image' or (
            cls.gen.color_coherence == 'Video Input' and cls.gen.hybrid_available)):
        image = maintain_colors(cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR), cls.gen.color_match_sample,
                                cls.gen.color_coherence)
        cls.gen.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif cls.gen.color_match_sample is not None and cls.gen.color_coherence != 'None' and not cls.gen.legacy_colormatch:
        image = maintain_colors(cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR), cls.gen.color_match_sample,
                                cls.gen.color_coherence)
        cls.gen.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return
def overlay_mask_cls(cls):
    # intercept and override to grayscale
    if cls.gen.color_force_grayscale:
        image = ImageOps.grayscale(cls.gen.image)
        cls.gen.image = ImageOps.colorize(image, black="black", white="white")

    # overlay mask
    if cls.gen.overlay_mask and (cls.gen.use_mask_video or cls.gen.use_mask):
        cls.gen.image = do_overlay_mask(cls.gen, cls.gen, cls.gen.image, cls.gen.frame_idx)

    # on strength 0, set color match to generation
    if ((not cls.gen.legacy_colormatch and not cls.gen.use_init) or (
            cls.gen.legacy_colormatch and cls.gen.strength == 0)) and not cls.gen.color_coherence in ['Image',
                                                                                                  'Video Input']:
        cls.gen.color_match_sample = cv2.cvtColor(np.asarray(cls.gen.image), cv2.COLOR_RGB2BGR)
    return
def post_gen_cls(cls):

    if cls.gen.frame_idx < cls.gen.max_frames:

        cls.gen.opencv_image = cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR)
        cls.images.append(cls.gen.opencv_image.copy())

        if not cls.gen.using_vid_init:
            cls.gen.prev_img = cls.gen.opencv_image

        if cls.gen.turbo_steps > 1:
            cls.gen.turbo_prev_image, cls.gen.turbo_prev_frame_idx = cls.gen.turbo_next_image, cls.gen.turbo_next_frame_idx
            cls.gen.turbo_next_image, cls.gen.turbo_next_frame_idx = cls.gen.opencv_image, cls.gen.frame_idx
            cls.gen.frame_idx += cls.gen.turbo_steps
        else:
            filename = f"{cls.gen.timestring}_{cls.gen.frame_idx:09}.png"

            #TODO IMPLEMENT CLS SAVING
            if not cls.gen.store_frames_in_ram:

                save_image(cls.gen.image, 'PIL', filename, cls.gen, cls.gen, cls.gen)

            if cls.gen.save_depth_maps:
                # if cmd_opts.lowvram or cmd_opts.medvram:
                #     lowvram.send_everything_to_cpu()
                #     sd_hijack.model_hijack.undo_hijack(sd_model)
                #     devices.torch_gc()
                #     depth_model.to(root.device)
                cls.gen.depth = cls.depth_model.predict(cls.gen.opencv_image, cls.gen.midas_weight, cls.gen.half_precision)
                cls.depth_model.save(os.path.join(cls.gen.outdir, f"{cls.gen.timestring}_depth_{cls.gen.frame_idx:09}.png"), cls.gen.depth)
                # if cmd_opts.lowvram or cmd_opts.medvram:
                #     depth_model.to('cpu')
                #     devices.torch_gc()
                #     lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
                #     sd_hijack.model_hijack.hijack(sd_model)
            cls.gen.frame_idx += 1
        # state.assign_current_image(image)
        done = cls.datacallback({"image": cls.gen.image})
        #TODO IMPLEMENT CLS NEXT SEED
        cls.gen.seed = next_seed(cls.gen, cls.gen)


    return



def make_cadence_frames(cls):
    if cls.gen.turbo_steps > 1:
        tween_frame_start_idx = max(cls.gen.start_frame, cls.gen.frame_idx - cls.gen.turbo_steps)
        cadence_flow = None
        for tween_frame_idx in range(tween_frame_start_idx, cls.gen.frame_idx):
            # update progress during cadence
            # state.job = f"frame {tween_frame_idx + 1}/{cls.gen.max_frames}"
            # state.job_no = tween_frame_idx + 1
            # cadence vars
            tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(cls.gen.frame_idx - tween_frame_start_idx)
            advance_prev = cls.gen.turbo_prev_image is not None and tween_frame_idx > cls.gen.turbo_prev_frame_idx
            advance_next = tween_frame_idx > cls.gen.turbo_next_frame_idx

            # optical flow cadence setup before animation warping
            if cls.gen.animation_mode in ['2D', '3D'] and cls.gen.optical_flow_cadence != 'None':
                if cls.gen.keys.strength_schedule_series[tween_frame_start_idx] > 0:
                    if cadence_flow is None and cls.gen.turbo_prev_image is not None and cls.gen.turbo_next_image is not None:
                        cadence_flow = get_flow_from_images(cls.gen.turbo_prev_image, cls.gen.turbo_next_image,
                                                            cls.gen.optical_flow_cadence, cls.raft_model) / 2
                        turbo_next_image = image_transform_optical_flow(cls.gen.turbo_next_image, -cadence_flow, 1)

                # if opts.data.get("deforum_save_gen_info_as_srt"):
                #     params_to_print = opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])
                #     params_string = format_animation_params(keys, prompt_series, tween_frame_idx, params_to_print)
                #     write_frame_subtitle(srt_filename, tween_frame_idx, srt_frame_duration,
                #                          f"F#: {tween_frame_idx}; Cadence: {tween < 1.0}; Seed: {cls.gen.seed}; {params_string}")
                params_string = None

            # print(
            #     f"Creating in-between {'' if cadence_flow is None else cls.gen.optical_flow_cadence + ' optical flow '}cadence frame: {tween_frame_idx}; tween:{tween:0.2f};")

            if cls.depth_model is not None:
                assert (cls.gen.turbo_next_image is not None)
                depth = cls.depth_model.predict(cls.gen.turbo_next_image, cls.gen.midas_weight, cls.gen.half_precision)

            if advance_prev:
                cls.gen.turbo_prev_image, _ = anim_frame_warp_cls_image(cls, cls.gen.turbo_prev_image)
            if advance_next:
                cls.gen.turbo_next_image, _ = anim_frame_warp_cls_image(cls, cls.gen.turbo_prev_image)

            # hybrid video motion - warps turbo_prev_image or turbo_next_image to match motion
            if tween_frame_idx > 0:
                if cls.gen.hybrid_motion in ['Affine', 'Perspective']:
                    if cls.gen.hybrid_motion_use_prev_img:
                        matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx - 1, (cls.gen.W, cls.gen.H),
                                                                   cls.gen.inputfiles, cls.gen.prev_img, cls.gen.hybrid_motion)
                        if advance_prev:
                            cls.gen.turbo_prev_image = image_transform_ransac(cls.gen.turbo_prev_image, matrix,
                                                                      cls.gen.hybrid_motion)
                        if advance_next:
                            cls.gen.turbo_next_image = image_transform_ransac(cls.gen.turbo_next_image, matrix,
                                                                      cls.gen.hybrid_motion)
                    else:
                        matrix = get_matrix_for_hybrid_motion(tween_frame_idx - 1, (cls.gen.W, cls.gen.H), cls.gen.inputfiles,
                                                              cls.gen.hybrid_motion)
                        if advance_prev:
                            cls.gen.turbo_prev_image = image_transform_ransac(cls.gen.turbo_prev_image, matrix,
                                                                      cls.gen.hybrid_motion)
                        if advance_next:
                            cls.gen.turbo_next_image = image_transform_ransac(cls.gen.turbo_next_image, matrix,
                                                                      cls.gen.hybrid_motion)
                if cls.gen.hybrid_motion in ['Optical Flow']:
                    if cls.gen.hybrid_motion_use_prev_img:
                        cls.gen.flow = get_flow_for_hybrid_motion_prev(tween_frame_idx - 1, (cls.gen.W, cls.gen.H), cls.gen.inputfiles,
                                                               cls.gen.hybrid_frame_path, cls.gen.prev_flow, cls.gen.prev_img,
                                                               cls.gen.hybrid_flow_method, cls.raft_model,
                                                               cls.gen.hybrid_flow_consistency,
                                                               cls.gen.hybrid_consistency_blur,
                                                               cls.gen.hybrid_comp_save_extra_frames)
                        if advance_prev:
                            cls.gen.turbo_prev_image = image_transform_optical_flow(cls.gen.turbo_prev_image, cls.gen.flow,
                                                                            cls.gen.hybrid_comp_schedules['flow_factor'])
                        if advance_next:
                            cls.gen.turbo_next_image = image_transform_optical_flow(cls.gen.turbo_next_image, cls.gen.flow,
                                                                            cls.gen.hybrid_comp_schedules['flow_factor'])
                        cls.gen.prev_flow = cls.gen.flow
                    else:
                        cls.gen.flow = get_flow_for_hybrid_motion(tween_frame_idx - 1, (cls.gen.W, cls.gen.H), cls.gen.inputfiles,
                                                          cls.gen.hybrid_frame_path, cls.gen.prev_flow,
                                                          cls.gen.hybrid_flow_method, cls.raft_model,
                                                          cls.gen.hybrid_flow_consistency,
                                                          cls.gen.hybrid_consistency_blur,
                                                          cls.gen.hybrid_comp_save_extra_frames)
                        if advance_prev:
                            cls.gen.turbo_prev_image = image_transform_optical_flow(cls.gen.turbo_prev_image, cls.gen.flow,
                                                                            cls.gen.hybrid_comp_schedules['flow_factor'])
                        if advance_next:
                            cls.gen.turbo_next_image = image_transform_optical_flow(cls.gen.turbo_next_image, cls.gen.flow,
                                                                            cls.gen.hybrid_comp_schedules['flow_factor'])
                        cls.gen.prev_flow = cls.gen.flow

            # do optical flow cadence after animation warping
            if cadence_flow is not None:
                cadence_flow = abs_flow_to_rel_flow(cadence_flow, cls.gen.W, cls.gen.H)
                cadence_flow, _, _ = anim_frame_warp(cadence_flow, cls.gen, cls.gen, cls.gen.keys, tween_frame_idx, cls.depth_model,
                                                     depth=depth, device=cls.gen.device,
                                                     half_precision=cls.gen.half_precision)
                cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, cls.gen.W, cls.gen.H) * tween
                if advance_prev:
                    cls.gen.turbo_prev_image = image_transform_optical_flow(cls.gen.turbo_prev_image, cadence_flow_inc,
                                                                    cls.gen.cadence_flow_factor)
                if advance_next:
                    cls.gen.turbo_next_image = image_transform_optical_flow(cls.gen.turbo_next_image, cadence_flow_inc,
                                                                    cls.gen.cadence_flow_factor)

            cls.gen.turbo_prev_frame_idx = cls.gen.turbo_next_frame_idx = tween_frame_idx

            if cls.gen.turbo_prev_image is not None and tween < 1.0:
                cls.gen.img = cls.gen.turbo_prev_image * (1.0 - tween) + cls.gen.turbo_next_image * tween
            else:
                cls.gen.img = cls.gen.turbo_next_image

            # intercept and override to grayscale
            if cls.gen.color_force_grayscale:
                cls.gen.img = cv2.cvtColor(cls.gen.img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                cls.gen.img = cv2.cvtColor(cls.gen.img, cv2.COLOR_GRAY2BGR)

                # overlay mask
            if cls.gen.overlay_mask and (cls.gen.use_mask_video or cls.gen.use_mask):
                cls.gen.img = do_overlay_mask(cls.gen, cls.gen, cls.gen.img, tween_frame_idx, True)

            # get prev_img during cadence
            cls.gen.prev_img = cls.gen.img

            # current image update for cadence frames (left commented because it doesn't currently update the preview)
            # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

            # saving cadence frames
            if not cls.gen.store_frames_in_ram:
                filename = f"{cls.gen.timestring}_{tween_frame_idx:09}.png"
                cv2.imwrite(os.path.join(cls.gen.outdir, filename), cls.gen.img)
            done = cls.datacallback({"cadence_frame": Image.fromarray(cls.gen.img)})

            if cls.gen.save_depth_maps:
                cls.depth_model.save(os.path.join(cls.gen.outdir, f"{cls.gen.timestring}_depth_{tween_frame_idx:09}.png"),
                                 depth)


def color_match_video_input(cls):
    if int(cls.gen.frame_idx) % int(cls.gen.color_coherence_video_every_N_frames) == 0:
        prev_vid_img = Image.open(os.path.join(cls.outdir, 'inputframes', get_frame_name(
            cls.video_init_path) + f"{cls.gen.frame_idx:09}.jpg"))
        cls.gen.prev_vid_img = prev_vid_img.resize((cls.W, cls.H), PIL.Image.LANCZOS)
        color_match_sample = np.asarray(cls.gen.prev_vid_img)
        cls.gen.color_match_sample = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2BGR)
