import importlib
import json
import math
import os
import secrets
import sys
import textwrap
import time
from datetime import datetime
from typing import Optional, Callable

import numexpr
import numpy as np
import pandas as pd
import torch
from PIL import Image

from deforum.animation.animation_key_frames import DeformAnimKeys, LooperAnimKeys
from deforum.animation.base_args import DeforumAnimPrompts
from deforum.animation.new_args import DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, RootArgs, ParseqArgs, LoopArgs
from deforum.avfunctions.hybridvideo.hybrid_video import hybrid_generation
from deforum.avfunctions.image.load_images import load_img, prepare_mask, check_mask_for_errors
from deforum.avfunctions.interpolation.RAFT import RAFT
from deforum.cmd import extract_values
from deforum.datafunctions.prompt import check_is_number, split_weighted_subprompts
from deforum.exttools.depth import DepthModel
from deforum.general_utils import pairwise_repl

from deforum.pipelines.animation_elements import (anim_frame_warp_cls,
                                                 hybrid_composite_cls,
                                                 affine_persp_motion,
                                                 optical_flow_motion,
                                                 color_match_cls,
                                                 set_contrast_image,
                                                 handle_noise_mask,
                                                 add_noise_cls,
                                                 get_generation_params,
                                                 optical_flow_redo,
                                                 main_generate_with_cls,
                                                 post_hybrid_composite_cls,
                                                 post_gen_cls,
                                                 post_color_match_with_cls,
                                                 film_interpolate_cls,
                                                 overlay_mask_cls,
                                                 make_cadence_frames,
                                                 color_match_video_input,
                                                 diffusion_redo)


#root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deforum.shared import root_path, other_model_dir

default_cache_folder = os.path.join(root_path, "models/checkpoints")

script_start_time = time.time()


class DeforumBase:
    """
    Base class for the Deforum animation processing.

    Provides methods for initializing the Deforum animation pipeline using specific generator and pipeline configurations.
    """

    @classmethod
    def from_civitai(cls,
                     modelid: str = None,
                     generator: str = "comfy",
                     pipeline: str = "DeforumAnimationPipeline",
                     cache_dir: str = default_cache_folder,
                     lcm: bool = False) -> 'DeforumBase':
        """
        Class method to initialize a Deforum animation pipeline using specific configurations.

        Args:
            modelid (str, optional): Identifier for the model to fetch from CivitAi. Defaults to None.
            generator (str, optional): The generator to use for the Deforum animation. Defaults to "comfy".
            pipeline (str, optional): The pipeline to use for the animation processing. Defaults to "DeforumAnimationPipeline".
            cache_dir (str, optional): Directory for caching models. Defaults to default_cache_folder.
            lcm (bool, optional): Flag to determine if low-complexity mode should be activated. Defaults to False.

        Returns:
            DeforumBase: Initialized Deforum animation pipeline object.

        Raises:
            AssertionError: Raised if the specified generator or pipeline is not available or if cache directory issues occur.
        """

        from deforum import available_engines
        assert generator in available_engines, f"Make sure to use one of the available engines: {available_engines}"

        from deforum import available_pipelines
        assert pipeline in available_pipelines, f"Make sure to use one of the available pipelines: {available_pipelines}"

        # Ensure cache directory exists
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        assert os.path.isdir(
            cache_dir), "Could not create the requested cache dir, make sure the application has permissions"

        # Download model from CivitAi if specified
        if modelid is not None:
            from deforum import fetch_and_download_model
            model_file_name = fetch_and_download_model(modelId=modelid)
            model_path = os.path.join(cache_dir, model_file_name)
        else:
            model_path = None

        # Initialize the generator
        from deforum import ComfyDeforumGenerator
        generator = ComfyDeforumGenerator(model_path=model_path, lcm=lcm)

        # Import the relevant pipeline class
        deforum_module = importlib.import_module(cls.__module__.split(".")[0])
        pipeline_class = getattr(deforum_module, cls.__name__)

        # Create and return pipeline object
        pipe = pipeline_class(generator)
        return pipe


class DeforumGenerationObject:
    """
    Class representing the generation object for Deforum animations.

    This class contains all the required attributes and methods for defining, managing, and manipulating the animation generation object.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the generation object with default values and any provided arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        # Extract default values from various argument classes
        base_args = extract_values(DeforumArgs())
        anim_args = extract_values(DeforumAnimArgs())
        parseg_args = extract_values(ParseqArgs())
        loop_args = extract_values(LoopArgs())
        root = RootArgs()
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        merged_args = {**base_args, **anim_args, **parseg_args, **loop_args, **output_args_dict, **root}

        # Set all default values as attributes
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

        # Handle seed initialization
        if self.seed == -1 or self.seed == "-1":
            setattr(self, "seed", secrets.randbelow(999999999999999999))
            setattr(self, "raw_seed", int(self.seed))
            setattr(self, "seed_internal", 0)
        else:
            self.seed = int(self.seed)



        # Further attribute initializations
        self.prompts = None
        self.frame_interpolation_engine = None
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
        self.hybrid_use_full_video = False
        self.turbo_steps = self.diffusion_cadence

        # Set all provided keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, attribute, default=None):
        """
        Retrieve the value of a specified attribute or a default value if not present.

        Args:
            attribute (str): Name of the attribute to retrieve.
            default (any, optional): Default value to return if attribute is not present.

        Returns:
            any: Value of the attribute or the default value.
        """
        return getattr(self, attribute, default)

    def to_dict(self) -> dict:
        """
        Convert all instance attributes to a dictionary.

        Returns:
            dict: Dictionary containing all instance attributes.
        """
        return self.__dict__.copy()

    def update_from_kwargs(self, *args, **kwargs):
        """
        Update object attributes using provided keyword arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_settings_file(cls, settings_file_path: str = None) -> 'DeforumGenerationObject':
        """
        Create an instance of the generation object using settings from a provided file.

        Args:
            settings_file_path (str, optional): Path to the settings file.

        Returns:
            DeforumGenerationObject: Initialized generation object instance.

        Raises:
            ValueError: If the provided file type is unsupported.
        """
        instance = cls()

        # Load data from provided file
        if settings_file_path and os.path.isfile(settings_file_path):
            file_ext = os.path.splitext(settings_file_path)[1]
            if file_ext == '.json':
                with open(settings_file_path, 'r') as f:
                    data = json.load(f)
            elif file_ext == '.txt':
                with open(settings_file_path, 'r') as f:
                    content = f.read()
                    data = json.loads(content)
            else:
                raise ValueError("Unsupported file type")

            # Update instance attributes using loaded data
            for key, value in data.items():
                setattr(instance, key, value)

        # Additional attribute updates based on loaded data
        if hasattr(instance, "diffusion_cadence"):
            instance.turbo_steps =  int(instance.diffusion_cadence)
        if hasattr(instance, "using_video_init") and instance.using_video_init:
            instance.turbo_steps = 1
        if instance.prompts is not None:
            instance.animation_prompts = instance.prompts

        return instance



class DeforumKeyFrame:
    """
    Class representing the key frame for Deforum animations.

    This class contains attributes that define a specific frame's characteristics in the Deforum animation process.
    """

    def get(self, attribute, default=None) -> any:
        """
        Retrieve the value of a specified attribute or a default value if not present.

        Args:
            attribute (str): Name of the attribute to retrieve.
            default (any, optional): Default value to return if attribute is not present.

        Returns:
            any: Value of the attribute or the default value.
        """
        return getattr(self, attribute, default)

    @classmethod
    def from_keys(cls, keys, frame_idx) -> 'DeforumKeyFrame':
        """
        Create an instance of the key frame object using settings from provided keys and frame index.

        Args:
            keys: Object containing animation schedule series attributes.
            frame_idx (int): Index of the frame to retrieve settings for.

        Returns:
            DeforumKeyFrame: Initialized key frame object instance.
        """
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
    """
    Logger class for logging messages to a file with optional timestamps.

    Provides functionalities to start a logging session, log messages, and close the logging session.
    """

    def __init__(self, root_path: str):
        """
        Initialize the Logger object.

        Args:
            root_path (str): Root directory path where the log files will be stored.
        """
        self.root_path = root_path
        self.log_file = None
        self.current_datetime = datetime.now()
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.terminal_width = self.get_terminal_width()

    def get_terminal_width(self) -> int:
        """Get the width of the terminal.

        Returns:
            int: Width of the terminal, or a default width if the terminal size cannot be determined.
        """
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except (ImportError, AttributeError):
            # Default width
            return 80

    def start_session(self):
        """Start a logging session by creating or appending to a log file."""
        year, month, day = self.current_datetime.strftime('%Y'), self.current_datetime.strftime(
            '%m'), self.current_datetime.strftime('%d')
        log_path = os.path.join(self.root_path, 'logs', year, month, day)
        os.makedirs(log_path, exist_ok=True)

        self.log_file = open(os.path.join(log_path, f"metrics_{self.timestamp}.log"), "a")
        self.log_file.write("=" * self.terminal_width + "\n")
        self.log_file.write("Log Session Started: " + self.timestamp.center(self.terminal_width - 20) + "\n")
        self.log_file.write("=" * self.terminal_width + "\n")

    def log(self, message: str, timestamped: bool = True):
        """
        Log a message to the log file.

        Args:
            message (str): The message to be logged.
            timestamped (bool, optional): If True, add a timestamp prefix to the message. Default is True.
        """
        if timestamped:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f"[{timestamp}] {message}"

        # Wrap the message to the terminal width
        wrapped_text = "\n".join(textwrap.wrap(message, width=self.terminal_width))
        self.log_file.write(f"{wrapped_text}\n")

    def __call__(self, message: str, timestamped: bool = True, *args, **kwargs):
        """Allow the Logger object to be called as a function."""
        self.log(message, timestamped)

    def close_session(self):
        """End the logging session and close the log file."""
        if self.log_file:
            self.log_file.write("\n" + "=" * self.terminal_width + "\n")
            self.log_file.write("Log Session Ended".center(self.terminal_width) + "\n")
            self.log_file.write("=" * self.terminal_width + "\n")
            self.log_file.close()

# class DeforumPipeline(DeforumBase):
#
#     def __init__(self,
#                  generator:Callable,
#                  logger:Optional[Callable]=None):
#
#         super().__init__()
#
#         # assert generator in available_engine_classes, f"Make sure to use one of the available engines: {available_engine_classes}"
#
#         self.generator = generator
#         self.logger = logger
#
#         self.prep_fns = []
#         self.shoot_fns = []
#         self.post_fns = []



class DeforumAnimationPipeline(DeforumBase):
    """
    Animation pipeline for Deforum.

    Provides a mechanism to run an animation generation process using the provided generator.
    Allows for pre-processing, main loop, and post-processing steps.
    Uses a logger to record the metrics and timings of each step in the pipeline.
    """
    def __init__(self, generator: Callable, logger: Optional[Callable] = None):
        """
        Initialize the DeforumAnimationPipeline.

        Args:
            generator (Callable): The generator function for producing animations.
            logger (Optional[Callable], optional): Optional logger function. Defaults to None.
        """
        super().__init__()

        self.generator = generator

        if logger == None:
            self.logger = Logger(root_path)
        else:
            self.logger = logger

        self.prep_fns = []
        self.shoot_fns = []
        self.post_fns = []
        self.images = []

    def __call__(self, settings_file: str = None, *args, **kwargs) -> DeforumGenerationObject:
        """
        Execute the animation pipeline.

        Args:
            settings_file (str, optional): Path to the settings file. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            DeforumGenerationObject: The generated object after the pipeline execution.
        """
        self.logger.start_session()

        start_total_time = time.time()

        duration = (start_total_time - script_start_time) * 1000
        self.logger.log(f"Script startup / model loading took {duration:.2f} ms")

        if settings_file:
            self.gen = DeforumGenerationObject.from_settings_file(settings_file)
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
            self.depth_model = DepthModel(other_model_dir, device, self.gen.half_precision,
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
        """
        Set up the list of functions to be executed during the main loop of the animation pipeline.

        This method populates the `shoot_fns` list with functions based on the configuration set in the `gen` object.
        Certain functions are added to the list based on the conditions provided by the attributes of the `gen` object.
        Additionally, post-processing functions can be added to the `post_fns` list.
        """
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

        if self.gen.frame_interpolation_engine == "FILM":
            self.post_fns.append(film_interpolate_cls)


    def reset(self, *args, **kwargs) -> None:
        self.prep_fns = []
        self.shoot_fns = []
        self.post_fns = []

    def datacallback(self, data):
        pass

    def generate(self):
        """
        Generates an image or animation using the given prompts, settings, and generator.

        This method sets up the necessary arguments, handles conditional configurations, and then
        uses the provided generator to produce the output.

        Returns:
            processed (Image): The generated image or animation frame.
        """
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




