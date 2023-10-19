import os
import sys

import torch

sys.path.extend([os.path.join(os.getcwd(), "deforum", "exttools")])

import pandas as pd
import cv2
import numpy as np
import numexpr
import gc
import random
import PIL
import math
from PIL import Image, ImageOps

from deforum.animation.animation import anim_frame_warp
from deforum.animation.animation_key_frames import DeformAnimKeys, LooperAnimKeys
from deforum.avfunctions.colors.colors import maintain_colors
from deforum.exttools.depth import DepthModel
from deforum.avfunctions.hybridvideo.hybrid_video import hybrid_generation, get_flow_from_images, \
    image_transform_optical_flow, get_matrix_for_hybrid_motion_prev, image_transform_ransac, \
    get_matrix_for_hybrid_motion, get_flow_for_hybrid_motion_prev, get_flow_for_hybrid_motion, abs_flow_to_rel_flow, \
    rel_flow_to_abs_flow, hybrid_composite
from deforum.avfunctions.image.image_sharpening import unsharp_mask
from deforum.avfunctions.image.load_images import load_img, get_mask_from_file, get_mask, load_image
from deforum.avfunctions.image.save_images import save_image
from deforum.avfunctions.interpolation.RAFT import RAFT
from deforum.avfunctions.masks.composable_masks import compose_mask_with_check
from deforum.avfunctions.masks.masks import do_overlay_mask
from deforum.avfunctions.noise.noise import add_noise
from deforum.avfunctions.video_audio_utilities import get_next_frame, get_frame_name
from deforum.datafunctions.parseq_adapter import ParseqAnimKeys
from deforum.datafunctions.prompt import prepare_prompt
from deforum.datafunctions.resume import get_resume_vars
from deforum.datafunctions.seed import next_seed
from deforum.datafunctions.subtitle_handler import format_animation_params, write_frame_subtitle, init_srt_file
from deforum.general_utils import isJson

from deforum.avfunctions.image.load_images import check_mask_for_errors, prepare_mask, load_img
from deforum.datafunctions.prompt import check_is_number, split_weighted_subprompts
from deforum.general_utils import isJson, pairwise_repl, substitute_placeholders
from deforum.torchfuncs.torch_gc import torch_gc


frame_warp_modes = ['2D', '3D']
hybrid_motion_modes = ['Affine', 'Perspective', 'Optical Flow']


class Deforum:

    def __init__(self, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root, opts=None,
                 state=None):

        super().__init__()

        self.args = args
        self.anim_args = anim_args
        self.video_args = video_args
        self.parseq_args = parseq_args
        self.loop_args = loop_args
        self.controlnet_args = controlnet_args
        self.root = root
        self.opts = opts
        self.state = state
        self.pipe = None

    def run_pre_checks(self):
        self.inputfiles = None
        srt_filename = None
        srt_frame_duration = None
        hybrid_frame_path = None
        self.prev_flow = None
        if self.opts is not None:
            if self.opts.data.get("deforum_save_gen_info_as_srt",
                                  False):  # create .srt file and set timeframe mechanism using FPS
                srt_filename = os.path.join(self.args.outdir, f"{self.root.timestring}.srt")
                srt_frame_duration = init_srt_file(srt_filename, self.video_args.fps)
            print("[ Saving SRT file. ]")
        if self.anim_args.animation_mode in frame_warp_modes:
            # handle hybrid video generation
            if self.anim_args.hybrid_composite != 'None' or self.anim_args.hybrid_motion in hybrid_motion_modes:
                self.args, self.anim_args, self.inputfiles = hybrid_generation(self.args, self.anim_args, self.root)
                # path required by hybrid functions, even if hybrid_comp_save_extra_frames is False
                hybrid_frame_path = os.path.join(self.args.outdir, 'hybridframes')

                print(f"[ Using Hybrid Motion: {self.anim_args.hybrid_motion}]")

            # initialize self.prev_flow
            if self.anim_args.hybrid_motion == 'Optical Flow':
                self.prev_flow = None
            if self.loop_args.use_looper:
                print(
                    "Using Guided Images mode: seed_behavior will be set to 'schedule' and 'strength_0_no_init' to False")
                if self.args.strength == 0:
                    raise RuntimeError("Strength needs to be greater than 0 in Init tab")
                self.args.strength_0_no_init = False
                self.args.seed_behavior = "schedule"
                if not isJson(self.loop_args.init_images):
                    raise RuntimeError("The images set for use with keyframe-guidance are not in a proper JSON format")
        # handle controlnet video input frames generation
        self.handle_controlnet(self.args, self.anim_args, self.controlnet_args)
        # if is_controlnet_enabled(self.controlnet_args):
        #    unpack_controlnet_vids(self.args, self.anim_args, self.controlnet_args)
        # create output folder for the batch
        os.makedirs(self.args.outdir, exist_ok=True)
        print(f"[ Saving animation frames to:\n{self.args.outdir} ]")

        # save settings.txt file for the current run
        self.save_settings_from_animation_run(self.args, self.anim_args, self.parseq_args, self.loop_args,
                                              self.controlnet_args, self.video_args, self.root)
        # resume from timestring
        if self.anim_args.resume_from_timestring:
            self.root.timestring = self.anim_args.resume_timestring

        print("[ Deforum Animation Pre-Checks done. ]")

        return srt_filename, srt_frame_duration, hybrid_frame_path, self.prev_flow

    def __call__(self, *args, **kwargs):

        self.cadence_flow = None
        srt_filename = None
        srt_frame_duration = None
        raft_model = None
        self.inputfiles = None
        hybrid_frame_path = None
        hybrid_comp_schedules = None
        self.cadence_flow_factor = None

        srt_filename, srt_frame_duration, hybrid_frame_path, self.prev_flow = self.run_pre_checks()

        # use parseq if manifest is provided
        if self.parseq_args is not None:
            print(f"[ Using Parseq Args ]")
            use_parseq = self.parseq_args.parseq_manifest is not None and self.parseq_args.parseq_manifest.strip()
        else:
            use_parseq = False
        # expand key frame strings to values
        keys = DeformAnimKeys(self.anim_args, self.args.seed) if not use_parseq else ParseqAnimKeys(self.parseq_args,
                                                                                                    self.video_args)
        loopSchedulesAndData = LooperAnimKeys(self.loop_args, self.anim_args, self.args.seed)

        # Always enable pseudo-3d with parseq. No need for an extra toggle:
        # Whether it's used or not in practice is defined by the schedules
        if use_parseq:
            if not self.anim_args.flip_2d_perspective:
                print(f"[ Enabling Flip 2D perspective because of Parseq ]")
                self.anim_args.flip_2d_perspective = True
            # expand prompts out to per-frame
        if use_parseq and keys.manages_prompts():
            prompt_series = keys.prompts
        else:
            prompt_series = pd.Series([np.nan for a in range(self.anim_args.max_frames)])
            for i, prompt in self.root.animation_prompts.items():
                if str(i).isdigit():
                    prompt_series[int(i)] = prompt
                else:
                    prompt_series[int(numexpr.evaluate(i))] = prompt
            prompt_series = prompt_series.ffill().bfill()
        self.prompt_series = prompt_series
        # check for video inits
        using_vid_init = self.anim_args.animation_mode == 'Video Input'

        # load depth model for 3D
        predict_depths = (
                                     self.anim_args.animation_mode == '3D' and self.anim_args.use_depth_warping) or self.anim_args.save_depth_maps
        predict_depths = predict_depths or (
                self.anim_args.hybrid_composite and self.anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth'])
        if predict_depths:
            if self.opts is not None:
                self.keep_in_vram = self.opts.data.get("deforum_keep_3d_models_in_vram")
            else:
                self.keep_in_vram = True
            # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
            # TODO Set device in root in webui
            device = self.root.device
            depth_model = DepthModel(self.root.models_path, device, self.root.half_precision,
                                     keep_in_vram=self.keep_in_vram,
                                     depth_algorithm=self.anim_args.depth_algorithm, Width=self.args.W,
                                     Height=self.args.H,
                                     midas_weight=self.anim_args.midas_weight)
            print(f"[ Loaded Depth model ]")
            # depth-based hybrid composite mask requires saved depth maps
            if self.anim_args.hybrid_composite != 'None' and self.anim_args.hybrid_comp_mask_type == 'Depth':
                self.anim_args.save_depth_maps = True
        else:
            depth_model = None
            self.anim_args.save_depth_maps = False

        raft_model = None
        load_raft = (self.anim_args.optical_flow_cadence == "RAFT" and int(self.anim_args.diffusion_cadence) > 0) or \
                    (self.anim_args.hybrid_motion == "Optical Flow" and self.anim_args.hybrid_flow_method == "RAFT") or \
                    (self.anim_args.optical_flow_redo_generation == "RAFT")
        if load_raft:
            print("[ Loading RAFT model ]")
            raft_model = RAFT()

        # state for interpolating between diffusion steps
        turbo_steps = 1 if using_vid_init else int(self.anim_args.diffusion_cadence)
        self.turbo_prev_image, self.turbo_prev_frame_idx = None, 0
        self.turbo_next_image, self.turbo_next_frame_idx = None, 0

        # initialize vars
        self.prev_img = None
        color_match_sample = None
        start_frame = 0

        # resume animation (requires at least two frames - see function)
        if self.anim_args.resume_from_timestring:

            print(f"[ Resuming Animation from timestring: {self.anim_args.resume_timestring} ]")

            # determine last frame and frame to start on
            prev_frame, next_frame, self.prev_img, next_img = get_resume_vars(
                folder=self.args.outdir,
                timestring=self.anim_args.resume_timestring,
                cadence=turbo_steps
            )

            # set up turbo step vars
            if turbo_steps > 1:
                self.turbo_prev_image, self.turbo_prev_frame_idx = self.prev_img, prev_frame
                self.turbo_next_image, self.turbo_next_frame_idx = next_img, next_frame

            # advance start_frame to next frame
            start_frame = next_frame + 1

        self.frame_idx = start_frame

        # reset the mask vals as they are overwritten in the compose_mask algorithm
        mask_vals = {}
        noise_mask_vals = {}

        mask_vals['everywhere'] = Image.new('1', (self.args.W, self.args.H), 1)
        noise_mask_vals['everywhere'] = Image.new('1', (self.args.W, self.args.H), 1)

        mask_image = None

        if self.args.use_init and self.args.init_image != None and self.args.init_image != '':
            _, mask_image = load_img(self.args.init_image,
                                     shape=(self.args.W, self.args.H),
                                     use_alpha_as_mask=self.args.use_alpha_as_mask)
            mask_vals['video_mask'] = mask_image
            noise_mask_vals['video_mask'] = mask_image

            print(f"[ Loaded Mask from Init Image ]")

        # Grab the first frame masks since they wont be provided until next frame
        # Video mask overrides the init image mask, also, won't be searching for init_mask if use_mask_video is set
        # Made to solve https://github.com/deforum-art/deforum-for-automatic1111-webui/issues/386
        if self.anim_args.use_mask_video:
            print(f"[ Using Mask Video ]")

            self.args.mask_file = get_mask_from_file(
                get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True),
                self.args)
            self.root.noise_mask = get_mask_from_file(
                get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True), self.args)

            mask_vals['video_mask'] = get_mask_from_file(
                get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True), self.args)
            noise_mask_vals['video_mask'] = get_mask_from_file(
                get_next_frame(self.args.outdir, self.anim_args.video_mask_path, frame_idx, True), self.args)
        elif mask_image is None and self.args.use_mask:
            mask_vals['video_mask'] = get_mask(self.args)
            noise_mask_vals['video_mask'] = get_mask(self.args)  # TODO?: add a different default noisc mask

        # get color match for 'Image' color coherence only once, before loop
        if self.anim_args.color_coherence == 'Image':

            print(f"[ Setting Color Match Sample to given Image Path ]")

            color_match_sample = self.get_color_match_sample(self.anim_args.color_coherence_image_path)

        # Webui
        done = self.datacallback({"max_frames": self.anim_args.max_frames})
        self.prev_flow = None
        # state.job_count = self.anim_args.max_frames
        from tqdm import tqdm
        for _ in tqdm(range(self.anim_args.max_frames), desc="Processing frames", position=0, leave=True):
            # Webui

            done = self.datacallback({"job": f"frame {self.frame_idx + 1}/{self.anim_args.max_frames}",
                                      "job_no": self.frame_idx + 1})

            #print(f"\033[36mAnimation frame: \033[0m{self.frame_idx}/{self.anim_args.max_frames}  ")

            noise = keys.noise_schedule_series[self.frame_idx]
            strength = keys.strength_schedule_series[self.frame_idx]
            scale = keys.cfg_scale_schedule_series[self.frame_idx]
            contrast = keys.contrast_schedule_series[self.frame_idx]
            kernel = int(keys.kernel_schedule_series[self.frame_idx])
            sigma = keys.sigma_schedule_series[self.frame_idx]
            amount = keys.amount_schedule_series[self.frame_idx]
            threshold = keys.threshold_schedule_series[self.frame_idx]
            self.cadence_flow_factor = keys.cadence_flow_factor_schedule_series[self.frame_idx]
            redo_flow_factor = keys.redo_flow_factor_schedule_series[self.frame_idx]
            hybrid_comp_schedules = {
                "alpha": keys.hybrid_comp_alpha_schedule_series[self.frame_idx],
                "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[self.frame_idx],
                "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[self.frame_idx],
                "mask_auto_contrast_cutoff_low": int(
                    keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[self.frame_idx]),
                "mask_auto_contrast_cutoff_high": int(
                    keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[self.frame_idx]),
                "flow_factor": keys.hybrid_flow_factor_schedule_series[self.frame_idx]
            }
            scheduled_sampler_name = None
            scheduled_clipskip = None
            scheduled_noise_multiplier = None
            scheduled_ddim_eta = None
            scheduled_ancestral_eta = None

            mask_seq = None
            noise_mask_seq = None
            if self.anim_args.enable_steps_scheduling and keys.steps_schedule_series[self.frame_idx] is not None:
                self.args.steps = int(keys.steps_schedule_series[self.frame_idx])
            if self.anim_args.enable_sampler_scheduling and keys.sampler_schedule_series[self.frame_idx] is not None:
                scheduled_sampler_name = keys.sampler_schedule_series[self.frame_idx].casefold()
            if self.anim_args.enable_clipskip_scheduling and keys.clipskip_schedule_series[self.frame_idx] is not None:
                scheduled_clipskip = int(keys.clipskip_schedule_series[self.frame_idx])
            if self.anim_args.enable_noise_multiplier_scheduling and keys.noise_multiplier_schedule_series[
                self.frame_idx] is not None:
                scheduled_noise_multiplier = float(keys.noise_multiplier_schedule_series[self.frame_idx])
            if self.anim_args.enable_ddim_eta_scheduling and keys.ddim_eta_schedule_series[self.frame_idx] is not None:
                scheduled_ddim_eta = float(keys.ddim_eta_schedule_series[self.frame_idx])
            if self.anim_args.enable_ancestral_eta_scheduling and keys.ancestral_eta_schedule_series[
                self.frame_idx] is not None:
                scheduled_ancestral_eta = float(keys.ancestral_eta_schedule_series[self.frame_idx])
            if self.args.use_mask and keys.mask_schedule_series[self.frame_idx] is not None:
                mask_seq = keys.mask_schedule_series[self.frame_idx]
            if self.anim_args.use_noise_mask and keys.noise_mask_schedule_series[self.frame_idx] is not None:
                noise_mask_seq = keys.noise_mask_schedule_series[self.frame_idx]

            if self.args.use_mask and not self.anim_args.use_noise_mask:
                noise_mask_seq = mask_seq

            depth = None
            done = self.datacallback({"webui": "sd_to_cpu"})
            if self.anim_args.animation_mode == '3D':
                if predict_depths: depth_model.to(self.root.device)
            if self.opts is not None:
                if turbo_steps == 1 and self.opts.data.get("deforum_save_gen_info_as_srt"):
                    params_string = format_animation_params(keys, prompt_series, self.frame_idx)
                    write_frame_subtitle(srt_filename, self.frame_idx, srt_frame_duration,
                                         f"F#: {self.frame_idx}; Cadence: false; Seed: {self.args.seed}; {params_string}")
                    params_string = None
            self.tween_frame_start_idx = max(start_frame, self.frame_idx - turbo_steps)

            # emit in-between frames
            if turbo_steps > 1:
                self.generate_in_between_frames(raft_model, keys, prompt_series, srt_filename, srt_frame_duration,
                                   depth_model, hybrid_frame_path, hybrid_comp_schedules)

            # get color match for video outside of self.prev_img conditional
            hybrid_available = self.anim_args.hybrid_composite != 'None' or self.anim_args.hybrid_motion in [
                'Optical Flow',
                'Affine',
                'Perspective']
            if self.anim_args.color_coherence == 'Video Input' and hybrid_available:
                if int(self.frame_idx) % int(self.anim_args.color_coherence_video_every_N_frames) == 0:
                    prev_vid_img = Image.open(os.path.join(self.args.outdir, 'inputframes', get_frame_name(
                        self.anim_args.video_init_path) + f"{self.frame_idx:09}.jpg"))
                    prev_vid_img = prev_vid_img.resize((self.args.W, self.args.H), PIL.Image.LANCZOS)
                    color_match_sample = self.get_color_match_sample(image=prev_vid_img)

            # after 1st frame, self.prev_img exists
            if self.prev_img is not None:
                # apply transforms to previous frame
                self.prev_img, depth,  mask = anim_frame_warp(self.prev_img, self.args, self.anim_args, keys, self.frame_idx, depth_model,
                                                  depth=None,
                                                  device=self.root.device, half_precision=self.root.half_precision)
                if mask is not None:
                    self.prev_img = self.generate_inpaint(self.args, keys, self.anim_args, self.loop_args,
                                  self.controlnet_args, self.root,
                                  sampler_name=scheduled_sampler_name, image=self.prev_img, mask=mask)
                    #self.prev_img = cv2.cvtColor(self.prev_img, cv2.COLOR_BGR2RGB)
                # do hybrid compositing before motion
                if self.anim_args.hybrid_composite == 'Before Motion':
                    self.args, self.prev_img = hybrid_composite(self.args, self.anim_args, self.frame_idx, self.prev_img, depth_model,
                                                           hybrid_comp_schedules, self.root)

                # hybrid video motion - warps self.prev_img to match motion, usually to prepare for compositing
                if self.anim_args.hybrid_motion in ['Affine', 'Perspective']:
                    if self.anim_args.hybrid_motion_use_prev_img:
                        matrix = get_matrix_for_hybrid_motion_prev(self.frame_idx - 1, (self.args.W, self.args.H),
                                                                   self.inputfiles,
                                                                   self.prev_img, self.anim_args.hybrid_motion)
                    else:
                        matrix = get_matrix_for_hybrid_motion(self.frame_idx - 1, (self.args.W, self.args.H), self.inputfiles,
                                                              self.anim_args.hybrid_motion)
                    self.prev_img = image_transform_ransac(self.prev_img, matrix, self.anim_args.hybrid_motion)
                if self.anim_args.hybrid_motion in ['Optical Flow']:
                    if self.anim_args.hybrid_motion_use_prev_img:
                        flow = get_flow_for_hybrid_motion_prev(self.frame_idx - 1, (self.args.W, self.args.H), self.inputfiles,
                                                               hybrid_frame_path, self.prev_flow, self.prev_img,
                                                               self.anim_args.hybrid_flow_method, raft_model,
                                                               self.anim_args.hybrid_flow_consistency,
                                                               self.anim_args.hybrid_consistency_blur,
                                                               self.anim_args.hybrid_comp_save_extra_frames)
                    else:

                        flow = get_flow_for_hybrid_motion(self.frame_idx - 1, (self.args.W, self.args.H), self.inputfiles,
                                                          hybrid_frame_path, self.prev_flow,
                                                          self.anim_args.hybrid_flow_method,
                                                          raft_model,
                                                          self.anim_args.hybrid_flow_consistency,
                                                          self.anim_args.hybrid_consistency_blur,
                                                          self.anim_args.hybrid_comp_save_extra_frames)
                    self.prev_img = image_transform_optical_flow(self.prev_img, flow, hybrid_comp_schedules['flow_factor'])
                    self.prev_flow = flow

                # do hybrid compositing after motion (normal)
                if self.anim_args.hybrid_composite == 'Normal':
                    self.args, self.prev_img = hybrid_composite(self.args, self.anim_args, self.frame_idx, self.prev_img, depth_model,
                                                           hybrid_comp_schedules, self.root)

                # apply color matching
                if self.anim_args.color_coherence != 'None':
                    if color_match_sample is None:
                        color_match_sample = self.prev_img.copy()
                    else:
                        self.prev_img = maintain_colors(self.prev_img, color_match_sample, self.anim_args.color_coherence)

                # intercept and override to grayscale
                if self.anim_args.color_force_grayscale:
                    self.prev_img = cv2.cvtColor(self.prev_img, cv2.COLOR_BGR2GRAY)
                    self.prev_img = cv2.cvtColor(self.prev_img, cv2.COLOR_GRAY2BGR)

                # apply scaling
                contrast_image = (self.prev_img * contrast).round().astype(np.uint8)
                # anti-blur
                if amount > 0:
                    contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold,
                                                  mask_image if self.args.use_mask else None)
                # apply frame noising
                if self.args.use_mask or self.anim_args.use_noise_mask:
                    self.root.noise_mask = compose_mask_with_check(self.root, self.args, noise_mask_seq,
                                                                   noise_mask_vals,
                                                                   Image.fromarray(
                                                                       cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)))
                noised_image = add_noise(contrast_image, noise, self.args.seed, self.anim_args.noise_type,
                                         (self.anim_args.perlin_w, self.anim_args.perlin_h,
                                          self.anim_args.perlin_octaves,
                                          self.anim_args.perlin_persistence),
                                         self.root.noise_mask, self.args.invert_mask)

                # use transformed previous frame as init for current
                self.args.use_init = True
                self.root.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
                self.args.strength = max(0.0, min(1.0, strength))

            self.args.scale = scale

            # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
            self.args.pix2pix_img_cfg_scale = float(keys.pix2pix_img_cfg_scale_series[self.frame_idx])

            # grab prompt for current frame
            self.args.prompt = prompt_series[self.frame_idx]

            if self.args.seed_behavior == 'schedule' or use_parseq:
                self.args.seed = int(keys.seed_schedule_series[self.frame_idx])

            if self.anim_args.enable_checkpoint_scheduling:
                self.args.checkpoint = keys.checkpoint_schedule_series[self.frame_idx]
            else:
                self.args.checkpoint = None

            # SubSeed scheduling
            if self.anim_args.enable_subseed_scheduling:
                self.root.subseed = int(keys.subseed_schedule_series[self.frame_idx])
                self.root.subseed_strength = float(keys.subseed_strength_schedule_series[self.frame_idx])

            if use_parseq:
                self.anim_args.enable_subseed_scheduling = True
                self.root.subseed = int(keys.subseed_schedule_series[self.frame_idx])
                self.root.subseed_strength = keys.subseed_strength_schedule_series[self.frame_idx]

            # set value back into the prompt - prepare and report prompt and seed
            self.args.prompt = prepare_prompt(self.args.prompt, self.anim_args.max_frames, self.args.seed, self.frame_idx)

            # grab init image for current frame
            if using_vid_init:
                init_frame = get_next_frame(self.args.outdir, self.anim_args.video_init_path, self.frame_idx, False)
                print(f"Using video init frame {init_frame}")
                self.args.init_image = init_frame
                self.args.strength = max(0.0, min(1.0, strength))
            if self.anim_args.use_mask_video:
                self.args.mask_file = get_mask_from_file(
                    get_next_frame(self.args.outdir, self.anim_args.video_mask_path, self.frame_idx, True), self.args)
                self.root.noise_mask = get_mask_from_file(
                    get_next_frame(self.args.outdir, self.anim_args.video_mask_path, self.frame_idx, True), self.args)

                mask_vals['video_mask'] = get_mask_from_file(
                    get_next_frame(self.args.outdir, self.anim_args.video_mask_path, self.frame_idx, True), self.args)

            if self.args.use_mask:
                self.args.mask_image = compose_mask_with_check(self.root, self.args, mask_seq, mask_vals,
                                                               self.root.init_sample) if self.root.init_sample is not None else None  # we need it only after the first frame anyway

            # setting up some arguments for the looper
            self.loop_args.imageStrength = loopSchedulesAndData.image_strength_schedule_series[self.frame_idx]
            self.loop_args.blendFactorMax = loopSchedulesAndData.blendFactorMax_series[self.frame_idx]
            self.loop_args.blendFactorSlope = loopSchedulesAndData.blendFactorSlope_series[self.frame_idx]
            self.loop_args.tweeningFrameSchedule = loopSchedulesAndData.tweening_frames_schedule_series[self.frame_idx]
            self.loop_args.colorCorrectionFactor = loopSchedulesAndData.color_correction_factor_series[self.frame_idx]
            self.loop_args.use_looper = loopSchedulesAndData.use_looper
            self.loop_args.imagesToKeyframe = loopSchedulesAndData.imagesToKeyframe
            if self.opts is not None:
                self.run_opts_scheduler(scheduled_clipskip, scheduled_noise_multiplier, scheduled_ddim_eta, scheduled_ancestral_eta)

            self.datacallback({"webui": "sd_to_gpu"})

            # optical flow redo before generation
            if self.anim_args.optical_flow_redo_generation != 'None' and self.prev_img is not None and strength > 0:
                self.root.init_sample = self.generate_disposable_image(keys, scheduled_sampler_name, raft_model, redo_flow_factor)

            # diffusion redo
            if int(self.anim_args.diffusion_redo) > 0 and self.prev_img is not None and strength > 0:
                stored_seed = self.args.seed
                for n in range(0, int(self.anim_args.diffusion_redo)):
                    print(f"Redo generation {n + 1} of {int(self.anim_args.diffusion_redo)} before final generation")
                    self.args.seed = random.randint(0, 2 ** 32 - 1)
                    disposable_image = self.generate(self.args, keys, self.anim_args, self.loop_args,
                                                     self.controlnet_args, self.root, self.frame_idx,
                                                     sampler_name=scheduled_sampler_name)
                    disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
                    # color match on last one only
                    if n == int(self.anim_args.diffusion_redo):
                        disposable_image = maintain_colors(self.prev_img, color_match_sample, self.anim_args.color_coherence)
                    self.args.seed = stored_seed
                    print("SETTING INIT SAMPLE #3")

                    self.root.init_sample = Image.fromarray(cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB))
                del (disposable_image, stored_seed)
                gc.collect()

            # generation
            image = self.generate(self.args, keys, self.anim_args, self.loop_args, self.controlnet_args, self.root, sampler_name=scheduled_sampler_name)
            if image is None:
                break
            # do hybrid video after generation
            if self.frame_idx > 0 and self.anim_args.hybrid_composite == 'After Generation':
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                self.args, image = hybrid_composite(self.args, self.anim_args, self.frame_idx, image, depth_model,
                                                    hybrid_comp_schedules,
                                                    self.root)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # color matching on first frame is after generation, color match was collected earlier, so we do an extra generation to avoid the corruption introduced by the color match of first output
            if self.frame_idx == 0 and (self.anim_args.color_coherence == 'Image' or (
                    self.anim_args.color_coherence == 'Video Input' and hybrid_available)):
                image = maintain_colors(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), color_match_sample,
                                        self.anim_args.color_coherence)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif color_match_sample is not None and self.anim_args.color_coherence != 'None' and not self.anim_args.legacy_colormatch:
                image = maintain_colors(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), color_match_sample,
                                        self.anim_args.color_coherence)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # intercept and override to grayscale
            if self.anim_args.color_force_grayscale:
                image = ImageOps.grayscale(image)
                image = ImageOps.colorize(image, black="black", white="white")

            # overlay mask
            if self.args.overlay_mask and (self.anim_args.use_mask_video or self.args.use_mask):
                image = do_overlay_mask(self.args, self.anim_args, image, self.frame_idx)
            # on strength 0, set color match to generation
            if ((not self.anim_args.legacy_colormatch and not self.args.use_init) or (
                    self.anim_args.legacy_colormatch and strength == 0)) and not self.anim_args.color_coherence in [
                'Image',
                'Video Input']:
                color_match_sample = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            if not using_vid_init:
                self.prev_img = opencv_image

            if turbo_steps > 1:
                self.turbo_prev_image, self.turbo_prev_frame_idx = self.turbo_next_image, self.turbo_next_frame_idx
                self.turbo_next_image, self.turbo_next_frame_idx = opencv_image, self.frame_idx
                self.frame_idx += turbo_steps
            else:
                filename = f"{self.root.timestring}_{self.frame_idx:09}.png"
                save_image(image, 'PIL', filename, self.args, self.video_args, self.root)
                if self.anim_args.save_depth_maps:
                    done = self.datacallback({"webui": "sd_to_cpu"})
                    depth = depth_model.predict(opencv_image, self.anim_args.midas_weight, self.root.half_precision)
                    depth_model.save(os.path.join(self.args.outdir, f"{self.root.timestring}_depth_{self.frame_idx:09}.png"),
                                     depth)
                    done = self.datacallback({"webui": "sd_to_cpu"})
                self.frame_idx += 1
            done = self.datacallback({"image": image})
            self.args.seed = next_seed(self.args, self.root)
        self.cleanup(predict_depths, load_raft, depth_model, raft_model)
        return True

    def cleanup(self, predict_depths, load_raft, depth_model, raft_model):
        if predict_depths and not self.keep_in_vram:
            depth_model.delete_model()  # handles adabins too
        if load_raft:
            raft_model.delete_model()

    def datacallback(self, data):
        return None

    def generate(self, args, keys, anim_args, loop_args, controlnet_args, root, sampler_name):

        assert args.prompt is not None

        # Setup the pipeline
        # p = get_webui_sd_pipeline(args, root, frame)
        prompt, negative_prompt = split_weighted_subprompts(args.prompt, self.frame_idx, anim_args.max_frames)

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

        next_prompt, blend_value = get_next_prompt_and_blend(self.frame_idx, self.prompt_series)
        # print("DEBUG", next_prompt, blend_value)

        # blend_value = 1.0
        # next_prompt = ""
        if not args.use_init and args.strength > 0 and args.strength_0_no_init:
            args.strength = 0
        processed = None
        mask_image = None
        init_image = None
        image_init0 = None

        if loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']:
            args.strength = loop_args.imageStrength
            tweeningFrames = loop_args.tweeningFrameSchedule
            blendFactor = .07
            colorCorrectionFactor = loop_args.colorCorrectionFactor
            jsonImages = json.loads(loop_args.imagesToKeyframe)
            # find which image to show
            parsedImages = {}
            frameToChoose = 0
            max_f = anim_args.max_frames - 1

            for key, value in jsonImages.items():
                if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
                    parsedImages[key] = value
                else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                    parsedImages[int(numexpr.evaluate(key))] = value

            framesToImageSwapOn = list(map(int, list(parsedImages.keys())))

            for swappingFrame in framesToImageSwapOn[1:]:
                frameToChoose += (frame >= int(swappingFrame))

            # find which frame to do our swapping on for tweening
            skipFrame = 25
            for fs, fe in pairwise_repl(framesToImageSwapOn):
                if fs <= frame <= fe:
                    skipFrame = fe - fs
            if skipFrame > 0:
                # print("frame % skipFrame", frame % skipFrame)

                if frame % skipFrame <= tweeningFrames:  # number of tweening frames
                    blendFactor = loop_args.blendFactorMax - loop_args.blendFactorSlope * math.cos(
                        (frame % tweeningFrames) / (tweeningFrames / 2))
            else:
                print("LOOPER ERROR, AVOIDING DIVISION BY 0")
            init_image2, _ = load_img(list(jsonImages.values())[frameToChoose],
                                      shape=(args.W, args.H),
                                      use_alpha_as_mask=args.use_alpha_as_mask)
            image_init0 = list(jsonImages.values())[0]
            # print(" TYPE", type(image_init0))


        else:  # they passed in a single init image
            image_init0 = args.init_image

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

        # if args.checkpoint is not None:
        #    info = sd_models.get_closet_checkpoint_match(args.checkpoint)
        #    if info is None:
        #        raise RuntimeError(f"Unknown checkpoint: {args.checkpoint}")
        #    sd_models.reload_model_weights(info=info)

        if root.init_sample is not None:
            # TODO: cleanup init_sample remains later
            img = root.init_sample
            init_image = img
            image_init0 = img
            if loop_args.use_looper and isJson(loop_args.imagesToKeyframe) and anim_args.animation_mode in ['2D', '3D']:
                init_image = Image.blend(init_image, init_image2, blendFactor)
                correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
                color_corrections = [correction_colors]

        # this is the first pass
        elif (loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']) or (
                args.use_init and ((args.init_image != None and args.init_image != ''))):
            init_image, mask_image = load_img(image_init0,  # initial init image
                                              shape=(args.W, args.H),
                                              use_alpha_as_mask=args.use_alpha_as_mask)

        else:

            # if anim_args.animation_mode != 'Interpolation':
            #    print(f"Not using an init image (doing pure txt2img)")
            """p_txt = StableDiffusionProcessingTxt2Img( 
                sd_model=sd_model,
                outpath_samples=root.tmp_deforum_run_duplicated_folder,
                outpath_grids=root.tmp_deforum_run_duplicated_folder,
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
            #                            self.frame_idx=frame)

            processed = self.generate_txt2img(prompt, next_prompt, blend_value, negative_prompt, args, root, self.frame_idx,
                                           init_image)

        if processed is None:
            # Mask functions
            if args.use_mask:
                mask_image = args.mask_image
                mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                                    (args.W, args.H),
                                    args.mask_contrast_adjust,
                                    args.mask_brightness_adjust)
                inpainting_mask_invert = args.invert_mask
                inpainting_fill = args.fill
                inpaint_full_res = args.full_res_mask
                inpaint_full_res_padding = args.full_res_mask_padding
                # prevent loaded mask from throwing errors in Image operations if completely black and crop and resize in webui pipeline
                # doing this after contrast and brightness adjustments to ensure that mask is not passed as black or blank
                mask = check_mask_for_errors(mask, args.invert_mask)
                args.noise_mask = mask

            else:
                mask = None

            assert not ((mask is not None and args.use_mask and args.overlay_mask) and (
                    args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

            image_mask = mask
            image_cfg_scale = args.pix2pix_img_cfg_scale

            # print_combined_table(args, anim_args, p, keys, frame)  # print dynamic table to cli

            # if is_controlnet_enabled(controlnet_args):
            #    process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img=True,
            #                            self.frame_idx=frame)

            processed = self.generate_txt2img(prompt, next_prompt, blend_value, negative_prompt, args, root, self.frame_idx,
                                           init_image)

        if root.first_frame == None:
            root.first_frame = processed

        return processed

        image = Image.new("RGB", (256,256))

        return image
    def generate_inpaint(self, args, keys, anim_args, loop_args, controlnet_args, root, sampler_name, image=None, mask=None):

        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        mask = mask.cpu().reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        mask_array = np.array(mask)
        # Check if any values are above 0
        has_values_above_zero = (np.array(mask) > 1e-05).any()
        # Count the number of values above 0
        count_values_above_zero = (mask_array > 0).sum()
        threshold = 40000

        if has_values_above_zero and count_values_above_zero > threshold and self.anim_args.padding_mode == 'zeros':
            print(f"[ Mask pixels above {threshold} by {count_values_above_zero-threshold}, generating inpaing image ]")
            mask = tensor2pil(mask[0])
            mask = dilate_mask(mask, dilation_size=48)
            change_pipe = False
            if gs.should_run:
                if not self.pipe or change_pipe:
                    from diffusers import StableDiffusionInpaintPipeline
                    self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                                "https://huggingface.co/XpucT/Deliberate/blob/main/Deliberate-inpainting.safetensors",
                                use_safetensors=True,
                                torch_dtype=torch.float16).to(gs.device.type)
                    # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    #             "runwayml/stable-diffusion-inpainting",
                    #             torch_dtype=torch.float16).to(gs.device.type)
                prompt, negative_prompt = split_weighted_subprompts(args.prompt, self.frame_idx, anim_args.max_frames)
                generation_args = {"generator":torch.Generator(gs.device.type).manual_seed(args.seed),
                                   "num_inference_steps":args.steps,
                                   "prompt":prompt,
                                   "image":image,
                                   "mask_image":mask,
                                   "width" : image.size[0],
                                   "height" : image.size[1],
                                   }
                #image.save("inpaint_image.png", "PNG")
                image = np.array(self.pipe(**generation_args).images[0]).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # # Composite the original image and the generated image using the mask
                mask_arr = np.array(mask).astype(np.uint8)[:, :, 0]  # Convert to grayscale mask for boolean indexing
                mask_bool = mask_arr > 0  # Convert to boolean mask
                original_image[mask_bool] = image[mask_bool]
                #test = Image.fromarray(original_image).save("test_result.png", "PNG")


        return original_image

    def generate_txt2img(self, prompt, next_prompt, blend_value, negative_prompt, args, root, frame,
                                           init_image=None):

        if self.pipe == None:
            from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline,
                                   StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline,
                                   StableDiffusionXLImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline,
                                   StableDiffusionXLControlNetPipeline, StableDiffusionXLInpaintPipeline, AutoPipelineForImage2Image)
            torch.backends.cuda.matmul.allow_tf32 = True
            self.pipe = StableDiffusionXLPipeline.from_single_file()m(
                "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
                # "segmind/SDXL-Mini",
                torch_dtype=torch.float16
            ).to("cuda")
            self.pipe.unet.to(memory_format=torch.channels_last)
            #self.pipe.unet = torch.compile(self.pipe.unet, mode="max-autotune", fullgraph=True)
            #self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)


            self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe).to("cuda")

        generator = torch.Generator("cuda").manual_seed(args.seed)
        gen_args = {

            "prompt":prompt,
            "negative_prompt":negative_prompt,
            "num_inference_steps":args.steps,
            "generator":generator,
            "guidance_scale":args.scale
        }

        if init_image is not None:
            image = self.img2img_pipe(image=init_image, strength=args.strength, **gen_args).images[0]
        else:
            image = self.pipe(**gen_args, width=args.W, height=args.H).images[0]

        return image

    def save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root,
                                         full_out_file_path=None):
        return

    def handle_controlnet(self, args, anim_args, controlnet_args):
        print("DEFORUM DUMMY CONTROLNET HANDLER, REPLACE ME WITH YOUR UI's, or API's CONTROLNET HANDLER")
        return None

    def generate_in_between_frames(self, raft_model, keys, prompt_series, srt_filename, srt_frame_duration,
                                   depth_model, hybrid_frame_path, hybrid_comp_schedules):  # Add other required static parameters
        for tween_frame_idx in range(self.tween_frame_start_idx, self.frame_idx):
            # update progress during cadence
            done = self.datacallback({"job": f"frame {tween_frame_idx + 1}/{self.anim_args.max_frames}",
                                      "job_no": tween_frame_idx + 1})
            # state.job = f"frame {tween_frame_idx + 1}/{self.anim_args.max_frames}"
            # state.job_no = tween_frame_idx + 1
            # cadence vars
            tween = float(tween_frame_idx - self.tween_frame_start_idx + 1) / float(
                self.frame_idx - self.tween_frame_start_idx)
            advance_prev = self.turbo_prev_image is not None and tween_frame_idx > self.turbo_prev_frame_idx
            advance_next = tween_frame_idx > self.turbo_next_frame_idx

            # optical flow cadence setup before animation warping
            if self.anim_args.animation_mode in ['2D', '3D'] and self.anim_args.optical_flow_cadence != 'None':
                if keys.strength_schedule_series[self.tween_frame_start_idx] > 0:
                    if self.cadence_flow is None and self.turbo_prev_image is not None and self.turbo_next_image is not None:
                        self.cadence_flow = get_flow_from_images(self.turbo_prev_image, self.turbo_next_image,
                                                            self.anim_args.optical_flow_cadence, raft_model) / 2
                        self.turbo_next_image = image_transform_optical_flow(self.turbo_next_image, -self.cadence_flow, 1)
                        self.prev_flow = self.cadence_flow
            if self.opts is not None:
                if self.opts.data.get("deforum_save_gen_info_as_srt"):
                    params_string = format_animation_params(keys, prompt_series, tween_frame_idx)
                    write_frame_subtitle(srt_filename, tween_frame_idx, srt_frame_duration,
                                         f"F#: {tween_frame_idx}; Cadence: {tween < 1.0}; Seed: {self.args.seed}; {params_string}")
                    params_string = None

            print(
                f"Creating in-between {'' if self.cadence_flow is None else self.anim_args.optical_flow_cadence + ' optical flow '}cadence frame: {tween_frame_idx}; tween:{tween:0.2f};")

            if depth_model is not None:
                assert (self.turbo_next_image is not None)
                depth = depth_model.predict(self.turbo_next_image, self.anim_args.midas_weight,
                                            self.root.half_precision)

            if advance_prev:
                self.turbo_prev_image, _, mask = anim_frame_warp(self.turbo_prev_image, self.args, self.anim_args, keys,
                                                      tween_frame_idx,
                                                      depth_model, depth=depth, device=self.root.device,
                                                      half_precision=self.root.half_precision)
                if mask is not None:
                    self.turbo_prev_image = self.generate_inpaint(self.args, keys, self.anim_args, self.loop_args,
                                  self.controlnet_args, self.root, sampler_name=None, image=self.turbo_prev_image, mask=mask)

            if advance_next:
                self.turbo_next_image, _, mask = anim_frame_warp(self.turbo_next_image, self.args, self.anim_args, keys,
                                                      tween_frame_idx,
                                                      depth_model, depth=depth, device=self.root.device,
                                                      half_precision=self.root.half_precision)
                if mask is not None:
                    self.turbo_next_image = self.generate_inpaint(self.args, keys, self.anim_args, self.loop_args,
                                  self.controlnet_args, self.root,
                                  sampler_name=None, image=self.turbo_next_image, mask=mask)

            # hybrid video motion - warps self.turbo_prev_image or self.turbo_next_image to match motion
            if tween_frame_idx > 0:
                if self.anim_args.hybrid_motion in ['Affine', 'Perspective']:
                    if self.anim_args.hybrid_motion_use_prev_img:
                        matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx - 1,
                                                                   (self.args.W, self.args.H),
                                                                   self.inputfiles, self.prev_img,
                                                                   self.anim_args.hybrid_motion)
                        if advance_prev:
                            self.turbo_prev_image = image_transform_ransac(self.turbo_prev_image, matrix,
                                                                      self.anim_args.hybrid_motion)
                        if advance_next:
                            self.turbo_next_image = image_transform_ransac(self.turbo_next_image, matrix,
                                                                      self.anim_args.hybrid_motion)
                    else:
                        matrix = get_matrix_for_hybrid_motion(tween_frame_idx - 1, (self.args.W, self.args.H),
                                                              self.inputfiles,
                                                              self.anim_args.hybrid_motion)
                        if advance_prev:
                            self.turbo_prev_image = image_transform_ransac(self.turbo_prev_image, matrix,
                                                                      self.anim_args.hybrid_motion)
                        if advance_next:
                            self.turbo_next_image = image_transform_ransac(self.turbo_next_image, matrix,
                                                                      self.anim_args.hybrid_motion)
                if self.anim_args.hybrid_motion in ['Optical Flow']:
                    if self.anim_args.hybrid_motion_use_prev_img:
                        flow = get_flow_for_hybrid_motion_prev(tween_frame_idx - 1, (self.args.W, self.args.H),
                                                               self.inputfiles, hybrid_frame_path, self.prev_flow,
                                                               self.prev_img, self.anim_args.hybrid_flow_method,
                                                               raft_model,
                                                               self.anim_args.hybrid_flow_consistency,
                                                               self.anim_args.hybrid_consistency_blur,
                                                               self.anim_args.hybrid_comp_save_extra_frames)
                        if advance_prev:
                            self.turbo_prev_image = image_transform_optical_flow(self.turbo_prev_image, flow,
                                                                            hybrid_comp_schedules[
                                                                                'flow_factor'])
                        if advance_next:
                            self.turbo_next_image = image_transform_optical_flow(self.turbo_next_image, flow,
                                                                            hybrid_comp_schedules[
                                                                                'flow_factor'])
                        self.prev_flow = flow
                    else:
                        flow = get_flow_for_hybrid_motion(tween_frame_idx - 1, (self.args.W, self.args.H),
                                                          self.inputfiles,
                                                          hybrid_frame_path, self.prev_flow,
                                                          self.anim_args.hybrid_flow_method, raft_model,
                                                          self.anim_args.hybrid_flow_consistency,
                                                          self.anim_args.hybrid_consistency_blur,
                                                          self.anim_args.hybrid_comp_save_extra_frames)
                        if advance_prev:
                            self.turbo_prev_image = image_transform_optical_flow(self.turbo_prev_image, flow,
                                                                            hybrid_comp_schedules[
                                                                                'flow_factor'])
                        if advance_next:
                            self.turbo_next_image = image_transform_optical_flow(self.turbo_next_image, flow,
                                                                            hybrid_comp_schedules[
                                                                                'flow_factor'])
                        self.prev_flow = flow

            # do optical flow cadence after animation warping
            if self.cadence_flow is not None:
                self.cadence_flow = abs_flow_to_rel_flow(self.cadence_flow, self.args.W, self.args.H)
                self.cadence_flow, _, mask = anim_frame_warp(self.cadence_flow, self.args, self.anim_args, keys,
                                                  tween_frame_idx,
                                                  depth_model, depth=depth, device=self.root.device,
                                                  half_precision=self.root.half_precision)
                # if mask is not None:
                #     self.cadence_flow = self.generate_inpaint(self.args, keys, self.anim_args, self.loop_args,
                #                   self.controlnet_args, self.root, self.frame_idx,
                #                   sampler_name=None, image=self.cadence_flow, mask=mask)


                self.prev_flow = self.cadence_flow
                self.cadence_flow_inc = rel_flow_to_abs_flow(self.cadence_flow, self.args.W, self.args.H) * tween
                if advance_prev:
                    self.turbo_prev_image = image_transform_optical_flow(self.turbo_prev_image, self.cadence_flow_inc,
                                                                    self.cadence_flow_factor)
                if advance_next:
                    self.turbo_next_image = image_transform_optical_flow(self.turbo_next_image, self.cadence_flow_inc,
                                                                    self.cadence_flow_factor)
            self.turbo_prev_frame_idx = self.turbo_next_frame_idx = tween_frame_idx
            if self.turbo_prev_image is not None and tween < 1.0:
                img = self.turbo_prev_image * (1.0 - tween) + self.turbo_next_image * tween
            else:
                img = self.turbo_next_image
            # intercept and override to grayscale
            if self.anim_args.color_force_grayscale:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # overlay mask
            if self.args.overlay_mask and (self.anim_args.use_mask_video or self.args.use_mask):
                img = do_overlay_mask(self.args, self.anim_args, img, tween_frame_idx, True)
            # get self.prev_img during cadence
            self.prev_img = img
            # saving cadence frames
            filename = f"{self.root.timestring}_{tween_frame_idx:09}.png"
            im = img.copy()
            self.datacallback(
                {"cadence_frame": Image.fromarray(cv2.cvtColor(im.astype("uint8"), cv2.COLOR_BGR2RGB))})
            cv2.imwrite(os.path.join(self.args.outdir, filename), img)
            if self.anim_args.save_depth_maps:
                depth_model.save(
                    os.path.join(self.args.outdir, f"{self.root.timestring}_depth_{tween_frame_idx:09}.png"),
                    depth)
        return
    def get_color_match_sample(self, path=None, image=None):
        if path is not None:
            color_match_sample = load_image(path)
            color_match_sample = color_match_sample.resize((self.args.W, self.args.H), PIL.Image.LANCZOS)
        elif image is not None:
            color_match_sample = np.asarray(image)
        color_match_sample = cv2.cvtColor(np.array(color_match_sample), cv2.COLOR_RGB2BGR)
        return color_match_sample

    def run_opts_scheduler(self, scheduled_clipskip, scheduled_noise_multiplier, scheduled_ddim_eta, scheduled_ancestral_eta):
        if 'img2img_fix_steps' in self.opts.data and self.opts.data[
            "img2img_fix_steps"]:  # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
            self.opts.data["img2img_fix_steps"] = False
        if scheduled_clipskip is not None:
            self.opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip
        if scheduled_noise_multiplier is not None:
            self.opts.data["initial_noise_multiplier"] = scheduled_noise_multiplier
        if scheduled_ddim_eta is not None:
            self.opts.data["eta_ddim"] = scheduled_ddim_eta
        if scheduled_ancestral_eta is not None:
            self.opts.data["eta_ancestral"] = scheduled_ancestral_eta
    def generate_disposable_image(self, keys, scheduled_sampler_name, raft_model, redo_flow_factor):
        print(
            f"Optical flow redo is diffusing and warping using {self.anim_args.optical_flow_redo_generation} optical flow before generation.")
        stored_seed = self.args.seed
        self.args.seed = random.randint(0, 2 ** 32 - 1)
        disposable_image = self.generate(self.args, keys, self.anim_args, self.loop_args, self.controlnet_args,
                                         self.root, self.frame_idx,
                                         sampler_name=scheduled_sampler_name)
        disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
        disposable_flow = get_flow_from_images(self.prev_img, disposable_image,
                                               self.anim_args.optical_flow_redo_generation, raft_model)
        disposable_image = cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB)
        disposable_image = image_transform_optical_flow(disposable_image, disposable_flow, redo_flow_factor)
        self.args.seed = stored_seed
        disposable_image = Image.fromarray(disposable_image)
        return disposable_image
