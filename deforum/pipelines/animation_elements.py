import gc

import math
import os
import random
from typing import Any, Union, Tuple, Optional

import PIL
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageChops, ImageEnhance

from ..animation.animation import get_flip_perspective_matrix, flip_3d_perspective, transform_image_3d_new, \
    anim_frame_warp
from ..avfunctions.colors.colors import maintain_colors
from ..avfunctions.hybridvideo.hybrid_video import autocontrast_grayscale, get_matrix_for_hybrid_motion_prev, \
    get_matrix_for_hybrid_motion, image_transform_ransac, get_flow_for_hybrid_motion_prev, get_flow_for_hybrid_motion, \
    image_transform_optical_flow, get_flow_from_images, hybrid_composite, abs_flow_to_rel_flow, rel_flow_to_abs_flow
from ..avfunctions.image.image_sharpening import unsharp_mask
from ..avfunctions.image.load_images import load_image, get_mask_from_file
from ..avfunctions.image.save_images import save_image
from ..avfunctions.masks.composable_masks import compose_mask_with_check
from ..avfunctions.masks.masks import do_overlay_mask
from ..avfunctions.noise.noise import add_noise
from ..avfunctions.video_audio_utilities import get_frame_name, get_next_frame
from ..cmd import save_as_h264
from ..datafunctions.prompt import prepare_prompt, check_is_number, split_weighted_subprompts
from ..datafunctions.seed import next_seed
from ..exttools import py3d_tools as p3d
from ..pipelines.interpolator import Interpolator
from ..shared import root_path


def anim_frame_warp_cls(cls: Any) -> None:
    """
    Adjusts the animation frame warp for the given class instance based on various conditions.

    This function is an element of an animation pipeline that handles 2D/3D morphs on the given frame.
    It modifies parameters within the passed class instance based on the generation parameters object (cls.gen).

    Args:
        cls (Any): The class instance that contains generation parameters and needs frame warp adjustments.
                   This instance should have attributes like gen.prev_img, gen.use_depth_warping, etc.

    Returns:
        None
    """
    if cls.gen.prev_img is not None:
        cls.gen.mask = None
        if cls.gen.use_depth_warping:
            if cls.gen.depth is None and cls.depth_model is not None:
                cls.gen.depth = cls.depth_model.predict(cls.gen.opencv_image, cls.gen.midas_weight, cls.gen.half_precision)
        else:
            cls.gen.depth = None

        if cls.gen.animation_mode == '2D':
            cls.gen.prev_img = anim_frame_warp_2d_cls(cls, cls.gen.prev_img)
        else:  # '3D'
            cls.gen.prev_img, cls.gen.mask = anim_frame_warp_3d_cls(cls, cls.gen.prev_img)
    return


def anim_frame_warp_cls_image(cls: Any, image: Union[None, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Adjusts the animation frame warp for a given image and class instance.

    Args:
        cls: The class instance containing generation parameters.
        image: The image to be processed.

    Returns:
        Tuple containing the processed image and its mask.
    """
    if image is not None:
        cls.gen.mask = None
        if cls.gen.use_depth_warping:
            if cls.gen.depth is None and cls.depth_model is not None:
                cls.gen.depth = cls.depth_model.predict(image, cls.gen.midas_weight, cls.gen.half_precision)
        else:
            cls.gen.depth = None

        if cls.gen.animation_mode == '2D':
            cls.gen.image = anim_frame_warp_2d_cls(cls, image)
        else:  # '3D'
            cls.gen.image, cls.gen.mask = anim_frame_warp_3d_cls(cls, image)
    return cls.gen.image, cls.gen.mask


def anim_frame_warp_2d_cls(cls: Any, image: Union[None, Any]) -> Any:
    """
    Adjusts the 2D animation frame warp for a given image and class instance based on transformation parameters.

    Args:
        cls: The class instance containing generation parameters and transformation keys.
        image: The image to be processed.

    Returns:
        Processed image after 2D transformation.
    """
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


def anim_frame_warp_3d_cls(cls: Any, image: Union[None, Any]) -> Tuple[Any, Any]:
    """
    Adjusts the 3D animation frame warp for a given image and class instance based on transformation parameters.

    Args:
        cls: The class instance containing generation parameters and transformation keys.
        image: The image to be processed.

    Returns:
        Tuple containing the processed image after 3D transformation and its mask.
    """
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


def hybrid_composite_cls(cls: Any) -> None:
    """
    Creates a hybrid composite frame for the given class instance based on various conditions and transformation parameters.

    Args:
        cls: The class instance containing generation parameters, image paths, transformation keys, and other settings.

    Returns:
        None: Modifies the class instance attributes in place.
    """
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


def affine_persp_motion(cls: Any) -> None:
    """
    Applies affine or perspective motion transformation to the previous image of the given class instance.

    Args:
        cls: The class instance containing generation parameters, motion settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.hybrid_motion_use_prev_img:
        matrix = get_matrix_for_hybrid_motion_prev(cls.gen.frame_idx - 1, (cls.gen.W, cls.gen.H), cls.gen.inputfiles, cls.gen.prev_img,
                                                   cls.gen.hybrid_motion)
    else:
        matrix = get_matrix_for_hybrid_motion(cls.gen.frame_idx - 1, (cls.gen.W, cls.gen.H), cls.gen.inputfiles,
                                              cls.gen.hybrid_motion)
    cls.gen.prev_img = image_transform_ransac(cls.gen.prev_img, matrix, cls.gen.hybrid_motion)
    return


def optical_flow_motion(cls: Any) -> None:
    """
    Applies optical flow motion transformation to the previous image of the given class instance.

    Args:
        cls: The class instance containing generation parameters, motion settings, optical flow methods, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
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


def color_match_cls(cls: Any) -> None:
    """
    Matches the color of the previous image to a reference sample in the given class instance.

    Args:
        cls: The class instance containing generation parameters, color match settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.color_match_sample is None and cls.gen.prev_img is not None:
            cls.gen.color_match_sample = cls.gen.prev_img.copy()
    elif cls.gen.prev_img is not None:
        cls.gen.prev_img = maintain_colors(cls.gen.prev_img, cls.gen.color_match_sample, cls.gen.color_coherence)
    return


def set_contrast_image(cls: Any) -> None:
    """
    Adjusts the contrast of the previous image in the given class instance.

    Args:
        cls: The class instance containing generation parameters, contrast settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
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


def handle_noise_mask(cls: Any) -> None:
    """
    Composes a noise mask for the contrast image in the given class instance.

    Args:
        cls: The class instance containing generation parameters, noise mask settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    cls.gen.noise_mask = compose_mask_with_check(cls.gen, cls.gen, cls.gen.noise_mask_seq, cls.gen.noise_mask_vals, Image.fromarray(
        cv2.cvtColor(cls.gen.contrast_image, cv2.COLOR_BGR2RGB)))
    return


def add_noise_cls(cls: Any) -> None:
    """
    Adds noise to the contrast image in the given class instance.

    Args:
        cls: The class instance containing generation parameters, noise settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
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


def get_generation_params(cls: Any) -> None:
    """
    Fetches and sets generation parameters for the given class instance based on various conditions and schedules.

    Args:
        cls: The class instance containing various generation parameters, schedules, settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """

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


def optical_flow_redo(cls: Any) -> None:
    """
    Applies optical flow redo transformation before generation based on given parameters and conditions.

    Args:
        cls: The class instance containing generation parameters, optical flow settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
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


def diffusion_redo(cls: Any) -> None:
    """
    Applies diffusion redo transformation before the final generation based on given parameters and conditions.

    Args:
        cls: The class instance containing generation parameters, diffusion redo settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
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

def main_generate_with_cls(cls: Any) -> None:
    """
    Executes the main generation process for the given class instance.

    Args:
        cls: The class instance containing generation parameters and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    cls.gen.image = cls.generate()

    return


def post_hybrid_composite_cls(cls: Any) -> None:
    """
    Executes the post-generation hybrid compositing process for the given class instance.

    Args:
        cls: The class instance containing generation parameters, hybrid compositing settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    # do hybrid video after generation
    if cls.gen.frame_idx > 0 and cls.gen.hybrid_composite == 'After Generation':
        image = cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR)
        cls.gen, image = hybrid_composite(cls.gen, cls.gen, cls.gen.frame_idx, image, cls.depth_model, cls.gen.hybrid_comp_schedules, cls.gen)
        cls.gen.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return


def post_color_match_with_cls(cls: Any) -> None:
    """
    Executes the post-generation color matching process for the given class instance.

    Args:
        cls: The class instance containing generation parameters, color matching settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
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


def overlay_mask_cls(cls: Any) -> None:
    """
    Overlays a mask onto the generated image in the given class instance.

    Args:
        cls: The class instance containing generation parameters, overlay mask settings, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
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


def post_gen_cls(cls: Any) -> None:
    """
    Executes post-generation processes for the given class instance including saving, updating images,
    and handling depth maps.

    Args:
        cls: The class instance containing generation parameters, image data, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if cls.gen.frame_idx < cls.gen.max_frames:
        # cls.images.append(cls.gen.opencv_image.copy())
        cls.images.append(np.array(cls.gen.image))

        cls.gen.opencv_image = cv2.cvtColor(np.array(cls.gen.image), cv2.COLOR_RGB2BGR)

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


def make_cadence_frames(cls: Any) -> None:
    """
    Generates intermediate frames between keyframes to create smoother animations.

    This function uses optical flow or hybrid motion methods to determine motion between frames.
    It then generates in-between frames (cadence frames) for smoother animations. The function also
    handles grayscale conversion, mask overlay, and saving cadence frames.

    Args:
        cls: The class instance containing generation parameters, image data, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
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


def color_match_video_input(cls: Any) -> None:
    """
    Matches the color of the generated image to the corresponding frame from the input video.

    If the current frame index is divisible by the specified interval (color_coherence_video_every_N_frames),
    the function fetches the corresponding frame from the input video and uses it as the reference
    for color matching. The fetched frame is resized to match the dimensions of the generated image
    and then used to set the color_match_sample attribute for later use in color coherence operations.

    Args:
        cls: The class instance containing generation parameters, image data, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place.
    """
    if int(cls.gen.frame_idx) % int(cls.gen.color_coherence_video_every_N_frames) == 0:
        prev_vid_img = Image.open(os.path.join(cls.outdir, 'inputframes', get_frame_name(
            cls.video_init_path) + f"{cls.gen.frame_idx:09}.jpg"))
        cls.gen.prev_vid_img = prev_vid_img.resize((cls.W, cls.H), PIL.Image.LANCZOS)
        color_match_sample = np.asarray(cls.gen.prev_vid_img)
        cls.gen.color_match_sample = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2BGR)

def film_interpolate_cls(cls: Any) -> None:
    """
    Performs frame interpolation on a sequence of images using FILM and saves the interpolated video.

    The function calculates the number of in-between frames to add based on the frame_interpolation_x_amount attribute.
    It then uses FILM to interpolate between the original frames and generate the in-between frames.
    The interpolated video is then saved as an H264-encoded MP4 file.

    Args:
        cls: The class instance containing generation parameters, image data, and other attributes.

    Returns:
        None: Modifies the class instance attributes in place and saves the interpolated video.
    """
    # "frame_interpolation_engine": "FILM",
    # "frame_interpolation_x_amount": 2,
    # "frame_interpolation_slow_mo_enabled": false,
    # "frame_interpolation_slow_mo_amount": 2,
    # "frame_interpolation_keep_imgs": false,
    # "frame_interpolation_use_upscaled": false,
    dir_path = os.path.join(root_path, 'output/video')
    os.makedirs(dir_path, exist_ok=True)
    output_filename_base = os.path.join(dir_path, cls.gen.timestring)
    interpolator = Interpolator()

    film_in_between_frames_count = calculate_frames_to_add(len(cls.images), cls.gen.frame_interpolation_x_amount)
    print("Interpolating with", film_in_between_frames_count)
    interpolated = interpolator(cls.images, film_in_between_frames_count)
    save_as_h264(interpolated, output_filename_base + "_FILM.mp4", fps=30)

def calculate_frames_to_add(total_frames: int, interp_x: float) -> int:
    """
    Calculates the number of frames to add for interpolation based on the desired multiplier.

    Args:
        total_frames (int): The total number of original frames in the sequence.
        interp_x (float): The desired multiplier for frame interpolation.

    Returns:
        int: The number of frames to add between each original frame.
    """
    frames_to_add = (total_frames * interp_x - total_frames) / (total_frames - 1)
    return int(round(frames_to_add))