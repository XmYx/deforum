import os

from .shared import root_path

default_cache_folder = os.path.join(root_path, "models/checkpoints")


from .fileutils.civit_ai_downloader import fetch_and_download_model
from .pipelines.comfy_generator import ComfyDeforumGenerator
from .pipelines.deforum_pipeline import DeforumAnimationPipeline


pipelines = {
    "deforum":DeforumAnimationPipeline
}

engines = {
    "comfy": ComfyDeforumGenerator
}

available_engines = [key for key, _ in engines.items()]
available_engine_classes = [value for _, value in engines.items()]

available_pipelines = [value.__name__ for _, value in pipelines.items()]

animation_modes = ['2D', '3D', 'None']


from .pipelines.animation_elements import (anim_frame_warp_cls,
                                          anim_frame_warp_cls_image,
                                          anim_frame_warp_2d_cls,
                                          anim_frame_warp_3d_cls,
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