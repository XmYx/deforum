import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_cache_folder = os.path.join(root_path, "models/checkpoints")


from deforum.fileutils.civit_ai_downloader import fetch_and_download_model
from deforum.pipelines.comfy_generator import ComfyDeforumGenerator
# from deforum.pipelines.deforum_pipeline import DeforumPipeline
#
#
# pipelines = {
#     "deforum":DeforumPipeline
# }
#
# engines = {
#     "comfy": ComfyDeforumGenerator
# }
#
# available_engines = [key for key, _ in engines.items()]
# available_engine_classes = [value for _, value in engines.items()]
#
# available_pipelines = [key for key, _ in pipelines.items()]
#
# animation_modes = ['2D', '3D', 'None']