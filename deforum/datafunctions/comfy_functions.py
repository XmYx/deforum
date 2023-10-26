
import os, sys
import numpy as np
import torch
import torchsde
from torch import nn
from torch.cuda import nvtx

from deforum.shared import root_path
from trt_yank.model_manager import modelmanager
from trt_yank.utilities import Engine

# # 1. Check if the "src" directory exists
# if not os.path.exists(os.path.join(root_path, "src")):
#     os.makedirs(os.path.join(root_path, 'src'))
# # 2. Check if "ComfyUI" exists
# if not os.path.exists(comfy_path):
#     # Clone the repository if it doesn't exist
#     subprocess.run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI", comfy_path])
# else:
#     # 3. If "ComfyUI" does exist, check its commit hash
#     current_folder = os.getcwd()
#     os.chdir(comfy_path)
#     current_commit = subprocess.getoutput("git rev-parse HEAD")
#
#     # 4. Reset to the desired commit if necessary
#     if current_commit != "4185324":  # replace with the full commit hash if needed
#         subprocess.run(["git", "fetch", "origin"])
#         subprocess.run(["git", "reset", "--hard", "b935bea3a0201221eca7b0337bc60a329871300a"])  # replace with the full commit hash if needed
#         subprocess.run(["git", "pull", "origin", "master"])
#     os.chdir(current_folder)

comfy_path = os.path.join(root_path, "src/ComfyUI")
sys.path.append(comfy_path)

from collections import namedtuple


# Define the namedtuple structure based on the properties identified
CLIArgs = namedtuple('CLIArgs', [
    'cpu',
    'normalvram',
    'lowvram',
    'novram',
    'highvram',
    'gpu_only',
    'disable_xformers',
    'use_pytorch_cross_attention',
    'use_split_cross_attention',
    'use_quad_cross_attention',
    'fp16_vae',
    'bf16_vae',
    'fp32_vae',
    'force_fp32',
    'force_fp16',
    'disable_smart_memory',
    'disable_ipex_optimize',
    'listen',
    'port',
    'enable_cors_header',
    'extra_model_paths_config',
    'output_directory',
    'temp_directory',
    'input_directory',
    'auto_launch',
    'disable_auto_launch',
    'cuda_device',
    'cuda_malloc',
    'disable_cuda_malloc',
    'dont_upcast_attention',
    'bf16_unet',
    'directml',
    'preview_method',
    'dont_print_server',
    'quick_test_for_ci',
    'windows_standalone_build',
    'disable_metadata'
])

# Update the mock args object with default values for the new properties
mock_args = CLIArgs(
    cpu=False,
    normalvram=False,
    lowvram=False,
    novram=False,
    highvram=True,
    gpu_only=True,
    disable_xformers=True,
    use_pytorch_cross_attention=True,
    use_split_cross_attention=False,
    use_quad_cross_attention=False,
    fp16_vae=True,
    bf16_vae=False,
    fp32_vae=False,
    force_fp32=False,
    force_fp16=False,
    disable_smart_memory=False,
    disable_ipex_optimize=True,
    listen="127.0.0.1",
    port=8188,
    enable_cors_header=None,
    extra_model_paths_config=None,
    output_directory=None,
    temp_directory=None,
    input_directory=None,
    auto_launch=False,
    disable_auto_launch=True,
    cuda_device=0,
    cuda_malloc=False,
    disable_cuda_malloc=True,
    dont_upcast_attention=True,
    bf16_unet=True,
    directml=None,
    preview_method="none",
    dont_print_server=True,
    quick_test_for_ci=False,
    windows_standalone_build=False,
    disable_metadata=False
)
from comfy.cli_args import LatentPreviewMethod as lp
class MockCLIArgsModule:
    args = mock_args
    LatentPreviewMethod = lp

# Add the mock module to sys.modules under the name 'comfy.cli_args'
sys.modules['comfy.cli_args'] = MockCLIArgsModule()


class DeforumBatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if torch.abs(t0 - t1) < 1e-6:  # or some other small value
            # Handle this case, e.g., return a zero tensor of appropriate shape
            return torch.zeros_like(t0)
        if self.cpu_tree:
            w = torch.stack(
                [tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (
                            self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]


#from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel









import comfy.k_diffusion.sampling
comfy.k_diffusion.sampling.BatchedBrownianTree = DeforumBatchedBrownianTree
