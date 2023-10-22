import os, sys
import subprocess
import secrets
import time
import argparse

from types import SimpleNamespace
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, WebSocket, Depends

from deforum.FILM.interpolator import Interpolator
from deforum.general_utils import substitute_placeholders
from deforum.main import Deforum
from deforum.animation.new_args import DeforumArgs, DeforumAnimArgs, ParseqArgs, LoopArgs, RootArgs, DeforumOutputArgs
from pydantic import BaseModel

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
comfy_path = os.path.join(root_path, "src/ComfyUI")
sys.path.extend([os.path.join(os.getcwd(), "deforum", "exttools")])

frames = []
cadence_frames = []

class Settings(BaseModel):
    file_content: str

async def get_deforum():
    deforum = setup_deforum()
    return deforum

def get_canny_image(image):
    import cv2
    import numpy as np
    image = cv2.Canny(np.array(image), 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

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
    import subprocess
    import time

    if len(frames) > 0:
        width = frames[0].size[0]
        height = frames[0].size[1]

        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
               '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-',
               '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0',
               '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23', '-an', filename]

        video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for frame in tqdm(frames, desc="Saving MP4 (ffmpeg)"):
            frame_np = np.array(frame)  # Convert the PIL image to numpy array
            video_writer.stdin.write(frame_np.tobytes())

        _, stderr = video_writer.communicate()

        if video_writer.returncode != 0:
            print(f"FFmpeg encountered an error: {stderr.decode('utf-8')}")
            return

        # if audio path is provided, merge the audio and the video
        if audio_path is not None:
            output_filename = f"output/mp4s/{time.strftime('%Y%m%d%H%M%S')}_with_audio.mp4"
            cmd = ['ffmpeg', '-y', '-i', filename, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', output_filename]

            result = subprocess.run(cmd, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Audio file merge failed from path {audio_path}\n{result.stderr.decode('utf-8')}")
    else:
        print("The buffer is empty, cannot save.")

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


def datacallback(data=None):

    if data:
        image = data.get("image")
        cadence_frame = data.get("cadence_frame")

    if image:
        frames.append(image)
    elif cadence_frame:
        cadence_frames.append(cadence_frame)


def setup_deforum():
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

    deforum.args.W = 1024
    deforum.args.H = 576
    deforum.args.steps = 20
    deforum.anim_args.animation_mode = "3D"
    deforum.anim_args.zoom = "0: (1.0)"
    deforum.anim_args.translate_z = "0: (4)"
    deforum.anim_args.strength_schedule = "0: (0.52)"
    deforum.anim_args.diffusion_cadence = 1
    deforum.anim_args.max_frames = 375
    deforum.video_args.store_frames_in_ram = True
    deforum.datacallback = datacallback
    from deforum.pipelines.comfy_generator import generate_txt2img_comfy
    deforum.generate_txt2img = generate_txt2img_comfy
    return deforum


def reset_deforum(deforum):
    args = SimpleNamespace(**extract_values(DeforumArgs()))
    anim_args = SimpleNamespace(**extract_values(DeforumAnimArgs()))
    parseg_args = SimpleNamespace(**extract_values(ParseqArgs()))
    loop_args = SimpleNamespace(**extract_values(LoopArgs()))
    root = SimpleNamespace(**RootArgs())

    output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}

    video_args = SimpleNamespace(**output_args_dict)
    controlnet_args = None
    deforum.args = args
    deforum.anim_args = anim_args
    deforum.parseq_args = parseg_args
    deforum.loop_args = loop_args
    deforum.root = root
    deforum.video_args = video_args
    deforum.controlnet_args = None
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

    deforum.args.W = 1024
    deforum.args.H = 576
    deforum.args.steps = 20
    deforum.anim_args.animation_mode = "3D"
    deforum.anim_args.zoom = "0: (1.0)"
    deforum.anim_args.translate_z = "0: (4)"
    deforum.anim_args.strength_schedule = "0: (0.52)"
    deforum.anim_args.diffusion_cadence = 1
    deforum.anim_args.max_frames = 375
    deforum.video_args.store_frames_in_ram = True
    # deforum.datacallback = datacallback
    # deforum.generate_txt2img = generate_txt2img_comfy



app = FastAPI()


@app.post("/start_deforum")
async def start_deforum(settings: Settings, deforum=Depends(get_deforum)):
    merged_data = json.loads(settings.file_content)

    # Update the SimpleNamespace objects as you did in the main() function
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

    return {"status": "success"}


@app.websocket("/ws/datacallback")
async def websocket_endpoint(websocket: WebSocket, deforum=Depends(get_deforum)):
    global ws
    ws = websocket
    await websocket.accept()
    deforum.datacallback = ws_datacallback

    success = deforum()

    return {"status": "done"}

async def ws_datacallback(data=None):
    if data:
        image = data.get("image")
        # Send image via WebSocket
        if image:
            await ws.send_bytes(image)

def main():
    process = None
    parser = argparse.ArgumentParser(description="Load settings from a txt file and run the deforum process.")
    parser.add_argument("--file", type=str, help="Path to the txt file containing dictionaries to merge.")
    parser.add_argument("--pipeline", type=str, default="deforum", help="Path to the txt file containing dictionaries to merge.")
    args_main = parser.parse_args()


    #deforum.enable_internal_controlnet()
    try:
        if args_main.pipeline == "deforum":

            global deforum
            deforum = setup_deforum()

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
            if deforum.args.seed == -1:
                deforum.args.seed = secrets.randbelow(18446744073709551615)

            success = deforum()
            output_filename_base = os.path.join(deforum.args.timestring)

            interpolator = Interpolator()

            interpolated = interpolator(frames, 1)

            save_as_h264(frames, output_filename_base + ".mp4", fps=15)
            save_as_h264(interpolated, output_filename_base + "_FILM.mp4", fps=30)
            if len(cadence_frames) > 0:
                save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")

        elif args_main.pipeline == "real2real":
            from deforum.pipelines.r2r_pipeline import Real2RealPipeline
            real2real = Real2RealPipeline()

            prompts = [
                "Starry night, Abstract painting by picasso",
                "PLanets and stars on the night sky, Abstract painting by picasso",
                "Galaxy, Abstract painting by picasso",
            ]

            keys = [30,30,30,30]

            real2real(fixed_seed=True,
                      mirror_conds=False,
                      use_feedback_loop=True,
                      prompts=prompts,
                      keys=keys,
                      strength=0.45)

        elif args_main.pipeline == "webui":
            # from deforum import streamlit_ui
            # cmd = ["streamlit", "run", f"{root_path}/deforum/streamlit_ui.py"]
            # process = subprocess.Popen(cmd)
            import streamlit.web.cli as stcli

            stcli.main(["run", f"{root_path}/deforum/streamlit_ui.py"])
        elif args_main.pipeline == "api":
            import uvicorn


            uvicorn.run(app, host="0.0.0.0", port=8000)

    except KeyboardInterrupt:
        if process:  # Check if there's a process reference
            process.terminate()  # Terminate the process
        print("\nKeyboardInterrupt detected. Interpolating and saving frames before exiting...")
        try:
            interpolator = Interpolator()
            interpolated = interpolator(frames, 1)
            output_filename_base = os.path.join(deforum.args.timestring)
            save_as_h264(frames, output_filename_base + ".mp4", fps=15)
            save_as_h264(interpolated, output_filename_base + "_FILM.mp4", fps=30)
            if len(cadence_frames) > 0:
                save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")
        except:
            pass

if __name__ == "__main__":
    main()
