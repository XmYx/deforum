import argparse
import json
import os, sys
from types import SimpleNamespace

from deforum.animation.new_args import DeforumArgs, DeforumAnimArgs, ParseqArgs, LoopArgs, RootArgs, DeforumOutputArgs

print(os.getcwd())

#sys.path.extend([os.path.join(os.getcwd(), "deforum", "exttools")])
sys.path.extend([os.path.join(os.getcwd(), "deforum", "exttools")])

import secrets
import time

from deforum.general_utils import substitute_placeholders
from deforum.main import Deforum

import subprocess

frames = []
cadence_frames = []

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

        print(data_dict)

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




def save_as_h264(frames, filename, audio_path=None, fps=24):
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
                output_filename = f"output/mp4s/{timestamp}_with_audio.mp4"
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


def main():

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
    args_main = parser.parse_args()

    if args_main.file:
        merged_data = merge_dicts_from_txt(args_main.file)

        # 3. Update the SimpleNamespace objects
        for key, value in merged_data.items():
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

    print(deforum.anim_args.strength_schedule)

    success = deforum()


    output_filename_base = os.path.join(deforum.args.timestring)
    save_as_h264(frames, output_filename_base + ".mp4")
    if len(cadence_frames) > 0:
        save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")



if __name__ == "__main__":
    main()
