import secrets

import streamlit as st
import os
import json
import argparse

from deforum.cmd import setup_deforum, merge_dicts_from_txt, Interpolator, save_as_h264, reset_deforum


frames = []
cadence_frames = []



def generate_ui():
    with st.form(key="controls_form"):
        st.subheader("Basic Settings")
        W = st.number_input("Width", value=768, min_value=1)
        H = st.number_input("Height", value=768, min_value=1)
        # show_info_on_ui = st.checkbox("Show Info on UI", value=False)
        # tiling = st.checkbox("Tiling", value=False)
        # restore_faces = st.checkbox("Restore Faces", value=False)
        # seed_resize_from_w = st.number_input("Seed Resize From Width", value=768, min_value=1)
        # seed_resize_from_h = st.number_input("Seed Resize From Height", value=768, min_value=1)
        seed = st.number_input("Seed", value=-1)
        # sampler = st.text_input("Sampler", value="DPM++ 2M SDE Karras")
        # steps = st.number_input("Steps", value=20, min_value=1)
        # batch_name = st.text_input("Batch Name", value="Bot-Preset-tests")
        seed_behavior = st.selectbox("Seed Behavior", ["fixed", "random"])
        # seed_iter_N = st.number_input("Seed Iteration N", value=1, min_value=1)
        # use_init = st.checkbox("Use Initialization", value=False)
        # strength = st.slider("Strength", value=0.8, min_value=0.0, max_value=1.0)
        # strength_0_no_init = st.checkbox("Strength 0 No Initialization", value=False)
        # init_image = st.text_input("Init Image", value="https://deforum.github.io/a1/I1.png")
        # use_mask = st.checkbox("Use Mask", value=False)
        # use_alpha_as_mask = st.checkbox("Use Alpha as Mask", value=False)
        # mask_file = st.text_input("Mask File", value="https://deforum.github.io/a1/M1.jpg")
        # invert_mask = st.checkbox("Invert Mask", value=False)
        # mask_contrast_adjust = st.slider("Mask Contrast Adjust", value=1.0, min_value=0.0, max_value=2.0)
        # mask_brightness_adjust = st.slider("Mask Brightness Adjust", value=1.0, min_value=0.0, max_value=2.0)
        # overlay_mask = st.checkbox("Overlay Mask", value=True)
        # mask_overlay_blur = st.number_input("Mask Overlay Blur", value=4, min_value=0)
        # fill = st.slider("Fill", value=1, min_value=0, max_value=2)
        # full_res_mask = st.checkbox("Full Resolution Mask", value=True)
        # full_res_mask_padding = st.number_input("Full Resolution Mask Padding", value=4, min_value=0)
        # reroll_blank_frames = st.selectbox("Reroll Blank Frames", ["ignore", "reroll", "stop"])
        # reroll_patience = st.slider("Reroll Patience", value=10.0, min_value=0.0, max_value=100.0)
        # motion_preview_mode = st.checkbox("Motion Preview Mode", value=False)
        prompts = st.text_area("Prompts", value=json.dumps({
            "0":"Hyperrealistic art Dog . Extremely high-resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike --neg simplified, abstract, unrealistic, impressionistic, low resolution"
        }))
        # animation_prompts_positive = st.text_input("Animation Prompts Positive")
        # animation_prompts_negative = st.text_input("Animation Prompts Negative")
        animation_mode = st.selectbox("Animation Mode", ["2D", "3D"])
        max_frames = st.number_input("Max Frames", value=375, min_value=1)
        border = st.selectbox("Border", ["replicate", "reflect", "constant"])
        # ... [continue adding widgets for all other parameters]

        # Assemble the parameters into a dictionary
        params = {
            "W": int(W),
            "H": int(H),
            # "show_info_on_ui": show_info_on_ui,
            # "tiling": tiling,
            # "restore_faces": restore_faces,
            # "seed_resize_from_w": seed_resize_from_w,
            # "seed_resize_from_h": seed_resize_from_h,
            "seed": seed,
            # "sampler": sampler,
            # "steps": steps,
            # "batch_name": batch_name,
            "seed_behavior": seed_behavior,
            # "seed_iter_N": seed_iter_N,
            # "use_init": use_init,
            # "strength": strength,
            # "strength_0_no_init": strength_0_no_init,
            # "init_image": init_image,
            # "use_mask": use_mask,
            # "use_alpha_as_mask": use_alpha_as_mask,
            # "mask_file": mask_file,
            # "invert_mask": invert_mask,
            # "mask_contrast_adjust": mask_contrast_adjust,
            # "mask_brightness_adjust": mask_brightness_adjust,
            # "overlay_mask": overlay_mask,
            # "mask_overlay_blur": mask_overlay_blur,
            # "fill": fill,
            # "full_res_mask": full_res_mask,
            # "full_res_mask_padding": full_res_mask_padding,
            # "reroll_blank_frames": reroll_blank_frames,
            # "reroll_patience": reroll_patience,
            # "motion_preview_mode": motion_preview_mode,
            "prompts": json.loads(prompts),
            # "animation_prompts_positive": animation_prompts_positive,
            # "animation_prompts_negative": animation_prompts_negative,
            "animation_mode": animation_mode,
            "max_frames": max_frames,
            "border": border,
            # ... [include all parameters in this dictionary]
        }
        submit_button = st.form_submit_button("Generate Deforum Animation")
    return params, submit_button


def datacallback_streamlit(data=None):
    if data:
        image = data.get("image")
        cadence_frame = data.get("cadence_frame")
    if image:
        frames.append(image)
        # 2. Update the placeholder with the latest image
        frame_placeholder.image(image, caption="Latest Frame", use_column_width=True)
    elif cadence_frame:
        cadence_frames.append(cadence_frame)
        # 2. Update the cadence frame placeholder with the latest cadence frame
        cadence_frame_placeholder.image(cadence_frame, caption="Latest Cadence Frame", use_column_width=True)



def main():
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


def update_deforum(data):
    for key, value in data.items():

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
def save_frames():
    interpolator = Interpolator()
    interpolated = interpolator(frames, 1)
    output_filename_base = os.path.join(deforum.args.timestring)
    save_as_h264(frames, output_filename_base + ".mp4", fps=15)
    save_as_h264(interpolated, output_filename_base + "_FILM.mp4", fps=30)
    if len(cadence_frames) > 0:
        save_as_h264(cadence_frames, output_filename_base + f"_cadence{deforum.anim_args.diffusion_cadence}.mp4")
    st.session_state.generation_complete = True  # Check if the generation was stopped prematurely

if __name__ == "__main__":




    st.set_page_config(layout="wide")
    # if "deforum" not in st.session_state:
    deforum = setup_deforum()
    deforum.datacallback = datacallback_streamlit
    st.session_state.generation_started = False
    st.session_state.generation_complete = False
    col1, col2 = st.columns([2,8])
    with col1:
        # Setup for Streamlit
        st.title("Deforum Animation Generator")
        st.write("### Upload Settings File")
        uploaded_file = st.file_uploader("Choose a settings txt file", type="txt")
        params, submitted = generate_ui()
    with col2:
        frame_placeholder = st.empty()
        cadence_frame_placeholder = st.empty()
    if uploaded_file:
        # Load settings from uploaded file
        merged_data = json.loads(uploaded_file.getvalue())
        # Update the SimpleNamespace objects
        update_deforum(merged_data)



    if submitted:

        st.session_state.generation_started = True
        st.session_state.generation_complete = False


        reset_deforum(deforum)
        deforum.anim_args.seed_schedule = "0:(-1)"

        # Call the main function to generate the animation
        if not uploaded_file:
            update_deforum(params)
        #deforum.generate_txt2img = generate_txt2img_comfy
        main()
