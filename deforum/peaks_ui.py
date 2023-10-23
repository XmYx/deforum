import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def extract_peaks(y, sr, pre_max, post_max, pre_avg, post_avg, delta, wait):
    # Find peaks using onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr,
                                              pre_max=pre_max, post_max=post_max,
                                              pre_avg=pre_avg, post_avg=post_avg,
                                              delta=delta, wait=wait)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


st.title("Audio Peak Extractor")

uploaded_file = st.file_uploader("Choose an MP3 or WAV file", type=["mp3", "wav"])

def plot_keyframes(keyframes):
    # Extract x and y values from keyframes
    x = list(keyframes.keys())
    y = list(keyframes.values())

    # Plot the keyframes
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.xlabel('Keyframe Index')
    plt.ylabel('Time (s)')
    plt.title('Keyframes')
    plt.grid(True)
    st.pyplot(plt.gcf())


if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        # Load the audio file
        y, sr = librosa.load(uploaded_file, sr=None)

        # Add sliders for tweaking parameters
        pre_max = st.slider("Pre Max", 1, 100, 20)
        post_max = st.slider("Post Max", 1, 100, 50)
        pre_avg = st.slider("Pre Avg", 1, 100, 10)
        post_avg = st.slider("Post Avg", 1, 100, 30)
        delta = st.slider("Delta", 0.0, 2.0, 0.2)
        wait = st.slider("Wait", 0, 50, 5)

        # Extract peaks
        peak_times = extract_peaks(y, sr, pre_max, post_max, pre_avg, post_avg, delta, wait)

    with col2:
        # Convert peak times to peak frames
        peak_frames = librosa.time_to_frames(peak_times, sr=sr)

        # Extract amplitude values at the peak frames
        peak_amplitudes = y[peak_frames]

        # Switches to minimize or maximize keyframe values at 0
        minimize_switch = st.checkbox("Minimize Keyframes at 0")
        minimize_at = st.slider("Minimize at", -1000.0, 1000.0, 0.0, 0.1)
        maximize_at = st.slider("Maximize at", -1000.0, 1000.0, 0.0, 0.1)
        maximize_switch = st.checkbox("Maximize Keyframes at 0")

        # Slider to adjust the amplitude of keyframes
        amplitude_multiplier = st.slider("Amplitude Multiplier", 0.0, 100.0, 1.0, 0.001)
        amplitude_multiplier_2 = st.slider("Amplitude Multiplier", 0.0, 5000.0, 1.0, 0.01)

        # Adjusting the amplitude of keyframes based on user choices
        if minimize_switch:
            peak_amplitudes = np.clip(peak_amplitudes, minimize_at, None)  # Set negative values to 0
        if maximize_switch:
            peak_amplitudes = np.clip(peak_amplitudes, None, maximize_at)  # Set positive values to 0

        peak_amplitudes *= amplitude_multiplier
        peak_amplitudes *= amplitude_multiplier_2

        # Plotting the waveform with peaks
        # plt.figure(figsize=(10, 6))
        # librosa.display.waveshow(y, sr=sr)
        # plt.vlines(peak_times, -1, 1, color='r', alpha=0.5)
        # plt.title("Waveform with Peaks")
        # st.pyplot(plt.gcf())

        # Displaying the keyframes
        keyframes = {i: round(amplitude, 5) for i, amplitude in enumerate(peak_amplitudes)}
        st.text_area("Keyframes:", str(keyframes))

        plot_keyframes(keyframes)