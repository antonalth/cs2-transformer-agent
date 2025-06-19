import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import moviepy.editor as mpy

def extract_features(path, time_ms, n_mels=80):
    """
    Extracts features for full audio sliding windows of size time_ms.
    Returns arrays of shape [n_windows, n_mels] for each feature and sampling info.
    """
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    window_samples = int(sr * time_ms / 1000)
    n_fft = 2 ** int(np.ceil(np.log2(window_samples)))
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    window = np.hanning(n_fft)

    total_samples = y.shape[1]
    n_windows = int(np.ceil(total_samples / window_samples))
    pad_amount = n_windows * window_samples - total_samples
    if pad_amount > 0:
        y = np.pad(y, ((0,0),(0,pad_amount)), mode='constant')

    mags_L, mags_R, coss, sins = [], [], [], []
    for i in range(n_windows):
        chunk = y[:, i*window_samples:(i+1)*window_samples]
        if chunk.shape[1] < n_fft:
            chunk = np.pad(chunk, ((0,0),(0,n_fft-chunk.shape[1])), mode='constant')
        D_L = librosa.stft(chunk[0], n_fft=n_fft, hop_length=n_fft,
                           win_length=n_fft, window=window, center=False)
        D_R = librosa.stft(chunk[1], n_fft=n_fft, hop_length=n_fft,
                           win_length=n_fft, window=window, center=False)
        Cmel_L = mel_fb.dot(D_L)
        Cmel_R = mel_fb.dot(D_R)
        mag_L = np.log(np.abs(Cmel_L[:,0]) + 1e-6)
        mag_R = np.log(np.abs(Cmel_R[:,0]) + 1e-6)
        phase_L = np.angle(Cmel_L[:,0])
        phase_R = np.angle(Cmel_R[:,0])
        ipd = phase_L - phase_R
        mags_L.append(mag_L)
        mags_R.append(mag_R)
        coss.append(np.cos(ipd))
        sins.append(np.sin(ipd))

    return np.array(mags_L), np.array(mags_R), np.array(coss), np.array(sins), sr, window_samples

def create_video(wavfile, output, time_ms, width, height):
    # Extract features
    mags_L, mags_R, coss, sins, sr, window_samples = extract_features(wavfile, time_ms)
    n_windows = mags_L.shape[0]
    fps = 1000.0 / time_ms
    duration = n_windows / fps

    # Compute y-limits at 95th percentile
    y_lims = []
    for arr in (mags_L, mags_R, coss, sins):
        y_min = float(arr.min())
        y_max = float(np.percentile(arr, 95))
        y_lims.append((y_min, y_max))

    # Prepare matplotlib figure
    fig, axs = plt.subplots(2,2, figsize=(width/100, height/100), dpi=100)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    titles = ['Log-Mel Left', 'Log-Mel Right', 'cos(IPD)', 'sin(IPD)']
    lines = []
    for ax, title, ylim in zip(axs.flatten(), titles, y_lims):
        ax.set_title(title)
        ax.set_xlim(0, mags_L.shape[1]-1)
        ax.set_ylim(ylim)
        line, = ax.plot(np.zeros_like(mags_L[0]))
        ax.set_xlabel('Mel Band')
        ax.set_ylabel('Value')
        lines.append(line)
    canvas = FigureCanvas(fig)

    def make_frame(t):
        idx = min(int(t * fps), n_windows-1)
        # Update line data
        for line, data in zip(lines, (mags_L[idx], mags_R[idx], coss[idx], sins[idx])):
            line.set_ydata(data)
        # Draw canvas
        canvas.draw()
        buf = canvas.buffer_rgba()
        arr = np.asarray(buf)[:,:,:3]  # drop alpha
        return arr

    # Create video clip
    video_clip = mpy.VideoClip(make_frame, duration=duration)
    audio_clip = mpy.AudioFileClip(wavfile).subclip(0, duration)
    video_clip = video_clip.set_audio(audio_clip).set_fps(fps)

    # Write file
    video_clip.write_videofile(output, codec='libx264', audio_codec='aac', 
                               fps=fps, bitrate='2000k')

def main():
    parser = argparse.ArgumentParser(description="Create video of audio features with audio")
    parser.add_argument("wavfile", help="Path to stereo .wav")
    parser.add_argument("--time", type=float, required=True, help="Window size in ms")
    parser.add_argument("--output", required=True, help="Output MP4 file path")
    parser.add_argument("--width", type=int, default=800, help="Video width")
    parser.add_argument("--height", type=int, default=600, help="Video height")
    args = parser.parse_args()

    create_video(args.wavfile, args.output, args.time, args.width, args.height)

if __name__ == "__main__":
    main()
