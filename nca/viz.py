import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import numpy as np
import subprocess
import os
import shutil
import imageio.v2 as imageio

## Animation

def animate_states(states):
    states = states.permute(0, 2, 3, 1)
    states_display = states[..., :4]
    states_display = torch.clamp(states_display, 0.0, 1.0) #clamp to avoid warning
    
    fig, ax = plt.subplots()
    im = ax.imshow(states_display[0])
    # ax.axis("off")

    def update(t):
        im.set_data(states_display[t])
        ax.set_title(f"t={t}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(states))
    return ani

def frame_to_rgb_uint8(frame):
    """
    frame: (C, H, W) in [~0, 1] (torch.Tensor)
    returns: (H, W, 3) uint8 (numpy.ndarray)
    """
    frame = frame.detach().cpu()

    rgb = frame[0:3]  # (3, H, W) torch.Tensor
    alpha = frame[3:4]  # (1, H, W) torch.Tensor

    img = rgb * alpha  # (3, H, W) torch.Tensor
    img = torch.clamp(img, 0.0, 1.0)  # Clamp values using PyTorch

    # Permute to (H, W, 3) using PyTorch, then convert to numpy
    img_np = img.permute(1, 2, 0).numpy()  # (H, W, 3) numpy.ndarray
    return (img_np * 255).astype(np.uint8)

def frame_to_rgb_uint8_signed(frame, eps=1e-8):
    frame = frame.detach().cpu()

    rgb = frame[0:3]
    alpha = frame[3:4]

    img = rgb * alpha

    m = img.abs().max()
    img = img / (m + eps)        # → [-1,1]
    img = (img + 1) / 2          # → [0,1]

    img_np = img.permute(1, 2, 0).numpy()
    return (img_np * 255).astype(np.uint8)

def channel_to_rgb(channel, cmap="vanimo", eps=1e-8):
    """
    channel: (H,W) tensor
    returns: (H,W,3) uint8 RGB
    """
    m = channel.abs().max()
    ch = channel/(m+eps)    # [-1,1]
    ch = (ch + 1) / 2       # [0,1]

    cmap_fn = plt.get_cmap(cmap)
    rgb = cmap_fn(ch, bytes=True)[..., :3]  # drop alpha

    return rgb

def tile_rgb_frames(rows, cols, frames):
    """
    frames: list of tensors with shape (H,W,3)
    """
    f_height = frames[0].shape[0]
    f_width = frames[0].shape[1]
    H = f_height * rows
    W = f_width * cols
    img = np.zeros((H,W,3),dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i*cols) + j > len(frames):
                return img
            img[f_height*i:f_height*(i+1), f_width*j:f_width*(j+1), :] = frames[(i*cols)+j]
    return img

def make_tiled_channel_frames(states):
    frames = []
    for state in states:
        tiled_img = tile_rgb_frames(4, 4, [channel_to_rgb(channel) for channel in state])
        frames.append(tiled_img)
    return frames

def save_crisp_mp4(
    frames,
    output_file="output.mp4",
    fps=30,
    upscale=1,
    frames_dir="frames",
    quiet=True
):
    """
    Save a list of numpy frames as a crisp MP4 using ffmpeg.

    Args:
        frames: List of numpy arrays (H, W, 3 or 4)
        output_file: Output video filename
        fps: Frames per second
        upscale: Integer factor to upscale frames (nearest neighbor)
        frames_dir: Temporary folder to store frames (will be overwritten)
    """
    # --- Prepare frames directory ---
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # --- Optionally upscale frames ---
    def upscale_frame(frame, scale):
        if scale == 1:
            return frame
        return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)

    # --- Save frames as PNG ---
    for i, frame in enumerate(frames):
        frame_to_save = upscale_frame(frame, upscale)
        # Remove alpha if present
        if frame_to_save.shape[2] == 4:
            frame_to_save = frame_to_save[:, :, :3]
        path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        imageio.imwrite(path, frame_to_save)

    # --- Build ffmpeg command ---
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output_file if exists
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv444p",
        "-crf", "0",
        "-preset", "veryslow",
        output_file
    ]

    # --- Run ffmpeg (suppress output) ---
    if quiet:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,  # hide normal messages
            stderr=subprocess.STDOUT     # merge errors into stdout (and hide)
        )
    else:
        subprocess.run(cmd, check=True)

## Plotting

def plot_means_by_parts(cell_states, stats):
    means = stats["means"]

    plt.figure(figsize=(12,12))

    plt.subplot(4,2,1)
    plt.plot(means["image"])
    plt.title("Total Image Average")

    num_alive = [state.shape[1] for state in cell_states['alive']]
    plt.subplot(4,2,2)
    plt.plot(num_alive)
    plt.title("Number of alive cells")

    plt.subplot(4,2,3)
    plt.plot(means["alive"])
    plt.title("Alive Mean")

    plt.subplot(4,2,4)
    plt.plot(means["body"])
    plt.title("Body Mean")

    plt.subplot(4,2,5)
    plt.plot(means["tongue"])
    plt.title("Tongue Mean")

    plt.subplot(4,2,6)
    plt.plot(means["eyes"])
    plt.title("Eyes Mean")

    plt.subplot(4,2,7)
    plt.plot(means["l_eye"])
    plt.title("Left Eye Mean")

    plt.subplot(4,2,8)
    plt.plot(means["r_eye"])
    plt.title("Right Eye Mean")

    plt.tight_layout()
    plt.show()

def plot_variances_by_part(stats):
    vars_ = stats["vars"]

    plt.figure(figsize=(12,12))

    plt.subplot(4,2,1)
    plt.plot(vars_["image"])
    plt.title("Total Image Variance")

    plt.subplot(4,2,2)
    plt.plot(vars_["alive"])
    plt.title("Alive Variance")

    plt.subplot(4,2,3)
    plt.plot(vars_["body"])
    plt.title("Body Variance")

    plt.subplot(4,2,4)
    plt.plot(vars_["tongue"])
    plt.title("Tongue Variance")

    plt.subplot(4,2,5)
    plt.plot(vars_["eyes"])
    plt.title("Eyes Variance")

    plt.subplot(4,2,6)
    plt.plot(vars_["l_eye"])
    plt.title("Left Eye Variance")

    plt.subplot(4,2,7)
    plt.plot(vars_["r_eye"])
    plt.title("Right Eye Variance")

    plt.tight_layout()
    plt.show()

def plot_alive_vs_r_eye_difference(stats):
    means = stats["means"]
    vars_ = stats["vars"]

    mean_diff = means["alive"] - means["r_eye"]
    var_diff = vars_["alive"] - vars_["r_eye"]

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(mean_diff)
    plt.title("Mean Difference (Alive - Right Eye)")

    plt.subplot(1,2,2)
    plt.plot(var_diff)
    plt.title("Variance Difference (Alive - Right Eye)")

    plt.tight_layout()
    plt.show()

def rank_channels_alive_vs_r_eye(stats, time_index=-1):
    means = stats["means"]

    diff = torch.abs(means["alive"][time_index] - means["r_eye"][time_index])
    ranked = torch.argsort(diff, descending=True)

    sorted_diff = diff[ranked]

    k = len(ranked)
    ranked_k = ranked[:k].cpu().numpy()
    diff_k = sorted_diff[:k].cpu().numpy()

    plt.figure()
    plt.bar(range(k), diff_k)

    plt.xticks(range(k), ranked_k.tolist())
    plt.xlabel("Channel ID (ranked)")
    plt.ylabel("|mean_alive - mean_r_eye|")
    plt.title("Top channel differences: alive vs r_eye")

    plt.tight_layout()
    plt.show()

    return ranked, sorted_diff

def plot_heatmap(data, title):
    plt.figure(figsize=(10,6))
    plt.imshow(data.T, aspect='auto')
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Channel")
    plt.title(title)
    plt.show()

def plot_means_heat(stats):
    means = stats["means"]

    plot_heatmap(means["alive"], "Alive Means")
    plot_heatmap(means["body"], "Body Means")
    plot_heatmap(means["r_eye"], "Right Eye Means")
