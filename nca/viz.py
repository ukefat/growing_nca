import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import numpy as np

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
    # Ensure frame is on CPU and detached
    frame = frame.detach().cpu()

    rgb = frame[0:3]  # (3, H, W) torch.Tensor
    alpha = frame[3:4]  # (1, H, W) torch.Tensor

    img = rgb * alpha  # (3, H, W) torch.Tensor
    img = torch.clamp(img, 0.0, 1.0)  # Clamp values using PyTorch

    # Permute to (H, W, 3) using PyTorch, then convert to numpy
    img_np = img.permute(1, 2, 0).numpy()  # (H, W, 3) numpy.ndarray
    return (img_np * 255).astype(np.uint8)

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

    return ranked, diff[ranked]

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
