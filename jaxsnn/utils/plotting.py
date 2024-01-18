from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

import numpy as onp
import jax.numpy as np


def plot_morphology(ax, morph, colors=None):
    """Plot the morphology of a neuron.

    Args:
        ax: matplotlib axis
        morph: Morphology object
        colors: array of colors for each branch
    """
    nb = morph.num_branches

    min_x = 0
    max_x = 1
    min_y = 0
    max_y = 1

    lines = []
    idx = 0

    for i in range(nb):
        segments = morph.branch_segments(i)
        xs = []
        ys = []

        for seg in segments:
            xs.append(seg.prox.x)
            xs.append(seg.dist.x)
            ys.append(seg.prox.y)
            ys.append(seg.dist.y)
            lines.append(Line2D(xs, ys, c=cm.viridis(colors[idx])))

            idx = idx + 1

        min_x = min(min_x, min(xs))
        max_x = max(max_x, max(xs))
        min_y = min(min_y, min(ys))
        max_y = max(max_y, max(ys))

    for line in lines:
        ax.add_line(line)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)


def animate_dynamics(recording, morph, filename, n_frames=300):
    """Animate the dynamics of a neuron.

    Args:
        recording: Recording object
        morph: Morphology object
        filename: name of the output file
        n_frames: number of frames in the animation
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    step = 0
    ax.set_axis_off()
    plot_morphology(ax, morph, colors=onp.array(recording.v[step]))
    fig.tight_layout()

    def update(step):
        ax.clear()
        ax.set_axis_off()
        plot_morphology(ax, morph, colors=onp.array(recording.v[step]))
        fig.tight_layout()
        return ax

    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(0, n_frames, 1), interval=100
    )
    ani.save(filename, writer="ffmpeg", fps=30)
