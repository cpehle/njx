import arbor
import jax.numpy as np
import matplotlib.pyplot as plt
from arbor import mpoint
from njx.base.tree_solver import TreeMatrix
from matplotlib.lines import Line2D


def branched_geometry():
    tree = arbor.segment_tree()
    mnpos = arbor.mnpos

    # Start with a cylinder segment for the soma (with tag 1)
    tree.append(mnpos, mpoint(0, 0.0, 0, 2.0), mpoint(4, 0.0, 0, 2.0), tag=1)
    # Construct the first section of the dendritic tree,
    # comprised of segments 1 and 2, attached to soma segment 0.
    tree.append(0, mpoint(4, 0.0, 0, 0.8), mpoint(8, 0.0, 0, 0.8), tag=3)
    tree.append(1, mpoint(8, 0.0, 0, 0.8), mpoint(12, -0.5, 0, 0.8), tag=3)
    # Construct the rest of the dendritic tree.
    tree.append(2, mpoint(12, -0.5, 0, 0.8), mpoint(20, 4.0, 0, 0.4), tag=3)
    tree.append(3, mpoint(20, 4.0, 0, 0.4), mpoint(26, 6.0, 0, 0.2), tag=3)
    tree.append(2, mpoint(12, -0.5, 0, 0.5), mpoint(19, -3.0, 0, 0.5), tag=3)
    tree.append(5, mpoint(19, -3.0, 0, 0.5), mpoint(24, -7.0, 0, 0.2), tag=3)
    tree.append(5, mpoint(19, -3.0, 0, 0.5), mpoint(23, -1.0, 0, 0.2), tag=3)
    tree.append(7, mpoint(23, -1.0, 0, 0.2), mpoint(26, -2.0, 0, 0.2), tag=3)
    # Two segments that define the axon, with the first at the root, where its proximal
    # end will be connected with the proximal end of the soma segment.
    tree.append(mnpos, mpoint(0, 0.0, 0, 2.0), mpoint(-7, 0.0, 0, 0.4), tag=2)
    tree.append(9, mpoint(-7, 0.0, 0, 0.4), mpoint(-10, 0.0, 0, 0.4), tag=2)

    morph = arbor.morphology(tree)
    return morph


def y_geometry():
    mnpos = arbor.mnpos
    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint(0.0, 0.0, 0.0, 1.0), mpoint(10.0, 0.0, 0.0, 0.5), tag=1)
    tree.append(0, mpoint(15.0, 3.0, 0.0, 0.2), tag=3)
    tree.append(0, mpoint(15.0, -3.0, 0.0, 0.2), tag=3)
    morph = arbor.morphology(tree)
    return morph


def linear_geometry(n=5):
    mnpos = arbor.mnpos
    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint(0.0, 0.0, 0.0, 1.0), mpoint(1.0, 0.0, 0.0, 1.0), tag=1)

    for i in range(n):
        tree.append(
            i, mpoint(i + 1, 0.0, 0.0, 1.0), mpoint(i + 2, 0.0, 0.0, 1.0), tag=3
        )
    morph = arbor.morphology(tree)
    return morph


def swc_geometry(filename):
    # Construct the morphology from an SWC file.
    return arbor.load_swc_arbor(filename)


def swc_neuron_geometry(filename):
    # Construct the morphology from an SWC file according to Neuron implementation.
    return arbor.load_swc_neuron(filename)


def compute_tree_matrix(morph, policy=arbor.cv_policy_every_segment()):
    # generate control volumes
    decor = arbor.decor()
    decor = decor.discretization(policy)
    # define regions using standard SWC tags
    labels = arbor.label_dict(
        {"soma": "(tag 1)", "axon": "(tag 2)", "dend": "(join (tag 3) (tag 4))"}
    )
    cell = arbor.cable_cell(morphology=morph, decor=decor, labels=labels)
    cv_data = arbor.cv_data(cell)
    ncv = cv_data.num_cv

    # compute parents in the discretisation
    p = np.array([cv_data.parent(i) for i in range(ncv)])

    # use heaviside to determine whether a parent or not
    sign = np.heaviside(p + 1, 0.0) + np.heaviside(-p, 0.0)

    # this matrix is the "graph laplacian"
    d = -1 * (
        np.array(
            [
                len(cv_data.children(i)) if len(cv_data.children(i)) > 0 else 1
                for i in range(ncv)
            ]
        )
        + sign
    )
    u = np.ones(ncv - 1)
    tree_matrix = TreeMatrix(d=d, p=p, u=u)
    return tree_matrix


def plot_morphology(ax, morph, colors=None):
    nb = morph.num_branches

    min_x = 0
    max_x = 1
    min_y = 0
    max_y = 1

    lines = []

    for i in range(nb):
        segs = morph.branch_segments(i)
        xs = []
        ys = []

        for seg in segs:
            xs.append(seg.prox.x)
            xs.append(seg.dist.x)
            ys.append(seg.prox.y)
            ys.append(seg.dist.y)

        min_x = min(min_x, min(xs))
        max_x = max(max_x, max(xs))
        min_y = min(min_y, min(ys))
        max_y = max(max_y, max(ys))

        lines.append(Line2D(xs, ys, color="grey"))

    for line in lines:
        ax.add_line(line)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)


if __name__ == "__main__":
    morph = linear_geometry(n=20)
    morph = swc_geometry(
        "data/morphologies/allen/Cux2-CreERT2_Ai14-211772.05.02.01_674408996_m.swc"
    )
    fig, ax = plt.subplots()
    plot_morphology(ax, morph)
    # plt.show()

    morph = y_geometry()
    tree_matrix = compute_tree_matrix(morph)
    print(tree_matrix)
    tree_matrix = compute_tree_matrix(morph, policy=arbor.cv_policy_fixed_per_branch(1))
