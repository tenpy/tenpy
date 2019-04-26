#!/usr/bin/env python3

import matplotlib.pyplot as plt
from tenpy.models import lattice, toric_code


def plot_lattice(lat, ax, name):
    lat.plot_coupling(ax, lat.nearest_neighbors, linewidth=3.)
    lat.plot_order(ax=ax, linestyle=':')
    #  lat.plot_coupling(ax, lat.next_nearest_neighbors, color='g')
    #  lat.plot_coupling(ax, lat.next_next_nearest_neighbors, color='orange')
    lat.plot_sites(ax)
    lat.plot_basis(ax, color='g', linewidth=2.)
    if name == "Chain":
        ax.set_ylim([-0.5, 0.5])
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(name + ".png")
    plt.savefig(name + ".pdf")


def plot_bc_shift_comparison(name="square_bc_shift"):
    print(name)
    Lx, Ly = 4, 3
    fig, axes = plt.subplots(1, 3, True, True, figsize=(5, 2))

    for shift, ax in zip([1, 0, -1], axes):
        lat = lattice.Square(Lx, Ly, None, bc=['periodic', shift])
        lat.plot_sites(ax)
        lat.plot_coupling(ax)
        lat.plot_basis(ax, color='g', linewidth=2.)
        lat.plot_bc_identified(ax)
        ax.set_title("shift = " + str(shift))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    axes[0].set_aspect('equal')
    plt.savefig(name + ".png")
    plt.savefig(name + ".pdf")


if __name__ == "__main__":
    # 1D lattices
    for name, figsize in zip(["Chain", "Ladder"], [(5, 1.5), (5, 1.5)]):
        print(name)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        lat = lattice.__dict__.get(name)(4, None, bc='periodic')
        plot_lattice(lat, ax, name)
    # 2D lattices
    for name, figsize in zip(["Square", "Triangular", "Honeycomb", "Kagome"], [(5, 5), (4, 5),
                                                                               (5, 6), (5, 4)]):
        print(name)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        lat = lattice.__dict__.get(name)(4, 4, None, bc='periodic')
        plot_lattice(lat, ax, name)
    print("DualSquare")
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    lat = toric_code.DualSquare(4, 4, None, bc='periodic')
    plot_lattice(lat, ax, "DualSquare")

    # comparison of bc shift for :doc:`intro_model`
    plot_bc_shift_comparison()
