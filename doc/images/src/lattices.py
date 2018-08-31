#!/usr/bin/env python3

import matplotlib.pyplot as plt
from tenpy.models import lattice


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
    #  plt.show()
    plt.savefig(name+".png")
    plt.savefig(name+".pdf")

for name, figsize in zip(["Square", "Honeycomb", "Kagome"],
                         [(5, 5),   (5, 6),      (5, 4)]):
    print(name)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    lat = lattice.__dict__.get(name)(4, 4, None, bc='periodic')
    plot_lattice(lat, ax, name)

for name, figsize in zip(["Chain", "Ladder"],
                [(5, 1.5), (5, 1.5)]):
    print(name)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    lat = lattice.__dict__.get(name)(4, None, bc='periodic')
    plot_lattice(lat, ax, name)

