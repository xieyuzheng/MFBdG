"""Plot the bulk free-fermion band, Fermi contour and density of states."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch
from scipy.spatial import Voronoi


def reciprocal_vectors(lattice):
    """Lattice spacing a = 1, with the first primitive vector along x."""
    if lattice == "Square":
        return 2*np.pi*np.eye(2)
    if lattice == "Triangle":
        return np.array([[2*np.pi, -2*np.pi/np.sqrt(3)], [0, 4*np.pi/np.sqrt(3)]])
    raise ValueError("Use lattice = Square or Triangle.")


def dispersion(k, lattice, hopping):
    """Bare band with the same -t hopping and three-shell convention as MFBdG."""
    x, y = k[..., 0], k[..., 1]
    t1, t2, t3 = hopping
    if lattice == "Square":
        return (-2*t1*(np.cos(x) + np.cos(y))
                -4*t2*np.cos(x)*np.cos(y)
                -2*t3*(np.cos(2*x) + np.cos(2*y)))
    if lattice == "Triangle":
        y = np.sqrt(3)*y
        return (-2*t1*(np.cos(x) + 2*np.cos(x/2)*np.cos(y/2))
                -2*t2*(np.cos(y) + 2*np.cos(3*x/2)*np.cos(y/2))
                -2*t3*(np.cos(2*x) + 2*np.cos(x)*np.cos(y)))
    raise ValueError("Use lattice = Square or Triangle.")


def number_solution(lattice, hopping, filling, nk):
    """Zero-T Fermi energy on an nk by nk primitive reciprocal-cell grid."""
    b1, b2 = reciprocal_vectors(lattice)
    qx, qy = np.meshgrid(np.arange(nk) / nk, np.arange(nk) / nk)
    k = qx[..., None] * b1 + qy[..., None] * b2
    band = dispersion(k, lattice, hopping)

    # Each k state holds two electrons. Fill the lowest energies first.
    # Between filled and empty levels use their midpoint; a partially filled
    # (possibly degenerate) level has mu equal to that level's energy.
    mu = float(np.quantile(band, filling / 2, method="averaged_inverted_cdf"))
    return mu, band


def density_of_states(band):
    """Histogram DOS per site including both spins; integral over energy equals two."""
    counts, edges = np.histogram(band.ravel(), bins=400)
    dos = 2 * counts / (band.size * np.diff(edges))
    return dos, edges


def brillouin_zone(lattice):
    """Wigner-Seitz cell about the origin in the reciprocal lattice."""
    reciprocal = reciprocal_vectors(lattice)
    grid = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    voronoi = Voronoi(grid @ reciprocal)
    region = voronoi.regions[voronoi.point_region[4]]
    vertices = voronoi.vertices[region]
    angle = np.arctan2(vertices[:, 1], vertices[:, 0])
    vertices = vertices[np.argsort(angle)]
    return np.vstack([vertices, vertices[0]])


def plot_bands(lattice, hopping, filling, mu, nk, band, output=None, show=True):
    b1, b2 = reciprocal_vectors(lattice)
    if lattice == "Square":
        points = np.array([[0, 0], b1 / 2, (b1 + b2) / 2, [0, 0]])
        labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    else:
        points = np.array([[0, 0], b1 / 2, (2*b1 + b2) / 3, [0, 0]])
        labels = [r"$\Gamma$", "M", "K", r"$\Gamma$"]
    segments = [np.linspace(a, b, nk, endpoint=False) for a, b in zip(points[:-1], points[1:])]
    path = np.vstack([*segments, points[-1:]])
    distance = np.r_[0, np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    ticks = np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]

    fig, (ax_band, ax_fs, ax_dos) = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    ax_band.plot(distance, dispersion(path, lattice, hopping), color="#215c91")
    ax_band.axhline(mu, color="0.4", linestyle="--", linewidth=0.8, label=rf"$\mu={mu:.4g}$")
    ax_band.legend(frameon=False)
    for position in ticks[1:-1]:
        ax_band.axvline(position, color="0.8", linewidth=0.7)
    ax_band.set(xticks=ticks, xticklabels=labels, xlim=(0, distance[-1]),
                ylabel=r"$\varepsilon(\mathbf{k})$", title="Band structure")

    boundary = brillouin_zone(lattice)
    limits = np.array([boundary.min(axis=0), boundary.max(axis=0)])
    kx, ky = np.meshgrid(np.linspace(*limits[:, 0], nk+1),
                         np.linspace(*limits[:, 1], nk+1))
    grid = np.stack([kx, ky], axis=-1)
    energy = dispersion(grid, lattice, hopping)
    polygon = PlotPath(boundary)
    mask = polygon.contains_points(grid.reshape(-1, 2), radius=1e-9).reshape(kx.shape)
    energy = np.ma.array(energy, mask=~mask)
    scale = max(float(np.max(np.abs(energy))), 1e-12)
    image = ax_fs.pcolormesh(kx, ky, energy, cmap="RdBu_r", vmin=-scale, vmax=scale,
                             shading="auto", rasterized=True)
    clip = PathPatch(polygon, transform=ax_fs.transData)
    image.set_clip_path(clip)
    if energy.min() < mu < energy.max():
        contour = ax_fs.contour(kx, ky, energy, levels=[mu], colors="black", linewidths=1.5)
        contour.set_clip_path(clip)
    else:
        message = (r"Flat band at $\mu$" if np.all(energy.compressed() == mu)
                   else r"No $\varepsilon(\mathbf{k})=\mu$ contour")
        ax_fs.text(0.5, 0.5, message, transform=ax_fs.transAxes, ha="center", va="center")
    ax_fs.plot(*boundary.T, color="0.25", linewidth=1)
    ax_fs.set(aspect="equal", xlabel=r"$k_x$", ylabel=r"$k_y$",
              title=r"Fermi contour  $\varepsilon(\mathbf{k})=\mu$",
              xlim=limits[:, 0], ylim=limits[:, 1])
    fig.colorbar(image, ax=ax_fs, label=r"$\varepsilon(\mathbf{k})$", shrink=0.75)
    dos, edges = density_of_states(band)
    ax_dos.stairs(dos, edges, color="#215c91", linewidth=1.1)
    ax_dos.axvline(mu, color="0.4", linestyle="--", linewidth=0.8, label=rf"$\mu={mu:.4g}$")
    ax_dos.legend(frameon=False)
    ax_dos.set(xlabel=r"$\varepsilon$", ylabel="DOS / site / energy (both spins)",
               title="Density of states", ylim=(0, None))
    fig.suptitle(f"{lattice}, n = {filling:g}, T = 0, mu = {mu:.6g}\n"
                 f"t = ({hopping[0]:g}, {hopping[1]:g}, {hopping[2]:g}), "
                 f"k-grid = {nk} x {nk}")
    if output is not None:
        fig.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
