"""Plot local charge, spin magnetization and pairing from a Square or Triangle BdG result."""

import argparse
import json
from pathlib import Path

import numpy as np


# Also support running CPU/Plot.py directly on a saved result.
if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CPU import square_pairing, triangle_pairing
from CPU.result_io import plotting_orders
from CPU.lattice_structure import SQUARE_BONDS, TRIANGLE_BASIS, TRIANGLE_BONDS


def pairing_channels(pair, param):
    """Return complex (Ny, Nx) gap maps; triplets have a final (uu, t0, dd) axis.

    Each directed link contributes its own U or V times the saved pair_order.
    The singlet uses the even part of opposite bonds at a site, the triplet
    the odd part. For a nonuniform state these are local projections.
    This also counts links correctly on small periodic cells where +r and -r
    can reach the same site. Missing open-boundary links contribute zero.
    """
    nx, ny = param["Nx"], param["Ny"]
    boundary = param.get("Boundary_Condition", "PBC")
    if param["lattice"] not in ("Square", "Triangle") or boundary not in ("PBC", "OBC"):
        raise ValueError("Use Square or Triangle with PBC or OBC.")
    n = nx * ny
    if pair.shape == (n, n):  # Older singlet-only files store P_up,down.
        pair = np.kron(pair, [[0, 1], [-1, 0]])
    spin = pair.reshape(n, 2, n, 2)
    y, x = np.indices((ny, nx))
    site = y * nx + x

    def bond(dx, dy):
        j = ((y + dy) % ny) * nx + (x + dx) % nx
        block = spin[site, :, j, :]
        if boundary == "OBC":
            valid = (0 <= x+dx) & (x+dx < nx) & (0 <= y+dy) & (y+dy < ny)
            block = block * valid[..., None, None]
        return block

    def direction(dx, dy):
        plus, minus = bond(dx, dy), bond(-dx, -dy)
        even = (plus + minus) / 2
        odd = (plus - minus) / 2
        singlet = (even[..., 0, 1] - even[..., 1, 0]) / 2
        t0 = (odd[..., 0, 1] + odd[..., 1, 0]) / 2
        triplet = np.stack((odd[..., 0, 0], t0, odd[..., 1, 1]), axis=-1)
        return singlet, triplet

    channels = {}
    if param.get("U", 0) or not any(param.get(k, 0) for k in ("V1", "V2", "V3")):
        channels["onsite s"] = param.get("U", 0) * spin[site, 0, site, 1]
    lattice_pairing = triangle_pairing if param["lattice"] == "Triangle" else square_pairing
    for shell in (1, 2, 3):
        coupling = param.get(f"V{shell}", 0)
        if coupling:
            channels.update(lattice_pairing.pairing_channels(shell, direction, coupling))
    return channels


def momentum_plotting_data(state, param):
    """Project directed cell bonds without folding distinct translations together."""
    if 'bond_a' not in state:
        from CPU.cell_bdg import model_from_input
        model = model_from_input(param)
        a, b, offset, onsite = model.a, model.b, model.offset, model.onsite
    else:
        a, b, offset, onsite = (state[k] for k in ('bond_a','bond_b','bond_offset','bond_onsite'))
    n = param['Nx']*param['Ny']
    shape = (param['Ny'], param['Nx'])
    links = {}
    local_pair = np.zeros(n,complex)
    for i,j,d,on,value in zip(a,b,offset,onsite,state['pair_order']):
        if on:
            local_pair[i//2] = value
            continue
        dx,dy = map(int,d)
        forward = links.setdefault((dx,dy),np.zeros((n,2,2),complex))
        reverse = links.setdefault((-dx,-dy),np.zeros((n,2,2),complex))
        forward[i//2,i%2,j%2] = value
        reverse[j//2,j%2,i%2] = -value
    if 'pair_local_order' in state:
        local_pair = state['pair_local_order'][:,0,1]
    def direction(dx,dy):
        plus = links.get((dx,dy),np.zeros((n,2,2),complex)).reshape(*shape,2,2)
        minus = links.get((-dx,-dy),np.zeros((n,2,2),complex)).reshape(*shape,2,2)
        even, odd = (plus+minus)/2, (plus-minus)/2
        return ((even[...,0,1]-even[...,1,0])/2,
                np.stack((odd[...,0,0],(odd[...,0,1]+odd[...,1,0])/2,odd[...,1,1]),axis=-1))
    channels = {}
    if param.get('U',0) or not any(param.get(f'V{i}',0) for i in (1,2,3)):
        channels['onsite s'] = param.get('U',0)*local_pair.reshape(shape)
    pairing = triangle_pairing if param['lattice']=='Triangle' else square_pairing
    for shell in (1,2,3):
        if param.get(f'V{shell}',0):
            channels.update(pairing.pairing_channels(shell,direction,param[f'V{shell}']))
    rho = np.zeros((2*n,2*n),complex)
    up = 2*np.arange(n)
    for i in (0,1):
        for j in (0,1):
            rho[up+i,up+j] = state['rho_local'][:,i,j]
    return state['density'], rho, channels


def plotting_data(state, param):
    if 'k_mesh' in state or 'bond_shift' in state:
        return momentum_plotting_data(state,param)
    density, rho, pair = plotting_orders(state)
    return density, rho, pairing_channels(pair,param)


def gap_magnitude(gap):
    if gap.ndim == 2:
        return abs(gap)
    # Half the squared Frobenius norm of the triplet spin block.
    uu, t0, dd = np.moveaxis(gap, -1, 0)
    return np.sqrt((abs(uu)**2 + abs(dd)**2) / 2 + abs(t0)**2)


def print_channels(channels):
    print(f"{'channel':24} {'mean real':>12} {'mean imag':>12} {'RMS':>12}")
    for name, gap in channels.items():
        if gap.ndim == 3:
            components = zip(("uu", "t0", "dd"), np.moveaxis(gap, -1, 0))
        else:
            components = [("", gap)]
        for spin, component in components:
            mean = component.mean()
            rms = np.sqrt(np.mean(abs(component)**2))
            print(f"{(name+' '+spin).strip():24} {mean.real:12.5g} {mean.imag:12.5g} {rms:12.5g}")


def site_magnetization(rho, shape):
    """m = <c† sigma c>, using rho[a,b] = <c†_a c_b> and alternating up/down."""
    up = 2 * np.arange(np.prod(shape))
    coherence = rho[up, up + 1]
    mz = rho[up, up].real - rho[up + 1, up + 1].real
    return np.stack((2 * coherence.real, 2 * coherence.imag, mz)).reshape(3, *shape)


def channel_phases(channels, triplet_spin, cutoff):
    """Reference every complex coefficient to the same amplitude in the saved spin basis."""
    spin = ("uu", "t0", "dd").index(triplet_spin)
    fields = {name: gap if gap.ndim == 2 else gap[..., spin] for name, gap in channels.items()}
    singlets = [name for name, gap in channels.items() if gap.ndim == 2 and abs(gap).max() > cutoff]
    name = max(singlets or fields, key=lambda key: abs(fields[key]).max())
    y, x = np.unravel_index(abs(fields[name]).argmax(), fields[name].shape)
    ref = fields[name][y, x]
    label = f"{name} at ({x}, {y})" if abs(ref) > cutoff else "no resolved reference"
    phases = {key: np.ma.masked_where(abs(field) <= cutoff, np.angle(field * ref.conjugate()))
              for key, field in fields.items()}
    return phases, label


def draw_bond_channel(ax, width, phase, color_map, shell, channel, scale, boundary):
    """Each endpoint supplies the width and phase on its half of a full lattice bond."""
    from matplotlib.collections import LineCollection

    ny, nx = width.shape
    # One layer of periodic image sites completes the visible halves of wrapped bonds.
    pad = int(boundary == "PBC")
    y, x = np.mgrid[-pad:ny+pad, -pad:nx+pad]
    sites = np.column_stack((x.ravel(), y.ravel()))
    t = np.linspace(0, 0.5, 13)[:, None]
    # Multiplying a complex coefficient by -1 shifts its phase by pi.
    positive = color_map.to_rgba(phase)
    negative = color_map.to_rgba(np.ma.where(phase >= 0, phase - np.pi, phase + np.pi))
    for dx, dy in SQUARE_BONDS[shell-1]:
        for sense in (1, -1):
            dx1, dy1 = sense * dx, sense * dy
            factor = {"extended s": 1, "d_x2-y2": dx1**2 - dy1**2, "d_xy": dx1*dy1,
                      "p_x": dx1, "p_y": dy1}[channel]
            if factor == 0:
                continue
            colors = positive if factor > 0 else negative
            offset = np.array([dx1, dy1])
            neighbor = sites + offset
            valid = np.ones(len(sites), dtype=bool)
            if boundary == "OBC":
                valid = ((neighbor >= 0) & (neighbor < (nx, ny))).all(axis=1)
            starts = sites[valid]
            path = t * offset
            # The same bend on both halves joins them smoothly at the midpoint.
            # Split the diagonal p components and bow 3NN links past the intervening site.
            if shell == 2 and channel in ("p_x", "p_y"):
                bend = 0.13 if channel == "p_x" else -0.13
                path = path + bend * np.sin(np.pi*t) * np.array([-dy, dx]) / np.sqrt(2)
            elif shell == 3:
                path = path + 0.24 * np.sin(np.pi*t) * np.array([-dy, dx]) / 2
            segments = starts[:, None, :] + path
            index = starts[:, 1] % ny, starts[:, 0] % nx
            ax.add_collection(LineCollection(segments, linewidths=scale * width[index],
                                             colors=colors[index]))


def draw_triangle_bonds(ax, shell, group, widths, phases, color_map, scale, shape, boundary):
    """Draw half-bonds in Cartesian coordinates and clip periodic images to the cell."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon

    ny, nx = shape
    cell = (np.array([[0, 0], [nx, 0], [nx, ny], [0, ny]]) - 0.5) @ TRIANGLE_BASIS
    clip = Polygon(cell, transform=ax.transData)
    pad = 2 if boundary == "PBC" else 0
    y, x = np.mgrid[-pad:ny+pad, -pad:nx+pad]
    sites = np.column_stack((x.ravel(), y.ravel()))
    t = np.linspace(0, 0.5, 9)[:, None]
    for j, offset in enumerate(TRIANGLE_BONDS[shell-1]):
        key = f"{shell}NN {group} {j}"
        vector = offset @ TRIANGLE_BASIS
        for sign in (1, -1):
            neighbor = sites + sign*np.array(offset)
            starts = sites
            if boundary == "OBC":
                starts = sites[((neighbor >= 0) & (neighbor < (nx, ny))).all(axis=1)]
            index = starts[:, 1] % ny, starts[:, 0] % nx
            phase = phases[key][index]
            if sign < 0 and group in ("p", "f"):
                phase = np.ma.where(phase >= 0, phase - np.pi, phase + np.pi)
            path = sign*t*vector
            if shell == 3:
                path += 0.12*np.sin(np.pi*t)*np.array([-vector[1], vector[0]])
            lines = LineCollection((starts @ TRIANGLE_BASIS)[:, None, :] + path,
                                   linewidths=scale*widths[key][index], colors=color_map.to_rgba(phase))
            ax.add_collection(lines)
            lines.set_clip_path(clip)
    ax.set(xlim=(cell[:, 0].min(), cell[:, 0].max()), ylim=(cell[:, 1].min(), cell[:, 1].max()),
           aspect="equal", xlabel="x")


def plot_orders(channels, param, density, rho, output=None, nn_bonds=False, triplet_spin="t0",
                iteration=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, hsv_to_rgb

    nx, ny = param["Nx"], param["Ny"]
    triangle = param["lattice"] == "Triangle"
    shape = (ny, nx)
    density = density.reshape(shape)
    charge = density - density.mean()
    magnetic = site_magnetization(rho, shape) if rho is not None else None
    active = [s for s in (1, 2, 3) if f"{s}NN extended s" in channels]
    shell_rows = [group for group in ([s for s in active if s == 1],
                                      [s for s in active if s > 1]) if group]
    rows = 1 + len(shell_rows)
    has_bonds = bool(active)
    fig = plt.figure(figsize=(12, 3.1 * rows + 0.3), layout="constrained")
    if iteration is not None:
        fig.suptitle(f"Iteration {iteration}")
    grid = fig.add_gridspec(rows, 1)
    top_widths = [1, 1, 1] + ([0.82] if has_bonds else [])
    top = grid[0].subgridspec(1, len(top_widths), width_ratios=top_widths)
    y, x = np.indices(shape)

    if triangle:
        x, y = np.moveaxis(np.stack((x, y), axis=-1) @ TRIANGLE_BASIS, -1, 0)

    def site_map(ax, values, title, cmap, limits):
        if triangle:
            cy, cx = np.indices((ny+1, nx+1)) - 0.5
            corners = np.stack((cx, cy), axis=-1) @ TRIANGLE_BASIS
            im = ax.pcolormesh(corners[..., 0], corners[..., 1], values, cmap=cmap,
                               vmin=limits[0], vmax=limits[1], shading="flat")
            ax.set_aspect("equal")
        else:
            im = ax.imshow(values, origin="lower", interpolation="nearest", cmap=cmap,
                           vmin=limits[0], vmax=limits[1])
        ax.set_title(title)
        ax.set(xlabel="x", ylabel="y")
        return im

    def panel_colorbar(im, label=""):
        # The inset follows the displayed panel after its equal aspect ratio is applied.
        bar_ax = im.axes.inset_axes([1.04, 0, 0.035, 1])
        fig.colorbar(im, cax=bar_ax, label=label, ticks=np.linspace(*im.get_clim(), 3),
                     format="{x:.3g}")

    # Keep tiny numerical fluctuations visually neutral on the density/spin scale.
    charge_limit = max(abs(charge).max(), 0.01)
    ax = fig.add_subplot(top[0, 0])
    im = site_map(ax, charge, r"Charge  $n_i-\bar n$", "RdBu_r", (-charge_limit, charge_limit))
    panel_colorbar(im)
    ax = fig.add_subplot(top[0, 1])
    if magnetic is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Magnetization unavailable\nNo spin density matrix in file",
                ha="center", va="center")
    else:
        mx, my, mz = magnetic
        z_limit = max(abs(mz).max(), 0.01)
        m_peak = np.linalg.norm(magnetic, axis=0).max()
        im = site_map(ax, mz, f"Magnetization\npeak |m| = {m_peak:.2g}",
                      "RdBu_r", (-z_limit, z_limit))
        panel_colorbar(im, "$m_z$")
        # The reference arrow spans 0.8 lattice spacings. Keep roundoff-sized spins invisible.
        reference = max(np.hypot(mx, my).max(), 0.01)
        ax.quiver(x, y, mx, my, angles="xy", scale_units="xy", scale=reference / 0.8,
                  pivot="middle", units="xy", width=0.035, minlength=0, color="#202020")

    if triangle:
        channels = triangle_pairing.bond_fields(channels, active)
    magnitudes = {name: gap_magnitude(gap) for name, gap in channels.items()}
    peak = max(v.max() for v in magnitudes.values())
    max_width = 8  # points
    if triangle:
        # Four panels share the figure width. Keep adjacent NN directions visible.
        panel_width = fig.get_figwidth() * 72 / 4
        site_spacing = panel_width / (nx + ny / 2)
        max_width = min(4, 0.14 * site_spacing)
    else:
        # Keep crossing bond phases visible as more sites share each panel.
        # A fixed eight-point width nearly fills a 20x20 panel with the bonds
        # drawn last, hiding the opposite sign on perpendicular d-wave bonds.
        panel_width = fig.get_figwidth() * 72 / 3
        panel_height = fig.get_figheight() * 72 / rows
        site_spacing = min(panel_width / nx, panel_height / ny)
        max_width = min(max_width, 0.22 * site_spacing)
    scale = max_width / peak if peak else 0
    if has_bonds:
        # Convert the input correlation tolerance to a display floor in gap units.
        coupling = max(abs(param.get(k, 0)) for k in ("U", "V1", "V2", "V3"))
        cutoff = max(1e-12 * peak, param.get("tolerate", 0) * coupling)
        phases, _ = channel_phases(channels, triplet_spin, cutoff)
        # Include both ends of the hue cycle so -pi and +pi give identical red.
        hue = np.linspace(0, 1, 257)
        palette = ListedColormap(hsv_to_rgb(np.column_stack((hue, np.ones((len(hue), 2))))))
        color_map = plt.cm.ScalarMappable(plt.Normalize(-np.pi, np.pi),
                                          palette.with_extremes(bad="#888888"))
    for row, shells in enumerate(shell_rows, 1):
        shell_label = " + ".join(f"{s}NN" for s in shells)
        if triangle:
            panels = grid[row].subgridspec(1, 4)
            for col, (group, title) in enumerate(zip(("s", "d", "p", "f"),
                                                     ("Extended $s$", "$d$", "$p$", "$f$"))):
                ax = fig.add_subplot(panels[0, col])
                for shell in shells:
                    draw_triangle_bonds(ax, shell, group, magnitudes, phases, color_map,
                                        scale, shape, param.get("Boundary_Condition", "PBC"))
                ax.scatter(x, y, s=5, color="#293338", zorder=3)
                ax.set_title(f"{shell_label}  {title}")
                if col == 0:
                    ax.set_ylabel("y")
            continue
        heatmap = shells == [1] and not nn_bonds
        panels = grid[row].subgridspec(1, 3)
        axes = [fig.add_subplot(panels[0, col]) for col in range(3)]
        for shell in shells:
            d_name = "d_xy" if shell == 2 else "d_x2-y2"
            for col, name in ((0, "extended s"), (1, d_name), (2, "p_x"), (2, "p_y")):
                ax = axes[col]
                key = f"{shell}NN {name}"
                values = magnitudes[key]
                if heatmap and col < 2:
                    im = site_map(ax, values, name, "viridis", (0, peak or 1))
                else:
                    draw_bond_channel(ax, values, phases[key], color_map, shell, name, scale,
                                      param.get("Boundary_Condition", "PBC"))
        d_title = " / ".join(dict.fromkeys("$d_{xy}$" if s == 2 else "$d_{x^2-y^2}$" for s in shells))
        for col, (ax, title) in enumerate(zip(axes, ("Extended $s$", d_title, "$p_x$, $p_y$"))):
            if not heatmap or col == 2:
                ax.scatter(x, y, s=5, color="#293338", zorder=3)
                ax.set(aspect="equal", xlim=(-0.5, nx-0.5), ylim=(-0.5, ny-0.5), xlabel="x")
            ax.set_title(f"{shell_label}  {title}")
        if heatmap:
            panel_colorbar(im, "Gap magnitude")
        axes[0].set_ylabel("y")
    ax = fig.add_subplot(top[0, 2])
    u = param.get("U", 0)
    onsite_gap = magnitudes["onsite s"] if u else np.zeros(shape)
    im = site_map(ax, onsite_gap, f"Onsite $s$  ($U={u:g}$)", "Blues", (0, peak or 1))
    panel_colorbar(im, r"$|\Delta_{ii}|$")
    if has_bonds:
        key_ax = fig.add_subplot(top[0, -1])
        key_ax.set(xlim=(0, 1), ylim=(0, 1))
        key_ax.set_axis_off()
        key_ax.text(0.05, 0.84, r"Peak gap $|\Delta|$", weight="bold")
        for row, shell in enumerate(active):
            value = max(v.max() for name, v in magnitudes.items() if name.startswith(f"{shell}NN "))
            y_key = 0.72 - 0.12 * row
            key_ax.text(0.05, y_key, f"{shell}NN", va="center")
            key_ax.plot([0.31, 0.58], [y_key, y_key], lw=scale * value,
                        color="#48545a", solid_capstyle="butt")
            key_ax.text(0.65, y_key, f"{value:.3g}", va="center")
        key_ax.text(0.05, 0.32, f"Bond phase  ·  {triplet_spin}", weight="bold")
        bar_ax = key_ax.inset_axes([0.05, 0.22, 0.90, 0.05])
        bar = fig.colorbar(color_map, cax=bar_ax, orientation="horizontal",
                           ticks=np.linspace(-np.pi, np.pi, 5))
        bar.ax.set_xticklabels(["−π", "−π/2", "0", "π/2", "π"])
    if output is None:
        plt.show()
    else:
        fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_iteration_state(param, step, state):
    density, rho, channels = plotting_data(state, param)
    plot_orders(channels, param, density, rho, nn_bonds=True, iteration=step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="path to a saved BdG .npz file")
    parser.add_argument("--nn-bonds", action="store_true", help="use bond widths for the 1NN row too")
    parser.add_argument("--phase-spin", choices=("uu", "t0", "dd"), default="t0",
                        help="triplet spin component used for bond color (default: t0)")
    args = parser.parse_args()
    with np.load(args.data) as data:
        param = json.loads(data["parameters"].item())
        density, rho, channels = plotting_data(data, param)
    print_channels(channels)
    suffix = "_nn_bonds.png" if args.nn_bonds else "_channels.png"
    output = args.data.with_name(args.data.stem + suffix)
    plot_orders(channels, param, density, rho, output, args.nn_bonds, args.phase_spin)
    print(output)
