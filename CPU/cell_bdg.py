"""Periodic-cell, full-spin Hartree-Fock-BdG with streamed Bloch diagonalization.

Nx, Ny count primitive sites inside a rectangular cell (also for Triangle).
Nkx, Nky count cell momenta, q_i = 2*pi*j/Nki. Bloch phases use cell
translations only: c_Ra = sum_q exp(i*q.R) c_qa / sqrt(Nk).
rho_ab(d) = <c_0a^dagger c_db>, pair_ab(d) = <c_db c_0a>.
Only independent forward interaction links are packed; reverse links are
recovered by Hermiticity / fermionic antisymmetry, including their translation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.special import entr, expit

from CPU import MFSolver
from CPU.parallel import WorkerRuntime, allocation_from_input, print_allocation, validate_thread_input
from CPU.bond_bdg import print_energy
from CPU import triangle_pairing
from CPU.lattice_structure import SQUARE_BONDS, TRIANGLE_BONDS
from CPU.solver.solver1d import secant_method


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@dataclass
class CellModel:
    nx: int
    ny: int
    nkx: int
    nky: int
    lattice: str
    hop_a: np.ndarray
    hop_b: np.ndarray
    hop_shift: np.ndarray
    hopping: np.ndarray
    a: np.ndarray
    b: np.ndarray
    shift: np.ndarray
    coupling: np.ndarray
    offset: np.ndarray
    onsite: np.ndarray

    @property
    def nsite(self):
        return self.nx * self.ny

    @property
    def nmode(self):
        return 2 * self.nsite

    @property
    def nk(self):
        return self.nkx * self.nky

    def momenta(self, batch_size):
        for start in range(0, self.nk, batch_size):
            indices = np.arange(start, min(start + batch_size, self.nk))
            yield 2*np.pi*np.column_stack((indices % self.nkx / self.nkx,
                                           indices // self.nkx / self.nky))


def cell_model(nx, ny, nkx, nky, *, t=(1, 0, 0), V=(-2, 0, 0), U=0,
               lattice="Square"):
    """Keep distinct geometric bonds even when their basis-site labels coincide."""
    nx, ny, nkx, nky = [_positive_integer(v, name) for v, name in
                       zip((nx, ny, nkx, nky), ("Nx", "Ny", "Nkx", "Nky"))]
    if len(t) != 3 or len(V) != 3 or not np.all(np.isfinite([*t, *V, U])):
        raise ValueError("Use three finite real hopping and interaction strengths and finite U.")
    if lattice not in ("Square", "Triangle"):
        raise ValueError("lattice must be Square or Triangle.")
    bonds = SQUARE_BONDS if lattice == "Square" else TRIANGLE_BONDS
    ha, hb, hs, ht, aa, bb, shifts, strengths, offsets, onsite = ([] for _ in range(10))
    for directions, hopping, strength in zip(bonds, t, V):
        if hopping == 0 and strength == 0:
            continue
        for y in range(ny):
            for x in range(nx):
                i = x + nx*y
                for dx, dy in directions:
                    cy, yy = divmod(y + dy, ny)
                    cx, xx = divmod(x + dx, nx)
                    j = xx + nx*yy
                    if hopping:
                        for spin in (0, 1):
                            ha.append(2*i + spin)
                            hb.append(2*j + spin)
                            hs.append((cx, cy))
                            ht.append(-hopping)
                    if strength:
                        for s in (0, 1):
                            for r in (0, 1):
                                aa.append(2*i + s)
                                bb.append(2*j + r)
                                shifts.append((cx, cy))
                                strengths.append(strength)
                                offsets.append((dx, dy))
                                onsite.append(False)
    if U:
        for i in range(nx*ny):
            aa.append(2*i)
            bb.append(2*i + 1)
            shifts.append((0, 0))
            strengths.append(U)
            offsets.append((0, 0))
            onsite.append(True)
    return CellModel(nx, ny, nkx, nky, lattice,
                     np.asarray(ha, int), np.asarray(hb, int), np.asarray(hs, int).reshape(-1, 2),
                     np.asarray(ht, float), np.asarray(aa, int), np.asarray(bb, int),
                     np.asarray(shifts, int).reshape(-1, 2), np.asarray(strengths, float),
                     np.asarray(offsets, int).reshape(-1, 2), np.asarray(onsite, bool))


def validate_boundary_condition(param):
    """Reject unsupported boundaries before constructing a model or workers."""
    boundary = param.get("Boundary_Condition", "PBC")
    if boundary != "PBC":
        raise ValueError(f"Boundary_Condition={boundary!r} is not supported by the unit-cell solver. Use 'PBC'.")


def validate_constraint_input(param):
    """Validate the unit-cell formulation and onsite constraints."""
    if "backend" in param:
        raise ValueError("Remove backend: every calculation uses the unit-cell solver. Set Nkx=Nky=1 for a finite periodic cell.")
    validate_boundary_condition(param)
    from CPU.bond_bdg import onsite_constraint_method
    method = onsite_constraint_method(param)
    if param.get('order_ansatz','full') == 'full' and param.get('constraint_method') is not None:
        raise ValueError('Use constraint_method=null for full; m_max sets its optional spin barrier.')
    bound = param.get('m_max')
    if method == 'local_barrier':
        bound = param.get('local_max', 1e-6)
    if bound is not None and not (np.isfinite(bound) and bound > 0
            and np.isfinite(param.get('tau', 1e-5)) and param.get('tau', 1e-5) > 0):
        raise ValueError('Constraint bounds and tau must be positive and finite.')
    if method == 'local_barrier' and bound is None:
        raise ValueError('local_max must be positive and finite.')
    if (param.get('order_ansatz') == 'pure_SC' and method == 'local_barrier'
            and param.get('n_filling') is None):
        raise ValueError('The original pure_SC local barrier requires n_filling.')


def model_from_input(param):
    validate_constraint_input(param)
    return cell_model(param["Nx"], param["Ny"], param.get("Nkx", 1), param.get("Nky", 1),
                      t=tuple(param.get(f"t{i}", 1 if i == 1 else 0) for i in (1, 2, 3)),
                      V=tuple(param.get(f"V{i}", 0) for i in (1, 2, 3)), U=param.get("U", 0),
                      lattice=param["lattice"])


def initial_magnetization(param):
    """Return m_i = n_up - n_down for the first iteration."""
    seed = param.get("normal_seed", "uniform").lower()
    if seed == "uniform":
        return 0.0
    n = param["n_filling"] if param["n_filling"] is not None else 1.0
    m = min(0.1, n, 2 - n)
    if seed == "fm":
        return m
    if seed == "afm" and param["lattice"] == "Square":
        y, x = np.indices((param["Ny"], param["Nx"]))
        return (m * (-1)**(x + y)).ravel()
    raise ValueError("Use normal_seed = uniform, fm, or afm (square checkerboard).")

def initial_order(model, param):
    """Return cell occupations and independent normal/pair link contractions."""
    filling = param.get("n_filling")
    filling = 1.0 if filling is None else filling
    magnetization = np.broadcast_to(initial_magnetization(param), (model.nsite,))
    n = np.column_stack(((filling + magnetization)/2, (filling - magnetization)/2)).ravel()
    chi = np.zeros(len(model.a), complex)
    pair = np.zeros_like(chi)
    seed = param.get("pair_seed")
    if seed is None:
        seed = "mixed" if np.any(~model.onsite) else 0.1
    sector = seed if isinstance(seed, str) else None
    presets = triangle_pairing.PAIR_SEEDS if model.lattice == 'Triangle' else ('dwave','pwave','extended_s')
    if sector not in (None, "mixed", "singlet", "triplet", *presets):
        raise ValueError(f"Unknown pair seed {seed!r}.")
    random_seed = sector in ("mixed", "singlet", "triplet")
    pair_noise = param.get("pair_noise")
    if pair_noise is None:
        pair_noise = 0.05 if random_seed else 0.0
    normal_noise = param.get("normal_noise", 0.0)
    if not np.all(np.isfinite([pair_noise, normal_noise])) or min(pair_noise, normal_noise) < 0:
        raise ValueError("Pair and normal noise must be finite and nonnegative.")
    spin_a, spin_b = model.a % 2, model.b % 2
    singlet_sign = np.where(spin_a != spin_b, 1 - 2*spin_a, 0)
    dx, dy = model.offset.T
    if sector is None:
        if np.ndim(seed):
            matrix = np.asarray(seed,dtype=complex)
            if matrix.shape == (model.nsite,model.nsite):
                matrix = np.kron(matrix,[[0,1],[-1,0]])
            if matrix.shape != (model.nmode,model.nmode) or not np.allclose(matrix,-matrix.T,rtol=0,atol=1e-14):
                raise ValueError('Use an N x N symmetric or 2N x 2N antisymmetric pair seed.')
            pair = matrix[model.a,model.b].copy()
        else:
            pair = complex(seed) * singlet_sign
    elif model.lattice == 'Triangle' and not random_seed:
        from CPU.lattice_structure import TRIANGLE_BONDS
        odd = sector in ('pwave','py','p+ip','p-ip','fwave','f_y')
        for shell, offsets in enumerate(TRIANGLE_BONDS,1):
            forms = triangle_pairing.triangle_forms(shell)
            weights = {'extended_s':forms['extended s'],'dwave':forms['d_x2-y2'],'dxy':forms['d_xy'],
                'd+id':forms['d_x2-y2']+1j*forms['d_xy'],'d-id':forms['d_x2-y2']-1j*forms['d_xy'],
                'pwave':forms['p_x'],'py':forms['p_y'],'p+ip':forms['p_x']+1j*forms['p_y'],
                'p-ip':forms['p_x']-1j*forms['p_y'],'fwave':forms.get('f_x',np.zeros(3)),
                'f_y':forms.get('f_y',np.zeros(3))}[sector]
            for offset, weight in zip(offsets,weights):
                selected = np.all(model.offset == offset,axis=1) & ~model.onsite
                pair[selected] = .1*weight*((spin_a!=spin_b) if odd else singlet_sign)[selected]
    elif sector == "dwave":
        pair = 0.1 * ((dy == 0).astype(float) - (dx == 0)) * singlet_sign
    elif sector == "extended_s":
        pair = 0.1 * (np.abs(dx) + np.abs(dy) == 1) * singlet_sign
    elif sector == "pwave":
        pair = 0.1 * (dx == 1) * (dy == 0) * (spin_a != spin_b)
    pair = np.asarray(pair, complex)
    if pair_noise:
        rng = np.random.default_rng(0)
        pair += pair_noise * (rng.normal(size=len(pair)) + 1j*rng.normal(size=len(pair)))
    if sector in ("singlet", "triplet"):
        blocks = pair[~model.onsite].reshape(-1, 2, 2)
        singlet = ((blocks[:, 0, 1] - blocks[:, 1, 0])/2)[:, None, None] * np.array([[0, 1], [-1, 0]])
        pair[~model.onsite] = (singlet if sector == "singlet" else blocks - singlet).ravel()
        if sector == "triplet":
            pair[model.onsite] = 0
    if normal_noise:
        rng = np.random.default_rng(1)
        dn = rng.normal(scale=normal_noise, size=len(n))
        n += dn - dn.mean()
        chi = normal_noise * (rng.normal(size=len(chi)) + 1j*rng.normal(size=len(chi)))
    return n, chi, pair


def project_pairing(model, pair, sector="mixed"):
    """Project independent bond contractions into the requested spin sector.

    Reverse spatial links are still reconstructed by fermionic antisymmetry.
    Onsite terms already contain only the independent up/down singlet entry.
    """
    if sector not in ("mixed", "singlet"):
        raise ValueError("pairing_sector must be mixed or singlet.")
    result = np.asarray(pair, complex).copy()
    if sector == "singlet":
        bonds = ~model.onsite
        blocks = result[bonds].reshape(-1, 2, 2)
        amplitude = (blocks[:, 0, 1] - blocks[:, 1, 0])/2
        result[bonds] = (amplitude[:, None, None]*np.array([[0, 1], [-1, 0]])).ravel()
    return result


def _add_links(matrix, a, b, values):
    np.add.at(matrix, (np.arange(len(matrix))[:, None], a[None, :], b[None, :]), values)


def bloch_blocks(model, order, q, onsite_fields=None):
    """Return h(q), delta(q), and bare h0(q), excluding chemical potential."""
    n, chi, pair = order
    m = model.nmode
    h0 = np.zeros((len(q), m, m), complex)
    hop = np.exp(1j*q @ model.hop_shift.T) * model.hopping
    _add_links(h0, model.hop_a, model.hop_b, hop)
    _add_links(h0, model.hop_b, model.hop_a, hop.conj())
    h = h0.copy()
    hartree = np.zeros(m)
    np.add.at(hartree, model.a, model.coupling*n[model.b])
    np.add.at(hartree, model.b, model.coupling*n[model.a])
    h[:, np.arange(m), np.arange(m)] += hartree
    phase = np.exp(1j*q @ model.shift.T)
    exchange = -model.coupling * chi.conj() * phase
    _add_links(h, model.a, model.b, exchange)
    _add_links(h, model.b, model.a, exchange.conj())
    delta = np.zeros_like(h)
    _add_links(delta, model.a, model.b, model.coupling * pair * phase)
    _add_links(delta, model.b, model.a, -model.coupling * pair * phase.conj())
    if onsite_fields is not None:
        charge, bx, by, bz, dr, di = np.asarray(onsite_fields).T
        up = 2*np.arange(model.nsite)
        dn = up + 1
        h[:, up, up] += charge + bz
        h[:, dn, dn] += charge - bz
        h[:, up, dn] += bx - 1j*by
        h[:, dn, up] += bx + 1j*by
        delta[:, up, dn] += dr + 1j*di
        delta[:, dn, up] -= dr + 1j*di
    return h, delta, h0


def _bloch_batches(model, order, batch_size, onsite_fields=None, *, momenta=None):
    for points in model.momenta(batch_size) if momenta is None else momenta:
        for q in points[:, None, :]:
            h, delta, h0 = bloch_blocks(model, order, q, onsite_fields)
            hm, _, _ = bloch_blocks(model, order, -q, onsite_fields)
            yield q, h, delta, h0, hm


def prepare_blocks(model, order, batch_size=128, max_bytes=64*1024**2):
    """Cache fields for one order evaluation, or stream if the budget is too small.

    The cached matrices exclude mu, and must never be reused after order changes.
    Diagonalization always streams one k-point at a time. Constraint searches
    change local fields and use the covariance operations instead.
    """
    required = model.nk * (4*model.nmode**2*np.dtype(complex).itemsize + 2*8)
    if required > max_bytes:
        return None
    return list(_bloch_batches(model, order, batch_size))


def _thermal_sums(model, temperature, mu, batches, number_only):
    """Unnormalized, additive contributions from one contiguous momentum range."""
    m = model.nmode
    n = np.zeros(m)
    chi = np.zeros(len(model.a), complex)
    pair = np.zeros_like(chi)
    rho_local = np.zeros((model.nsite, 2, 2), complex)
    pair_local = np.zeros_like(rho_local)
    kinetic = entropy = 0.0
    spectra = []
    up = 2*np.arange(model.nsite)
    for block in batches:
        for index in range(len(block[0])):
            q, h, delta, h0, hm = (array[index:index+1] for array in block)
            bdg = np.zeros((len(q), 2*m, 2*m), complex)
            bdg[:, :m, :m] = h - mu*np.eye(m)
            bdg[:, :m, m:] = delta
            bdg[:, m:, :m] = delta.conj().transpose(0, 2, 1)
            bdg[:, m:, m:] = -hm.transpose(0, 2, 1) + mu*np.eye(m)
            try:
                energies, vectors = np.linalg.eigh(bdg)
            except np.linalg.LinAlgError:
                energies, vectors = eigh(bdg[0], driver="ev")
                energies, vectors = energies[None, :], vectors[None, :, :]
            f = expit(-energies / temperature)
            if number_only:
                n += np.sum(np.abs(vectors[:, :m])**2 * f[:, None, :], axis=(0, 2))
                del vectors, bdg
                continue
            G = (vectors.conj() * f[:, None, :]) @ vectors.transpose(0, 2, 1)
            del vectors, bdg
            rho_k = G[:, :m, :m]
            pair_k = G[:, m:, :m].transpose(0, 2, 1)
            n += np.sum(rho_k.diagonal(axis1=1, axis2=2).real, axis=0)
            phase = np.exp(1j*q @ model.shift.T)
            chi += np.sum(phase * rho_k[:, model.a, model.b], axis=0)
            pair += np.sum(phase.conj() * pair_k[:, model.a, model.b], axis=0)
            for s in (0, 1):
                for r in (0, 1):
                    rho_local[:, s, r] += rho_k[:, up+s, up+r].sum(axis=0)
                    pair_local[:, s, r] += pair_k[:, up+s, up+r].sum(axis=0)
            kinetic += np.sum(h0 * rho_k).real
            entropy += np.sum(entr(f) + entr(1-f))
            spectra.append(energies)
            del G, rho_k, pair_k
    return dict(n=n, chi=chi, pair=pair, rho_local=rho_local, pair_local=pair_local,
                kinetic=kinetic, entropy=entropy, spectra=spectra)


def _momentum_partitions(model, workers):
    # Contiguous balanced ranges keep spectral ordering deterministic. Split
    # independently of coordinate batches; each worker diagonalizes one k at a time.
    size, extra = divmod(model.nk, workers)
    starts = np.arange(workers+1)*size + np.minimum(np.arange(workers+1), extra)
    return list(zip(starts[:-1], starts[1:]))


def _range_batches(model, order, batch_size, onsite_fields, blocks, bounds):
    start, stop = bounds
    if blocks is not None:
        offset = 0
        for block in blocks:
            end = offset + len(block[0])
            lo, hi = max(start, offset)-offset, min(stop, end)-offset
            if hi > lo:
                yield tuple(array[lo:hi] for array in block)
            offset = end
            if offset >= stop:
                break
        return

    def momenta():
        for first in range(start, stop, batch_size):
            indices = np.arange(first, min(first+batch_size, stop))
            yield 2*np.pi*np.column_stack((indices % model.nkx / model.nkx,
                                           indices // model.nkx / model.nky))
    yield from _bloch_batches(model, order, batch_size, onsite_fields, momenta=momenta())


def thermal_state(model, order, temperature, mu, *, batch_size=128, onsite_fields=None,
                  number_only=False, _blocks=None, _runtime=None, _lane=0):
    """Integrate momentum contributions, then evaluate the GLOBAL interaction."""
    workers = 1 if _runtime is None else _runtime.plan.momentum_workers
    if workers == 1:
        batches = _bloch_batches(model, order, batch_size, onsite_fields) if _blocks is None else _blocks
        total = _thermal_sums(model, temperature, mu, batches, number_only)
    else:
        def evaluate(bounds):
            batches = _range_batches(model, order, batch_size, onsite_fields, _blocks, bounds)
            return _thermal_sums(model, temperature, mu, batches, number_only)
        parts = _runtime.map_momentum(evaluate, _momentum_partitions(model, workers), _lane)
        total = parts[0]
        for part in parts[1:]:
            for key in total:
                total[key] += part[key]
    n = total['n'] / model.nk
    filling = n.sum() / model.nsite
    if number_only:
        return filling
    chi = total['chi'] / model.nk
    pair = total['pair'] / model.nk
    interaction = model.coupling @ (n[model.a]*n[model.b] - abs(chi)**2 + abs(pair)**2)
    energy = (total['kinetic'] / model.nk + interaction) / model.nsite
    entropy = total['entropy'] / (2 * model.nsite * model.nk)
    return dict(occupations=n, chi=chi, pair=pair, density=n.reshape(-1, 2).sum(axis=1),
                rho_local=total['rho_local']/model.nk, pair_local=total['pair_local']/model.nk,
                filling=filling, mu=mu, energy=energy, entropy=entropy,
                free=energy-temperature*entropy, energies=np.concatenate(total['spectra']))


def solve_number(model, order, temperature, target, mu, tol, batch_size=128, *, normal_first=False,
                 _blocks=None, _runtime=None, _lane=0):
    errors = {}
    def residual(value):
        if value not in errors:
            errors[value] = thermal_state(model, order, temperature, value, batch_size=batch_size,
                                          number_only=True, _blocks=_blocks,
                                          _runtime=_runtime, _lane=_lane) - target
        return errors[value]

    def normal_seed():
        if _runtime is not None and _runtime.plan.momentum_workers > 1:
            def normal_levels(bounds):
                batches = _range_batches(model, order, batch_size, None, _blocks, bounds)
                return np.concatenate([np.linalg.eigvalsh(h) for _, h, _, _, _ in batches])
            levels = np.concatenate(_runtime.map_momentum(normal_levels,
                _momentum_partitions(model, _runtime.plan.momentum_workers), _lane))
        elif _blocks is None:
            levels = np.concatenate([np.linalg.eigvalsh(h) for _, h, _, _, _
                                     in _bloch_batches(model, order, batch_size)])
        else:
            levels = np.concatenate([np.linalg.eigvalsh(h) for _, h, _, _, _ in _blocks])
        return secant_method(lambda value: 2*expit((value-levels)/temperature).mean() - target,
                             x1=mu, ftol=tol, print_info=False)

    # Use the normal-state seed and previous-mu fallback policy.
    # Different number-solve errors can be amplified by Broyden far from a root.
    for use_normal in (normal_first, not normal_first):
        seed = normal_seed() if use_normal else mu
        value = secant_method(residual, x1=seed, ftol=tol, print_info=False)
        if np.isfinite(value) and abs(residual(value)) <= tol:
            return value
    raise RuntimeError("Momentum number equation did not reach the requested filling.")


def solve(model, param, *, initial=None, _allocation=None, plot_iteration=None):
    """Solve with automatic allocation; the wrapper can pass its announced plan."""
    validate_constraint_input(param)
    validate_thread_input(param)
    plan = _allocation if _allocation is not None else allocation_from_input(param, momentum_tasks=model.nk)
    if _allocation is None:
        print_allocation(plan)
    metadata = dict(backend='momentum', cell_shape=np.array([model.nx, model.ny]),
                    k_mesh=np.array([model.nkx, model.nky]), bond_a=model.a, bond_b=model.b,
                    bond_shift=model.shift, bond_offset=model.offset, coupling=model.coupling,
                    bond_onsite=model.onsite, momentum_constraint_version=2)
    plot = None if plot_iteration is None else lambda i,s: plot_iteration(i,dict(s,**metadata))
    with WorkerRuntime(plan) as runtime:
        state = _solve(model, param, initial=initial, runtime=runtime, plot_iteration=plot)
    state.pop('_response_cache',None)
    state.update(metadata)
    # Scalar arrays serialize directly in the existing npz output format.
    state.update({f'parallel_{key}': value for key, value in plan.metadata().items()})
    return state


def _solve(model, param, *, initial=None, runtime=None, plot_iteration=None):
    """Solve cell fields with the existing mixing methods; observables are per site.

    Custom initial=(occupations, chi, pair) arrays follow model.a/b/shift.
    Saved chi/pair are thermal contractions; chi_order/pair_order are the
    final trial fields, whose maximum packed residual is reported as err.
    """
    T, tol = param["T"], param["tolerate"]
    # Resolve energy differences independently of
    # the outer order tolerance, so density errors cannot drive line searches.
    field_tol = min(tol, 2e-12) if param["solver"] == "energy_descent" else tol
    target = param.get("n_filling")
    batch_size = 128  # Internal coordinate batching; diagonalization is one k at a time.
    cache_mb = param.get("k_cache_mb", 64.)
    if not np.isfinite(cache_mb) or cache_mb < 0:
        raise ValueError("k_cache_mb must be finite and nonnegative.")
    _positive_integer(param["maxiterate"], "maxiterate")
    if not np.all(np.isfinite([T, tol, param.get("alpha", .4)])) or T <= 0 or tol <= 0 or not 0 < param.get("alpha", .4) <= 1:
        raise ValueError("Use positive finite T/tolerate and 0 < alpha <= 1.")
    if target is not None and not 0 < target < 2:
        raise ValueError("Filling must be between zero and two.")
    ansatz = param.get("order_ansatz", "full")
    if ansatz not in ("full", "pure_SC", "nonmagnetic"):
        raise ValueError("order_ansatz must be full or pure_SC.")
    from CPU import bond_bdg as constraints
    from CPU.momentum_backend import MomentumBackend
    method = constraints.onsite_constraint_method(param)
    local_barrier = ansatz == 'pure_SC' and method == 'local_barrier'
    pure = ansatz == "pure_SC" and not local_barrier
    magnetic = ansatz == 'nonmagnetic' or (ansatz == 'full' and param.get('m_max') is not None)
    if pure and param.get('m_max') is not None:
        field_tol = min(field_tol,param['m_max']/2)
    spin_cap = param.get('local_max', 1e-6) if ansatz == 'nonmagnetic' and method == 'local_barrier' else param.get('m_max')
    cap, tau = param.get('local_max', 1e-6), param.get('tau', 1e-5)
    pairing_sector = param.get("pairing_sector", "mixed")
    if pairing_sector not in ("mixed", "singlet"):
        raise ValueError("pairing_sector must be mixed or singlet.")
    active = ~model.onsite if pure or local_barrier else np.ones(len(model.a), bool)
    nlocal = int(target is None) if pure or local_barrier else model.nmode
    seed = initial_order(model, param) if initial is None else tuple(np.asarray(x).copy() for x in initial)
    if tuple(x.shape for x in seed) != ((model.nmode,), (len(model.a),), (len(model.a),)):
        raise ValueError("Initial arrays must have shapes (2*Nsite,), (Nterms,), (Nterms,).")
    if not all(np.all(np.isfinite(x)) for x in seed) or np.any(np.imag(seed[0]) != 0):
        raise ValueError("Initial contractions must be finite and occupations real.")
    seed = np.asarray(seed[0].real, float), np.asarray(seed[1], complex), np.asarray(seed[2], complex)
    mu = float(param.get("mu", 0.0))
    if not np.isfinite(mu):
        raise ValueError("mu must be finite.")
    if local_barrier and param['solver'] == 'energy_descent':
        from CPU.barrier_descent import mixed_barrier_descent
        ops = MomentumBackend(model, runtime, batch_size=batch_size, pairing_sector=pairing_sector)
        def compact(result):
            rho, pair = result['rho'], result['pair_order']
            n, chi, p = ops.order(rho, pair)
            thermal = ops.state(rho, pair, result['energies'], T, result['mu'])
            result = dict(result)
            for key in ('rho', 'h', 'delta', 'pair_bdg'):
                result.pop(key, None)
            for key in ('occupations', 'chi', 'pair', 'rho_local', 'pair_local'):
                result[key] = thermal[key]
            result.update(occupations_order=n, chi_order=chi, pair_order=p,
                          pair_local_order=thermal['pair_local'],
                          descent_policy_version=MFSolver.ENERGY_DESCENT_VERSION)
            return result
        plot = None if plot_iteration is None else lambda i, r: plot_iteration(i, compact(r))
        result = mixed_barrier_descent(ops.lift(seed), param, mu, cap,
                                       ops.vector, plot, ops=ops)
        return compact(result)
    fields = np.zeros((model.nsite, 4 if pure or local_barrier else 3)) if pure or magnetic or local_barrier else None
    first = True
    err_prev = np.inf
    mu_guess_threshold = param.get("mu_guess_threshold", .01)
    if not mu_guess_threshold >= 0:
        raise ValueError("mu_guess_threshold must be nonnegative.")

    def pack(n, chi, pair):
        local = ([n.mean()] if nlocal else []) if pure or local_barrier else n
        if pairing_sector == "singlet":
            pair = project_pairing(model, pair, pairing_sector)
        values = np.r_[chi[active], pair]
        packed = np.r_[local, values.real, values.imag]
        return packed

    normal_size = np.count_nonzero(active)
    def packed_state(state):
        return pack(state['occupations'], state['chi'], state['pair'])

    def unpack(values):
        n = np.full(model.nmode, values[0] if target is None else target/2) if pure or local_barrier else values[:nlocal].copy()
        re, im = np.split(values[nlocal:], 2)
        c, p = np.split(re + 1j*im, [normal_size])
        chi = np.zeros(len(model.a), complex)
        pair = np.zeros_like(chi)
        chi[active], pair[:] = c, p
        if pairing_sector == "singlet":
            pair = project_pairing(model, pair, pairing_sector)
        return n, chi, pair

    response_cache = {}
    constrained = pure or magnetic or local_barrier
    lanes = 1 if runtime is None else runtime.plan.descent_concurrency
    backends = [MomentumBackend(model,runtime,lane,batch_size,pairing_sector)
                for lane in range(lanes)] if constrained else []

    def evaluate_gap(values, mu, fields, first, err_prev, lane, cache):
        # These auxiliaries belong to this evaluation only. Concurrent descent
        # directions never mutate the accepted state or each other's guesses.
        cache = cache.copy()
        order = unpack(values)
        blocks = prepare_blocks(model, order, batch_size, cache_mb*1024**2) if not constrained else None
        if target is not None and (not constrained or first):
            mu = solve_number(model, order, T, target, mu, field_tol, batch_size,
                              normal_first=err_prev >= mu_guess_threshold, _blocks=blocks,
                              _runtime=runtime, _lane=lane)
        first = False
        if magnetic or local_barrier or pure:
            ops = backends[lane]
            h, delta = bloch_blocks(model, order, ops.q)[:2]
            extra = {}
            if pure:
                r, p, h, delta, levels, mu, fields = constraints._constrained_bdg(
                    h, delta, T, target, mu, fields, field_tol,
                    refine=param['solver']=='energy_descent', response_cache=cache, ops=ops)
                extra['onsite_fields'] = fields.copy()
                extra['_response_cache'] = cache
                extra['max_m'] = np.linalg.norm(ops.local_values(r,p,0.)[:,1:4],axis=1).max()
            elif magnetic:
                r, p, h, levels, mu, fields, cost, peak = constraints._normal_barrier_bdg(
                    h, delta, T, target, mu, fields, spin_cap, tau, field_tol, ops=ops)
                extra.update(spin_fields=fields.copy(), spin_barrier_field=fields.copy(),
                             spin_barrier=cost, max_m=peak, bounds_satisfied=True)
            else:
                def normal_feedback(local):
                    nonlocal order
                    dn, mx, my, mz = local.T
                    n = np.column_stack(((target+dn+mz)/2,(target+dn-mz)/2)).ravel()
                    chi = order[1].copy()
                    sites = model.a[model.onsite]//2
                    chi[model.onsite] = (mx+1j*my)[sites]/2
                    order = n, chi, order[2]
                    return bloch_blocks(model,order,ops.q)[0]
                r, p, h, levels, mu, fields, _, _ = constraints._normal_barrier_bdg(
                    h, delta, T, target, mu, fields, cap, tau, field_tol, ops=ops,
                    constrain_charge=True, normal_feedback=normal_feedback)
                extra['old_barrier_fields'] = fields.copy()
                extra.update(ops.barrier(r,p,target,cap,tau))
            state = ops.state(r,p,levels,T,mu)
            state.update(extra)
            if magnetic:
                state['objective'] = state['free'] + extra['spin_barrier']
                if target is None:
                    state['objective'] -= mu*state['filling']
            elif local_barrier:
                state['objective'] = state['free'] + extra['local_barrier']
        else:
            state = thermal_state(model, order, T, mu, batch_size=batch_size, _blocks=blocks,
                                  _runtime=runtime, _lane=lane)
        if target is not None and abs(state["filling"] - target) > tol:
            raise RuntimeError("Momentum filling missed its target.")
        residual = packed_state(state) - values
        err_prev = np.max(np.abs(residual), initial=0.)
        groups = np.r_[np.zeros(nlocal,int),np.tile(np.r_[~model.onsite[active],2*(~model.onsite)],2)]
        state['errset'] = np.array([abs(residual[groups==i]).max(initial=0.) for i in range(3)])
        state.update(order=values.copy(), occupations_order=order[0], chi_order=order[1], pair_order=order[2])
        state["unprojected_pair_residual"] = np.max(np.abs(state["pair"]-order[2]), initial=0.)
        return residual, state

    def gap(values, previous=None, previous_err=None, *, lane=0):
        nonlocal mu, fields, first, err_prev, response_cache
        if previous is not None:
            local_fields = previous.get('onsite_fields' if pure else 'spin_fields')
            local_fields = None if local_fields is None else local_fields.copy()
            return evaluate_gap(values, float(previous['mu']), local_fields, False, previous_err, lane,
                                previous.get('_response_cache',{}))
        residual, state = evaluate_gap(values, mu, None if fields is None else fields.copy(),
                                       first, err_prev, lane, response_cache)
        # Sequential Broyden/DFSANE calls keep their usual warm-start behavior.
        mu, fields, first = state['mu'], state.get('onsite_fields' if pure else 'old_barrier_fields' if local_barrier else 'spin_fields'), False
        response_cache = state.get('_response_cache',{})
        err_prev = np.max(np.abs(residual), initial=0.)
        return residual, state

    solvers = dict(broyden=MFSolver.Self_Consistent_broyden, dfsane=MFSolver.Self_Consistent_dfsane,
                   linear_mixing=MFSolver.Self_Consistent_linear_mixing, safe=MFSolver.Self_Consistent_linear_mixing,
                   energy_descent=MFSolver.Self_Consistent_energy_descent)
    if param["solver"] not in solvers:
        raise ValueError(f"Unknown solver {param['solver']!r}.")
    history = []
    def report(iterate, state):
        print_energy(state)
        history.append((iterate, state["mu"], state["energy"], state["entropy"], state["free"]))
        if plot_iteration:
            plot_iteration(iterate, state)
    options = {}
    if param['solver'] == 'energy_descent' and runtime is not None:
        options['evaluate_trials'] = lambda orders, previous, error: runtime.trials(gap, orders, previous, error)
    initial_values = pack(*seed)
    state, iterate, err = solvers[param["solver"]](initial_values, param, gap, report, **options)
    if not history or history[-1][0] != iterate:
        report(iterate, state)
    if err >= tol and iterate >= param["maxiterate"]:
        print(f"maximum iteration reached, error = {err}", flush=True)
    dtype = [("iterate", int), ("mu", float), ("energy", float), ("entropy", float), ("free", float)]
    state.update(converged=bool(err < tol and state.get('bounds_satisfied',True)),
                 iterate=iterate, err=err, history=np.array(history, dtype=dtype))
    return state
