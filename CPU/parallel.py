"""One CPU budget for descent trials, momentum tasks and native BLAS threads."""

import os
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from threading import local

import psutil
import numpy as np
from threadpoolctl import ThreadpoolController


def _quota_limits():
    """Read Linux cgroup CPU quotas, including tighter parent quotas."""
    try:
        memberships = Path('/proc/self/cgroup').read_text().splitlines()
    except OSError:
        return []
    limits = []
    for membership in memberships:
        _, controllers, relative = membership.split(':', 2)
        if controllers == '':
            roots, filename = [Path('/sys/fs/cgroup')], 'cpu.max'
        elif 'cpu' in controllers.split(','):
            # The cpu controller may share a mount with cpuacct, without a
            # separate /cpu alias. Check both conventional controller orders.
            roots = [Path('/sys/fs/cgroup') / name for name in
                     dict.fromkeys(('cpu', controllers, 'cpu,cpuacct', 'cpuacct,cpu'))]
            filename = 'cpu.cfs_quota_us'
        else:
            continue
        for root in roots:
            directory = root / relative.lstrip('/')
            while directory == root or root in directory.parents:
                try:
                    if filename == 'cpu.max':
                        quota, period = (directory / filename).read_text().split()
                    else:
                        quota = (directory / filename).read_text().strip()
                        period = (directory / 'cpu.cfs_period_us').read_text().strip()
                    if quota != 'max' and int(quota) > 0 and int(period) > 0:
                        limits.append(max(1, int(quota) // int(period)))
                except (OSError, ValueError):
                    pass
                if directory == root:
                    break
                directory = directory.parent
    return limits


def available_cpu_count():
    """Use process affinity and per-task scheduler limits, not just host CPUs."""
    limits = [os.cpu_count() or 1]
    try:
        limits.append(len(psutil.Process().cpu_affinity()))
    except (AttributeError, OSError, psutil.Error):
        pass
    for name in ('SLURM_CPUS_PER_TASK', 'NCPUS', 'NSLOTS'):
        value = os.environ.get(name)
        if value is not None:
            try:
                count = int(value)
            except ValueError as exc:
                raise ValueError(f'{name} must be a positive integer, got {value!r}.') from exc
            if count < 1:
                raise ValueError(f'{name} must be positive.')
            limits.append(count)
    limits.extend(_quota_limits())
    return max(1, min(limits))


@dataclass(frozen=True)
class Allocation:
    available_threads: int
    requested_diag_threads: int
    diag_threads: int
    descent_workers: int
    descent_concurrency: int
    momentum_workers: int

    @property
    def allocated_threads(self):
        return self.descent_concurrency * self.momentum_workers * self.diag_threads

    def metadata(self):
        return dict(asdict(self), allocated_threads=self.allocated_threads)


def allocation(solver, diag_threads=1, momentum_tasks=1):
    if isinstance(diag_threads, bool) or not isinstance(diag_threads, Integral) or diag_threads < 1:
        raise ValueError('diag_threads must be a positive integer.')
    if isinstance(momentum_tasks, bool) or not isinstance(momentum_tasks, Integral) or momentum_tasks < 1:
        raise ValueError('momentum_tasks must be a positive integer.')
    available = available_cpu_count()
    diagonal = min(int(diag_threads), available)
    descent = 2 if solver == 'energy_descent' else 1
    concurrent = min(descent, available // diagonal)
    momentum = min(int(momentum_tasks), available // (concurrent * diagonal))
    return Allocation(available, int(diag_threads), diagonal, descent, concurrent, momentum)


def validate_thread_input(param):
    """Worker counts are derived outputs, not independent input controls."""
    unsupported = sorted(set(param).intersection({
        'descent_workers', 'descent_worker', 'momentum_workers', 'momentum_worker',
        'momentum-worker', 'worker_threads', 'k_workers',
    }))
    if unsupported:
        raise ValueError(f"Remove {', '.join(unsupported)} from the input. Only diag_threads is "
                         'configurable; descent and momentum workers are determined automatically.')
    diag = param.get('diag_threads', 1)
    if isinstance(diag, bool) or not isinstance(diag, Integral) or diag < 1:
        raise ValueError('diag_threads must be a positive integer.')


def allocation_from_input(param, *, momentum_tasks=None):
    """Plan a momentum solve before constructing any model or numerical pool."""
    validate_thread_input(param)
    if momentum_tasks is None:
        for key in ('Nkx', 'Nky'):
            value = param.get(key, 1)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{key} must be a positive integer.')
        momentum_tasks = int(param.get('Nkx', 1)) * int(param.get('Nky', 1))
    return allocation(param['solver'], param.get('diag_threads', 1), momentum_tasks)


def print_allocation(plan):
    diagonal = str(plan.diag_threads)
    if plan.diag_threads != plan.requested_diag_threads:
        diagonal += f' (requested={plan.requested_diag_threads})'
    print(f'CPU allocation: available_threads={plan.available_threads}, diag_threads={diagonal}, '
          f'descent_workers={plan.descent_workers} (concurrent={plan.descent_concurrency}), '
          f'momentum_workers={plan.momentum_workers}/descent worker, '
          f'allocated_threads={plan.allocated_threads}', flush=True)


class WorkerRuntime:
    """Persistent pools; all worker threads use the same native thread limit.

    Each descent lane has its own momentum pool. Control threads wait while
    their momentum pool runs. Native limits are restored only after ALL pools
    shut down; setting/restoring process-wide BLAS limits per task would race.
    A process must run independent solves sequentially (use processes for sweeps).
    """

    def __init__(self, plan):
        self.plan = plan
        self.stack = ExitStack()
        self.local = local()

    def __enter__(self):
        try:
            self.controller = ThreadpoolController()
            self.stack.enter_context(self.controller.limit(limits=self.plan.diag_threads))

            def initialize():
                # MKL limits can be thread-local. Apply them in every Python
                # worker too, retaining the limiter for that thread's lifetime.
                self.local.limit = self.controller.limit(limits=self.plan.diag_threads)

            self.momentum = []
            for _ in range(self.plan.descent_concurrency):
                pool = (self.stack.enter_context(ThreadPoolExecutor(
                    max_workers=self.plan.momentum_workers, initializer=initialize,
                    thread_name_prefix='bdg-k')) if self.plan.momentum_workers > 1 else None)
                self.momentum.append(pool)
            self.descent = (self.stack.enter_context(ThreadPoolExecutor(
                max_workers=2, initializer=initialize, thread_name_prefix='bdg-descent'))
                if self.plan.descent_concurrency == 2 else None)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def map_momentum(self, function, tasks, lane=0):
        pool = self.momentum[lane]
        if pool is None:
            return list(map(function, tasks))
        futures = [pool.submit(function, task) for task in tasks]
        try:
            return [future.result() for future in futures]
        except BaseException:
            # A rejected trial must finish/cancel its outstanding work before
            # starting the next trial or restoring the native thread limits.
            for future in futures:
                future.cancel()
            wait(futures)
            raise

    def trials(self, gap, orders, previous, previous_err):
        def evaluate(item):
            lane, order = item
            try:
                return gap(order, previous, previous_err, lane=lane)
            except (RuntimeError, np.linalg.LinAlgError):
                return None

        if self.descent is None:
            return [evaluate((0, order)) for order in orders]
        # Preserve direction order, even if the second direction finishes first.
        return list(self.descent.map(evaluate, enumerate(orders)))
