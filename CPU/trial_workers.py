"""Isolated energy-descent trials with bounded BLAS thread pools."""

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np


def initialize_trial_worker(order_guess, h0, terms, param, threads):
    global problem, thread_limit, solve_trial
    from CPU.bond_bdg import solve_from_order
    from threadpoolctl import threadpool_limits
    solve_trial = solve_from_order
    # Retain the controller for the lifetime of this process.
    thread_limit = threadpool_limits(limits=threads)
    problem = order_guess, h0, terms, param


def evaluate_trial(order, previous, previous_err):
    try:
        return solve_trial(*problem, _trial=(order, previous, previous_err))
    except (RuntimeError, np.linalg.LinAlgError):
        return None


class TrialWorkers:
    def __init__(self, order_guess, h0, terms, param):
        threads = max(1, (os.cpu_count() or 1) // 2)
        self.pool = ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn"),
            initializer=initialize_trial_worker,
            initargs=(order_guess, h0, terms, param, threads))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __call__(self, trials, previous, previous_err):
        # These are the only accepted-state fields consumed by FGap.
        fields = {key: previous[key] for key in ("mu", "onsite_fields", "spin_barrier_field")
                  if key in previous}
        futures = [self.pool.submit(evaluate_trial, order, fields, previous_err)
                   for order in trials]
        # Preserve direction order for deterministic energy/residual tie-breaking.
        return [future.result() for future in futures]
