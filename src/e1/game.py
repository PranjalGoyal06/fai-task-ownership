"""Finite empirical reporting game for the conjunctive replay."""
from dataclasses import dataclass

import numpy as np

from src.e1.dataset import MODULES

# Action zero reports the local estimate. Remaining actions use a 0.05 grid.
GRID = np.linspace(0, 1, 21)
ACTIONS = len(GRID) + 1
BITS = 1 << np.arange(len(MODULES))


def select(probabilities, reward):
    """Exact equation 2 for equal unit costs, favoring the shortest optimal prefix."""
    order = np.argsort(-probabilities, axis=-1, kind="stable")
    omitted = 1 - np.take_along_axis(probabilities, order, axis=-1)
    ones = np.ones((*order.shape[:-1], 1))
    coverage = np.concatenate((np.cumprod(omitted[..., ::-1], axis=-1)[..., ::-1], ones), axis=-1)
    size = (reward * coverage - np.arange(order.shape[-1] + 1)).argmax(axis=-1)
    return np.sum((1 << order) * (np.arange(order.shape[-1]) < size[..., None]), axis=-1)


def population(seed, size, phi, count):
    rng = np.random.default_rng(seed)
    ranks = rng.random((size, count)).argsort(axis=1).argsort(axis=1)
    return ranks < round(count * phi), rng.random((size, count)), rng.random((size, count))


@dataclass(frozen=True)
class Condition:
    ell: float = 0
    phi: float = 0
    scale: float = 0
    audit: float = 0
    credit: str = "off"
    reward: float = 8
    delta: float = .4
    fine: float = 2
    audit_cost: float = .1


def reports(actions, probabilities):
    index = actions % ACTIONS
    return np.where(index == 0, probabilities, GRID[np.maximum(index - 1, 0)])


def information(p, thresholds, types):
    return (p > thresholds).astype(int) + 2 * types


def submit(policy, p, info, draw, degree, prior, c):
    act = policy[np.arange(p.shape[1]), info]
    r = reports(act, p)
    checked, exposed = draw
    caught = (act >= ACTIONS) & (checked < c.audit) & (exposed < 1 - (1 - c.delta) ** degree)
    return r, np.where(caught, prior, r), caught


def fit_strategy(p, y, degree, prior, c, seed, start=0):
    """Coordinate best responses conditional on module, belief bin and private type."""
    n, m = p.shape
    thresholds = np.median(p, axis=0)
    types, checked, exposed = population(seed, n, c.phi, m)
    info = information(p, thresholds, types)
    target = y @ BITS
    counts = np.array([int(k).bit_count() for k in range(1 << m)])
    policy = np.full((m, 4), start, dtype=int)
    actions = np.arange(2 * ACTIONS) if c.audit else np.arange(ACTIONS)
    seen = set()

    def improvements(i):
        _, others, _ = submit(policy, p, info, (checked, exposed), degree, prior, c)
        r = reports(actions[:, None], p[:, i])
        profiles = np.broadcast_to(others, (*r.shape, m)).copy()
        profiles[:, :, i] = r
        selected = select(profiles, c.reward)
        profiles[:, :, i] = prior[i]
        corrected = select(profiles[:1], c.reward)
        q = (actions[:, None] >= ACTIONS) * c.audit * (1 - (1 - c.delta) ** degree[:, i])

        def payoff(mask):
            assigned = (mask & BITS[i]) > 0
            utility = -float(c.ell) * types[:, i] * assigned
            covered = (mask & target) == target
            if c.credit == "equal":
                utility += c.reward * covered * assigned / np.maximum(counts[mask], 1)
            if c.credit == "shapley":
                utility += c.reward * covered * y[:, i] / counts[target]
            return utility

        utility = (1 - q) * payoff(selected) + q * (payoff(corrected) - c.fine)
        utility -= c.scale * (r - p[:, i]) ** 2
        best, residual = policy[i].copy(), 0.
        for bin_id in np.unique(info[:, i]):
            values = utility[:, info[:, i] == bin_id].mean(axis=1)
            current = policy[i, bin_id]
            gain = values.max() - values[current]
            residual = max(residual, gain)
            if gain > 1e-8:
                best[bin_id] = np.flatnonzero(values >= values.max() - 1e-10)[0]
        return best, residual

    for iteration in range(1, 21):
        key = policy.tobytes()
        if key in seen:
            status = "cycle"
            break
        seen.add(key)
        changed = False
        for i in range(m):
            new, _ = improvements(i)
            changed |= not np.array_equal(policy[i], new)
            policy[i] = new
        if not changed:
            status = "converged"
            break
    else:
        status = "iteration limit"
    residual = max(improvements(i)[1] for i in range(m))
    return policy, thresholds, dict(status=status, residual=float(residual), iterations=iteration)
