#!/usr/bin/env python
"""
Generate figures for algorithmic game theory project.
Usage: python plots.py --all
"""

import argparse
import itertools
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12

def ensure_figs_dir():
    os.makedirs("figs", exist_ok=True)

def compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for x in range(n + 1):
        for rest in compositions(n - x, k - 1):
            yield (x,) + rest

def mean_ci(vals):
    v = np.array(vals, dtype=float)
    mu = float(np.mean(v))
    if len(v) <= 1:
        return mu, 0.0
    se = float(np.std(v, ddof=1) / math.sqrt(len(v)))
    return mu, se

def iqr_band(vals_matrix):
    med = np.median(vals_matrix, axis=0)
    q25 = np.percentile(vals_matrix, 25, axis=0)
    q75 = np.percentile(vals_matrix, 75, axis=0)
    return med, q25, q75


class FiniteGame:
    def num_players(self):
        raise NotImplementedError

    def actions(self, i):
        raise NotImplementedError

    def cost(self, profile, i):
        raise NotImplementedError

    def social_cost(self, profile):
        return sum(self.cost(profile, i) for i in range(self.num_players()))

    def best_response(self, profile, i):
        best_a = None
        best_c = float("inf")
        for a in self.actions(i):
            new_prof = list(profile)
            new_prof[i] = a
            c = self.cost(tuple(new_prof), i)
            if c < best_c - EPS:
                best_c = c
                best_a = a
        return best_a

    def is_pne(self, profile):
        for i in range(self.num_players()):
            br = self.best_response(profile, i)
            if br != profile[i]:
                new_prof = list(profile)
                new_prof[i] = br
                if self.cost(tuple(new_prof), i) < self.cost(profile, i) - EPS:
                    return False
        return True

# Generic async best response dynamics
def async_best_response_dynamics(
    game,
    init_profile,
    max_sweeps=400,
    noise_p=0.0,
    rng=None,
    frozen=None,
):
    if rng is None:
        rng = random.Random()
    if frozen is None:
        frozen = {}

    prof = list(init_profile)
    n = game.num_players()

    for _ in range(max_sweeps):
        changed = False
        order = list(range(n))
        rng.shuffle(order)

        for i in order:
            if i in frozen:
                if prof[i] != frozen[i]:
                    prof[i] = frozen[i]
                    changed = True
                continue

            if rng.random() < noise_p:
                a = rng.choice(game.actions(i))
            else:
                a = game.best_response(tuple(prof), i)

            if a != prof[i]:
                new_prof = prof.copy()
                new_prof[i] = a
                if game.cost(tuple(new_prof), i) <= game.cost(tuple(prof), i) - EPS:
                    prof[i] = a
                    changed = True

        if not changed:
            break

    return tuple(prof)

# Routing game classes
class LinkLatency:
    def __init__(self, fn):
        self.fn = fn

class ParallelLinksRoutingGame(FiniteGame):
    def __init__(self, n_players, latencies):
        self.n = n_players
        self.latencies = latencies
        self.m = len(latencies)

    def num_players(self):
        return self.n

    def actions(self, i):
        return list(range(self.m))

    @staticmethod
    def random_instance(n_players, m_links=3, seed=0):
        rng = random.Random(seed)
        latencies = []
        for _ in range(m_links):
            a = rng.choice([0.5, 1.0, 1.5, 2.0])
            b = rng.choice([0.0, 0.5, 1.0, 2.0])
            latencies.append(LinkLatency(fn=lambda x, a=a, b=b: a * x + b))
        return ParallelLinksRoutingGame(n_players, latencies)

    def load_vector(self, profile):
        loads = [0] * self.m
        for a in profile:
            loads[a] += 1
        return loads

    def cost(self, profile, i):
        a = profile[i]
        loads = self.load_vector(profile)
        return self.latencies[a].fn(loads[a])

    def social_cost_from_loads(self, loads):
        sc = 0.0
        for e, x in enumerate(loads):
            if x > 0:
                sc += x * self.latencies[e].fn(x)
        return sc

    def profile_from_loads(self, loads):
        prof = []
        for e, x in enumerate(loads):
            prof.extend([e] * x)
        return tuple(prof)

    def is_pne_loads(self, loads):
        for e, xe in enumerate(loads):
            if xe <= 0:
                continue
            cur = self.latencies[e].fn(xe)
            for f, xf in enumerate(loads):
                if f == e:
                    continue
                new_cost = self.latencies[f].fn(xf + 1)
                if new_cost < cur - EPS:
                    return False
        return True

    def compute_opt_loads(self):
        best_loads = None
        best_val = float("inf")
        for L in compositions(self.n, self.m):
            sc = self.social_cost_from_loads(L)
            if sc < best_val - EPS:
                best_val = sc
                best_loads = L
        return best_loads, best_val

    def compute_worst_pne_loads(self):
        worst_loads = None
        worst_val = -float("inf")
        for L in compositions(self.n, self.m):
            if self.is_pne_loads(L):
                sc = self.social_cost_from_loads(L)
                if sc > worst_val + EPS:
                    worst_val = sc
                    worst_loads = L
        if worst_loads is None:
            raise RuntimeError("No PNE loads found.")
        return worst_loads, worst_val

    def all_pne_loads(self):
        out = []
        for L in compositions(self.n, self.m):
            if self.is_pne_loads(L):
                out.append(L)
        return out

    def run_br_dynamics(
        self,
        init_profile,
        max_sweeps=600,
        noise_p=0.0,
        rng=None,
        frozen=None,
    ):
        if rng is None:
            rng = random.Random()
        if frozen is None:
            frozen = {}

        prof = list(init_profile)
        loads = [0] * self.m
        for a in prof:
            loads[a] += 1

        for _ in range(max_sweeps):
            changed = False
            order = list(range(self.n))
            rng.shuffle(order)

            for i in order:
                if i in frozen:
                    desired = frozen[i]
                    if prof[i] != desired:
                        loads[prof[i]] -= 1
                        prof[i] = desired
                        loads[desired] += 1
                        changed = True
                    continue

                cur_link = prof[i]
                cur_cost = self.latencies[cur_link].fn(loads[cur_link])

                if rng.random() < noise_p:
                    cand = rng.choice(list(range(self.m)))
                else:
                    # best response: choose link with minimal latency if i moved there
                    best_link = cur_link
                    best_cost = cur_cost
                    for e in range(self.m):
                        if e == cur_link:
                            continue
                        # if i moves cur->e, loads[cur]-=1, loads[e]+=1
                        cost_e = self.latencies[e].fn(loads[e] + 1)
                        if cost_e < best_cost - EPS:
                            best_cost = cost_e
                            best_link = e
                    cand = best_link

                if cand != cur_link:
                    # improvement check
                    new_cost = self.latencies[cand].fn(loads[cand] + 1)
                    if new_cost <= cur_cost - EPS:
                        loads[cur_link] -= 1
                        prof[i] = cand
                        loads[cand] += 1
                        changed = True

            if not changed:
                break

        return tuple(prof)

def piecewise_latency(t, a1, b1, a2, extra=0.0):
    def fn(x):
        if x <= 0:
            return 0.0
        if x <= t:
            return a1 * x + b1
        base = a1 * t + b1
        return base + a2 * (x - t) + extra
    return fn

def make_candidate_routing_instance(n, m, rng):
    # Generate instances with multiple equilibria by including
    # at least one "tempting then steep" link

    t = max(2, n // 3)

    a1 = rng.uniform(0.02, 0.08)
    b1 = rng.uniform(0.0, 0.3)
    a2 = rng.uniform(0.8, 2.2)
    l0 = LinkLatency(fn=piecewise_latency(t, a1, b1, a2, rng.uniform(0.0, 0.5)))

    a = rng.uniform(0.08, 0.25)
    b = rng.uniform(0.2, 1.2)
    l1 = LinkLatency(fn=lambda x, a=a, b=b: a * x + b)

    lats = [l0, l1]

    if m >= 3:
        a = rng.uniform(0.0, 0.06)
        b = rng.uniform(0.9, 2.0)
        lats.append(LinkLatency(fn=lambda x, a=a, b=b: a * x + b))

    if m >= 4:
        a = rng.uniform(0.06, 0.25)
        b = rng.uniform(0.4, 1.8)
        lats.append(LinkLatency(fn=lambda x, a=a, b=b: a * x + b))

    return ParallelLinksRoutingGame(n, lats)

def find_hard_routing_instance(n, m=3, target_poa=1.20, min_pne=2, max_tries=4000, seed=0):
    rng = random.Random(seed)
    fallback = None
    fallback_poa = 1.0

    for _ in range(max_tries):
        g = make_candidate_routing_instance(n, m, rng)
        optL, opt_val = g.compute_opt_loads()
        worstL, worst_val = g.compute_worst_pne_loads()
        poa = worst_val / (opt_val + 1e-10)
        pne_count = len(g.all_pne_loads())

        if poa > fallback_poa:
            fallback_poa = poa
            fallback = g

        if poa >= target_poa and pne_count >= min_pne:
            return g

    return fallback


class CostSharingPathGame(FiniteGame):
    # Fair cost-sharing game: edge costs split equally among users
    def __init__(self, n_players, paths, edge_costs):
        self.n = n_players
        self.paths = [list(p) for p in paths]
        self.edge_costs = [float(c) for c in edge_costs]
        self.K = len(paths)
        self.E = len(edge_costs)
        self.path_edge_sets = [set(p) for p in self.paths]

    def num_players(self):
        return self.n

    def actions(self, i):
        return list(range(self.K))

    def edge_counts_from_path_counts(self, counts):
        ec = [0] * self.E
        for p, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            for e in self.paths[p]:
                ec[e] += cnt
        return ec

    def social_cost_from_counts(self, counts):
        ec = self.edge_counts_from_path_counts(counts)
        sc = 0.0
        for e, cnt in enumerate(ec):
            if cnt > 0:
                sc += self.edge_costs[e]
        return sc

    def profile_from_counts(self, counts):
        prof = []
        for p, cnt in enumerate(counts):
            prof.extend([p] * cnt)
        return tuple(prof)

    def cost(self, profile, i):
        counts = [0] * self.K
        for a in profile:
            counts[a] += 1
        ec = self.edge_counts_from_path_counts(tuple(counts))

        p = profile[i]
        c = 0.0
        for e in self.paths[p]:
            c += self.edge_costs[e] / max(ec[e], 1)
        return c

    def deviation_cost(self, old_p, new_p, edge_counts):
        old_edges = self.path_edge_sets[old_p]
        new_edges = self.path_edge_sets[new_p]

        c = 0.0
        for e in new_edges:
            denom = edge_counts[e] + (0 if e in old_edges else 1)
            c += self.edge_costs[e] / max(denom, 1)
        return c

    def is_pne_counts(self, counts):
        ec = self.edge_counts_from_path_counts(counts)
        for p in range(self.K):
            if counts[p] <= 0:
                continue
            cur_cost = 0.0
            for e in self.paths[p]:
                cur_cost += self.edge_costs[e] / max(ec[e], 1)

            for q in range(self.K):
                if q == p:
                    continue
                new_cost = self.deviation_cost(old_p=p, new_p=q, edge_counts=ec)
                if new_cost < cur_cost - EPS:
                    return False
        return True

    def compute_opt_counts(self):
        best_counts = None
        best_val = float("inf")
        for C in compositions(self.n, self.K):
            sc = self.social_cost_from_counts(C)
            if sc < best_val - EPS:
                best_val = sc
                best_counts = C
        return best_counts, best_val

    def compute_worst_pne_counts(self):
        worst_counts = None
        worst_val = -float("inf")
        for C in compositions(self.n, self.K):
            if self.is_pne_counts(C):
                sc = self.social_cost_from_counts(C)
                if sc > worst_val + EPS:
                    worst_val = sc
                    worst_counts = C
        if worst_counts is None:
            raise RuntimeError("No PNE counts found.")
        return worst_counts, worst_val

    def run_br_dynamics(self, init_profile, max_sweeps=600, noise_p=0.0, rng=None, frozen=None):
        if rng is None:
            rng = random.Random()
        if frozen is None:
            frozen = {}

        prof = list(init_profile)
        counts = [0] * self.K
        for a in prof:
            counts[a] += 1

        edge_counts = self.edge_counts_from_path_counts(tuple(counts))

        for _ in range(max_sweeps):
            changed = False
            order = list(range(self.n))
            rng.shuffle(order)

            for i in order:
                if i in frozen:
                    desired = frozen[i]
                    if prof[i] != desired:
                        old = prof[i]
                        # update path counts
                        counts[old] -= 1
                        counts[desired] += 1
                        # update edge counts
                        for e in self.paths[old]:
                            edge_counts[e] -= 1
                        for e in self.paths[desired]:
                            edge_counts[e] += 1
                        prof[i] = desired
                        changed = True
                    continue

                old_p = prof[i]

                # current cost
                cur_cost = 0.0
                for e in self.paths[old_p]:
                    cur_cost += self.edge_costs[e] / max(edge_counts[e], 1)

                if rng.random() < noise_p:
                    cand = rng.choice(list(range(self.K)))
                else:
                    best_p = old_p
                    best_cost = cur_cost
                    for q in range(self.K):
                        if q == old_p:
                            continue
                        c = self.deviation_cost(
                            old_p=old_p, new_p=q, edge_counts=edge_counts
                        )
                        if c < best_cost - EPS:
                            best_cost = c
                            best_p = q
                    cand = best_p

                if cand != old_p:
                    new_cost = self.deviation_cost(
                        old_p=old_p, new_p=cand, edge_counts=edge_counts
                    )
                    if new_cost <= cur_cost - EPS:
                        # apply move
                        counts[old_p] -= 1
                        counts[cand] += 1
                        for e in self.paths[old_p]:
                            edge_counts[e] -= 1
                        for e in self.paths[cand]:
                            edge_counts[e] += 1
                        prof[i] = cand
                        changed = True

            if not changed:
                break

        return tuple(prof)


class ToyCostSharingGame(FiniteGame):
    # Simple 2-choice game: private edge vs shared edge
    def __init__(self, n_players=3, cS=6.0, cP=3.0):
        self.n = n_players
        self.cS = float(cS)
        self.cP = float(cP)

    def num_players(self):
        return self.n

    def actions(self, i):
        return [0, 1]  # 0=private, 1=shared

    def cost(self, profile, i):
        if profile[i] == 0:
            return self.cP
        k = sum(profile)
        if k <= 0:
            return float("inf")
        return self.cS / k

    def social_cost(self, profile):
        return sum(self.cost(profile, i) for i in range(self.n))

def make_threshold_costsharing_instance(n):
    # Threshold instance with shared vs private paths
    C_shared = 14.0
    C_priv = 5.0
    C_priv2 = 6.0

    paths = [[0], [1], [2]]
    edge_costs = [C_shared, C_priv, C_priv2]
    return CostSharingPathGame(n, paths, edge_costs)


class PartyAffiliationPrefsGame(FiniteGame):
    # Party affiliation game with edge weights and preference costs
    def __init__(self, W, pref, pref_w):
        self.W = W.astype(float)
        self.pref = pref.astype(int)
        self.pref_w = pref_w.astype(float)
        self.n = W.shape[0]

    def num_players(self):
        return self.n

    def actions(self, i):
        return [0, 1]

    def cost(self, profile, i):
        pi = profile[i]
        c = self.pref_w[i] * (1.0 if pi != self.pref[i] else 0.0)
        for j in range(self.n):
            if j == i:
                continue
            if profile[j] != pi:
                c += self.W[i, j]
        return c

    def social_cost(self, profile):
        sc = 0.0
        for i in range(self.n):
            sc += self.pref_w[i] * (1.0 if profile[i] != self.pref[i] else 0.0)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if profile[i] != profile[j]:
                    sc += self.W[i, j]
        return sc

    def compute_opt_bruteforce(self):
        best_val = float("inf")
        best_prof = None
        for prof in itertools.product([0, 1], repeat=self.n):
            sc = self.social_cost(tuple(prof))
            if sc < best_val - EPS:
                best_val = sc
                best_prof = tuple(prof)
        return best_prof, best_val

    @staticmethod
    def planted_instance(n, p_in, p_out, w_low, w_high, pref_w, seed):
        rng = random.Random(seed)
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                same = (i < n // 2 and j < n // 2) or (i >= n // 2 and j >= n // 2)
                p = p_in if same else p_out
                if rng.random() < p:
                    w = rng.uniform(w_low, w_high)
                    W[i, j] = w
                    W[j, i] = w

        pref = np.zeros(n, dtype=int)
        pref[n // 2:] = 1
        pref_w_vec = np.full(n, pref_w, dtype=float)
        return PartyAffiliationPrefsGame(W, pref, pref_w_vec)


# PSA functions
def pick_receptive_random(n, alpha, rng):
    return {i for i in range(n) if rng.random() < alpha}

def pick_receptive_targeted(game, s_rec, alpha):
    n = game.num_players()
    k = int(math.floor(alpha * n))
    if k <= 0:
        return set()

    if isinstance(game, ParallelLinksRoutingGame):
        loads = game.load_vector(s_rec)
        delta = []
        for e, lat in enumerate(game.latencies):
            x = loads[e]
            if x == 0:
                d = lat.fn(1)
            else:
                d = (x + 1) * lat.fn(x + 1) - x * lat.fn(x)
            delta.append((d, e))
        delta.sort(reverse=True)
        d_by = {e: d for (d, e) in delta}

        scores = []
        for i in range(n):
            scores.append((d_by.get(s_rec[i], 0.0), i))
        scores.sort(reverse=True)
        return {i for _, i in scores[:k]}

    if isinstance(game, CostSharingPathGame):
        return set(range(k))

    if isinstance(game, PartyAffiliationPrefsGame):
        deg = game.W.sum(axis=1)
        idx = sorted(range(n), key=lambda i: deg[i], reverse=True)
        return set(idx[:k])

    return set(range(k))

def run_psa(game, alpha, targeted, s_rec, rng, noise_p=0.0, temp_T=0):
    n = game.num_players()

    if targeted:
        R = pick_receptive_targeted(game, s_rec, alpha)
    else:
        R = pick_receptive_random(n, alpha, rng)

    init = []
    for i in range(n):
        if i in R:
            init.append(s_rec[i])
        else:
            init.append(rng.choice(game.actions(i)))
    init_prof = tuple(init)

    # temporary PSA = freeze for temp_T sweeps then release
    if temp_T > 0:
        frozen = {i: s_rec[i] for i in R}
        if isinstance(game, ParallelLinksRoutingGame):
            prof = init_prof
            for _ in range(temp_T):
                prof = game.run_br_dynamics(
                    prof, max_sweeps=1, noise_p=noise_p, rng=rng, frozen=frozen
                )
            prof = game.run_br_dynamics(
                prof, max_sweeps=700, noise_p=noise_p, rng=rng, frozen={}
            )
            return prof
        if isinstance(game, CostSharingPathGame):
            prof = init_prof
            for _ in range(temp_T):
                prof = game.run_br_dynamics(
                    prof, max_sweeps=1, noise_p=noise_p, rng=rng, frozen=frozen
                )
            prof = game.run_br_dynamics(
                prof, max_sweeps=700, noise_p=noise_p, rng=rng, frozen={}
            )
            return prof

        prof = init_prof
        for _ in range(temp_T):
            prof = async_best_response_dynamics(
                game, prof, max_sweeps=1, noise_p=noise_p, rng=rng, frozen=frozen
            )
        prof = async_best_response_dynamics(
            game, prof, max_sweeps=700, noise_p=noise_p, rng=rng, frozen={}
        )
        return prof

    # normal PSA
    if isinstance(game, ParallelLinksRoutingGame):
        return game.run_br_dynamics(init_prof, max_sweeps=700, noise_p=noise_p, rng=rng)
    if isinstance(game, CostSharingPathGame):
        return game.run_br_dynamics(init_prof, max_sweeps=700, noise_p=noise_p, rng=rng)

    return async_best_response_dynamics(
        game, init_prof, max_sweeps=700, noise_p=noise_p, rng=rng
    )


def estimate_psa_ratio_trials(game, alphas, targeted, s_opt, opt_val, trials, seed):
    rng = random.Random(seed)
    means, q25s, q75s = [], [], []
    for a in alphas:
        vals = []
        for _ in range(trials):
            out = run_psa(game, a, targeted, s_opt, rng=rng)
            vals.append(game.social_cost(out) / (opt_val + 1e-10))
        v = np.array(vals, dtype=float)
        means.append(float(np.mean(v)))
        q25s.append(float(np.percentile(v, 25)))
        q75s.append(float(np.percentile(v, 75)))
    return means, q25s, q75s

def baseline_br_distribution(game, opt_val, restarts, seed):
    rng = random.Random(seed)
    vals = []
    for _ in range(restarts):
        init = tuple(rng.choice(game.actions(i)) for i in range(game.num_players()))
        if isinstance(game, ParallelLinksRoutingGame):
            out = game.run_br_dynamics(init, max_sweeps=700, rng=rng)
        elif isinstance(game, CostSharingPathGame):
            out = game.run_br_dynamics(init, max_sweeps=700, rng=rng)
        else:
            out = async_best_response_dynamics(game, init, max_sweeps=700, rng=rng)
        vals.append(game.social_cost(out) / (opt_val + 1e-10))
    return vals


# Coordination mechanisms
def hard_gating_latency(base_fn, thresh, bigM=1e6):
    def fn(x):
        return base_fn(x) + (bigM if x > thresh else 0.0)
    return fn

def smooth_gating_latency_poly(base_fn, thresh, M, p=2.0):
    def fn(x):
        over = max(0, x - thresh)
        return base_fn(x) + M * (over**p)
    return fn

def apply_mechanism_parallel_links(game, mode, link_idx, thresh, M, p=2.0):
    new_lat = []
    for e, lat in enumerate(game.latencies):
        if e != link_idx:
            new_lat.append(lat)
        else:
            if mode == "hard":
                new_lat.append(LinkLatency(fn=hard_gating_latency(lat.fn, thresh)))
            elif mode == "smooth":
                new_lat.append(LinkLatency(fn=smooth_gating_latency_poly(lat.fn, thresh, M, p)))
            else:
                raise ValueError("mode must be hard or smooth")
    return ParallelLinksRoutingGame(game.num_players(), new_lat)

def engineered_ratio_worst_pne(original, modified):
    worstL, _ = modified.compute_worst_pne_loads()
    _, opt_val = original.compute_opt_loads()
    return original.social_cost_from_loads(worstL) / (opt_val + 1e-10)

def pick_mechanism_params_for_demo(game, seed=0):
    n = game.num_players()
    cand_thresh = sorted(set([max(1, n // 4), max(1, n // 3), max(1, n // 2),
                               max(1, n // 3 - 1), max(1, n // 3 + 1)]))
    best = float("inf")
    best_params = (0, cand_thresh[0])
    for e in range(game.m):
        for t in cand_thresh:
            mod = apply_mechanism_parallel_links(game, mode="hard", link_idx=e, thresh=t, M=10.0)
            r = engineered_ratio_worst_pne(game, mod)
            if r < best - EPS:
                best = r
                best_params = (e, t)
    return best_params


# Figure generation functions
def fig_routing_psa_alpha(path="figs/routing_psa_alpha.pdf"):
    ensure_figs_dir()

    n = 10
    game = ParallelLinksRoutingGame.random_instance(n_players=n, m_links=3, seed=1)
    opt_loads, opt_val = game.compute_opt_loads()
    s_opt = game.profile_from_loads(opt_loads)

    alphas = np.linspace(0.0, 1.0, 11).tolist()

    mean_r, _, _ = estimate_psa_ratio_trials(game, alphas, False, s_opt, opt_val, 200, 1)
    mean_t, _, _ = estimate_psa_ratio_trials(game, alphas, True, s_opt, opt_val, 200, 2)

    plt.figure()
    plt.plot(alphas, mean_r, marker="o", label="Random PSA")
    plt.plot(alphas, mean_t, marker="o", label="Targeted PSA")
    plt.axhline(1.0, linestyle="--", label="OPT baseline")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("PSA-ratio")
    plt.title("Routing (parallel links): PSA-ratio vs α")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig_routing_mech_vs_psa(path="figs/routing_mech_vs_psa.pdf"):
    ensure_figs_dir()

    n = 60
    game = find_hard_routing_instance(n=n, m=3, target_poa=1.15, min_pne=2, seed=2)
    optL, opt_val = game.compute_opt_loads()
    s_opt = game.profile_from_loads(optL)

    baseline_vals = baseline_br_distribution(game, opt_val, 300, 20)
    base_mean, base_se = mean_ci(baseline_vals)

    alphas = [0.2, 0.4, 0.6, 0.8]
    psa_means, psa_ses = [], []
    for a in alphas:
        rng = random.Random(100 + int(1000 * a))
        vals = []
        for _ in range(260):
            out = run_psa(game, a, True, s_opt, rng)
            vals.append(game.social_cost(out) / (opt_val + 1e-10))
        mu, se = mean_ci(vals)
        psa_means.append(mu)
        psa_ses.append(se)

    link_idx, thresh = pick_mechanism_params_for_demo(game, seed=0)
    hard_mod = apply_mechanism_parallel_links(game, "hard", link_idx, thresh, 10.0)
    smooth_mod = apply_mechanism_parallel_links(game, "smooth", link_idx, thresh, 8.0, 2.0)

    hard_vals = baseline_br_distribution(hard_mod, opt_val, 300, 30)
    smooth_vals = baseline_br_distribution(smooth_mod, opt_val, 300, 31)
    hard_mean, hard_se = mean_ci(hard_vals)
    smooth_mean, smooth_se = mean_ci(smooth_vals)

    labels = [f"PSA α={a:.1f}" for a in alphas] + ["Hard gating", "Smooth gating", "Baseline"]
    means = psa_means + [hard_mean, smooth_mean, base_mean]
    ses = psa_ses + [hard_se, smooth_se, base_se]

    plt.figure(figsize=(9.0, 4.4))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=ses, capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Mean outcome / OPT (error bars = stderr)")
    plt.title(f"Routing (hard instance): mechanisms vs PSA (gate link {link_idx}, thresh {thresh})")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig_smooth_vs_hard_gating(path="figs/smooth_vs_hard_gating.pdf"):
    ensure_figs_dir()

    n = 6
    latencies = [
        LinkLatency(fn=lambda x, a=1.0, b=0.0: a * x + b),
        LinkLatency(fn=lambda x, a=0.0, b=5.0: a * x + b),
    ]
    base = ParallelLinksRoutingGame(n, latencies)

    baseline_ratio = engineered_ratio_worst_pne(base, base)
    hard_mod = apply_mechanism_parallel_links(base, "hard", 0, 3, 10.0)
    hard_ratio = engineered_ratio_worst_pne(base, hard_mod)

    Ms = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    smooth_ratios = []
    for M in Ms:
        smooth_mod = apply_mechanism_parallel_links(base, "smooth", 0, 3, M, 2.0)
        smooth_ratios.append(engineered_ratio_worst_pne(base, smooth_mod))

    plt.figure(figsize=(7.0, 4.2))
    plt.axhline(baseline_ratio, linestyle=":", label="No mechanism")
    plt.axhline(hard_ratio, linestyle="--", label="Hard gating")
    plt.plot(Ms, smooth_ratios, marker="o", label="Smooth gating")

    plt.xlabel(r"Penalty scale $M$")
    plt.ylabel("Worst outcome / OPT (engineered ratio)")
    plt.title("Smooth penalties vs hard gating (Pigou-style atomic routing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig_costsharing_psa(path="figs/costsharing_psa.pdf"):
    ensure_figs_dir()

    n = 40
    game = make_threshold_costsharing_instance(n)

    optC, opt_val = game.compute_opt_counts()
    s_opt = game.profile_from_counts(optC)

    alphas = np.linspace(0.0, 1.0, 11).tolist()

    mean_r, q25_r, q75_r = estimate_psa_ratio_trials(game, alphas, False, s_opt, opt_val, 320, 50)
    mean_t, q25_t, q75_t = estimate_psa_ratio_trials(game, alphas, True, s_opt, opt_val, 320, 51)

    plt.figure(figsize=(8.0, 4.6))
    plt.plot(alphas, mean_r, marker="o", label="Random PSA (mean)")
    plt.fill_between(alphas, q25_r, q75_r, alpha=0.2)
    plt.plot(alphas, mean_t, marker="o", label="Targeted PSA (mean)")
    plt.fill_between(alphas, q25_t, q75_t, alpha=0.2)
    plt.axhline(1.0, linestyle="--", label="OPT baseline")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Outcome / OPT")
    plt.title("Cost-sharing (threshold instance): PSA vs α (bands = IQR over PSA trials)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig_party_threshold(path="figs/party_threshold.pdf"):
    ensure_figs_dir()

    n = 14
    alphas = np.linspace(0.0, 1.0, 11).tolist()

    seeds = list(range(30))
    rand_mat = []
    targ_mat = []

    for sd in seeds:
        game = PartyAffiliationPrefsGame.planted_instance(n, 0.55, 0.18, 0.5, 2.0, 1.0, 1000 + sd)
        s_opt, opt_val = game.compute_opt_bruteforce()

        mean_r, _, _ = estimate_psa_ratio_trials(game, alphas, False, s_opt, opt_val, 220, 2000 + sd)
        mean_t, _, _ = estimate_psa_ratio_trials(game, alphas, True, s_opt, opt_val, 220, 3000 + sd)
        rand_mat.append(mean_r)
        targ_mat.append(mean_t)

    rand_mat = np.array(rand_mat, dtype=float)
    targ_mat = np.array(targ_mat, dtype=float)

    r_med, r_q25, r_q75 = iqr_band(rand_mat)
    t_med, t_q25, t_q75 = iqr_band(targ_mat)

    plt.figure(figsize=(8.0, 4.8))
    plt.plot(alphas, r_med, marker="o", label="Random PSA (median over graph seeds)")
    plt.fill_between(alphas, r_q25, r_q75, alpha=0.2)
    plt.plot(alphas, t_med, marker="o", label="Targeted PSA (median over graph seeds)")
    plt.fill_between(alphas, t_q25, t_q75, alpha=0.2)
    plt.axhline(1.0, linestyle="--", label="OPT baseline")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Outcome / OPT")
    plt.title("Party affiliation (prefs): PSA vs α (bands = IQR over graph seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def choose_alpha_near_tipping(game, s_opt, opt_val, seed=0):
    rng = random.Random(seed)
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    best_a = 0.4
    best_var = -1.0
    for a in alphas:
        vals = []
        for _ in range(180):
            out = run_psa(game, a, True, s_opt, rng)
            vals.append(game.social_cost(out) / (opt_val + 1e-10))
        v = float(np.var(np.array(vals, dtype=float)))
        if v > best_var + EPS:
            best_var = v
            best_a = a
    return best_a

def fig_robustness_noise_temp(path="figs/robustness_noise_temp.pdf"):
    ensure_figs_dir()

    n = 60
    game = find_hard_routing_instance(n, 3, 1.15, 2, 4000, 4)
    optL, opt_val = game.compute_opt_loads()
    s_opt = game.profile_from_loads(optL)

    alpha = choose_alpha_near_tipping(game, s_opt, opt_val, 77)

    rng = random.Random(123)

    ps = [0.0, 0.02, 0.05, 0.1, 0.2]
    noise_means, noise_ses = [], []
    for p in ps:
        vals = []
        for _ in range(260):
            out = run_psa(game, alpha, True, s_opt, rng, p, 0)
            vals.append(game.social_cost(out) / (opt_val + 1e-10))
        mu, se = mean_ci(vals)
        noise_means.append(mu)
        noise_ses.append(se)

    Ts = [0, 1, 2, 4, 8, 16]
    temp_means, temp_ses = [], []
    for T in Ts:
        vals = []
        for _ in range(260):
            out = run_psa(game, alpha, True, s_opt, rng, 0.0, T)
            vals.append(game.social_cost(out) / (opt_val + 1e-10))
        mu, se = mean_ci(vals)
        temp_means.append(mu)
        temp_ses.append(se)

    fig = plt.figure(figsize=(10.5, 4.6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.errorbar(ps, noise_means, yerr=noise_ses, marker="o", capsize=4)
    ax1.axhline(1.0, linestyle="--")
    ax1.set_xlabel("Noise probability p")
    ax1.set_ylabel("Outcome / OPT")
    ax1.set_title(f"Noisy BR (α={alpha:.2f})")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.errorbar(Ts, temp_means, yerr=temp_ses, marker="o", capsize=4)
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel("Temporary PSA horizon T (sweeps frozen)")
    ax2.set_ylabel("Outcome / OPT")
    ax2.set_title(f"Temporary PSA (α={alpha:.2f})")

    fig.suptitle("Robustness (routing hard instance): noise and temporary advice", y=1.02)
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def generate_all():
    fig_routing_psa_alpha()
    fig_routing_mech_vs_psa()
    fig_robustness_noise_temp()
    fig_costsharing_psa()
    fig_party_threshold()
    fig_smooth_vs_hard_gating()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--routing_alpha", action="store_true")
    parser.add_argument("--routing_mech", action="store_true")
    parser.add_argument("--smooth_vs_hard", action="store_true")
    parser.add_argument("--costsharing", action="store_true")
    parser.add_argument("--party", action="store_true")
    parser.add_argument("--robust", action="store_true")
    args = parser.parse_args()

    if args.all:
        generate_all()
        return
    if args.routing_alpha:
        fig_routing_psa_alpha()
    if args.routing_mech:
        fig_routing_mech_vs_psa()
    if args.smooth_vs_hard:
        fig_smooth_vs_hard_gating()
    if args.costsharing:
        fig_costsharing_psa()
    if args.party:
        fig_party_threshold()
    if args.robust:
        fig_robustness_noise_temp()

if __name__ == "__main__":
    main()