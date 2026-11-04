"""
Section 6.4 scenario-based adaptability analysis (self-contained, seeded).

Model: paper Eqs. 13-22 with Table 6 constants.
  Cost_norm(X)    = Gp * Gu(Bs) / (Bs * Cost_max),  Gu(Bs) = 21000 + 6000*Bs
  Latency(X)      = T_prop + T_val + T_block + k*Bs + gamma/Gp
  Latency_norm(X) = Latency / Latency_max
  F(X)            = alpha*Cost_norm + beta*Latency_norm
Normalisation constants (Cost_max, Latency_max) are FIXED at their Table 6 /
base-case values for every scenario so that fitness values are comparable.

Scenarios are defined from gamma (congestion coefficient) and (alpha, beta);
gas price Gp and batch size Bs are both free decision variables.
"""
import csv
import numpy as np

# ---- Table 6 constants -------------------------------------------------
T_PROP, T_VAL, T_BLOCK = 0.2736, 1.0212, 19.9124
GAMMA0 = 69.17
K_BATCH = 0.133
GP_BOUNDS = (1.0, 100.0)
BS_BOUNDS = (1.0, 50.0)
BOUNDS = np.array([GP_BOUNDS, BS_BOUNDS])

COST_MAX = GP_BOUNDS[1] * (21000.0 + 6000.0 * BS_BOUNDS[0]) / BS_BOUNDS[0]
LATENCY_MAX = T_PROP + T_VAL + T_BLOCK + K_BATCH * BS_BOUNDS[1] + GAMMA0 / GP_BOUNDS[0]

# ---- PSO settings (Table 5) --------------------------------------------
N_PART, N_ITER, W, C1, C2 = 30, 100, 0.7, 1.5, 1.5
N_SEEDS = 30
SEEDS = list(range(2000, 2000 + N_SEEDS))
CONV_TOL = 1e-6

BASE_GP, BASE_BS = 20.0, 1.0          # fixed unoptimised baseline (Sec. 5.3)


def gas_usage(bs):
    return 21000.0 + 6000.0 * bs


def cost_per_tx(gp, bs):
    return gp * gas_usage(bs) / bs


def latency(gp, bs, gamma):
    return T_PROP + T_VAL + T_BLOCK + K_BATCH * bs + gamma / gp


def fitness(x, states, alpha, beta):
    """states: list of (gamma, probability). Expected fitness over states."""
    gp, bs = x
    cn = gp * gas_usage(bs) / (bs * COST_MAX)
    f = 0.0
    for gamma, p in states:
        f += p * (alpha * cn + beta * latency(gp, bs, gamma) / LATENCY_MAX)
    return f


def run_pso(states, alpha, beta, seed):
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    pos = rng.uniform(lo, hi, size=(N_PART, 2))
    vel = np.zeros((N_PART, 2))
    pbest = pos.copy()
    pfit = np.array([fitness(p, states, alpha, beta) for p in pos])
    g = int(np.argmin(pfit))
    gbest, gfit = pbest[g].copy(), pfit[g]
    hist = [gfit]
    for _ in range(N_ITER):
        r1, r2 = rng.random((N_PART, 2)), rng.random((N_PART, 2))
        vel = W * vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)
        for i in range(N_PART):
            f = fitness(pos[i], states, alpha, beta)
            if f < pfit[i]:
                pfit[i], pbest[i] = f, pos[i].copy()
        g = int(np.argmin(pfit))
        if pfit[g] < gfit:
            gfit, gbest = pfit[g], pbest[g].copy()
        hist.append(gfit)
    conv = next((i for i in range(1, len(hist)) if abs(hist[i] - hist[i - 1]) < CONV_TOL),
                len(hist) - 1)
    return gbest, gfit, conv


def grid_opt(states, alpha, beta, n=600):
    gps = np.linspace(*GP_BOUNDS, n)
    bss = np.linspace(*BS_BOUNDS, n)
    G, B = np.meshgrid(gps, bss, indexing="ij")
    cn = G * gas_usage(B) / (B * COST_MAX)
    F = np.zeros_like(G)
    for gamma, p in states:
        F += p * (alpha * cn + beta * (T_PROP + T_VAL + T_BLOCK + K_BATCH * B + gamma / G) / LATENCY_MAX)
    i = np.unravel_index(np.argmin(F), F.shape)
    return gps[i[0]], bss[i[1]], F[i]


# ---- scenario definitions (fixed a priori) ------------------------------
G_LOW, G_HIGH = 0.5 * GAMMA0, 2.0 * GAMMA0
SCENARIOS = [
    # name, states [(gamma, prob)], alpha, beta
    ("Reference (gamma0)",            [(GAMMA0, 1.0)],                 0.5, 0.5),
    ("A: Low congestion",             [(G_LOW, 1.0)],                  0.5, 0.5),
    ("B: High congestion",            [(G_HIGH, 1.0)],                 0.5, 0.5),
    ("C: Volatile (50/50 low/high)",  [(G_LOW, 0.5), (G_HIGH, 0.5)],   0.5, 0.5),
    ("D: Cost-priority (a=0.8)",      [(GAMMA0, 1.0)],                 0.8, 0.2),
    ("E: Latency-priority (a=0.2)",   [(GAMMA0, 1.0)],                 0.2, 0.8),
]


def summarise(name, states, alpha, beta):
    res = [run_pso(states, alpha, beta, s) for s in SEEDS]
    gp = np.array([r[0][0] for r in res])
    bs = np.array([r[0][1] for r in res])
    fit = np.array([r[1] for r in res])
    conv = np.array([r[2] for r in res])
    gp_m, bs_m = gp.mean(), bs.mean()
    # expected latency / baseline comparison over the scenario's states
    lat_opt = sum(p * latency(gp_m, bs_m, g) for g, p in states)
    lat_base = sum(p * latency(BASE_GP, BASE_BS, g) for g, p in states)
    c_opt, c_base = cost_per_tx(gp_m, bs_m), cost_per_tx(BASE_GP, BASE_BS)
    ggp, gbs, gf = grid_opt(states, alpha, beta)
    return dict(
        scenario=name, alpha=alpha, beta=beta,
        gamma=" / ".join(f"{g:.2f}" for g, _ in states),
        Gp_mean=gp_m, Gp_std=gp.std(ddof=1), Bs_mean=bs_m, Bs_std=bs.std(ddof=1),
        fitness_mean=fit.mean(), fitness_std=fit.std(ddof=1), conv_mean=conv.mean(),
        cost_per_tx=c_opt, latency=lat_opt,
        cost_reduction_pct=(c_base - c_opt) / c_base * 100,
        latency_change_pct=(lat_opt - lat_base) / lat_base * 100,
        grid_Gp=ggp, grid_Bs=gbs, grid_F=gf,
    )


def main():
    rows = [summarise(*sc) for sc in SCENARIOS]
    # adaptation gain: fitness of a STATIC config tuned for gamma0 vs re-optimised config
    ref = next(r for r in rows if r["scenario"].startswith("Reference"))
    static = np.array([ref["Gp_mean"], ref["Bs_mean"]])
    gains = []
    for name, states, a, b in SCENARIOS:
        f_static = fitness(static, states, a, b)
        r = next(x for x in rows if x["scenario"] == name)
        f_adapt = fitness(np.array([r["Gp_mean"], r["Bs_mean"]]), states, a, b)
        gains.append((name, f_static, f_adapt, (f_static - f_adapt) / f_static * 100))

    print(f"LATENCY_MAX={LATENCY_MAX:.4f}  COST_MAX={COST_MAX:,.0f}  gamma_low={G_LOW:.3f}  gamma_high={G_HIGH:.3f}")
    for r in rows:
        print(f"{r['scenario']:32s} gamma={r['gamma']:14s} a/b={r['alpha']}/{r['beta']} "
              f"Gp*={r['Gp_mean']:.3f}±{r['Gp_std']:.3f} Bs*={r['Bs_mean']:.3f}±{r['Bs_std']:.3f} "
              f"F={r['fitness_mean']:.6f}±{r['fitness_std']:.6f} conv={r['conv_mean']:.1f} | "
              f"cost/tx={r['cost_per_tx']:,.0f} lat={r['latency']:.2f}s "
              f"dCost={r['cost_reduction_pct']:.2f}% dLat={r['latency_change_pct']:+.2f}% | "
              f"grid=({r['grid_Gp']:.2f},{r['grid_Bs']:.2f},F={r['grid_F']:.6f})")
    print("\nAdaptation gain vs static reference configuration "
          f"(Gp={static[0]:.2f}, Bs={static[1]:.2f}):")
    for n, fs, fa, gpct in gains:
        print(f"  {n:32s} F_static={fs:.6f}  F_adapted={fa:.6f}  gain={gpct:.2f}%")

    out = "/tmp/claude-0/-home-claude/bae4b5e7-b790-56bd-ae52-7cfaabe445a8/scratchpad/fixed_code/scenario_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", out)
    return rows, gains


if __name__ == "__main__":
    main()
