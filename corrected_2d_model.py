"""
corrected_2d_model.py
----------------------
Path A fix: implements the paper's ACTUAL declared model (Section 4.4,
Eq. 13-21), which prior code never fully implemented.

Decision vector: X = [Gp, Bs]  (D = 2, matches Eq. 13 / Table 5 bounds)

Cost_norm(X)    = Gp * Gu(Bs) / (Bs * Cost_max)      [Eq. 14]
  Gu(Bs)        = 21000 + 6000*Bs   (total batch gas; matches
                    pso_blockchain_batch_optimization.py and Eq. 14's
                    division by Bs -- resolves the "Gu constant" vs
                    "Gu depends on Bs" contradiction in favor of the
                    equation and code, since the prose was the odd one out)
  Cost_max      = Gp_max * Gu(Bs_min) / Bs_min = 100 * 27000 / 1 = 2,700,000
                    (matches the constant already used in optimization_combined.py)

Latency(X)      = T_prop + T_val + T_block + T_batch(Bs) + T_confirm(Gp)  [Eq. 16]
  T_confirm(Gp) = gamma / Gp                                              [Eq. 17]
  T_batch(Bs)   = k * Bs

Fixed environmental parameters (T_prop, T_val, T_block) and the
congestion coefficient (gamma) are NOT arbitrary -- they are derived
from the already-validated blockchain_simulation_calibrated_N1000.csv
(the same file that exactly reproduces Table 7), so this script is
consistent with the rest of the paper's empirical calibration instead
of inventing new constants:

  T_prop  = 0.2736 s  (dataset column mean)
  T_val   = 1.0212 s  (dataset column mean)
  T_block = 19.9124 s (dataset column mean)
  gamma   = 69.17     (mean of T_confirm * gas_price across the dataset,
                        i.e. solving T_confirm = gamma/Gp for gamma)
  k       = 0.133 s/tx (chosen so T_batch(Bs=6) ~= 0.8s, matching the
                        dataset's T_batch column mean at the paper's
                        previously-reported optimal batch size)

This is the single script that should now back Section 6.3 (Figure 8)
and Section 6.5 (GA vs BP-PSO), replacing both the old 2D script that
silently dropped T_confirm(Gp) and the old 5D script used only for the
GA comparison.
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ======================================================================
# CALIBRATED CONSTANTS (derived from blockchain_simulation_calibrated_N1000.csv)
# ======================================================================

T_PROP = 0.2736
T_VAL = 1.0212
T_BLOCK = 19.9124
GAMMA = 69.17          # congestion coefficient, Eq. 17
K_BATCH = 0.133         # s per transaction, T_batch(Bs) = K_BATCH * Bs

GAS_PRICE_BOUNDS = (1.0, 100.0)
BATCH_SIZE_BOUNDS = (1.0, 50.0)
BOUNDS = [GAS_PRICE_BOUNDS, BATCH_SIZE_BOUNDS]
DIMENSIONS = 2

ALPHA = 0.5
BETA = 0.5

COST_MAX = GAS_PRICE_BOUNDS[1] * (21000.0 + 6000.0 * BATCH_SIZE_BOUNDS[0]) / BATCH_SIZE_BOUNDS[0]  # 2,700,000
LATENCY_MAX = T_PROP + T_VAL + T_BLOCK + K_BATCH * BATCH_SIZE_BOUNDS[1] + GAMMA / GAS_PRICE_BOUNDS[0]

NUM_PARTICLES = 30
MAX_ITERATIONS = 100
W, C1, C2 = 0.7, 1.5, 1.5

POP_SIZE = 30
N_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

PAPER_SEED = 42
N_STAT_RUNS = 30
CONVERGENCE_TOL = 1e-6


def gas_usage(bs):
    return 21000.0 + 6000.0 * bs


def fitness_function(position):
    gp, bs = position
    gu = gas_usage(bs)

    cost = gp * gu
    cost_norm = cost / (bs * COST_MAX)

    t_confirm = GAMMA / gp
    t_batch = K_BATCH * bs
    latency = T_PROP + T_VAL + T_BLOCK + t_batch + t_confirm
    latency_norm = latency / LATENCY_MAX

    fitness = ALPHA * cost_norm + BETA * latency_norm
    return fitness, cost, latency


def detect_convergence_iteration(history, tol=CONVERGENCE_TOL):
    final = history[-1]
    for i in range(len(history)):
        if all(abs(history[j] - final) <= tol for j in range(i, len(history))):
            return i + 1
    return len(history)


# ======================================================================
# PSO (real, seeded, no fabrication)
# ======================================================================

class Particle:
    def __init__(self):
        self.position = np.array([np.random.uniform(lo, hi) for lo, hi in BOUNDS])
        self.velocity = np.array(
            [np.random.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1) for lo, hi in BOUNDS]
        )
        self.best_position = self.position.copy()
        self.best_fitness = float("inf")


def run_pso(seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    swarm = [Particle() for _ in range(NUM_PARTICLES)]
    global_best_position = None
    global_best_fitness = float("inf")

    initial_fitnesses = []
    for particle in swarm:
        fit, *_ = fitness_function(particle.position)
        particle.best_fitness = fit
        particle.best_position = particle.position.copy()
        initial_fitnesses.append(fit)
        if fit < global_best_fitness:
            global_best_fitness = fit
            global_best_position = particle.position.copy()

    f_worst = max(initial_fitnesses)
    worst_position = swarm[int(np.argmax(initial_fitnesses))].position.copy()

    history = [global_best_fitness]

    for _ in range(1, MAX_ITERATIONS + 1):
        for particle in swarm:
            r1 = np.random.rand(DIMENSIONS)
            r2 = np.random.rand(DIMENSIONS)
            inertia = W * particle.velocity
            cognitive = C1 * r1 * (particle.best_position - particle.position)
            social = C2 * r2 * (global_best_position - particle.position)
            particle.velocity = inertia + cognitive + social
            particle.position = particle.position + particle.velocity

            for d, (lo, hi) in enumerate(BOUNDS):
                particle.position[d] = np.clip(particle.position[d], lo, hi)

            fit, *_ = fitness_function(particle.position)
            if fit < particle.best_fitness:
                particle.best_fitness = fit
                particle.best_position = particle.position.copy()

        min_idx = int(np.argmin([p.best_fitness for p in swarm]))
        if swarm[min_idx].best_fitness < global_best_fitness:
            global_best_fitness = swarm[min_idx].best_fitness
            global_best_position = swarm[min_idx].best_position.copy()

        history.append(global_best_fitness)

    return global_best_position, global_best_fitness, history, f_worst, worst_position


# ======================================================================
# GA (real, seeded, same 2D model)
# ======================================================================

def run_ga(seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    pop = np.array([[np.random.uniform(lo, hi) for lo, hi in BOUNDS] for _ in range(POP_SIZE)])

    best_fitness = float("inf")
    best_ind = None
    history = []

    initial_fitnesses = [fitness_function(ind)[0] for ind in pop]
    f_worst = max(initial_fitnesses)
    worst_position = pop[int(np.argmax(initial_fitnesses))].copy()

    for gen in range(1, N_GENERATIONS + 1):
        fitnesses = np.array([fitness_function(ind)[0] for ind in pop])

        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_fitness:
            best_fitness = fitnesses[min_idx]
            best_ind = pop[min_idx].copy()
        history.append(best_fitness)

        new_pop = []
        for _ in range(POP_SIZE):
            i1, i2 = np.random.randint(0, POP_SIZE, 2)
            winner = pop[i1] if fitnesses[i1] < fitnesses[i2] else pop[i2]
            new_pop.append(winner.copy())

        for i in range(0, POP_SIZE, 2):
            if i + 1 < POP_SIZE and np.random.rand() < CROSSOVER_RATE:
                r = np.random.rand(DIMENSIONS)
                child1 = r * new_pop[i] + (1 - r) * new_pop[i + 1]
                child2 = (1 - r) * new_pop[i] + r * new_pop[i + 1]
                new_pop[i], new_pop[i + 1] = child1, child2

        for i in range(POP_SIZE):
            if np.random.rand() < MUTATION_RATE:
                dim = np.random.randint(0, DIMENSIONS)
                new_pop[i][dim] = np.random.uniform(BOUNDS[dim][0], BOUNDS[dim][1])

        new_pop[np.random.randint(0, POP_SIZE)] = best_ind.copy()
        pop = np.array(new_pop)

        for d, (lo, hi) in enumerate(BOUNDS):
            pop[:, d] = np.clip(pop[:, d], lo, hi)

    return best_ind, best_fitness, history, f_worst, worst_position


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 72)
    print("  CORRECTED 2D BP-PSO MODEL  (Eq. 13-21, with real T_confirm(Gp)=gamma/Gp)")
    print("=" * 72)
    print(f"  T_prop={T_PROP}  T_val={T_VAL}  T_block={T_BLOCK}  gamma={GAMMA}  k={K_BATCH}")
    print(f"  Cost_max={COST_MAX:,.0f}  Latency_max={LATENCY_MAX:.4f}s")

    pso_pos, pso_fit, pso_hist, pso_fworst, pso_worst_pos = run_pso(seed=PAPER_SEED)
    ga_pos, ga_fit, ga_hist, ga_fworst, ga_worst_pos = run_ga(seed=PAPER_SEED)

    pso_conv_iter = detect_convergence_iteration(pso_hist)
    ga_conv_iter = detect_convergence_iteration(ga_hist)

    _, pso_cost, pso_lat = fitness_function(pso_pos)
    _, ga_cost, ga_lat = fitness_function(ga_pos)
    _, pso_worst_cost, pso_worst_lat = fitness_function(pso_worst_pos)
    _, ga_worst_cost, ga_worst_lat = fitness_function(ga_worst_pos)

    pso_cost_reduction = (pso_worst_cost - pso_cost) / pso_worst_cost * 100
    pso_latency_reduction = (pso_worst_lat - pso_lat) / pso_worst_lat * 100
    ga_cost_reduction = (ga_worst_cost - ga_cost) / ga_worst_cost * 100
    ga_latency_reduction = (ga_worst_lat - ga_lat) / ga_worst_lat * 100

    print(f"\n[Seed={PAPER_SEED} single run]")
    print(f"{'Metric':<26} {'GA':>15} {'BP-PSO':>15}")
    print("-" * 58)
    print(f"{'Gas Price (Gwei)':<26} {ga_pos[0]:>15.4f} {pso_pos[0]:>15.4f}")
    print(f"{'Batch Size':<26} {ga_pos[1]:>15.4f} {pso_pos[1]:>15.4f}")
    print(f"{'Latency (s)':<26} {ga_lat:>15.4f} {pso_lat:>15.4f}")
    print(f"{'Best Fitness':<26} {ga_fit:>15.6f} {pso_fit:>15.6f}")
    print(f"{'Cost Reduction (%)':<26} {ga_cost_reduction:>15.2f} {pso_cost_reduction:>15.2f}")
    print(f"{'Latency Reduction (%)':<26} {ga_latency_reduction:>15.2f} {pso_latency_reduction:>15.2f}")
    print(f"{'Convergence Iteration':<26} {ga_conv_iter:>15} {pso_conv_iter:>15}")

    print(f"\n[Statistical comparison over {N_STAT_RUNS} independent seeds]")
    pso_fits, ga_fits = [], []
    for run_idx in range(N_STAT_RUNS):
        _, pf, _, _, _ = run_pso(seed=2000 + run_idx)
        _, gf, _, _, _ = run_ga(seed=2000 + run_idx)
        pso_fits.append(pf)
        ga_fits.append(gf)
    pso_fits = np.array(pso_fits)
    ga_fits = np.array(ga_fits)
    stat, p_value = mannwhitneyu(pso_fits, ga_fits, alternative="two-sided")

    print(f"{'':<26} {'GA':>15} {'BP-PSO':>15}")
    print("-" * 58)
    print(f"{'Mean Best Fitness':<26} {ga_fits.mean():>15.6f} {pso_fits.mean():>15.6f}")
    print(f"{'Std Dev':<26} {ga_fits.std():>15.6f} {pso_fits.std():>15.6f}")
    print(f"\nMann-Whitney U test: p={p_value:.4f}")

    # Real convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pso_hist) + 1), pso_hist, label="BP-PSO (2D, real T_confirm)", color="royalblue", linewidth=2)
    plt.plot(range(1, len(ga_hist) + 1), ga_hist, label="GA (2D, real T_confirm)", color="tomato", linewidth=2, linestyle="--")
    plt.title("Corrected 2D Model Convergence: BP-PSO vs GA", fontweight="bold")
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best Fitness F(X)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("corrected_2d_convergence.png", dpi=200)
    print("\n[INFO] Saved corrected_2d_convergence.png")

    # Cost-latency trade-off sanity plot (replaces Figure 8)
    gps = np.linspace(1, 100, 60)
    bss = np.linspace(1, 50, 60)
    GP, BS = np.meshgrid(gps, bss)
    LAT = np.zeros_like(GP)
    for i in range(GP.shape[0]):
        for j in range(GP.shape[1]):
            _, _, LAT[i, j] = fitness_function([GP[i, j], BS[i, j]])
    plt.figure(figsize=(8, 6))
    cs = plt.contourf(GP, BS, LAT, levels=30, cmap="viridis")
    plt.colorbar(cs, label="Latency (s)")
    plt.scatter([pso_pos[0]], [pso_pos[1]], color="red", marker="*", s=250, label="BP-PSO gBest")
    plt.xlabel("Gas Price (Gwei)")
    plt.ylabel("Batch Size")
    plt.title("Latency surface now DOES depend on Gas Price (real T_confirm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("corrected_2d_latency_surface.png", dpi=200)
    print("[INFO] Saved corrected_2d_latency_surface.png (shows real Gp-latency coupling)")

    with open("corrected_2d_results.csv", "w") as f:
        f.write("Metric,GA,BP-PSO\n")
        f.write(f"Gas Price (Gwei),{ga_pos[0]:.4f},{pso_pos[0]:.4f}\n")
        f.write(f"Batch Size,{ga_pos[1]:.4f},{pso_pos[1]:.4f}\n")
        f.write(f"Latency (s),{ga_lat:.4f},{pso_lat:.4f}\n")
        f.write(f"Best Fitness (seed=42),{ga_fit:.6f},{pso_fit:.6f}\n")
        f.write(f"Mean Best Fitness (30 runs),{ga_fits.mean():.6f},{pso_fits.mean():.6f}\n")
        f.write(f"Std Dev (30 runs),{ga_fits.std():.6f},{pso_fits.std():.6f}\n")
        f.write(f"Cost Reduction (%),{ga_cost_reduction:.2f},{pso_cost_reduction:.2f}\n")
        f.write(f"Latency Reduction (%),{ga_latency_reduction:.2f},{pso_latency_reduction:.2f}\n")
        f.write(f"Convergence Iteration (seed=42),{ga_conv_iter},{pso_conv_iter}\n")
        f.write(f"Mann-Whitney U p-value,{p_value:.4f},{p_value:.4f}\n")
    print("[INFO] Saved corrected_2d_results.csv")


if __name__ == "__main__":
    main()
