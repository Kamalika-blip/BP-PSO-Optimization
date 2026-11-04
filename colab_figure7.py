# ============================================================
# Reproduces the BP-PSO parameter-evolution plot (Figure 6/7):
# gBest Gas Price and gBest Batch Size vs. iteration, using the
# corrected 2D model with T_confirm(Gp) = gamma / Gp (Eq. 16-17).
# Paste this whole cell into Google Colab and run.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Fixed environmental parameters (calibrated) ----------------
# Simulated-dataset constants (from blockchain_simulation_calibrated_N1000.csv)
T_PROP  = 0.2736
T_VAL   = 1.0212
T_BLOCK = 19.9124
GAMMA   = 69.17       # congestion coefficient, Eq. 17
K_BATCH = 0.133        # seconds per transaction, T_batch(Bs) = K_BATCH * Bs

# ---------------- Search space ----------------
GAS_PRICE_BOUNDS  = (1.0, 100.0)
BATCH_SIZE_BOUNDS = (1.0, 50.0)
BOUNDS = [GAS_PRICE_BOUNDS, BATCH_SIZE_BOUNDS]
DIMENSIONS = 2

# ---------------- Fitness weights ----------------
ALPHA, BETA = 0.5, 0.5

COST_MAX = GAS_PRICE_BOUNDS[1] * (21000.0 + 6000.0 * BATCH_SIZE_BOUNDS[0]) / BATCH_SIZE_BOUNDS[0]
LATENCY_MAX = (T_PROP + T_VAL + T_BLOCK
               + K_BATCH * BATCH_SIZE_BOUNDS[1]
               + GAMMA / GAS_PRICE_BOUNDS[0])

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

# ---------------- PSO hyperparameters (Table 5) ----------------
NUM_PARTICLES  = 30
MAX_ITERATIONS = 100
W  = 0.7
C1 = 1.5
C2 = 1.5

CONVERGENCE_TOL = 1e-6
PAPER_SEED = 42

class Particle:
    def __init__(self):
        self.position = np.array([np.random.uniform(lo, hi) for lo, hi in BOUNDS])
        self.velocity = np.zeros(DIMENSIONS)
        self.best_position = self.position.copy()
        self.best_fitness = float("inf")

def detect_convergence_iteration(history, tol=CONVERGENCE_TOL):
    for i in range(1, len(history)):
        if abs(history[i] - history[i - 1]) < tol:
            return i
    return len(history) - 1

def run_pso_with_history(seed=None):
    if seed is not None:
        np.random.seed(seed)

    swarm = [Particle() for _ in range(NUM_PARTICLES)]
    global_best_position = None
    global_best_fitness = float("inf")

    for particle in swarm:
        fit, *_ = fitness_function(particle.position)
        particle.best_fitness = fit
        particle.best_position = particle.position.copy()
        if fit < global_best_fitness:
            global_best_fitness = fit
            global_best_position = particle.position.copy()

    gp_history = [global_best_position[0]]
    bs_history = [global_best_position[1]]
    fitness_history = [global_best_fitness]

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

        gp_history.append(global_best_position[0])
        bs_history.append(global_best_position[1])
        fitness_history.append(global_best_fitness)

    return gp_history, bs_history, fitness_history

# ---------------- Run and plot ----------------
gp_hist, bs_hist, fit_hist = run_pso_with_history(seed=PAPER_SEED)
conv_iter = detect_convergence_iteration(fit_hist)

print(f"Initial Gp = {gp_hist[0]:.4f} Gwei  ->  Final Gp = {gp_hist[-1]:.4f} Gwei")
print(f"Initial Bs = {bs_hist[0]:.4f}       ->  Final Bs = {bs_hist[-1]:.4f}")
print(f"Convergence iteration: {conv_iter}")
print(f"Final best fitness: {fit_hist[-1]:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(gp_hist, color="tab:blue", linewidth=1.8)
axes[0].axvline(conv_iter, color="gray", linestyle="--", linewidth=1,
                label=f"Convergence (iter {conv_iter})")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("gBest Gas Price (Gwei)")
axes[0].set_title("Evolution of Gas Price")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(bs_hist, color="tab:orange", linewidth=1.8)
axes[1].axvline(conv_iter, color="gray", linestyle="--", linewidth=1,
                label=f"Convergence (iter {conv_iter})")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("gBest Batch Size")
axes[1].set_title("Evolution of Batch Size")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("Evolution of Gas and Batch Parameters under BP-PSO", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figure7_parameter_evolution.png", dpi=150)
plt.show()
