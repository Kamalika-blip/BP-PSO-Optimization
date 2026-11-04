import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/bae4b5e7-b790-56bd-ae52-7cfaabe445a8/scratchpad/fixed_code")
import corrected_2d_model as m

def cost_per_tx(gp, bs):
    gu = m.gas_usage(bs)
    return gp * gu / bs

def latency(gp, bs):
    t_batch = m.K_BATCH * bs
    t_confirm = m.GAMMA / gp
    return m.T_PROP + m.T_VAL + m.T_BLOCK + t_batch + t_confirm

# Optimum (from corrected_2d_results_final.csv)
GP_OPT, BS_OPT = 15.2688, 9.3079
cost_opt = cost_per_tx(GP_OPT, BS_OPT)
lat_opt = latency(GP_OPT, BS_OPT)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

bs_range = np.linspace(1, 50, 200)
gp_levels = [1, 25, 50, 75, 100]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(gp_levels)))

ax = axes[0, 0]
for gp, c in zip(gp_levels, colors):
    ax.plot(bs_range, [cost_per_tx(gp, bs) for bs in bs_range], label=f"Gp={gp} Gwei", color=c)
ax.set_xlabel("Batch Size (Bs)")
ax.set_ylabel("Cost per Transaction")
ax.set_title("Transaction Cost vs. Batch Size")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
for gp, c in zip(gp_levels, colors):
    ax.plot(bs_range, [latency(gp, bs) for bs in bs_range], label=f"Gp={gp} Gwei", color=c)
ax.set_xlabel("Batch Size (Bs)")
ax.set_ylabel("Latency (s)")
ax.set_title("Latency vs. Batch Size")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
gp_range = np.linspace(1, 100, 200)
bs_levels = [1, 10, 20, 35, 50]
colors2 = plt.cm.plasma(np.linspace(0, 0.9, len(bs_levels)))
for bs, c in zip(bs_levels, colors2):
    ax.plot(gp_range, [cost_per_tx(gp, bs) for gp in gp_range], label=f"Bs={bs}", color=c)
ax.set_xlabel("Gas Price (Gp, Gwei)")
ax.set_ylabel("Cost per Transaction")
ax.set_title("Transaction Cost vs. Gas Price")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
rng = np.random.default_rng(42)
n = 800
gp_s = rng.uniform(1, 100, n)
bs_s = rng.uniform(1, 50, n)
cost_s = np.array([cost_per_tx(gp, bs) for gp, bs in zip(gp_s, bs_s)])
lat_s = np.array([latency(gp, bs) for gp, bs in zip(gp_s, bs_s)])
sc = ax.scatter(cost_s, lat_s, c=bs_s, s=gp_s * 1.2, cmap="viridis", alpha=0.6, edgecolors="none")
ax.scatter([cost_opt], [lat_opt], color="red", marker="*", s=300, edgecolors="black",
           linewidths=0.8, label=f"BP-PSO Optimum\n(Gp={GP_OPT:.2f}, Bs={BS_OPT:.2f})", zorder=5)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Batch Size")
ax.set_xlabel("Cost per Transaction")
ax.set_ylabel("Latency (s)")
ax.set_title("Cost-Latency Trade-off")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

fig.suptitle("Cost-Latency Analysis", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/tmp/claude-0/-home-claude/bae4b5e7-b790-56bd-ae52-7cfaabe445a8/scratchpad/fixed_code/figure8_cost_latency_analysis.png"
fig.savefig(out, dpi=150)
print("saved", out)
print("cost_opt", cost_opt, "lat_opt", lat_opt)
