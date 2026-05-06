import numpy as np
import matplotlib.pyplot as plt

L, K, B = 6.9509, 3.3919, -13.0068
SLOPE = 1.667e-7

def model(vgs, V0, VSAT):
    I_log = 10 ** (B + L / (1 + np.exp(-K * (np.minimum(vgs, VSAT) - V0))))
    I_lin = np.maximum(0, vgs - VSAT) * SLOPE
    return I_log + I_lin

vgs = np.linspace(-8, 5, 2000)

I_orig    = model(vgs, V0=0.0802,  VSAT=2.34)
I_shifted = model(vgs, V0=-4.9198, VSAT=-2.66)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("IGZO Model: Vth Shift by −5V\n"
             "Original: V0=+0.08V, VSAT=2.34V  |  Shifted: V0=−4.92V, VSAT=−2.66V",
             fontsize=11)

for ax, yscale in zip(axes, ["log", "linear"]):
    ax.plot(vgs, I_orig,    color="steelblue", linewidth=2, label="Original (Vth ≈ 0V)")
    ax.plot(vgs, I_shifted, color="tomato",    linewidth=2, linestyle="--",
            label="Shifted (Vth ≈ −5V)")
    ax.axvline(0.0802,  color="steelblue", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(-4.9198, color="tomato",    linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(2.34,    color="steelblue", linestyle="-.", linewidth=0.8, alpha=0.5,
               label="VSAT (orig=2.34V)")
    ax.axvline(-2.66,   color="tomato",    linestyle="-.", linewidth=0.8, alpha=0.5,
               label="VSAT (shifted=−2.66V)")
    ax.set_xlabel("Vgs [V]", fontsize=11)
    ax.set_ylabel("Ids [A]", fontsize=11)
    ax.set_xlim(-8, 5)
    ax.set_yscale(yscale)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_title("Log scale" if yscale == "log" else "Linear scale", fontsize=10)

    # Shade operating range of circuit (-3V to 3V)
    ax.axvspan(-3, 3, alpha=0.07, color="green", label="Circuit op. range (−3~3V)")

plt.tight_layout()
plt.savefig("igzo_vth_shift.png", dpi=150)
print("Saved: igzo_vth_shift.png")

print("\n[Circuit operating range -3V to 3V]")
for vg in [-3, -2, -1, 0, 1, 2, 3]:
    I_o = model(np.array([float(vg)]), 0.0802, 2.34)[0]
    I_s = model(np.array([float(vg)]), -4.9198, -2.66)[0]
    print(f"  Vgs={vg:+d}V:  orig={I_o:.2e} A   shifted={I_s:.2e} A")
