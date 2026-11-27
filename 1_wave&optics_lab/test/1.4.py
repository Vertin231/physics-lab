
import numpy as np
import matplotlib.pyplot as plt

from analysis.dispersion_analysis import DispersionAnalysis


# 1. Physical parameters
g = 9.81

# 2. Theoretical dispersion relation
k = np.linspace(0.1, 20, 500)
omega = np.sqrt(g * k)

# 3. Analyze
analysis = DispersionAnalysis(k, omega)

# 4. Theoretical velocities
vp_theory = np.sqrt(g / k)
vg_theory = 0.5 * np.sqrt(g / k)

# 5. Plot dispersion relation
fig, axes = plt.subplots(2, 1, figsize=(9, 9))

axes[0].plot(k, omega, label="Numerical dispersion data")
axes[0].set_xlabel("Wavenumber k")
axes[0].set_ylabel("Angular frequency ω")
axes[0].set_title("Deep-water Gravity Wave")
axes[0].grid()
axes[0].legend()

# 6. Compare numerical and theoretical velocities
axes[1].plot(
    k,
    analysis.phase_velocity,
    label="Numerical phase velocity"
)

axes[1].plot(
    k,
    analysis.group_velocity,
    label="Numerical group velocity"
)

axes[1].plot(
    k,
    vp_theory,
    "--",
    label="Theoretical phase velocity"
)

axes[1].plot(
    k,
    vg_theory,
    "--",
    label="Theoretical group velocity"
)

axes[1].set_xlabel("Wavenumber k")
axes[1].set_ylabel("Velocity")
axes[1].set_title("Phase Velocity vs Group Velocity")
axes[1].grid()
axes[1].legend()

plt.tight_layout()
plt.show()


# 7. Evaluate at a particular wave number
result = analysis.evaluate(5.0)

for name, value in result.items():
    print(f"{name}: {value}")
