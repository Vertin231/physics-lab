
import numpy as np
import matplotlib.pyplot as plt

from analysis.dispersion_analysis import DispersionAnalysis


# 1. Define dispersion relation
c = 3.0

k = np.linspace(0.1, 20, 200)
omega = c * k

# 2. Analyze
analysis = DispersionAnalysis(k, omega)

# 3. Plot dispersion relation
plt.figure(figsize=(9, 5))

plt.plot(k, omega, label="ω(k) = ck")

plt.xlabel("Wavenumber k")
plt.ylabel("Angular frequency ω")
plt.title("Linear Dispersion Relation")

plt.grid()
plt.legend()
plt.show()

# 4. Plot velocities
plt.figure(figsize=(9, 5))

plt.plot(
    k,
    analysis.phase_velocity,
    label="Phase velocity"
)

plt.plot(
    k,
    analysis.group_velocity,
    "--",
    label="Group velocity"
)

plt.xlabel("Wavenumber k")
plt.ylabel("Velocity")
plt.title("Phase Velocity and Group Velocity")

plt.grid()
plt.legend()
plt.show()

print("Mean phase velocity:", np.nanmean(analysis.phase_velocity))
print("Mean group velocity:", np.mean(analysis.group_velocity))
