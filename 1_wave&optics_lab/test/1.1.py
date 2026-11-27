import numpy as np
import matplotlib.pyplot as plt

from waves.models.wave_packect import WavePacket
from analysis.wave_analysis import WaveAnalysis


# 1. Create a wave packet
packet = WavePacket(
    amplitude=1.0,
    wave_number=10 * np.pi,
    angular_frequency=20 * np.pi,
    group_velocity=0.5,
    sigma=0.5,
    x0=-2.0
)

# 2. Generate coordinates
x = np.linspace(-5, 5, 1000)
t = np.linspace(0, 6, 300)

# 3. Analyze
analysis = WaveAnalysis(packet, x, t)

# 4. Plot the wave packet at several moments
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

for time in [0, 1, 2, 3]:
    index = np.argmin(np.abs(t - time))

    axes[0].plot(
        x,
        analysis.snapshot(index),
        label=f"t = {t[index]:.1f}"
    )

axes[0].set_title("Wave Packet Evolution")
axes[0].set_xlabel("Position x")
axes[0].set_ylabel("Displacement y")
axes[0].legend()
axes[0].grid()

# 5. Analyze the packet center
axes[1].plot(t, analysis.center, label="Measured center")

theoretical_center = (
    packet.x0 + packet.group_velocity * t
)

axes[1].plot(
    t,
    theoretical_center,
    "--",
    label="Theoretical center"
)

axes[1].set_title("Wave Packet Center")
axes[1].set_xlabel("Time t")
axes[1].set_ylabel("Position")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()


# 6. Print some analysis results
print("Maximum amplitude:", analysis.max_amplitude[:5])
print("RMS amplitude:", analysis.rms_amplitude[:5])
print("Packet center:", analysis.center[:5])
print("Packet width:", analysis.width[:5])
print("Center velocity:", analysis.center_velocity[:5])