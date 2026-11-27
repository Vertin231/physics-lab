
import numpy as np
import matplotlib.pyplot as plt

from waves.models.wave_packect import WavePacket
from analysis.spectral_analysis import SpectralAnalysis


# 1. Create a wave packet
packet = WavePacket(
    amplitude=1.0,
    wave_number=10 * np.pi,
    angular_frequency=20 * np.pi,
    group_velocity=0.5,
    sigma=0.5,
    x0=0.0
)

# 2. Generate coordinates
x = np.linspace(-10, 10, 2048)
t = np.linspace(0, 5, 1000)

# 3. Spectral analysis
analysis = SpectralAnalysis(
    packet,
    x,
    t,
    window="hann"
)

# 4. Calculate spatial spectrum at t = 0
k, amplitude = analysis.spatial_spectrum(0)

# 5. Plot
plt.figure(figsize=(10, 5))

plt.plot(k, amplitude)

plt.xlabel("Wavenumber k")
plt.ylabel("Amplitude")
plt.title("Spatial Spectrum of a Gaussian Wave Packet")

plt.grid()
plt.tight_layout()
plt.show()

print(
    "Dominant wavenumber:",
    analysis.dominant_wavenumber(0)
)

print(
    "Theoretical carrier wavenumber:",
    packet.wave_number
)


# Choose a position where the packet passes
position_index = np.argmin(np.abs(x - 0.5))

omega, temporal_amplitude = analysis.temporal_spectrum(
    position_index
)

plt.figure(figsize=(10, 5))

plt.plot(omega, temporal_amplitude)

plt.xlabel("Angular frequency ω")
plt.ylabel("Amplitude")
plt.title("Temporal Spectrum of a Gaussian Wave Packet")

plt.grid()
plt.tight_layout()
plt.show()

print(
    "Dominant angular frequency:",
    analysis.dominant_angular_frequency(position_index)
)

print(
    "Theoretical angular frequency:",
    packet.angular_frequency
)