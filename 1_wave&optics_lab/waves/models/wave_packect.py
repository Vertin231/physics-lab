
import numpy as np


class WavePacket:
    """A one-dimensional Gaussian-modulated wave packet."""

    def __init__(
        self,
        amplitude,
        wave_number,
        angular_frequency,
        group_velocity,
        sigma,
        x0=0.0,
        phase=0.0
    ):
        self.amplitude = float(amplitude)
        self.wave_number = float(wave_number)
        self.angular_frequency = float(angular_frequency)
        self.group_velocity = float(group_velocity)
        self.sigma = float(sigma)
        self.x0 = float(x0)
        self.phase = float(phase)

        self._validate()

    def _validate(self):
        parameters = [
            self.amplitude,
            self.wave_number,
            self.angular_frequency,
            self.group_velocity,
            self.sigma,
            self.x0,
            self.phase
        ]

        if not np.all(np.isfinite(parameters)):
            raise ValueError("All parameters must be finite.")

        if self.wave_number <= 0:
            raise ValueError("Wave number must be positive.")

        if self.angular_frequency <= 0:
            raise ValueError("Angular frequency must be positive.")

        if self.sigma <= 0:
            raise ValueError("Sigma must be positive.")

    @property
    def phase_velocity(self):
        return self.angular_frequency / self.wave_number

    @property
    def envelope_velocity(self):
        return self.group_velocity

    @property
    def wavelength(self):
        return 2 * np.pi / self.wave_number

    @property
    def frequency(self):
        return self.angular_frequency / (2 * np.pi)

    def envelope(self, x, t):
        """Return the Gaussian envelope."""
        center = self.x0 + self.group_velocity * t

        return self.amplitude * np.exp(
            -(x - center) ** 2 / (2 * self.sigma ** 2)
        )

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)

        if x.ndim > 1 or t.ndim > 1:
            raise ValueError(
                "x and t must be scalars or one-dimensional arrays."
            )

        scalar_input = x.ndim == 0 and t.ndim == 0

        x = np.atleast_1d(x)
        t = np.atleast_1d(t)

        envelope = self.envelope(
            x[None, :],
            t[:, None]
        )

        carrier = np.cos(
            self.wave_number * x[None, :]
            - self.angular_frequency * t[:, None]
            + self.phase
        )

        y = envelope * carrier

        return y[0, 0] if scalar_input else y

    def __repr__(self):
        return (
            f"WavePacket(amplitude={self.amplitude}, "
            f"wave_number={self.wave_number}, "
            f"angular_frequency={self.angular_frequency}, "
            f"group_velocity={self.group_velocity}, "
            f"sigma={self.sigma})"
        )
