
import numpy as np


class StandingWave:
    """A one-dimensional standing wave."""

    def __init__(
        self,
        amplitude,
        wave_number,
        angular_frequency,
        phase=0.0
    ):
        self.amplitude = float(amplitude)
        self.wave_number = float(wave_number)
        self.angular_frequency = float(angular_frequency)
        self.phase = float(phase)

        self._validate()

    def _validate(self):
        parameters = [
            self.amplitude,
            self.wave_number,
            self.angular_frequency,
            self.phase
        ]

        if not np.all(np.isfinite(parameters)):
            raise ValueError("All parameters must be finite.")

        if self.wave_number <= 0:
            raise ValueError("Wave number must be positive.")

        if self.angular_frequency <= 0:
            raise ValueError("Angular frequency must be positive.")

    @property
    def frequency(self):
        return self.angular_frequency / (2 * np.pi)

    @property
    def wavelength(self):
        return 2 * np.pi / self.wave_number

    @property
    def phase_velocity(self):
        return self.angular_frequency / self.wave_number

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

        y = (
            2 * self.amplitude
            * np.sin(self.wave_number * x[None, :] + self.phase)
            * np.cos(self.angular_frequency * t[:, None])
        )

        return y[0, 0] if scalar_input else y

    def __repr__(self):
        return (
            f"StandingWave(amplitude={self.amplitude}, "
            f"wave_number={self.wave_number}, "
            f"angular_frequency={self.angular_frequency})"
        )
