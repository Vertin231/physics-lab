import numpy as np


class HarmonicWave:
    """A one-dimensional wave represented by a finite harmonic series."""

    def __init__(self, frequency, wave_number, amplitude, phase=None):
        self.frequency = float(frequency)
        self.wave_number = float(wave_number)
        self.amplitude = np.asarray(amplitude, dtype=float)

        if phase is None:
            phase = np.zeros(len(self.amplitude), dtype=float)
        self.phase = np.asarray(phase, dtype=float)
        self._validate()

    def _validate(self):
        if not np.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError("Frequency must be a finite positive number.")
        if not np.isfinite(self.wave_number) or self.wave_number <= 0:
            raise ValueError("Wave number must be a finite positive number.")
        if self.amplitude.ndim != 1:
            raise ValueError("Amplitude must be a one-dimensional array.")
        if self.phase.ndim != 1:
            raise ValueError("Phase must be a one-dimensional array.")
        if len(self.amplitude) == 0:
            raise ValueError("At least one harmonic component is required.")
        if len(self.amplitude) != len(self.phase):
            raise ValueError("Amplitude and phase must have the same length.")
        if not np.all(np.isfinite(self.amplitude)):
            raise ValueError("Amplitudes must be finite.")
        if not np.all(np.isfinite(self.phase)):
            raise ValueError("Phases must be finite.")

    @property
    def harmonics(self):
        return np.arange(1, len(self.amplitude) + 1)

    @property
    def angular_frequency(self):
        return 2 * np.pi * self.harmonics * self.frequency

    @property
    def wave_numbers(self):
        return self.harmonics * self.wave_number

    @property
    def phase_velocity(self):
        return 2 * np.pi * self.frequency / self.wave_number

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)
        if x.ndim > 1 or t.ndim > 1:
            raise ValueError("x and t must be scalars or one-dimensional arrays.")

        scalar_input = x.ndim == 0 and t.ndim == 0
        x = np.atleast_1d(x)
        t = np.atleast_1d(t)

        phase = (
            self.wave_numbers[:, None, None] * x[None, None, :]
            - self.angular_frequency[:, None, None] * t[None, :, None]
            + self.phase[:, None, None]
        )
        y = np.sum(self.amplitude[:, None, None] * np.sin(phase), axis=0)
        return y[0, 0] if scalar_input else y

    def __repr__(self):
        return (
            f"HarmonicWave(frequency={self.frequency}, "
            f"wave_number={self.wave_number}, "
            f"harmonics={len(self.amplitude)})"
        )
