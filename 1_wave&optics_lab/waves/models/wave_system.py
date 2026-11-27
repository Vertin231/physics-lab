import numpy as np


class WaveSystem:
    """A superposition of callable wave models."""

    def __init__(self, waves=None):
        self.waves = list(waves) if waves is not None else []
        for wave in self.waves:
            self._validate_wave(wave)

    @staticmethod
    def _validate_wave(wave):
        if not callable(wave):
            raise TypeError("Each wave must be callable.")

    def add_wave(self, wave):
        """Add a wave model to this system."""
        self._validate_wave(wave)
        self.waves.append(wave)
        return self

    def __call__(self, x, t):
        x_arr = np.asarray(x, dtype=float)
        t_arr = np.asarray(t, dtype=float)

        if x_arr.ndim > 1 or t_arr.ndim > 1:
            raise ValueError("x and t must be scalars or one-dimensional arrays.")

        scalar_input = x_arr.ndim == 0 and t_arr.ndim == 0

        if not self.waves:
            x_arr = np.atleast_1d(x_arr)
            t_arr = np.atleast_1d(t_arr)
            result = np.zeros((len(t_arr), len(x_arr)), dtype=float)
            return result[0, 0] if scalar_input else result

        result = np.asarray(self.waves[0](x, t))
        for wave in self.waves[1:]:
            result = result + wave(x, t)

        return result

    def __add__(self, other):
        if isinstance(other, WaveSystem):
            return WaveSystem(self.waves + other.waves)
        self._validate_wave(other)
        return WaveSystem(self.waves + [other])

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __len__(self):
        return len(self.waves)

    def __repr__(self):
        return f"WaveSystem(number_of_waves={len(self.waves)})"
