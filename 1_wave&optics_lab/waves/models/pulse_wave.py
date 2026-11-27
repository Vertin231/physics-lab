import numpy as np


class PulseWave:
    """A one-dimensional pulse wave; currently supports a Gaussian pulse.

    Gaussian parameters are [velocity, amplitude, sigma].
    """

    def __init__(self, pulse_type: str, parameter: list):
        self.pulse_type = pulse_type
        self.parameter = np.asarray(parameter, dtype=float)
        self._validate()

    def _validate(self):
        if self.pulse_type != "gaussian":
            raise NotImplementedError(
                f"Unsupported pulse type: {self.pulse_type}"
            )
        if self.parameter.shape != (3,):
            raise ValueError(
                "Gaussian parameters must be [velocity, amplitude, sigma]."
            )
        if not np.all(np.isfinite(self.parameter)):
            raise ValueError("Parameters must be finite.")

        velocity, amplitude, sigma = self.parameter
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")

    def _gaussian_delta(self, x, t):
        velocity, amplitude, sigma = self.parameter
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)

        if x.ndim > 1 or t.ndim > 1:
            raise ValueError("x and t must be scalars or one-dimensional arrays.")

        scalar_input = x.ndim == 0 and t.ndim == 0
        x = np.atleast_1d(x)
        t = np.atleast_1d(t)

        y = amplitude * np.exp(
            -(x[None, :] - velocity * t[:, None]) ** 2
            / (2 * sigma ** 2)
        )
        return y[0, 0] if scalar_input else y

    def __call__(self, x, t):
        if self.pulse_type == "gaussian":
            return self._gaussian_delta(x, t)
        raise NotImplementedError(
            f"Unsupported pulse type: {self.pulse_type}"
        )

    def __repr__(self):
        return (
            f"PulseWave(type={self.pulse_type}, "
            f"parameters={self.parameter.tolist()})"
        )
