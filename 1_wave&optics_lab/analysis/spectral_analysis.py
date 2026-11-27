
import numpy as np


class SpectralAnalysis:
    """
    Perform spatial and temporal spectral analysis of a
    one-dimensional wave.

    Parameters
    ----------
    wave : callable
        A wave model satisfying wave(x, t).
    x : array_like
        Uniformly sampled spatial coordinates.
    t : array_like
        Uniformly sampled time coordinates.
    window : {None, "hann"}, optional
        Window function applied before FFT.
    remove_mean : bool, optional
        Whether to remove the signal mean before FFT.
    """

    def __init__(
        self,
        wave,
        x,
        t,
        window=None,
        remove_mean=True
    ):
        if not callable(wave):
            raise TypeError("wave must be callable.")

        self.wave = wave
        self.x = np.asarray(x, dtype=float)
        self.t = np.asarray(t, dtype=float)

        self.window = window
        self.remove_mean = bool(remove_mean)

        self._validate_coordinates()
        self._validate_window()

        self.data = np.asarray(
            self.wave(self.x, self.t),
            dtype=float
        )

        expected_shape = (len(self.t), len(self.x))

        if self.data.shape != expected_shape:
            raise ValueError(
                f"Wave data must have shape {expected_shape}, "
                f"got {self.data.shape}."
            )

        if not np.all(np.isfinite(self.data)):
            raise ValueError("Wave data contains non-finite values.")

    # ==================================
    # 1. Validation
    # ==================================

    def _validate_coordinates(self):
        if self.x.ndim != 1 or self.t.ndim != 1:
            raise ValueError("x and t must be one-dimensional.")

        if len(self.x) < 3 or len(self.t) < 3:
            raise ValueError(
                "At least three samples are required "
                "along both x and t."
            )

        if not np.all(np.isfinite(self.x)):
            raise ValueError("x must contain finite values.")

        if not np.all(np.isfinite(self.t)):
            raise ValueError("t must contain finite values.")

        dx = np.diff(self.x)
        dt = np.diff(self.t)

        if np.any(dx <= 0) or np.any(dt <= 0):
            raise ValueError(
                "x and t must be strictly increasing."
            )

        if not np.allclose(dx, dx[0], rtol=1e-5, atol=1e-12):
            raise ValueError("x must be uniformly sampled.")

        if not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-12):
            raise ValueError("t must be uniformly sampled.")

    def _validate_window(self):
        if self.window not in (None, "hann"):
            raise ValueError(
                "window must be None or 'hann'."
            )

    # ==================================
    # 2. Sampling properties
    # ==================================

    @property
    def dx(self):
        return self.x[1] - self.x[0]

    @property
    def dt(self):
        return self.t[1] - self.t[0]

    @property
    def spatial_sampling_rate(self):
        """Number of spatial samples per unit length."""
        return 1.0 / self.dx

    @property
    def temporal_sampling_rate(self):
        """Number of temporal samples per unit time."""
        return 1.0 / self.dt

    @property
    def spatial_nyquist_wavenumber(self):
        """Maximum resolvable angular spatial frequency."""
        return np.pi / self.dx

    @property
    def temporal_nyquist_angular_frequency(self):
        """Maximum resolvable angular temporal frequency."""
        return np.pi / self.dt

    # ==================================
    # 3. Window function
    # ==================================

    def _get_window(self, n):
        if self.window is None:
            return np.ones(n)

        if n < 3:
            raise ValueError(
                "Hann window requires at least three samples."
            )

        return np.hanning(n)

    # ==================================
    # 4. One-sided amplitude spectrum
    # ==================================

    def _amplitude_spectrum(self, signal, spacing):
        signal = np.asarray(signal, dtype=float)

        if self.remove_mean:
            signal = signal - np.mean(signal)

        window = self._get_window(len(signal))

        normalization = np.sum(window)

        if normalization <= 0:
            raise ValueError("Invalid window normalization.")

        transformed = np.fft.rfft(signal * window)

        amplitude = np.abs(transformed) / normalization

        # Convert the two-sided FFT amplitude to
        # a one-sided amplitude spectrum.
        if len(signal) % 2 == 0:
            amplitude[1:-1] *= 2
        else:
            amplitude[1:] *= 2

        frequency = np.fft.rfftfreq(
            len(signal),
            d=spacing
        )

        return frequency, amplitude

    # ==================================
    # 5. Spatial spectrum
    # ==================================

    def spatial_spectrum(self, time_index):
        """
        Calculate the spatial spectrum at a given time.

        Returns
        -------
        wavenumber : ndarray
            Angular spatial frequency k.
        amplitude : ndarray
            One-sided amplitude spectrum.
        """
        if not isinstance(time_index, (int, np.integer)):
            raise TypeError("time_index must be an integer.")

        if not 0 <= time_index < len(self.t):
            raise IndexError("Time index out of range.")

        frequency, amplitude = self._amplitude_spectrum(
            self.data[time_index],
            self.dx
        )

        wavenumber = 2 * np.pi * frequency

        return wavenumber, amplitude

    # ==================================
    # 6. Temporal spectrum
    # ==================================

    def temporal_spectrum(self, position_index):
        """
        Calculate the temporal spectrum at a given position.

        Returns
        -------
        angular_frequency : ndarray
            Angular temporal frequency omega.
        amplitude : ndarray
            One-sided amplitude spectrum.
        """
        if not isinstance(position_index, (int, np.integer)):
            raise TypeError("position_index must be an integer.")

        if not 0 <= position_index < len(self.x):
            raise IndexError("Position index out of range.")

        frequency, amplitude = self._amplitude_spectrum(
            self.data[:, position_index],
            self.dt
        )

        angular_frequency = 2 * np.pi * frequency

        return angular_frequency, amplitude

    # ==================================
    # 7. Dominant spectral components
    # ==================================

    def dominant_wavenumber(self, time_index):
        """Return the dominant nonzero spatial wavenumber."""
        k, amplitude = self.spatial_spectrum(time_index)

        if len(k) < 2:
            raise ValueError("Insufficient spectral resolution.")

        index = np.argmax(amplitude[1:]) + 1
        return k[index]

    def dominant_angular_frequency(self, position_index):
        """Return the dominant nonzero angular frequency."""
        omega, amplitude = self.temporal_spectrum(position_index)

        if len(omega) < 2:
            raise ValueError("Insufficient spectral resolution.")

        index = np.argmax(amplitude[1:]) + 1
        return omega[index]

    def __repr__(self):
        return (
            f"SpectralAnalysis("
            f"spatial_points={len(self.x)}, "
            f"time_points={len(self.t)}, "
            f"window={self.window})"
        )
