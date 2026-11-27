
import numpy as np


class WaveAnalysis:
    """
    Analyze a one-dimensional wave model.

    Parameters
    ----------
    wave : callable
        A wave model satisfying wave(x, t).
    x : array_like
        One-dimensional spatial coordinates.
    t : array_like
        One-dimensional time coordinates.

    Notes
    -----
    The wave data is calculated once during initialization.
    Its shape must be (len(t), len(x)).
    """

    def __init__(self, wave, x, t):
        if not callable(wave):
            raise TypeError("wave must be callable.")

        self.wave = wave
        self.x = np.asarray(x, dtype=float)
        self.t = np.asarray(t, dtype=float)

        self._validate_coordinates()

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

    def _validate_coordinates(self):
        if self.x.ndim != 1 or self.t.ndim != 1:
            raise ValueError("x and t must be one-dimensional.")

        if len(self.x) < 2:
            raise ValueError("At least two spatial points are required.")

        if len(self.t) < 1:
            raise ValueError("At least one time point is required.")

        if not np.all(np.isfinite(self.x)):
            raise ValueError("x must contain finite values.")

        if not np.all(np.isfinite(self.t)):
            raise ValueError("t must contain finite values.")

        if np.any(np.diff(self.x) <= 0):
            raise ValueError("x must be strictly increasing.")

        if np.any(np.diff(self.t) <= 0):
            raise ValueError("t must be strictly increasing.")

    # ==================================
    # 1. Wave snapshots
    # ==================================

    def snapshot(self, index):
        """Return the wave profile at a given time index."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer.")

        if index < 0 or index >= len(self.t):
            raise IndexError("Time index out of range.")

        return self.data[index].copy()

    def snapshot_at(self, time):
        """Return the wave profile at the nearest sampled time."""
        index = np.argmin(np.abs(self.t - time))
        return self.snapshot(index)

    # ==================================
    # 2. Amplitude analysis
    # ==================================

    @property
    def max_amplitude(self):
        """Maximum absolute displacement at each time."""
        return np.max(np.abs(self.data), axis=1)

    @property
    def rms_amplitude(self):
        """Spatial root-mean-square displacement at each time."""
        return np.sqrt(np.mean(self.data ** 2, axis=1))

    # ==================================
    # 3. Spatial integral
    # ==================================

    @property
    def intensity(self):
        """Squared displacement y(x,t)^2."""
        return self.data ** 2

    @property
    def spatial_integral(self):
        """
        Integrate y^2 over space at each time.

        This is a useful wave-strength measure, but is not
        generally the physical energy of the wave.
        """
        return np.trapezoid(
            self.intensity,
            x=self.x,
            axis=1
        )

    # ==================================
    # 4. Spatial center and width
    # ==================================

    @property
    def center(self):
        """The y^2-weighted spatial center at each time."""
        weight = self.intensity

        numerator = np.trapezoid(
            weight * self.x[None, :],
            x=self.x,
            axis=1
        )

        denominator = self.spatial_integral

        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0
        )

    @property
    def width(self):
        """The y^2-weighted spatial standard deviation."""
        center = self.center

        variance_integral = np.trapezoid(
            self.intensity
            * (self.x[None, :] - center[:, None]) ** 2,
            x=self.x,
            axis=1
        )

        variance = np.divide(
            variance_integral,
            self.spatial_integral,
            out=np.full_like(variance_integral, np.nan),
            where=self.spatial_integral > 0
        )

        return np.sqrt(np.maximum(variance, 0))

    # ==================================
    # 5. Propagation velocity
    # ==================================

    @property
    def center_velocity(self):
        """Estimate the velocity of the spatial center."""
        if len(self.t) < 2:
            raise ValueError(
                "At least two time points are required "
                "to calculate velocity."
            )

        return np.gradient(self.center, self.t)

    def __repr__(self):
        return (
            f"WaveAnalysis("
            f"spatial_points={len(self.x)}, "
            f"time_points={len(self.t)})"
        )
