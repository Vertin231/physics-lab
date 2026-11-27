import numpy as np


class DispersionAnalysis:
    """
    Analyze a dispersion relation omega = omega(k).

    Parameters
    ----------
    wave_number : array_like
        Sampled wave numbers k.
    angular_frequency : array_like
        Corresponding angular frequencies omega.

    Notes
    -----
    The input arrays represent a single dispersion branch.
    Wave numbers must be non-negative and strictly increasing.
    """

    def __init__(self, wave_number, angular_frequency):

        self.k = np.asarray(wave_number, dtype=float)
        self.omega = np.asarray(angular_frequency, dtype=float)

        self._validate()

    # ==================================
    # 1. Validation
    # ==================================

    def _validate(self):

        if self.k.ndim != 1 or self.omega.ndim != 1:
            raise ValueError(
                "k and omega must be one-dimensional arrays."
            )

        if len(self.k) != len(self.omega):
            raise ValueError(
                "k and omega must have the same length."
            )

        if len(self.k) < 3:
            raise ValueError(
                "At least three data points are required."
            )

        if not np.all(np.isfinite(self.k)):
            raise ValueError("Wave numbers must be finite.")

        if not np.all(np.isfinite(self.omega)):
            raise ValueError("Angular frequencies must be finite.")

        if np.any(self.k < 0):
            raise ValueError(
                "Wave numbers must be non-negative."
            )

        if np.any(np.diff(self.k) <= 0):
            raise ValueError(
                "Wave numbers must be strictly increasing."
            )

    # ==================================
    # 2. Phase velocity
    # ==================================

    @property
    def phase_velocity(self):
        """
        Calculate phase velocity:

            vp = omega / k

        The value at k = 0 is undefined and returned as NaN.
        """

        return np.divide(
            self.omega,
            self.k,
            out=np.full_like(self.omega, np.nan),
            where=self.k != 0
        )

    # ==================================
    # 3. Group velocity
    # ==================================

    @property
    def group_velocity(self):
        """
        Calculate group velocity numerically:

            vg = d(omega) / dk
        """

        return np.gradient(
            self.omega,
            self.k,
            edge_order=2
        )

    # ==================================
    # 4. Dispersion data
    # ==================================

    @property
    def dispersion_data(self):
        """Return k, omega, phase velocity and group velocity."""

        return {
            "wave_number": self.k.copy(),
            "angular_frequency": self.omega.copy(),
            "phase_velocity": self.phase_velocity,
            "group_velocity": self.group_velocity
        }

    # ==================================
    # 5. Interpolation
    # ==================================

    def evaluate(self, wave_number):
        """
        Evaluate the dispersion relation at a given k.

        Linear interpolation is used.
        Values outside the sampled range are rejected.
        """

        k_query = np.asarray(wave_number, dtype=float)

        if not np.all(np.isfinite(k_query)):
            raise ValueError("Requested wave numbers must be finite.")

        if np.any(k_query < self.k[0]) or np.any(k_query > self.k[-1]):
            raise ValueError(
                "Requested wave number is outside the sampled range."
            )

        omega_query = np.interp(
            k_query,
            self.k,
            self.omega
        )

        vp_query = np.divide(
            omega_query,
            k_query,
            out=np.full_like(
                np.asarray(omega_query, dtype=float),
                np.nan
            ),
            where=k_query != 0
        )

        vg_query = np.interp(
            k_query,
            self.k,
            self.group_velocity
        )

        return {
            "wave_number": k_query,
            "angular_frequency": omega_query,
            "phase_velocity": vp_query,
            "group_velocity": vg_query
        }

    # ==================================
    # 6. Summary
    # ==================================

    def __repr__(self):

        return (
            f"DispersionAnalysis("
            f"number_of_points={len(self.k)}, "
            f"k_range=({self.k[0]}, {self.k[-1]}))"
        )
