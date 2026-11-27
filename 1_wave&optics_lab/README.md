# 1. Wave & Optics Lab

This directory contains numerical models and experiments for wave phenomena.

## Current structure

```text
1_wave&optics_lab/
└── waves/
    └── models/
        ├── harmonic_wave.py
        ├── pulse_wave.py
        └── wave_system.py
```

All models are callable as `wave(x, t)`. For one-dimensional arrays of
positions and times, the result has shape `(Nt, Nx)`; scalar inputs produce
a scalar.

- `HarmonicWave`: finite superposition of harmonics.
- `PulseWave`: Gaussian pulse model.
- `WaveSystem`: superposition of callable wave models.
