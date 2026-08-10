# ADCS Simulator

Tamilarasan Ketheeswaran

A 3U cubesat attitude determination and control simulator: rigid-body dynamics on SO(3), an
orbit/environment model, sensor models, a multiplicative EKF, and detumbling/pointing control.

## Status

Just getting started. Repository skeleton is in place.

## Setup

```bash
cd "ADCS Simulator"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

- `adcs/`: package source
- `tests/`: pytest suite
- `docs/`: derivations (MEKF, control laws)
- `figures/`: tracked, curated result figures
- `output/`: regenerated on each run, gitignored

## License

MIT. See [LICENSE](LICENSE).
