# ABYSS: Adaptive Biodiversity Exploration

WHACK 2026, Marshall Wace New Worlds Challenge. Team: Rihaan, Cam and Ammar.

ABYSS is the decision-making software for a simulated underwater exploration robot. It
learns which environments are associated with biodiversity during a mission and uses
that to choose where to survey next, within a fixed energy budget.

**This is a simulation.** The terrain is real GEBCO bathymetry, but the biology on it is
always simulated and energy is in abstract units. It is not a validated AUV or a model
of a real ecosystem.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m pytest
```

```bash
python -m abyss.run_mission --scenario ridge --seed 0 --planner random
```

This writes a mission log and a `.reveal.json` file to `results/missions/`.

## Replay UI

```bash
python -m scripts.build_ui_logs
```

```bash
python -m http.server 8000 --directory ui
```

Then open <http://localhost:8000>. The first command runs real missions and rewrites
`ui/logs/`; rerun it after adding a planner so that planner appears in the UI. Both panels
share a clock in energy units, so they are always compared at equal spend. Reveal files
are only fetched when the viewer presses Reveal.

Press **Guided demo** for the five-stage walkthrough. The benchmark section appears once
`ui/logs/benchmark.json` exists.

## Results

See [docs/results.md](docs/results.md) for findings, limitations and full tables.

```bash
python -m abyss.evaluation.tune
```

```bash
python -m abyss.evaluation.benchmark
```

```bash
python -m abyss.evaluation.report
```

`tune` uses development seeds only. Copy any changed settings into `DEFAULT_CONFIG` in
`abyss/planners/adaptive.py` before running the benchmark. The benchmark uses separate
evaluation seeds and takes a few minutes on 12 CPU threads. Don't choose worlds or
settings based on evaluation results.

## Status

| Area | Owner | State |
| --- | --- | --- |
| Contract, GEBCO terrain, worlds, environment, mission runner, logs | Rihaan | Working |
| Replay UI | Rihaan | Working; belief layers tested only with temporary data until the adaptive planner logs them |
| Bayesian learner, ABYSS planner, baselines, ablations, benchmark | Cam's area, drafted by Rihaan | Working; for Cam to review, on the `abyss-planners` branch |
| Real perception flow in the simulator | Rihaan | Working; tested through the pipeline's real classes with a fake encoder, not yet with DINOv2 |
| Perception pipeline | Ammar | Working; only smoke-test results so far |

## Adding a planner (Cam)

Subclass `Planner` from `abyss/planners/base.py`, then add it to `PLANNERS` in
`abyss/planners/__init__.py`. It then works with `--planner <name>`.

```python
class MyPlanner(Planner):
    name = "mine"

    def reset(self, public_map, config, seed): ...
    def choose_target(self, public_state, observation_history): ...  # (row, col) or HOME
    def observe(self, observation): ...  # observation.richness is the model target
    def diagnostics(self): ...  # predicted maps and utility terms for the log
```

- `public_state.feasible` marks every target the simulator will accept, and
  `public_state.travel_cost` gives the cost of reaching each cell.
- `public_map.features` holds standardised depth, slope and BPI.
- A rejected target sends the robot home and is recorded as a violation in the log.

To show beliefs in the UI, return these optional keys from `diagnostics()`. It is called
after `observe()`, so `target_utility` describes the target just travelled to, and the maps
are the updated beliefs. Each map is a `[row][col]` nested list.

```python
{
    "predicted_richness": ...,
    "uncertainty": ...,
    "target_utility": {
        "expected_richness": ..., "information_gain": ..., "coverage": ...,
        "energy_cost": ..., "utility": ...,
    },
}
```

## Adding perception (Ammar)

The simulator side is ready and tested with a fake model. To try it:

```bash
python -m abyss.run_mission --scenario ridge --seed 0 --perception stub
```

Implement these two interfaces from `abyss/contract.py`. `abyss/stub_perception.py` is a
working example of both, and `run_mission()` takes them as `perception=` and `image_pool=`,
with `perception_mode="real"` in the config.

- **`ImagePool`** provides `categories` (common first; the count must equal `n_categories`),
  `withheld`, `images_for(category)` and `load(ref)`. Your categories replace the
  `category_00` names.
- **`Perception`** provides `name` (recorded in the log) and `predict(image)`, which returns
  a `PerceptionResult`. The same image must always give the same result.

Image refs reach the planner, so they must not reveal the category: the simulator refuses
a path like `images/squid/001.jpg`. Use opaque IDs. Refs are also the UI's image URLs, so
they should resolve from `ui/` for the gallery to show them.

Each survey runs the model once per cell. Images flagged unknown count towards richness,
but not towards discovered categories. True labels and the scores in `perception_summary`
stay in the `.reveal.json` file. These scores cover only images seen on that mission, so
they don't replace your held-out evaluation.

`abyss/stub_perception.py` has no real images and made-up error rates. Never report its
numbers as vision results.

`abyss/fathomnet_perception.py` connects the `perception` package to these interfaces.
After following the perception runbook to fit a model:

```bash
python -m abyss.run_mission --scenario ridge --seed 0 --perception fathomnet --manifest data/fathomnet/splits.json --artifact data/perception
```

This sets `n_categories` to the manifest's category count, and only held-out test images
are used. Familiar categories come first, so withheld categories are the rarest in the
world. A small pool means each image is seen repeatedly, with the same result each time,
so use enough test images per category for the final run.

## Rules the code enforces

- **No leakage.** Planners receive only `PublicMap`, `PublicState` and `Observation`.
  These exclude hidden ecology, including the scenario name and rho.
- **Safe return.** A target is accepted only if
  `E_remaining >= E_travel + E_survey + E_return + E_reserve`.
- **Fair worlds.** Encounters are generated once per world seed and do not depend on the
  planner. Repeat surveys return the same categories, so they cannot inflate discoveries.
- **Separate RNG streams.** Terrain, world and planner randomness never share a stream.
- **Separate truth.** Mission logs hold public data. Hidden maps go in `.reveal.json`.

## Scenarios

Each scenario keeps every category's mean prevalence the same, so scenarios change where
organisms are, not how many exist. `rho` in `[0, 1]` scales signal strength, and `rho = 0`
removes all signal.

| Scenario | Hidden distribution |
| --- | --- |
| `random` | Independent of terrain, location and neighbours, at any `rho` |
| `clustered` | Spatial clusters, with linear and squared terrain trends removed |
| `ridge` | Favours steep, ridge-like terrain, with per-category depth niches |
| `reversed` | Favours flat terrain, with the same depth niches as `ridge` |
| `nonlinear` | Depth-slope interaction plus a spatial field; outside the quadratic features |

## Data sources

**Bathymetry:** GEBCO Compilation Group (2025) GEBCO 2025 Grid
(doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29).

- The committed crop is `data/gebco/monterey_canyon_30x30.json`, so nobody needs to
  download anything. `python -m scripts.fetch_gebco` regenerates it from the
  [UNH image service](https://gis.ccom.unh.edu/server/rest/services/GEBCO2025/GEBCO_2025_IS/ImageServer).
- It is the all-ocean 30 x 30 window with the most varied slope, chosen from terrain alone.
- Cells are about 464 m by 372 m. They are regional planning units, not camera footprints
  or collision maps. Slope uses the true spacing, but travel costs treat the grid as square.
- All seeds share this one crop, so repeated seeds do not show generalisation to other
  regions. Set `terrain_source: synthetic` in `config.yaml` to vary terrain by seed.

**Imagery:** the perception pipeline uses [FathomNet](https://database.fathomnet.org/)
images. Check each image's licence before showing or redistributing it.

**Stretch goal, not integrated:** [OBIS](https://obis.org/) occurrence records.

## Ammar's perception pipeline

The `perception/` package implements FathomNet organism cropping, frozen
DINOv2-small embeddings, nearest-neighbour familiar-category recognition,
validation-calibrated reference-unfamiliarity detection, and held-out evaluation.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
# CPU-only PyTorch avoids downloading CUDA libraries on a laptop/server:
python -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[vision]'
python -m perception --help
python -m pytest -q
```

See [the perception runbook](docs/perception.md) for data preparation, offline
inference, evaluation, image licensing and the simulator integration interface.
