# Open-Meteo model selection

Approved scope: the user requested implementation of model selection within
Open-Meteo after discussing Auto / ECMWF / ICON / GFS. Keep LightGBM and battery
planning, with Auto preserving the existing installation.

## Approach

A small add-on-only module maps the four UI choices to Open-Meteo model IDs
and isolates model-dependent files. A generic multi-provider framework and
automatic model competition were considered and excluded: the user wants a
compact plugin and explicitly selected Open-Meteo only.

The `open_meteo_model` setting defaults to `auto`. The panel and Supervisor
configuration expose the same choices. Live and previous-runs requests use the
same selected model; timestamps and radiation interval semantics stay unchanged.
Auto retains the existing paths. Explicit models have separate weather archives,
training history, load/PV bundles, and retraining state. Calibration identities
include the weather model. A failed fetch may use only that model's archive.
No models or history are deleted on switching. A newly selected model is trained
on its own previous-runs data before regular planning; existing fallback behavior
applies if training is unavailable. Reports identify the selected weather model.

## Verification

Test request parameters, invalid choices, default compatibility, cache isolation,
history isolation, model loading/training paths, retraining after switching,
calibration identities, and panel/Supervisor schema consistency. Run the add-on
regression tests and lint changed Python files. No new dependencies, EMHASS core
changes, live inverter writes, deployment or publishing are needed.
