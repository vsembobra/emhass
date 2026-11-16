# Archive daily Open-Meteo forecasts

Run the standalone snapshot logger once per site each day from a host with Python 3.12 or newer.
Use the same site ID and coordinates on each run. A site ID can contain up to 64 lowercase ASCII
letters, digits, hyphens, and underscores, starting with a letter or digit. Use a persistent
directory on a filesystem that supports hard links, outside the source checkout.

```sh
python -m emhass.weather_snapshot \
  --site-id home \
  --latitude 52.52 \
  --longitude 13.41 \
  --archive-dir /path/to/weather-archive
```

Run this command from the environment where EMHASS is installed. From a source checkout, set
`PYTHONPATH=src` before the command. Choose a time in your external scheduler shortly before your
usual day-ahead decision if you want to replay the forecasts available at that point. The command
prints the path to the saved vintage on success. It reports a failed request to stderr and exits
with a nonzero status, so check the scheduler's exit status to detect gaps.

The live Forecast API request starts at the current UTC hour and asks for 48 hourly values of 2 m
air temperature, global horizontal, direct horizontal, diffuse horizontal, and direct normal solar
radiation, total and low, mid, and high cloud cover, and snow depth. The radiation fields are
means over the hour ending at their validity timestamps. The request uses `best_match` by default.
Use `--model` to select a single model ID supported by Open-Meteo at your site. Each file records the
requested model selection. [Open-Meteo Forecast API](https://open-meteo.com/en/docs)

## Archive records

Each distinct snapshot has its own `<site-id>/<sha256>.json` file. The hash covers the request
metadata and the parsed response fields except `generationtime_ms`, which measures API processing
duration. A duplicate fetch returns the existing file path. Concurrent runs publish complete files
atomically and preserve the first file for a given hash.

| Field | Meaning |
| --- | --- |
| `schema_version` | Archive format version, currently `1`. |
| `site_id` | Site supplied at invocation. |
| `provider` | `Open-Meteo`. |
| `request` | Endpoint and query parameters, including requested coordinates, model selection and hourly fields. |
| `received_at_utc` | UTC time on the logger host when the archived response body was received. |
| `valid_times_utc` | ISO 8601 UTC timestamp for each corresponding index in the response's hourly arrays. |
| `response` | Original provider JSON body with its returned grid coordinates, Unix timestamps, values, and `hourly_units`. Provider nulls are preserved. |

The live API stitches successive model runs into a continuously updated series. Its response
does not identify a model issue time for the vintage, so the record leaves `issued_at_utc` absent.
Use `received_at_utc` to select records available at or before the decision time being replayed.
Keep the logger host clock synchronized for that comparison. [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api)

Weather data by Open-Meteo are licensed under CC BY 4.0. Credit Open-Meteo and indicate any
modifications when sharing the data. The logger calls the free endpoint, which is restricted to
non-commercial use. A commercial deployment requires an appropriate provider endpoint and terms.
[Open-Meteo terms](https://open-meteo.com/en/terms) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
