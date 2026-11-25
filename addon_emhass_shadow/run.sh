#!/bin/sh
set -e

mkdir -p /data/shadow_reports /data/weather_archive /share/emhass_shadow 2>/dev/null || true

# Seed models and historical data on first run.
if [ ! -d /data/models ]; then
  cp -r /opt/models_default /data/models
fi
if [ ! -f /data/hourly_audit.jsonl ] && [ ! -f /data/hourly_audit.jsonl.gz ]; then
  if [ -f /share/emhass_shadow/seed/hourly_audit.jsonl.gz ]; then
    cp /share/emhass_shadow/seed/hourly_audit.jsonl.gz /data/hourly_audit.jsonl.gz
  elif [ -f /share/emhass_shadow/seed/hourly_audit.jsonl ]; then
    cp /share/emhass_shadow/seed/hourly_audit.jsonl /data/hourly_audit.jsonl
  fi
fi
if [ -f /share/emhass_shadow/seed/outages.jsonl ] && [ ! -f /data/outages.jsonl ]; then
  cp /share/emhass_shadow/seed/outages.jsonl /data/outages.jsonl
fi
if [ -f /share/emhass_shadow/seed/outages.sqlite ] && [ ! -f /data/outages.sqlite ]; then
  cp /share/emhass_shadow/seed/outages.sqlite /data/outages.sqlite
fi
if [ -d /share/emhass_shadow/seed/weather_archive ]; then
  for source in /share/emhass_shadow/seed/weather_archive/*.json.gz /share/emhass_shadow/seed/weather_archive/*.json; do
    [ -f "$source" ] || continue
    name=${source##*/}
    logical=${name%.gz}
    if [ ! -f "/data/weather_archive/$logical" ] && [ ! -f "/data/weather_archive/$logical.gz" ]; then
      cp "$source" "/data/weather_archive/$name"
    fi
  done
fi

exec python /opt/app/main.py
