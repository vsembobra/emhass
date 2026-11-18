#!/bin/sh
set -e

mkdir -p /data/shadow_reports /data/weather_archive /share/emhass_shadow 2>/dev/null || true

# Seed models and historical data on first run.
if [ ! -d /data/models ]; then
  cp -r /opt/models_default /data/models
fi
if [ -f /share/emhass_shadow/seed/hourly_audit.jsonl ] && [ ! -f /data/hourly_audit.jsonl ]; then
  cp /share/emhass_shadow/seed/hourly_audit.jsonl /data/hourly_audit.jsonl
fi
if [ -f /share/emhass_shadow/seed/outages.jsonl ] && [ ! -f /data/outages.jsonl ]; then
  cp /share/emhass_shadow/seed/outages.jsonl /data/outages.jsonl
fi
if [ -f /share/emhass_shadow/seed/outages.sqlite ] && [ ! -f /data/outages.sqlite ]; then
  cp /share/emhass_shadow/seed/outages.sqlite /data/outages.sqlite
fi
if [ -d /share/emhass_shadow/seed/weather_archive ]; then
  cp -n /share/emhass_shadow/seed/weather_archive/*.json /data/weather_archive/ 2>/dev/null || true
fi

exec python /opt/app/main.py
