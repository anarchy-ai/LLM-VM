#! /usr/bin/env bash
mypy  --config-file pyproject.toml src/  --check-untyped-defs --explicit-package-base
