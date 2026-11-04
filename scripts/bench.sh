#!/usr/bin/env bash
# Copyright (c) 2026 QDrift Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Thin wrapper around the single entry point: python -m main.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "${PYTHON:-python}" -m main benchmark "$@"
