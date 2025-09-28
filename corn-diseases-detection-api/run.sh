#!/usr/bin/env bash
# Workers: 1–2 para CPU pequeña; puedes subir --workers si tienes más vCPU
exec uvicorn app.main:app --host 0.0.0.0 --port 8000