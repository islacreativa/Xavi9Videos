# CLAUDE.md - Xavi9Videos

## Proyecto
App de generacion de video por IA para NVIDIA DGX Spark (128GB). Gradio UI + 3 modelos: Cosmos (NIM remoto), LTX-2 y SVD-XT (locales).

## Comandos
- **Tests:** `python3 -m pytest tests/ -v`
- **Run app:** `python -m app.main` (requiere GPU) o `docker compose up --build`
- **Setup:** `./scripts/setup.sh` + `./scripts/download_models.sh`

## Estructura clave
- `app/main.py` - Entry point y orquestador
- `app/config.py` - Pydantic Settings
- `app/models/` - BaseVideoModel ABC + cosmos.py, ltx2.py, svd.py
- `app/ui/components.py` - Gradio UI
- `app/utils/` - storage.py (cleanup) + video.py (ffmpeg)
- `docs/plan.md` - Plan de implementacion completo
- `docs/TASKS.md` - Seguimiento de tareas pendientes/completadas

## Convenciones
- Python async/await con asyncio.Lock (1 request a la vez)
- Type hints estilo Python 3.10+ (union con `|`)
- Logging por modulo: `logger = logging.getLogger(__name__)`
- Output files: `{model}_{timestamp}.mp4`
- Tests con pytest + mocks (sin GPU requerida)

## Estado actual
- Fases 1-4 completadas (infraestructura, backend, UI, tests)
- 25 tests pasando
- Pendiente: verificacion en DGX Spark real (Fase 5) y mejoras de produccion (Fase 6)
