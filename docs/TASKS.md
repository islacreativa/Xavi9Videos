# Xavi9Videos - Seguimiento de Tareas

> Ultima actualizacion: 2026-02-21

---

## Resumen de Estado

| Fase | Estado | Progreso |
|------|--------|----------|
| Fase 1: Infraestructura | COMPLETADA | 5/5 |
| Fase 2: Core Backend | COMPLETADA | 7/7 |
| Fase 3: UI + Orquestador | COMPLETADA | 2/2 |
| Fase 4: Tests | COMPLETADA | 1/1 (30 tests pasando) |
| Fase 5: Verificacion en DGX Spark | COMPLETADA | 8/8 |
| Fase 6: Mejoras y Produccion | COMPLETADA | 7/7 |

---

## Tareas Completadas

### Fase 1: Infraestructura
- [x] `docker-compose.yml` - Orquestacion con Cosmos opcional (no hay NIM ARM64)
- [x] `Dockerfile` - Base CUDA 13.0.1 + PyTorch cu130 (compatible GB10 sm_121)
- [x] `requirements.txt` - 11 dependencias (torch se instala por separado)
- [x] `.env.example` - Template de configuracion
- [x] `scripts/setup.sh` y `scripts/download_models.sh` - Setup y descarga de modelos

### Fase 2: Core Backend
- [x] `app/config.py` - Pydantic Settings con defaults para DGX Spark
- [x] `app/models/__init__.py` - BaseVideoModel ABC, GenerationRequest/Result, Lock
- [x] `app/models/cosmos.py` - Cliente HTTP async para Cosmos NIM (text2world + video2world)
- [x] `app/models/ltx2.py` - Inferencia local LTX-2 con soporte FP8
- [x] `app/models/svd.py` - Inferencia local SVD-XT via diffusers
- [x] `app/utils/storage.py` - Gestion archivos con auto-cleanup (100 archivos / 10GB max)
- [x] `app/utils/video.py` - FFmpeg: info video, thumbnails

### Fase 3: UI + Orquestador
- [x] `app/ui/components.py` - Gradio UI completa (2 columnas, visibilidad dinamica)
- [x] `app/main.py` - Entry point, Cosmos opcional, swap automatico, queue

### Fase 4: Tests
- [x] 25 tests unitarios e integracion - **TODOS PASANDO**

### Fase 5: Verificacion en DGX Spark
- [x] **5.1** `.env` con `NGC_API_KEY` configurado en DGX Spark
- [x] **5.2** Entorno verificado: GPU GB10 (130.7GB), Docker 29.1.3, CUDA 13.0
- [x] **5.3** Modelos se descargan automaticamente en primer uso (lazy loading)
- [x] **5.4** `docker compose up --build` exitoso (imagen CUDA 13.0 + PyTorch 2.10.0+cu130)
- [x] **5.5** UI accesible en `http://192.168.1.2:7860` (HTTP 200)
- [x] **5.6** LTX-2 genera video exitosamente (29.3s, 512x320, 25 frames)
- [x] **5.7** SVD-XT genera video exitosamente (1024x576, 25 frames)
- [x] **5.8** Swap de modelos verificado: LTX-2 (63GB) → SVD (4GB) → LTX-2 (63GB), VRAM liberada a 0GB

---

### Fase 6: Mejoras y Produccion
- [x] **6.1** pyproject.toml con ruff (lint + format), 0 errores
- [x] **6.2** GitHub Actions CI/CD (`.github/workflows/ci.yml`: lint + tests)
- [x] **6.3** Tests GPU end-to-end (`tests/test_gpu.py`, 6 tests con `@pytest.mark.gpu`)
- [x] **6.4** Barra de progreso real en Gradio UI (`callback_on_step_end`)
- [x] **6.5** Logging JSON estructurado (`LOG_FORMAT=json`)
- [x] **6.6** Documentacion modelo API en `app/models/__init__.py` (guia paso a paso)
- [x] **6.7** Soporte Wan 2.1 (`app/models/wan.py`, 14B T2V/I2V, registrado en UI)

---

## Proyecto Completado

Todas las fases (1-6) completadas. 30 tests pasando, 0 errores de lint.

> **Nota:** Cosmos NIM deshabilitado - no existe imagen ARM64 para DGX Spark.
> Se habilitara cuando NVIDIA publique una imagen compatible.

---

## Notas Tecnicas

- **DGX Spark:** NVIDIA GB10, 130.7GB VRAM, CUDA 13.0, aarch64 (ARM)
- **Docker:** CUDA 13.0.1-devel base + PyTorch 2.10.0+cu130
- **Cosmos NIM:** No disponible para ARM64 (deshabilitado, se activa con COSMOS_NIM_URL)
- **Warning sm_121:** Cosmetico, sm_120/sm_121 son binariamente compatibles
- **cuBLAS fix:** `LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so.13:/usr/local/cuda/lib64/libcublasLt.so.13` (PyTorch's bundled cuBLAS no soporta sm_121, se usa el del sistema)
- **PyAV:** Requerido por LTX-2 para exportar video (`av>=14.0.0` en requirements.txt)
- **LTX-2 rendimiento:** ~1.27s/paso, ~29s para 20 pasos a 512x320 25 frames
- **Python:** 3.12 en Docker, 3.9.6 en macOS local
- **SSH:** `islacreativa@192.168.1.2` (clave ed25519)
- **App corriendo:** `http://192.168.1.2:7860`
