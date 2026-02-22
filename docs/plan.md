# Xavi9Videos - Plan de Implementacion

## Contexto

Crear una aplicacion de generacion de video por IA que corre localmente en una **NVIDIA DGX Spark** (GB10, 128GB memoria unificada, soporte FP4/FP8). La app acepta prompts de texto y/o imagenes y genera videos usando 3 modelos: **Cosmos**, **LTX-2**, y **SVD-XT**.

## Arquitectura

```
+-------------------------------------------+     +------------------------------+
|  xavi9videos-app (Container Principal)    |     | cosmos-nim (NIM Container)   |
|                                           |     |                              |
|  Gradio UI (puerto 7860)                  |     |  Cosmos Predict1 7B          |
|    |                                      |     |  - text2world                |
|    v                                      |     |  - video2world               |
|  Orquestador Backend                      |     |                              |
|    |         |          |                 |     |  REST API (puerto 8000)      |
|    v         v          v                 |     |  POST /v1/infer              |
|  LTX-2    SVD-XT     Cosmos Client -------+---->|  GET  /v1/health/ready       |
|  (local)  (local)    (HTTP client)        |     +------------------------------+
|                                           |
|  Compartido: /app/outputs, /app/models    |
+-------------------------------------------+
```

**Dos contenedores Docker** orquestados con Docker Compose:
1. **cosmos-nim**: Contenedor NIM oficial (`nvcr.io/nim/nvidia/cosmos-predict1:1.0.0`)
2. **xavi9videos-app**: App principal con Gradio + LTX-2 + SVD locales

## Gestion de Memoria (128GB unificados)

- Solo un modelo local (LTX-2 o SVD) cargado a la vez
- `asyncio.Lock` + cola de Gradio serializa requests (1 a la vez)
- Swap automatico: al cambiar de modelo, se descarga el anterior (`torch.cuda.empty_cache()`)
- Cosmos corre en su propio contenedor, comparte memoria fisica

## Estructura del Proyecto

```
Xavi9Videos/
├── docker-compose.yml          # Orquestacion 2 servicios
├── Dockerfile                  # Imagen app principal (base: NGC PyTorch)
├── .env                        # NGC_API_KEY
├── requirements.txt            # Dependencias Python
├── app/
│   ├── __init__.py
│   ├── main.py                 # Entry point: registro modelos, generacion, Gradio launch
│   ├── config.py               # Pydantic Settings (env vars, defaults)
│   ├── models/
│   │   ├── __init__.py         # BaseVideoModel ABC, GenerationRequest/Result, Lock
│   │   ├── cosmos.py           # Cliente HTTP async para Cosmos NIM
│   │   ├── ltx2.py             # Inferencia local LTX-2 con FP8
│   │   └── svd.py              # Inferencia local SVD-XT via diffusers
│   ├── ui/
│   │   ├── __init__.py
│   │   └── components.py       # Interfaz Gradio completa
│   └── utils/
│       ├── __init__.py
│       ├── video.py            # FFmpeg: info video, thumbnails
│       └── storage.py          # Gestion archivos con auto-cleanup
├── outputs/                    # Videos generados
├── models/                     # Pesos de modelos (volumen Docker)
│   ├── ltx2/                   # ~36GB
│   └── svd/                    # ~10GB
├── scripts/
│   ├── setup.sh                # Verificacion entorno + NGC login
│   └── download_models.sh      # Descarga LTX-2 y SVD de HuggingFace
├── docs/
│   └── plan.md                 # Este documento
└── tests/
    ├── test_config.py
    ├── test_storage.py
    ├── test_models_cosmos.py
    ├── test_models_ltx2.py
    ├── test_models_svd.py
    └── test_integration.py
```

## Archivos Criticos y Que Hacen

### 1. `docker-compose.yml`
- Servicio `cosmos-nim`: imagen NIM, GPU passthrough, healthcheck con 5min grace period
- Servicio `xavi9videos-app`: build local, depende de cosmos healthy, puertos 7860
- Volumenes: `nim_cache`, `models_cache`, `outputs`

### 2. `app/config.py`
- Pydantic Settings con defaults para DGX Spark
- FP8 habilitado por defecto
- Concurrencia maxima: 1 request a la vez

### 3. `app/models/cosmos.py`
- `httpx.AsyncClient` con timeout de 600s
- Payloads para text2world e image2world (base64)
- Health check via `/v1/health/ready`

### 4. `app/models/ltx2.py`
- Carga lazy del pipeline `LTXPipeline`
- FP8 transformer con checkpoint distilado (27GB vs 43GB)
- Ejecucion en thread pool (`run_in_executor`) para no bloquear async
- `unload()` explicito con `gc.collect()` + `cuda.empty_cache()`

### 5. `app/models/svd.py`
- `StableVideoDiffusionPipeline` con `enable_model_cpu_offload()`
- Solo image-to-video (valida que hay imagen)
- Resolucion fija 1024x576, 25 frames

### 6. `app/ui/components.py`
- Layout 2 columnas: controles | preview
- Selector de modelo (4 opciones)
- Visibilidad dinamica segun modelo seleccionado
- Parametros: resolucion, frames, FPS, steps, guidance, seed
- Historial de archivos generados

### 7. `app/main.py`
- Registro de modelos (dict)
- `_ensure_model_loaded()`: swap de modelos locales
- `generate_video()`: funcion principal con lock, validacion, callbacks
- `demo.queue(max_size=5, default_concurrency_limit=1)`

## Modelos y Capacidades

| Modelo | Modo | Resolucion | Frames | FPS | Tiempo Est. |
|--------|------|-----------|--------|-----|-------------|
| Cosmos Text2World | texto→video | hasta 1280x720 | hasta 121 | 24 | 60-300s |
| Cosmos Video2World | imagen→video | hasta 1280x720 | hasta 121 | 24 | 60-300s |
| LTX-2 (FP8) | texto/imagen→video | hasta 1920x1080 | hasta 121 | 25/50 | 30-120s |
| SVD-XT | imagen→video | 1024x576 (fijo) | 25 (fijo) | 7 | 30-60s |

## Pasos de Implementacion

### Fase 1: Infraestructura (archivos base)
1. Crear `docker-compose.yml`
2. Crear `Dockerfile`
3. Crear `requirements.txt`

4. Crear `.env.example`
5. Crear `scripts/setup.sh` y `scripts/download_models.sh`

### Fase 2: Core Backend
6. Crear `app/config.py` (Settings)
7. Crear `app/models/__init__.py` (ABC, dataclasses, lock)
8. Crear `app/models/cosmos.py`
9. Crear `app/models/ltx2.py`
10. Crear `app/models/svd.py`
11. Crear `app/utils/storage.py`
12. Crear `app/utils/video.py`

### Fase 3: UI + Orquestador
13. Crear `app/ui/components.py`
14. Crear `app/main.py`

### Fase 4: Tests
15. Tests unitarios e integracion

## Manejo de Errores

- Cosmos caido → mensaje claro + sugerencia `docker compose restart`
- Timeout → "Intenta video mas corto o menos steps"
- OOM → `empty_cache()` + "Reduce resolucion"
- Modelo no encontrado → "Ejecuta download_models.sh"
- SVD sin imagen → "SVD requiere imagen"

## Verificacion

1. `docker compose up --build` sin errores
2. `http://localhost:7860` carga la UI
3. "Check Model Health" muestra todos los modelos Ready
4. Generar video con cada modelo (Cosmos, LTX-2, SVD)
5. Verificar swap de modelos (LTX-2 → SVD → LTX-2)
6. Verificar descarga de video
7. Verificar historial de generaciones
8. `nvidia-smi` confirma liberacion de memoria tras unload
