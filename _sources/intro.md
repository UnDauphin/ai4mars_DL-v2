# Segmentación Semántica de Terreno Marciano con Deep Learning

**Entregable 2 — Proyecto de Deep Learning**

---

## Integrantes del Grupo

<!-- Por favor completa los nombres aquí -->

| Nombre | Código |
|--------|--------|
| Alejandro David Moya Nieves | 200196366 |
| Mateo Jose Gomez Rojas | 200193297 |
| Mateo Andres Molinares Tellez |  |      
| David Alejandro Ibañez Barrios | 200195861 |

---

## Motivación

La exploración autónoma de Marte depende críticamente de la capacidad de los rovers para interpretar su entorno sin intervención humana en tiempo real. La distancia entre la Tierra y Marte introduce latencias de comunicación de entre 4 y 24 minutos, lo que hace imposible el control remoto reactivo frente a obstáculos inesperados. Un error de navegación —como confundir arena profunda con suelo firme— puede dejar al rover irrecuperable, como ocurrió con el Spirit en 2009.

La segmentación semántica a nivel de píxel es la herramienta que permite al rover "ver" su entorno de forma estructurada: distinguir suelo transitable de roca, arena o acumulaciones peligrosas de piedras sueltas. El avance en modelos de Deep Learning, desde las CNN clásicas hasta los Vision Transformers, ha abierto la posibilidad de dotar a los rovers de percepción visual comparable a la humana, pero en condiciones marcianas extremas: polvo, sombras duras, distorsión de las cámaras de ingeniería (HazCams) y distribuciones de clase altamente desbalanceadas.

Este proyecto evalúa de forma rigurosa y comparable **cinco modelos de segmentación semántica** sobre el dataset público **AI4MARS** del JPL/NASA, con el objetivo de responder: ¿qué arquitectura ofrece el mejor balance entre precisión, robustez estadística y viabilidad computacional para su eventual despliegue en un sistema de navegación real?

---

## Pregunta de Investigación

> ¿Cuál de los modelos de segmentación semántica benchmarkeados produce diferencias estadísticamente significativas en mIoU sobre el dataset AI4MARS, y qué arquitectura representa el mejor trade-off entre rendimiento y costo computacional?

---

## Estructura del Libro

El libro está organizado en los siguientes capítulos:

| Capítulo | Contenido |
|----------|-----------|
| **1. Preprocesamiento y Dataset** | Construcción del manifest, limpieza de máscaras inválidas, split estratificado por misión, rerredimensionamiento a 256×256. |
| **2. EDA** | Análisis exploratorio de distribución de clases, estadísticas por misión, visualización de imágenes y máscaras representativas. |
| **3. Estado del Arte** | Revisión sistemática de la literatura (2018–2026). Análisis crítico de 8 modelos top, comparación de tendencias CNN vs. Transformers, identificación de gaps. |
| **4. Benchmark modelos** | Tabla comparativa con métricas (media ± std, IC 95 %), ranking, análisis estadístico (Friedman + Nemenyi) y discusión crítica. |

---

## Entorno Experimental

Todos los experimentos fueron ejecutados bajo el mismo entorno para garantizar comparabilidad:

- **Hardware**: NVIDIA RTX 4050 Laptop (6 GB VRAM)
- **Framework**: PyTorch 2.x con AMP (Automatic Mixed Precision)
- **Dataset**: AI4MARS — 22.965 imágenes válidas (MSL, MER, M2020)
- **Resolución de entrada**: 256 × 256 px (prerredimensionado)
- **Protocolo de evaluación**: 3 seeds independientes por modelo (42, 123, 7); media, desviación estándar e IC 95 % reportados
- **Métricas principales**: mIoU, IoU por clase (soil, bedrock, sand, big\_rock), Pixel Accuracy

---

## Reproducibilidad

Todo el código es reproducible a partir del repositorio adjunto. El pipeline completo sigue este orden:

```
preprocess_resize.py          # Prerredimensionamiento único (ejecutar una vez)
01_preprocessing_and_dataset  # Limpieza y split
02_EDA                        # EDA del dataset
mars_util_fast.py             # Funciones necesarias para entrenar los modelos
03a–03e_model_*               # Entrenamiento de cada modelo
03_benchmark_models           # Análisis estadístico y benchmark final
```

Las semillas aleatorias están fijadas globalmente via `set_seed()` en `mars_utils_fast.py`. Los splits train/val/test se persisten en `processed/split_indices.pkl` para garantizar que todos los modelos sean evaluados sobre exactamente la misma partición.

---
**Enlace al Dataset:** [Zenodo - AI4MARS](https://zenodo.org/records/15995036)