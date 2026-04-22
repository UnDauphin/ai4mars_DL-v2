# Dataset: AI4MARS (Artificial Intelligence for Mars Rover Surveys)

Este dataset es el pilar fundamental para el entrenamiento de modelos de segmentación semántica aplicados a la navegación autónoma en Marte. Proporciona etiquetas de clasificación de terreno para garantizar la seguridad de los rovers.



## Fuente de los Datos
Los datos provienen del repositorio oficial en **Zenodo** (ID: 15995036), recopilados y curados por el **Jet Propulsion Laboratory (JPL) de la NASA**.

* **Misiones incluidas:** Mars Science Laboratory (Rover **Curiosity**) y Mars Exploration Rovers (**Spirit** y **Opportunity**).
* **Instrumentos:** Principalmente imágenes de las cámaras de ingeniería (**HazCams**), que tienen un gran angular para detectar obstáculos cercanos.

## Composición del Dataset
El dataset es único debido a su escala y al método de etiquetado:

| Componente | Descripción |
| :--- | :--- |
| **Imágenes RAW** | Más de 35,000 imágenes en escala de grises y color de la superficie marciana. |
| **Etiquetado Semántico** | Cerca de 326,000 etiquetas individuales generadas mediante *crowdsourcing*. |
| **Clases Principales** | 1. **Suelo** (Soil), 2. **Roca** (Bedrock), 3. **Arena** (Sand), 4. **Rocas sueltas** (Big Rocks). |
| **Validación Gold** | Un subconjunto de etiquetas verificadas por científicos de la misión para asegurar la precisión. |

## Desafíos Técnicos del Terreno
El uso de este dataset en Deep Learning presenta retos específicos que los modelos (como PSANet o SegFormer) deben resolver:

* **Ambigüedad Visual:** La transición entre "arena" y "suelo fino" es sutil, lo que requiere una fuerte extracción de características de textura.
* **Geometría de la Cámara:** Las HazCams tienen una distorsión de "ojo de pez" que afecta cómo se proyectan los objetos en los bordes de la imagen.
* **Condiciones de Iluminación:** Sombras densas proyectadas por el propio rover o por formaciones rocosas que pueden confundir al modelo.

## Importancia Científica
A diferencia de los datasets terrestres (como Cityscapes), aquí el error tiene un costo crítico: si un rover confunde arena profunda con suelo firme, puede quedar atrapado permanentemente (como le sucedió al rover Spirit en 2009).

---
**Enlace al Dataset:** [Zenodo - AI4MARS](https://zenodo.org/records/15995036)