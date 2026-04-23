# 05 — Modelo original — MambaSegNet

---

## 5.1 Selección del Modelo Original y Justificación

### Criterios de Selección

Se buscó un paper guía que cumpliera tres condiciones simultáneamente: (1) publicado en revista Q1 de WoS/Scopus con revisión por pares, (2) arquitectura no previamente aplicada al dominio de segmentación de terreno marciano, y (3) dominio de aplicación con características estructurales transferibles a AI4MARS. El modelo seleccionado es **MambaSegNet (Wen et al., 2025)**, publicado en *Remote Sensing* (MDPI, Q1).

| Criterio de selección | Evaluación de MambaSegNet |
|---|---|
| Revista Q1 indexada en Scopus/WoS |  *Remote Sensing*, MDPI |
| Nunca aplicado a segmentación de terreno marciano |  Primera aplicación propuesta en este proyecto |
| Dominio con alta analogía estructural a AI4MARS |  Teledetección satelital: fondos complejos, objetos pequeños dispersos |
| Arquitectura diferenciada de los benchmarks existentes |  State Space Model (Mamba) — no usado en ningún benchmark del proyecto |
| Costo computacional compatible con hardware disponible |  22.43M parámetros, complejidad lineal O(N) |

**MambaSegNet** es seleccionado como modelo original. Su dominio de teledetección comparte con AI4MARS las mismas condiciones adversas: fondos complejos y ruidosos, variabilidad de iluminación, presencia de objetos diminutos mal representados (big rocks ~2% del área etiquetada ↔ naves pequeñas en imágenes satelitales), y ausencia de estructura semántica fuerte. La arquitectura Mamba, basada en State Space Models (SSM), no ha sido aplicada previamente a segmentación de terreno marciano, convirtiéndolo en un modelo **genuinamente original** para este dominio.

---

## 5.2 Referencia Completa

> Wen, R., Yuan, Y., Xu, X., Yin, S., Chen, Z., Zeng, H., & Wang, Z. (2025). **MambaSegNet: A Fast and Accurate High-Resolution Remote Sensing Imagery Ship Segmentation Network**. *Remote Sensing*, 17(19), 3328. https://doi.org/10.3390/rs17193328

- **Revista**: *Remote Sensing* (MDPI) — indexada en Scopus/WoS, Q1 en teledetección
- **Fecha de publicación**: 29 de septiembre de 2025
- **Acceso al código**: https://github.com/wzp8023391/Ship_Segmentation

---

## 5.3 Descripción Detallada del Modelo Original

### 5.3.1 Arquitectura General

MambaSegNet es una red de segmentación semántica encoder-decoder ligera construida sobre el paradigma **Mamba (State Space Model, SSM)**, diseñada para procesar imágenes de teledetección de alta resolución. La arquitectura integra dos módulos centrales:

1. **MambaLayer**: capa de extracción de características mediante secuencias de tokens; aplana y embebe la imagen en parches, los procesa con un Mamba Block que opera en espacio de estados latente.
2. **ResMambaBlock**: bloque residual que combina convolución depthwise, normalización por capas y el MambaLayer para extraer representaciones multi-escala con bajo costo computacional.

El flujo de datos sigue el siguiente esquema:

```
Entrada (C × H × W)
    ↓  [Convolución depthwise inicial]
    ↓  [Flatten + Embedding → tokens H/4 × W/4 × C]
    ↓  [Encoder: 4 etapas con ResMambaBlock + Token Merging (downsampling)]
         H/4×W/4×C → H/8×W/8×2C → H/16×W/16×4C → H/32×W/32×8C
    ↓  [Bottleneck: 2× ResMambaBlock]
    ↓  [Decoder: ResUp Blocks + Conv3d + Skip Connections]
         H/16×W/16×4C → H/8×W/8×2C → H/4×W/4×C
    ↓  [Convolución final → segmentación]
Salida (num_classes × H × W)
```

Las **skip connections** conectan cada etapa del encoder con su etapa simétrica del decoder, preservando detalles espaciales de alta frecuencia perdidos durante el submuestreo. Este diseño es análogo al U-Net clásico, pero reemplazando las convoluciones estándar por bloques SSM.

### 5.3.2 Estado Latente (State Space Model)

El núcleo matemático de Mamba modela secuencias mediante ecuaciones diferenciales ordinarias lineales:

$$f'(t) = a \cdot f(t) + b \cdot x(t), \quad g(t) = c \cdot f(t) + d \cdot x(t)$$

En su versión discreta (discretización Zero-Order Hold con parámetro de escala temporal $\Delta$):

$$f_t = \bar{A} f_{k-2} + \bar{B} x_k, \quad g_t = C f_k + D x_k$$

donde $\bar{A} = e^{\Delta A}$, $\bar{B} \approx \Delta B$ (aproximación de primer orden Taylor). Esta formulación permite que cada token "recuerde" contexto a largo alcance de forma eficiente en **complejidad lineal** O(N), a diferencia del mecanismo de atención Transformer con complejidad O(N²). Para imágenes marcianas con texturas complejas y rocas dispersas en toda la imagen, esta capacidad de modelar dependencias a largo rango sin penalización cuadrática es una ventaja significativa.

### 5.3.3 Bloques Clave

**ResMambaBlock** (corazón del modelo):
- Entrada → embedding lineal → dos ramas paralelas
- Rama 1: Norm → DWConv → ReLU → DWConv → Norm → MambaLayer1 → Norm → ReLU → MambaLayer2
- Rama 2: identidad (residual)
- Salida: suma element-wise de ambas ramas
- Elimina el mecanismo de multi-head attention y el MLP del Transformer estándar → más bloques en el mismo presupuesto computacional

**ResUp Block** (decoder):
- DWConv + Norm + ReLU → upsampling por interpolación bilineal o ConvTranspose

**Función de pérdida combinada**:

$$\mathcal{L} = \alpha \mathcal{L}_{CE} + \beta \mathcal{L}_{IoU}$$

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$

$$\mathcal{L}_{IoU} = 1 - \frac{|P \cap G|}{|P \cup G|}$$

Con $\alpha = \beta = 0.5$. Esta combinación equilibra precisión pixel-a-pixel (CE) con cobertura de regiones (IoU), siendo especialmente relevante para clases raras como `big_rock` (~2% de píxeles en AI4MARS).

### 5.3.4 Supuestos del Modelo

1. **Supuesto de secuencialidad espacial**: los tokens de imagen procesados linealmente capturan dependencias espaciales suficientes; la estructura 2D se aproxima mediante scanning 1D secuencial de parches.
2. **Supuesto de transferibilidad de representaciones**: características aprendidas en teledetección satelital (texturas complejas, objetos pequeños en fondos homogéneos) son transferibles al dominio de imágenes HazCam marcianas.
3. **Supuesto de escalabilidad**: el modelo funciona con imágenes redimensionadas a 256×256 sin pérdida significativa de información semántica relevante para la navegación.
4. **Supuesto de complementariedad de pérdidas**: CE e IoU capturan aspectos ortogonales del error de segmentación; su combinación lineal con igual peso es suficiente sin requerir ajuste adaptativo.

---

## 5.4 Justificación de Selección

### 5.4.1 Relevancia para el Problema

La segmentación de terreno marciano en AI4MARS comparte con la segmentación de naves en teledetección cuatro características estructurales determinantes:

| Característica | AI4MARS (HazCam) | LEVIR_SHIP / DIOR_SHIP |
|---|---|---|
| Fondo dominante | Suelo/roca continua, textura variable | Mar/tierra, fondo extenso |
| Objeto minoritario difícil | `big_rock` (~2% píxeles, disperso) | Naves pequeñas (<450px perímetro) |
| Ruido e iluminación | Distorsión de lente gran angular, sombras | Nubes, bruma, destellos solares |
| Estructura semántica | Débil (no hay geometría rígida de escena) | Débil (naves en orientaciones libres) |

El MambaSegNet demostró **IoU = 0.8208** en DIOR_SHIP y **0.8094** en LEVIR_SHIP superando a SegFormer (0.7791, 0.5603), DeepLabV3+ (0.7923, 0.7789) y SwinTransformer (0.7991, 0.7333), todos ellos modelos benchmark ya evaluados en este proyecto. Esto sugiere capacidad superior para manejar exactamente las condiciones que hacen difícil AI4MARS.

### 5.4.2 Impacto en la Literatura

- El modelo supera a **12 arquitecturas mainstream** en datasets de referencia de teledetección de alta resolución
- Primera aplicación de arquitectura Mamba a segmentación semántica de buques en imágenes satelitales RGB
- Demuestra que SSMs con skip connections superan a Transformers de ventana (Swin) en escenarios de objetos pequeños con fondo complejo
- Publicado en *Remote Sensing* (MDPI) con revisión por pares, indexado en Scopus/WoS

### 5.4.3 Originalidad para el Dominio Marciano

A diferencia de los modelos benchmark implementados en este proyecto (DeepLabV3+, SegFormer, MarsSeg, TerSeg, DepthFormer), **ningún trabajo publicado ha aplicado arquitecturas Mamba (SSM) a segmentación semántica de terreno marciano**. Esto constituye una contribución genuina: si MambaSegNet supera los benchmarks en AI4MARS, cierra un gap real en la literatura.

---

## 5.5 Relación con los Modelos Benchmark del Proyecto

### 5.5.1 Similitudes

| MambaSegNet vs. | Similitud |
|---|---|
| **DeepLabV3+** | Ambos usan backbone preentrenado en ImageNet; ambos tienen arquitectura encoder-decoder con skip connections |
| **SegFormer** | Ambos procesan la imagen en parches/tokens; ambos usan una representación jerarquizada multi-escala; ambos eliminan positional encoding explícito |
| **TerSeg** | Ambos integran mecanismos de captura de contexto global más allá de la CNN local; ambos usan skip connections para preservar detalles espaciales |
| **DepthFormer** | Ambos usan encoder Transformer-like + decoder con upsample progresivo y conexiones residuales |
| **MarsSeg** | Ambos utilizan función de pérdida compuesta (CE + IoU/Dice) para manejar el desbalance de clases raras |

### 5.5.2 Diferencias Clave

| Dimensión | MambaSegNet | Benchmarks del Proyecto |
|---|---|---|
| **Mecanismo de contexto global** | State Space Model (SSM) — complejidad O(N) lineal | Atención (SegFormer, TerSeg, DepthFormer): O(N²); Atrous CNN (DeepLabV3+, MarsSeg): receptivo local |
| **Backbone** | DWConv inicial + Mamba puro (sin ResNet/MiT) | ResNet-50 (DeepLabV3+, MarsSeg), MiT-B2 (SegFormer), ResNet-34+Swin-Tiny (TerSeg), Swin-Tiny (DepthFormer) |
| **Parámetros** | 22.43M — el segundo más ligero después de SegFormer (27.35M) | 38.61M (MarsSeg), 42.00M (DeepLabV3+), 49.19M (TerSeg), 31.58M (DepthFormer) |
| **Dominio de preentrenamiento** | Sin preentrenamiento externo en la propuesta original | ImageNet-1K (todos los benchmarks) |
| **Loss function** | CE + IoU (combinación directa, α=β=0.5) | CE puro (SegFormer), CE+aux (DeepLabV3+), Focal-Dice (MarsSeg), FocalLoss (TerSeg), CE ponderado (DepthFormer) |
| **Handling de objetos pequeños** | SSM con scanning secuencial → dependencias globales sin pérdida de resolución local | ASPP/multi-scale (MarsSeg, DeepLabV3+), window attention limitada (DepthFormer, TerSeg) |
| **Originalidad para Marte** | **Nunca aplicado** al dominio marciano | Todos ya evaluados en AI4MARS o adaptar metodologías previas |

### 5.5.3 Hipótesis de Superioridad

Se hipotetiza que MambaSegNet puede superar los benchmarks en AI4MARS porque:

1. **Ventaja en `big_rock`**: Los SSMs capturan dependencias de largo rango eficientemente, permitiendo que el modelo relacione una roca aislada con el contexto global de la imagen (algo que ASPP o ventanas de atención locales no hacen bien con una clase que representa solo el 2% del área).

2. **Eficiencia computacional**: Con 22.43M parámetros y complejidad lineal, permite mayor batch size y más épocas de entrenamiento bajo las restricciones de la RTX 4050 laptop (6GB VRAM, 45W TGP), potencialmente llevando a mejor convergencia.

3. **Robustez a fondos heterogéneos**: El mecanismo SSM fue diseñado explícitamente para manejar fondos complejos en teledetección, condición análoga a la variabilidad de texturas en imágenes HazCam.

---

## 5.6 Adaptación a AI4MARS

Para adaptar MambaSegNet al dominio de segmentación de terreno marciano, se realizan las siguientes modificaciones respecto al paper original:

### Modificaciones Arquitectónicas

| Componente | Paper Original | Adaptación AI4MARS |
|---|---|---|
| Cabeza de clasificación | Binaria (naves vs. fondo) | 4 clases (soil, bedrock, sand, big_rock) + ignore_index=255 |
| Resolución de entrada | 800×800 (DIOR), 800×600 (LEVIR) | 256×256 (prerredimensionado con `preprocess_resize.py`) |
| Función de pérdida | CE + IoU (α=β=0.5) | Weighted CE (pesos: [1.0, 0.8, 2.0, 8.0]) + IoU, o Focal-Dice para enfatizar `big_rock` |
| Preentrenamiento backbone | Sin especificar en paper | Inicialización aleatoria; opcional: pesos de VMamba preentrenados en ImageNet |

### Protocolo Experimental (Consistente con el Proyecto)

```python
# Configuración alineada con mars_utils_fast.py
optimizer    : AdamW(lr=6e-5, weight_decay=0.01)
scheduler    : CosineAnnealingLR(T_max=30, eta_min=1e-6)
batch_size   : 8–10  (AMP habilitado)
num_epochs   : 80 con early stopping (patience=10)
seeds        : [42, 123, 7]  (3 runs para media ± std)
n_per_mission: 700 (subset estratificado por misión)
```

### Métricas de Evaluación

Consistentes con el resto del proyecto:
- **mIoU** (métrica principal): promedio de IoU sobre las 4 clases válidas
- **IoU por clase**: soil, bedrock, sand, big_rock (especial atención a big_rock)
- **Pixel Accuracy**: proporción de píxeles correctamente clasificados
- **IC 95%**: $\pm 1.96 \cdot \sigma / \sqrt{3}$ sobre las 3 semillas

---

## 5.7 Análisis Crítico del Paper Guía

### Fortalezas

- **Complejidad lineal en longitud de secuencia**: permite procesar imágenes de mayor resolución que los Transformers estándar con el mismo presupuesto de memoria
- **Diseño residual profundo**: los ResMambaBlocks facilitan el flujo de gradiente en redes profundas, reduciendo el riesgo de vanishing gradient
- **Validación cross-dataset**: demostrado en dos datasets con distribuciones muy diferentes (DIOR: resolución 0.5–30m; LEVIR: 0.2–1m), sugiriendo buena generalización
- **Dataset pequeño**: logra buen desempeño con solo ~1200–1400 imágenes por dataset, comparable al tamaño del conjunto de entrenamiento utilizado en nuestros experimentos (n_per_mission=700 → ~2100 imágenes)

### Limitaciones y Riesgos de Transferencia

- **Dominio target es binario** (nave vs. fondo), mientras AI4MARS requiere 4 clases con desbalance severo; la función de pérdida CE+IoU uniforme puede no ser óptima para `big_rock`
- **Sin preentrenamiento en ImageNet**: a diferencia de los benchmarks del proyecto que usan pesos ImageNet, MambaSegNet en su formulación original parte de cero; esto puede ser una desventaja en datasets pequeños
- **Scanning 1D de imagen 2D**: la aproximación de "leer" la imagen como una secuencia 1D puede perder algunas dependencias espaciales 2D que CNN y ViT 2D capturan naturalmente
- **Hardware**: los experimentos originales usan RTX 3090 Ti; nuestra RTX 4050 laptop (6GB VRAM, 45W TGP) requerirá ajuste de batch size y posiblemente reducción de dimensiones del modelo

---

*Paper guía: Wen et al. (2025), Remote Sensing, 17(19), 3328.*
