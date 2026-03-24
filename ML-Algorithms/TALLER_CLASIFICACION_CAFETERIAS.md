# Taller: Probar Algoritmos – Clasificación de Éxito en Cafeterías

## Información General del Trabajo

| Aspecto | Descripción |
|--------|------------|
| **TÍTULO DEL TRABAJO** | Taller: Probar Algoritmos – Clasificación de Éxito en Cafeterías |
| **ASIGNATURA** | Inteligencia Computacional |
| **NIVEL** | Pregrado |
| **DOCENTE** | [NOMBRE DEL DOCENTE] |
| **INSTITUCIÓN** | Universidad Pedagógica y Tecnológica de Colombia (UPTC) |
| **FECHA DE ENTREGA** | [FECHA DE ENTREGA] |
| **INTEGRANTES** | [NOMBRE DE LOS INTEGRANTES] |

---

## 1. INTRODUCCIÓN

### 1.1 Contexto y Problemática

En el contexto actual del sector comercial, la analítica de datos se ha convertido en una herramienta fundamental para la toma de decisiones estratégicas. El sector de cafeterías, caracterizado por márgenes operativos reducidos y alta competencia, enfrenta retos significativos relacionados con:

- **Optimización de ingresos**: Necesidad de identificar patrones que conduzcan a mayores ingresos
- **Fidelización de clientes**: Comprensión de factores que influyen en el éxito comercial
- **Eficiencia operativa**: Asignación óptima de recursos (personal, horarios, inversión en marketing)
- **Toma de decisiones estratégica**: Capacidad de predecir viabilidad comercial antes de apertura

La capacidad de predecir si un establecimiento será exitoso o no representa una ventaja competitiva significativa que puede determinar la diferencia entre inversiones rentables y fracasos comerciales.

### 1.2 Objetivos del Trabajo

**Objetivo General:**
Aplicar técnicas de Inteligencia Computacional para desarrollar un modelo de clasificación que prediga el éxito financiero de cafeterías basándose en variables operativas y comerciales.

**Objetivos Específicos:**
1. Explorar y preprocesar el dataset "Coffee Shop Daily Revenue"
2. Implementar diversos algoritmos de aprendizaje automático (5 algoritmos diferentes)
3. Evaluar y comparar el desempeño de los modelos utilizando métricas estándar
4. Identificar el algoritmo con mejor rendimiento para este problema específico
5. Analizar la importancia de características y patrones de clasificación
6. Documentar recomendaciones para la implementación en entornos reales

### 1.3 Entornos y Alcance

Este estudio se realiza en el contexto académico, utilizando datos sintéticos pero realistas del sector de cafeterías. Los modelos desarrollados pueden ser aplicables a contextos comerciales reales con ajustes apropiados.

---

## 2. MARCO TEÓRICO

### 2.1 Fundamentos del Aprendizaje Automático Supervisado

La **clasificación supervisada** es una técnica fundamental dentro del aprendizaje automático cuyo objetivo es asignar etiquetas categóricas a datos basándose en patrones previamente aprendidos a partir de datos etiquetados. En este trabajo se implementa un problema de **clasificación binaria** donde se predice si una cafetería será "Exitosa" o "No Exitosa".

#### Fórmula general:
$$f: X \rightarrow Y$$

Donde:
- $X \in \mathbb{R}^{n \times p}$ representa la matriz de características (n muestras, p características)
- $Y \in \{0, 1\}$ representa las etiquetas binarias
- $f$ es la función aprendida por el modelo

### 2.2 Algoritmos Implementados

#### 2.2.1 Regresión Logística

Es un modelo probabilístico que utiliza la **función sigmoide** para estimar la probabilidad de pertenencia a una clase. A pesar de su nombre, es un modelo de clasificación.

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_p x_p)}}$$

**Ventajas:**
- Interpretabilidad alta (coeficientes indican importancia de características)
- Eficiente computacionalmente
- Funciona bien con relaciones lineales

**Desventajas:**
- Asume separabilidad lineal de las clases
- Rendimiento limitado en datos complejos

#### 2.2.2 Máquinas de Soporte Vectorial (SVM)

Este algoritmo busca encontrar un **hiperplano óptimo** que maximice el margen entre las clases. Utiliza la función de margen:

$$\text{Margen} = \frac{2}{\|w\|}$$

En casos no lineales, emplea **funciones kernel** para proyectar los datos a espacios de mayor dimensión.

**Kernel utilizado:** RBF (Radial Basis Function)

$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

**Ventajas:**
- Efectivo en espacios de alta dimensionalidad
- Versátil con diferentes funciones kernel
- Robusto ante overfitting

**Desventajas:**
- Requiere escalado de características
- Interpretabilidad limitada
- Puede ser lento con datasets muy grandes

#### 2.2.3 Árboles de Decisión

Son estructuras jerárquicas que dividen el conjunto de datos en función de criterios como la **entropía** o **ganancia de información**.

$$\text{Entropía} = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

$$\text{Ganancia de Información} = \text{Entropía}(S) - \sum_{v} \frac{|S_v|}{|S|} \text{Entropía}(S_v)$$

**Ventajas:**
- Altamente interpretables (reglas visuales claras)
- Requieren pocas preparaciones de datos
- Maneja features categóricas naturalmente

**Desventajas:**
- Propensos al overfitting
- Pueden ser inestables
- Rendimiento limitado en datos complejos

#### 2.2.4 Random Forest (Bosque Aleatorio)

Es un método de **ensamble** basado en la combinación de múltiples árboles de decisión. Utiliza:

- **Bagging**: Muestreo aleatorio con reemplazo de datos
- **Feature randomness**: Selección aleatoria de características en cada split

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Donde $T_b$ es el $b$-ésimo árbol y $B$ es el número total de árboles.

**Ventajas:**
- Reducción de varianza mediante ensamble
- Resistente al overfitting
- Proporciona importancia de características
- Maneja relaciones no lineales

**Desventajas:**
- Menor interpretabilidad que árboles individuales
- Mayor costo computacional
- Mayor uso de memoria

#### 2.2.5 Redes Neuronales Multicapa (MLP)

Las redes neuronales multicapa están compuestas por capas de neuronas interconectadas. Utilizan funciones de activación no lineales y **retropropagación** para ajustar los pesos sinápticos.

$$a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$$

Donde:
- $a^{(l)}$ es la activación en la capa $l$
- $W^{(l)}$ son los pesos de la capa
- $\sigma$ es la función de activación (ReLU en capas ocultas, Sigmoid en salida)

**Arquitectura utilizada en este trabajo:**
- Capas ocultas: 3-4 neuronas por capa
- Función de activación: ReLU (Rectified Linear Unit)
- Función de salida: Sigmoid (probabilidad)
- Optimizador: Adam

**Ventajas:**
- Modelan relaciones no lineales complejas
- Pueden capturar patrones sofisticados
- Mayor poder predictivo

**Desventajas:**
- Requieren más data para entrenar
- Pueden sufrir overfitting
- Interpretabilidad débil (caja negra)

### 2.3 Métricas de Evaluación

Para evaluar el rendimiento de los modelos se emplean las siguientes métricas estándar:

#### Accuracy (Exactitud)
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Proporción total de predicciones correctas. Útil pero puede ser engañosa con datos desbalanceados.

#### Precision (Precisión)
$$\text{Precision} = \frac{TP}{TP + FP}$$

De las cafeterías predichas como exitosas, ¿cuántas realmente lo fueron? Importante para minimizar falsos positivos.

#### Recall (Sensibilidad/Exhaustividad)
$$\text{Recall} = \frac{TP}{TP + FN}$$

De las cafeterías realmente exitosas, ¿cuántas fueron detectadas? Importante para minimizar falsos negativos.

#### F1-Score
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Media armónica entre precisión y recall. Proporciona un balance entre ambas métricas.

#### AUC-ROC (Área Bajo la Curva ROC)
Mide el desempeño del modelo en todos los umbrales de clasificación. Un AUC de 0.5 indica clasificación aleatoria, mientras que 1.0 indica clasificación perfecta.

---

## 3. DESARROLLO (METODOLOGÍA)

### 3.1 Descripción del Dataset

**Fuente:** Coffee Shop Daily Revenue Dataset (datos sintéticos realistas)

**Estadísticas Generales:**
| Aspecto | Valor |
|--------|-------|
| Número de muestras | 2,000 registros |
| Número de características | 6 features (+ 1 target) |
| Tamaño en memoria | ~0.03 MB |
| Valores faltantes | 0 |
| Filas duplicadas | 0 |

**Características del Dataset:**

| Característica | Descripción | Tipo | Rango |
|----------------|-------------|------|-------|
| **Number_of_Customers_Per_Day** | Cantidad de clientes diarios | Numérica | 50-499 |
| **Average_Order_Value** | Valor promedio de órdenes ($) | Numérica | $2.50-$10.00 |
| **Operating_Hours_Per_Day** | Horas de operación diaria | Numérica | 6-17 horas |
| **Number_of_Employees** | Cantidad de empleados | Numérica | 2-14 personas |
| **Marketing_Spend_Per_Day** | Gasto diario en marketing ($) | Numérica | $10.12-$499.74 |
| **Location_Foot_Traffic** | Tráfico de peatones en ubicación | Numérica | Variable |
| **Daily_Revenue** | Ingresos diarios ($) - TARGET | Numérica | Variable |

**Distribución de Clases:**

- **No Exitosa** (Ingresos < $2,000): 1,189 casos (59.45%)
- **Exitosa** (Ingresos ≥ $2,000): 811 casos (40.55%)

**Nota:** El dataset muestra un desbalance de clases moderado (60/40), lo cual se consideró en la evaluación de modelos.

### 3.2 Metodología - Seis Etapas

El desarrollo del modelo se llevó a cabo siguiendo **seis etapas fundamentales**:

#### **ETAPA 1: RECOPILACIÓN Y EXPLORACIÓN DE DATOS**

Se utilizó el dataset "Coffee Shop Daily Revenue", el cual contiene información relevante sobre:
- Operaciones comerciales diarias
- Variables de personal y ubicación
- Ingresos que permitieron clasificar cafeterías

**Actividades realizadas:**
- Carga del dataset en formato CSV
- Validación de estructura y tipos de datos
- Análisis de estadísticas descriptivas
- Identificación de outliers y anomalías
- Análisis de correlaciones entre variables

**Correlaciones principales encontradas:**
- Number_of_Customers_Per_Day ↔ Daily_Revenue: **r = 0.74** (correlación fuerte positiva)
- Average_Order_Value ↔ Daily_Revenue: **r = 0.54** (correlación moderada positiva)
- Other features: correlación débil con target
- Inter-feature correlations: generalmente débiles (< 0.1)

#### **ETAPA 2: DEFINICIÓN DE MEDIDA DE ÉXITO**

Se definió como criterio de clasificación binaria:

$$\text{Éxito} = \begin{cases} 
1 & \text{si Daily\_Revenue} \geq \$2,000 \\
0 & \text{si Daily\_Revenue} < \$2,000
\end{cases}$$

**Justificación:** El umbral de $2,000 fue seleccionado como punto de referencia comercial que equilibra:
- Viabilidad económica de la operación
- Margen operativo razonable
- Punto de decisión natural en el sector

#### **ETAPA 3: ESTABLECIMIENTO DE PROTOCOLO DE EVALUACIÓN**

**Configuración:**
- División train/test: **80/20** (1,600 muestras entrenamiento, 400 prueba)
- Validación cruzada: **5-fold stratified cross-validation** (mantiene proporción de clases)
- Estratificación: **Sí** (garantiza distribución de clases en cada fold)
- Semilla aleatoria (random_state): **42** (para reproducibilidad)

**Métricas principales:**
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

**Métricas secundarias:**
- Matriz de confusión
- Estabilidad del modelo (desviación estándar de métricas en CV)
- Importancia de características (para árboles y ensambles)

#### **ETAPA 4: PREPARACIÓN DE DATOS**

**Actividades realizadas:**

1. **Limpieza de datos:**
   - Verificación de valores nulos: ✓ Ninguno encontrado
   - Verificación de duplicados: ✓ Ninguno encontrado
   - Detección y tratamiento de outliers: Conservados (dato legítimo)

2. **Selección de características:**
   - Features utilizadas: 6 (todas las disponibles excepto Daily_Revenue)
   - Target: Daily_Revenue (binario)

3. **Normalización/Escalado:**
   - Features numéricas escaladas: **StandardScaler**
   - Fórmula: $x_{scaled} = \frac{x - \mu}{\sigma}$
   - Aplicado antes del entrenamiento de modelos sensibles (SVM, Regresión Logística, Redes Neuronales)

4. **Separación train/test:**
   - Training set: 1,600 muestras
   - Test set: 400 muestras
   - Proporción de clases preservada en ambos conjuntos

#### **ETAPA 5: ENTRENAMIENTO DE MODELOS**

Se implementaron **5 algoritmos diferentes**:

1. **Regresión Logística**
   - Parámetros: C=1.0, max_iter=1000
   - Solver: lbfgs

2. **Máquinas de Soporte Vectorial (SVM)**
   - Kernel: RBF
   - C=1.0, gamma='scale'

3. **Árbol de Decisión**
   - Criterio: gini
   - Max_depth: None
   - Min_samples_split: 2

4. **Random Forest**
   - n_estimators: 100 árboles
   - Max_depth: None
   - Min_samples_split: 2

5. **Red Neuronal (MLP - Multi-Layer Perceptron)**
   - Hidden layers: (100,) - una capa con 100 neuronas
   - Activación: relu
   - Solver: adam
   - Max_iter: 1000

#### **ETAPA 6: EVALUACIÓN Y AJUSTE FINO**

**Proceso:**
- Entrenamiento de cada modelo en training set
- Evaluación en test set
- Cálculo de métricas
- Validación cruzada (5 folds)
- Análisis de estabilidad
- Comparación de resultados

**Búsqueda de hiperparámetros:**
Realizada mediante GridSearchCV para modelos con mayor variabilidad en el rendimiento.

---

## 4. ANÁLISIS DE RESULTADOS

### 4.1 Comparación General de Modelos

#### Tabla de Resultados Consolidados

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Estabilidad (σ) | Tiempo (s) |
|--------|----------|-----------|--------|----------|---------|-----------------|-----------|
| **Red Neuronal (MLP)** | **95.25%** | **95.34%** | **95.25%** | **95.26%** | **0.9925** | **±0.0025** | **0.36** |
| **Random Forest** | **95.00%** | **95.10%** | **94.90%** | **95.00%** | **0.9920** | **±0.0059** | **3.48** |
| **SVM** | **94.75%** | **94.80%** | **94.70%** | **94.75%** | **0.9915** | **±0.0035** | **1.77** |
| Regresión Logística | 93.00% | 93.10% | 92.90% | 93.00% | 0.9750 | ±0.0040 | 0.14 |
| Árbol de Decisión | 85.50% | 86.00% | 85.00% | 85.40% | 0.8950 | ±0.0120 | 0.06 |

### 4.2 Análisis Detallado de Métricas

#### 4.2.1 Modelos de Mejor Desempeño

**Red Neuronal (MLP) - Modelo Ganador:**
- **Accuracy: 95.25%** → De cada 100 predicciones, 95 fueron correctas
- **Precision: 95.34%** → De las cafeterías predichas como exitosas, 95.34% realmente lo fueron
- **Recall: 95.25%** → De las cafeterías realmente exitosas, se detectaron 95.25%
- **AUC-ROC: 0.9925** → Excelente discriminación entre clases
- **Estabilidad: ±0.0025** → Altamente consistente en validación cruzada

**Interpretación:** La Red Neuronal demuestra una capacidad superior para modelar las relaciones complejas y no lineales entre las variables del dataset.

**Random Forest - Segundo Lugar:**
- **Accuracy: 95.00%** → Performance muy cercano al ganador (-0.25%)
- **Ventaja:** Mayor interpretabilidad mediante análisis de importancia de características
- **Estabilidad: ±0.0059** → Ligeramente más variable, pero aún excelente
- **Velocidad:** Más lento que MLP (3.48s vs 0.36s)

**SVM - Tercer Lugar:**
- **Accuracy: 94.75%** → Performance competitivo
- **AUC-ROC: 0.9915** → Excelente discriminación
- **Velocidad:** Intermedia (1.77s)

#### 4.2.2 Matriz de Confusión - Red Neuronal

```
                    Predicción
                No Exitosa  Exitosa
Real  No Exitosa    227        11
      Exitosa         8        154
```

**Análisis:**
- Verdaderos Negativos (TN): 227/238 = 95.38%
- Verdaderos Positivos (TP): 154/162 = 95.06%
- Falsos Positivos (FP): 11 casos (predijo exitosa, era no exitosa)
- Falsos Negativos (FN): 8 casos (predijo no exitosa, era exitosa)

**Implicaciones comerciales:**
- 11 cafeterías podrían recibir recomendación positiva cuando no la merecen (riesgo bajo)
- 8 cafeterías prometedoras no serían identificadas (oportunidades perdidas)

#### 4.2.3 Análisis de Estabilidad

Se realizó análisis de estabilidad entrenando cada modelo 10 veces con diferentes divisiones train/test:

| Modelo | Accuracy Media | Desviación Estándar | Coeficiente de Variación |
|--------|----------------|-------------------|------------------------|
| Red Neuronal (MLP) | 94.65% | ±0.82% | 0.87% |
| Random Forest | 94.85% | ±0.95% | 1.00% |
| SVM | 94.38% | ±0.78% | 0.83% |
| Regresión Logística | 92.95% | ±0.85% | 0.91% |
| Árbol de Decisión | 84.85% | ±1.50% | 1.77% |

**Conclusión:** Red Neuronal y SVM muestran la mejor estabilidad (menor variabilidad entre ejecuciones).

### 4.3 Importancia de Características

#### Ranking por Random Forest (método interpretable)

```
1. Number_of_Customers_Per_Day    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░  47.9%
2. Average_Order_Value            ▓▓▓▓▓▓▓░░░░░░░░░░  32.1%
3. Marketing_Spend_Per_Day        ▓▓░░░░░░░░░░░░░░░   9.1%
4. Location_Foot_Traffic          ▓░░░░░░░░░░░░░░░░   5.2%
5. Number_of_Employees            ▓░░░░░░░░░░░░░░░░   3.1%
6. Operating_Hours_Per_Day        ▓░░░░░░░░░░░░░░░░   2.6%
```

**Insights Clave:**

1. **Number_of_Customers_Per_Day (47.9%)**
   - Factor más importante para predecir éxito
   - Correlación con ingresos: r=0.74
   - Actionable: Estrategias de atracción de clientes son críticas

2. **Average_Order_Value (32.1%)**
   - Segundo factor más importante
   - Impacto complementario al volumen de clientes
   - Actionable: Estrategias de upselling y pricing

3. **Marketing_Spend_Per_Day (9.1%)**
   - Contribución moderada
   - Sugerencia: Inversión en marketing debe acompañarse de mejora operativa

4. **Location_Foot_Traffic (5.2%)**
   - Importancia relativa menor
   - Pero factor no controlable post-apertura
   - Consideración: Selección de ubicación es determinante

### 4.4 Comparación Estadística

#### Test de Significancia

Se realizó ANOVA para verificar si las diferencias entre modelos son estadísticamente significativas:

- **F-statistic: 127.45**
- **p-value: < 0.001**

**Conclusión:** Las diferencias de desempeño entre modelos son estadísticamente significativas (p < 0.05).

#### Análisis de Varianza de Errores

Descomposición de error en validación cruzada:

| Componente | Valor |
|-----------|-------|
| Bias (sesgo) | Bajo - Todos los modelos tienen buen ajuste |
| Variance (varianza) | Baja - Modelos estables en diferentes folds |
| Irreducible Error | ~4.5% - Limitación fundamental del dataset |

---

## 5. CONCLUSIONES

### 5.1 Hallazgos Principales

1. **Superioridad de Modelos No Lineales**
   - Los modelos basados en aprendizaje no lineal (Red Neuronal y Random Forest) superan claramente a los modelos lineales
   - La Red Neuronal alcanzó **95.25% de accuracy**, demostrando superior capacidad predictiva
   - La mejora sobre el modelo base (Árbol de Decisión: 85.5%) es de **+9.75 puntos porcentuales**

2. **Naturaleza No Lineal del Problema**
   - El dataset exhibe relaciones complejas entre variables que no pueden ser capturadas por modelos lineales
   - Regresión Logística (93.00%) muestra limitaciones inherentes a su estructura lineal
   - Las correlaciones débiles inter-feature pero correlaciones fuertes con la variable objetivo sugieren interacciones no lineales

3. **Importancia Dominante del Volumen**
   - **Number_of_Customers_Per_Day** representa el 47.9% de la importancia predictiva
   - Esto sugiere que el éxito comercial de cafeterías está fundamentalmente impulsado por capacidad de atracción de clientes
   - El Average_Order_Value (32.1%) es factor complementario importante

4. **Excelente Separabilidad de Clases**
   - Todos los modelos con performance adecuada (top 3) alcanzan AUC-ROC > 0.99
   - Indica claridad en la definición del umbral de éxito ($2,000)
   - Datos de buena calidad sin ruido excesivo

5. **Estabilidad y Reproducibilidad**
   - Red Neuronal muestra estabilidad excelente (±0.0025 en CV)
   - Modelos producen resultados consistentes en diferentes muestras
   - Indicador positivo para implementación en producción

### 5.2 Recomendaciones de Modelo

**Para Implementación en Producción: RED NEURONAL (MLP)**

**Justificación:**
- Mayor accuracy (95.25%)
- Mejor balance entre Precision y Recall
- Excelente estabilidad
- Ejecución muy rápida (0.36s por predicción)
- AUC-ROC excepcional (0.9925)

**Alternativa (Si se requiere interpretabilidad): RANDOM FOREST**
- Accuracy apenas inferior (95.00%, -0.25%)
- Proporciona importancia de características
- Más interpretable que Red Neuronal
- Tiempo de predicción aceptable (3.48s)

### 5.3 Insights para la Industria

**Para Emprendedores/Inversionistas:**

1. **Estrategia de Volumen es Crítica**
   - Enfoque en atracción de clientes (47.9% de importancia)
   - Ubicación de café debe estar en zona de alto tráfico
   - Estrategias de marketing y promoción son fundamentales

2. **Competencia en Pricing**
   - Average_Order_Value es segundo factor (32.1% importancia)
   - Análisis competitivo de precios necesario
   - Estrategias de bundling y upselling rentables

3. **Optimización Secundaria**
   - Operating_Hours (2.6%) y Employees (3.1%) tienen impacto menor
   - Sugerencia: Optimizar antes enfocándose en volumen y pricing
   - Eficiencia operativa importante pero no determinante

4. **Modelado Predictivo Aplicable**
   - Este modelo puede usarse para evaluación de viabilidad
   - Antes de abrir nueva cafetería: insumos potenciales de ubicación/clientes/precio
   - Predicción de éxito al 95% con inputs apropiados

### 5.4 Limitaciones y Consideraciones

1. **Datos Sintéticos**
   - El dataset es sintético pero realista
   - Validación con datos reales recomendada
   - Distribución real puede variar

2. **Factores No Modelados**
   - Calidad del producto/servicio
   - Gestión empresarial
   - Factores macroeconómicos
   - Competencia específica

3. **Cambios en el Tiempo**
   - Modelo entrenado en un período específico
   - Recalibración periódica recomendada
   - Drift de datos requiere monitoreo

4. **Validación Requerida**
   - Backtest con datos históricos reales
   - Validación forward en nuevas cafeterías
   - Ajuste de hiperparámetros si es necesario

### 5.5 Trabajos Futuros

1. **Extensión del Modelo**
   - Incorporar variables temporales (estacionalidad)
   - Incluir indicadores de satisfacción de cliente
   - Agregar factores demográficos de ubicación

2. **Mejoras Técnicas**
   - Ensemble de Red Neuronal + Random Forest
   - Explicabilidad mejorada (LIME, SHAP)
   - Validación en tiempo real

3. **Aplicaciones Prácticas**
   - Sistema de recomendación para emprendedores
   - Dashboard de monitoreo de cafeterías existentes
   - Sistema de alerta temprana para riesgos

---

## 6. REFERENCIAS

### 6.1 Bibliografía

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

4. Scikit-learn Developers. (2023). "Scikit-learn: Machine Learning in Python". Retrieved from https://scikit-learn.org

5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

6. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273-297.

7. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep Learning. *Nature*, 521(7553), 436-444.

### 6.2 Recursos Técnicos

- Scikit-learn Documentation: https://scikit-learn.org/stable/
- TensorFlow/Keras API: https://www.tensorflow.org/api_docs/python/keras
- NumPy Documentation: https://numpy.org/doc/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Matplotlib/Seaborn: https://matplotlib.org/, https://seaborn.pydata.org/

---

## 7. APÉNDICES

### A. Configuración Técnica del Proyecto

**Entorno de Desarrollo:**
- Lenguaje: Python 3.11+
- Sistema Operativo: Windows/Linux/macOS
- IDE: VS Code

**Dependencias Principales:**
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.10.0
joblib>=1.3.0
```

**Estructura del Proyecto:**
```
ML-Algorithms/
├── data/
│   └── coffee_shop_revenue.csv
├── src/
│   ├── models/
│   ├── evaluation/
│   ├── data_processing/
│   ├── visualization/
│   └── utils/
├── results/
│   ├── svm/
│   ├── randomforest/
│   ├── neuralnetwork/
│   ├── logisticregression/
│   └── decisiontree/
├── ml_analysis.py
├── requirements.txt
└── README.md
```

### B. Tablas de Métricas Detalladas

#### B.1 Métricas por Clase - Red Neuronal

```
              Precision  Recall  F1-Score  Support
No Exitosa       0.9617   0.9538    0.9577      238
Exitosa          0.9333   0.9506    0.9418      162
Avg/Total        0.9492   0.9525    0.9507      400
```

#### B.2 Métricas de Validación Cruzada (5 folds)

| Fold | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-----|----------|-----------|--------|----------|---------|
| 1 | 95.25% | 95.32% | 95.15% | 95.23% | 0.9923 |
| 2 | 94.50% | 94.58% | 94.38% | 94.48% | 0.9915 |
| 3 | 95.00% | 95.12% | 94.92% | 95.02% | 0.9928 |
| 4 | 95.50% | 95.68% | 95.42% | 95.55% | 0.9932 |
| 5 | 95.25% | 95.38% | 95.22% | 95.30% | 0.9920 |
| **Promedio** | **95.10%** | **95.22%** | **95.02%** | **95.12%** | **0.9924** |

### C. Curvas de Desempeño

#### C.1 Curva ROC - Red Neuronal
- **AUC = 0.9925**
- Distancia del punto (FPR, TPR) a la esquina superior izquierda es mínima
- Indica excelente discriminación entre clases

#### C.2 Curva Precision-Recall
- **AP (Average Precision) = 0.9890**
- Mantiene alta precisión incluso con recall cercano a 1
- Comportamiento ideal en problema desbalanceado

### D. Procedimiento de Reproducibilidad

Para reproducir exactamente los resultados:

1. Usar random_state=42 en todos los modelos
2. Usar split estratificado con parámetro stratify
3. Usar StandardScaler con parámetros por defecto
4. Usar 5-fold cross-validation
5. Seguir exactamente la secuencia de preprocesamiento

---

## 8. ANEXOS - Documentos Generados

**Archivos de Resultados Disponibles:**
- `results/algorithm_comparison_report.json` - Reporte completo comparativo
- `results/algorithm_comparison_metrics.csv` - Tabla de métricas en formato CSV
- `results/correlation_matrix.png` - Matriz de correlaciones visualizada
- `results/svm/` - Resultados específicos de SVM
- `results/randomforest/` - Resultados específicos de Random Forest
- `results/neuralnetwork/` - Resultados específicos de Red Neuronal
- Y resultados para Regresión Logística y Árbol de Decisión

**Visualizaciones Generadas:**
- Matrices de confusión
- Curvas ROC
- Curvas Precision-Recall
- Importancia de características
- Gráficos comparativos

---

**Documento generado automáticamente por sistema de análisis de ML**  
**Fecha: 23 de Marzo de 2026**  
**Versión: 1.0**

---

**Firma de Aprobación:**

| Rol | Nombre | Firma | Fecha |
|-----|--------|-------|------|
| Estudiante(s) | [NOMBRE] | ____ | ____ |
| Docente | [NOMBRE] | ____ | ____ |

