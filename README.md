# CarPrice AI

**CarPrice AI** es una aplicación interactiva desarrollada en Streamlit que permite predecir el precio estimado de un coche usado en Alemania. Basado en un modelo de Machine Learning, el proyecto busca ofrecer una herramienta confiable para concesionarios, compradores y plataformas de compra-venta que necesitan una referencia objetiva de precios.

## Objetivo

El propósito de este proyecto es predecir el precio de coches usados a partir de características técnicas como marca, tipo de combustible, transmisión, año de fabricación, kilometraje y potencia (CV). La aplicación está diseñada para facilitar estimaciones rápidas y visuales, basadas en datos históricos reales.

## Fuente de datos

El conjunto de datos fue obtenido de [Kaggle](https://www.kaggle.com/datasets/ander289386/cars-germany), e incluye más de 40.000 registros de coches usados en Alemania, recopilados entre los años 2011 y 2021. Cada registro contiene información detallada sobre el coche, incluyendo:

- Marca
- Año
- Kilometraje
- Potencia (CV)
- Tipo de combustible
- Tipo de transmisión
- Precio

## Preprocesamiento de los datos

Se realizaron las siguientes tareas de limpieza y transformación:

- Eliminación de valores nulos y duplicados.
- Detección y eliminación de registros sospechosos con valores idénticos en múltiples campos clave.
- Recorte de outliers:
  - Kilometraje máximo: 350.000 km
  - Precio máximo: 400.000 €
- Codificación de variables categóricas mediante One-Hot Encoding.

El resultado fue un dataset limpio, equilibrado y adecuado para entrenar modelos de regresión.

## Exploración de datos (EDA)

Se analizaron las principales variables con visualizaciones que muestran:

- Predominio de coches a gasolina y con transmisión manual.
- Distribución del kilometraje y precios con fuerte sesgo hacia los valores bajos.
- Correlaciones relevantes:
  - CV y precio: fuerte correlación positiva (0.79)
  - Año y precio: correlación moderada (0.46)
  - Año y kilometraje: correlación negativa (-0.69)

## Modelos de Machine Learning

Se probaron distintos algoritmos de regresión:

- Regresión Lineal
- Bagging Regressor
- Pasting Regressor
- AdaBoost
- Gradient Boosting
- Random Forest Regressor

El mejor rendimiento lo obtuvo el modelo **Random Forest**, con los siguientes resultados:

- **MAE**: 2.129 €
- **RMSE**: 4.664 €
- **R²**: 0.931

Esto indica que el modelo explica el 93% de la variabilidad en el precio del coche, con un error medio muy razonable.

## App interactiva

La aplicación está desarrollada en **Streamlit**, e incluye:

- Navegación lateral por secciones: objetivo, dataset, EDA, modelos, predicción, mejoras futuras y agradecimientos.
- Visualización de gráficos y correlaciones.
- Formulario interactivo para predecir el precio de un coche ingresando sus características.
- Resultado visual y estimación del margen de error.

## Mejoras futuras

Algunas posibles extensiones del proyecto:

- Incluir el modelo específico del coche (por ejemplo: Audi A3 vs A6).
- Añadir variables como ubicación, número de dueños o estado general.
- Incluir marcas de alta gama y más datos recientes (post-2021).
- Explorar modelos avanzados como XGBoost o redes neuronales.
- Desplegar la app como API para integración en plataformas externas.

## Autor

**Pol Urbano Kinsel**  