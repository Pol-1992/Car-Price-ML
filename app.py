import sys, subprocess

# instalar dependencias en caliente
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "joblib>=1.3.2",
    "scikit-learn>=1.3.2",
    "pandas>=2.2.2",
    "numpy>=1.26.4",
    "pillow>=11.0.0"
])


import streamlit as st
import pandas as pd
from joblib import load

# Cargar modelo y datos
model = load("random_forest_model.pkl")
df = pd.read_csv("cars_limpio.csv")

# Valores únicos
brands = sorted(df["brand"].unique())
fuels = sorted(df["fuel"].unique())
gears = sorted(df["gear"].unique())

# Sidebar navigation
seccion = st.sidebar.radio("Navegar por", [
    "Portada",
    "Objetivo",
    "Dataset y Limpieza",
    "EDA",
    "Modelos",
    "Predicción",
    "Mejoras Futuras",
    "Agradecimientos"
])

# --------------------- PORTADA ---------------------
if seccion == "Portada":
    st.title("CarPrice AI")
    st.subheader("Predicción inteligente de precios de coches")


    # Imagen centrada
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("Images/logo.png", width=250)

    # Nombre alineado a la derecha
    st.markdown("<p style='text-align: right;'>© Pol Urbano Kinsel</p>", unsafe_allow_html=True)

# --------------------- OBJETIVO ---------------------
elif seccion == "Objetivo":

    st.header("Objetivo del Proyecto")

    st.markdown("""
                
    El objetivo de este proyecto es desarrollar una aplicación interactiva capaz de predecir el **precio estimado de un coche** —desde nuevo hasta usado. 
                 
    Para ello se han aplicado técnicas de **machine learning**, en especial modelos de regresión, con el fin de encontrar patrones y relaciones entre variables como marca, año, kilometraje, potencia, tipo de combustible y transmisión.


    Este tipo de solución puede ser útil para:
    - **Concesionarios**, que buscan estimaciones rápidas y consistentes para fijar precios competitivos.
                
    - **Compradores particulares**, que desean saber si el precio de un coche usado está dentro de un rango razonable.
                 
    - **Plataformas de compra-venta**, que pueden integrar modelos así para mejorar su servicio o recomendar precios.

    En un mercado tan dinámico como el automovilístico, contar con una herramienta de predicción fiable puede suponer una ventaja tanto comercial como informativa.
    """)

# --------------------- DATASET Y LIMPIEZA ---------------------
elif seccion == "Dataset y Limpieza":

    st.header("Dataset y Preprocesamiento")

    st.markdown("""
                
   El conjunto de datos utilizado proviene de **Kaggle** e incluye más de **40.000 registros** de coches usados y no usados puestos a la venta en **Alemania** entre los años **2011 y 2021**.

### Principales pasos de limpieza y preparación:

- **Eliminación de valores nulos y duplicados**.
- **Filtrado de registros sospechosos**: se eliminaron casos con valores idénticos en múltiples campos clave (como año, km, potencia y precio), que podrían corresponder a entradas duplicadas o generadas artificialmente.
- **Recorte de valores extremos** para evitar distorsión en el modelo:
    - Kilometraje máximo: **350.000 km**
    - Precio máximo: **400.000 €**
- **Codificación de variables categóricas** mediante **One-Hot Encoding (dummies)** para hacerlas compatibles con los algoritmos de machine learning.

Este proceso permitió construir un dataset confiable, representativo del mercado de coches usados en Alemania, y óptimo para aplicar modelos de predicción.
                
    """)

# --------------------- EDA ---------------------
elif seccion == "EDA":

    st.header("Exploración de Datos (EDA)")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Combustible", "Transmision","Kilometraje", "Precio", "Precio < 100K", "Correlación"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
           st.image("Images/fuel.png")
        with col2:
            st.markdown("**Insights**")
            st.markdown("""
            - Gasolina: más del 60%  
            - Diésel: segunda opción más frecuente  
            - Eléctricos e híbridos: siguen siendo minoria  
            """)

    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image("Images/gear.png")
        with col2:
            st.markdown("**Insights**")
            st.markdown("""
            - Coches con transmisión manuel son casi el doble de Automatica aunque es posible que esta proporción haya cambiado en años recientes.
            """)

    with tab3:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image("Images/km.png")
        with col2:
            st.markdown("**Insights**")
            st.markdown("""
            - Mayoría de los coches tienen menos de  100.000 km  
            - Mas de 7000 coches tiene menos de 10.000 km 
            """)
        
    
    with tab4:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image("Images/price.png")
        with col2:
            st.markdown("**Insights**")
            st.markdown("""
            - Mayoría de los coches tiene un precio por debajo de los 40.000 €  
            - Precios altos: casos aislados  
            """)

    with tab5:
        col1, col2 = st.columns([2, 1])
        with col1:
             st.image("Images/price -100k.png")
        with col2:
             st.markdown("**Insights**")
             st.markdown("""
            - +80% de los coches tienen un precio de menos de 30.000 €  
            - Rango más común: 10k–20k €  
            """)
        
    with tab6:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image("Images/matriz.png")
        with col2:
            st.markdown("**Insights**")
            st.markdown("""
            - **CV y Precio**: correlación fuerte (0.79)  
            - **Año y Precio**: correlación media (0.46)  
            - **Kilometraje y Año**: correlación negativa (-0.69)  
            """)

# --------------------- MODELOS ---------------------
elif seccion == "Modelos":

    st.header("Modelos Utilizados")

    st.markdown("""
                
    Se probaron los siguientes modelos:
                
    - Random Forest 93%
    - Regresión Lineal 83%
    - Bagging 93%
    - Pasting 87%
    - AdaBoost 91%
    - Gradient Boosting 88%

    **Resultados del modelo Random Forest:**
    - MAE: 2.129 €  
    En promedio, el modelo se equivoca por unos **2.129 euros** al predecir el precio de un coche.
                
    - RMSE: 4.664 €  
    Mide el error total. Cuanto más bajo, mejor. Aquí el modelo comete errores algo más grandes en pocos casos extremos.
                
    - R²: 0.931
    Esto significa que el modelo es capaz de predecir **el 93% del comportamiento del precio** en base a los datos disponibles.
    """)

# --------------------- PREDICCIÓN ---------------------
elif seccion == "Predicción":

    st.header("Predicción Interactiva de Precio")

    brand = st.selectbox("Marca", brands)
    fuel = st.selectbox("Combustible", fuels)
    gear = st.selectbox("Transmisión", gears)
    year = st.number_input("Año", min_value=2011, max_value=2021)
    km = st.number_input("Kilometraje", min_value=0, max_value=350000, step=1000)
    cv = st.number_input("CV (potencia)", min_value=0, max_value=900, step=10)

    if st.button("Predecir Precio"):
        input_df = pd.DataFrame([{
            'brand': brand, 'fuel': fuel, 'gear': gear,
            'year': year, 'km': km, 'cv': cv
        }])

        input_encoded = pd.get_dummies(input_df)
        for col in model.feature_names_in_:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model.feature_names_in_]

        pred = model.predict(input_encoded)[0]
        mae = 2129.18
        st.markdown(f"""
        <h2 style='color: #4CAF50;'>Precio estimado: €{pred:,.2f}</h2>
        """, unsafe_allow_html=True)

        st.info(f"*Este valor puede variar aproximadamente ± €{mae:,.0f} según el modelo.*")

# --------------------- MEJORAS FUTURAS ---------------------
elif seccion == "Mejoras Futuras":

    st.header("Posibles Mejoras")

    st.markdown("""
                
    - Incluir más marcas premium para mejorar la precisión en coches de lujo.
                
    - Añadir más años y datos más recientes.
                
    - Incluir el modelo específico del coche (por ejemplo, Audi A3 vs A6).
                            
    - Incluir nuevas variables como ubicación, estado del coche, número de dueños, etc.
                
    - Implementar validaciones automáticas de entrada.
                
    - Probar modelos más avanzados como **XGBoost** o **Redes Neuronales** que son algoritmos de machine learning más sofisticados que podrían mejorar la precisión de las predicciones.
                
    """)

# --------------------- AGRADECIMIENTOS ---------------------
elif seccion == "Agradecimientos":

    st.header("Muchas Gracias")

    st.markdown("""
    Gracias por visitar este proyecto.  
                
    Si teneis preguntas o sugerencias, ¡estaré encantado de responder!

    © Pol Urbano Kinsel
    """)