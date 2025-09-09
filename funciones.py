import pandas as pd

def eliminar_fuel_other_en_duplicados(df):
    cols = ["price", "cv", "km", "year"]
    
    # Buscar grupos duplicados según esas columnas
    duplicados = df[df.duplicated(subset=cols, keep=False)]
    
    # Filtrar solo los que tienen fuel == "Other"
    otros_a_eliminar = duplicados[duplicados["fuel"] == "Other"]
    
    # Eliminar esos de df
    df_filtrado = df.drop(index=otros_a_eliminar.index)
    
    return df_filtrado