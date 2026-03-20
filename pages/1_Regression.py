import streamlit as st
import pandas as pd
from joblib import load
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
#import pyautogui


st.set_page_config(page_title="Regression", page_icon=":bar_chart:")


# Cargar el modelo de regresión
regressor = load('randomforest_model_reg.joblib')

# Cargar el encoder
#with open('encoderpipeline.pickle', 'rb') as f:
#    encoder = pickle.load(f)

# Inicializar variables
field_edad = field_nro_personas_trabajo = 1
field_horas_trabajo = 0

field_select_sexo = "Masculino"
field_select_NIVELI = 'Primaria'
field_select_Informal_P = 'Empleo Formal'

# Streamlit app
st.title("Modelo de Regresión")
st.markdown("##### Si colocas un valor negativo, aparecerá un error y no podrás completar otros campos. La predicción será incorrecta.")

# Sidebar para la entrada del usuario
st.sidebar.header("Campos a Evaluar")

# Entrada de la usuario para "Edad"
field_edad = st.sidebar.number_input("**edad (Min=0, Max=99)**", min_value=0, value=int(field_edad))

# Entrada del usuario para "nro_personas_trabajo"
field_nro_personas_trabajo = st.sidebar.number_input("**nro_personas_trabajo (Min=1, Max=31)**", min_value=1, value=int(field_nro_personas_trabajo))

# Entrada del usuario para "horas_trabajo"
field_horas_trabajo = st.sidebar.number_input("**horas_trabajo (Min=0, Max=112)**", min_value=0, value=int(field_horas_trabajo))

# Entrada del usuario para "Sexo"
st.sidebar.markdown("<h1 style='font-size: 24px;'>Sexo</h1>", unsafe_allow_html=True)
field_select_sexo = st.sidebar.selectbox("Select Sexo", ["Masculino", "Femenino"], index=["Masculino", "Femenino"].index(field_select_sexo))

# Entrada del usuario para "NIVELI"
st.sidebar.markdown("<h1 style='font-size: 24px;'>Nivel Educativo</h1>", unsafe_allow_html=True)
field_select_NIVELI = st.sidebar.selectbox("Select Nivel Educativo", ["Primaria", "Secundaria", "Superior"], index=["Primaria", "Secundaria", "Superior"].index(field_select_NIVELI))

# Entrada del usuario para "Informal_P"
st.sidebar.markdown("<h1 style='font-size: 24px;'>Situación de Informalidad</h1>", unsafe_allow_html=True)
field_select_Informal_P = st.sidebar.selectbox("Select Situación de Informalidad", ["Empleo Formal", "Empleo Informal"], index=["Empleo Formal", "Empleo Informal"].index(field_select_Informal_P))

# Función para resetear las entradas
def reset_inputs():
    global field_edad, field_horas_trabajo, field_select_sexo, field_select_NIVELI, field_select_Informal_P
    
    field_edad = field_nro_personas_trabajo = 1
    field_horas_trabajo = 0

    field_select_sexo = "Masculino"
    field_select_NIVELI = 'Primaria'
    field_select_Informal_P = 'Empleo Formal'

# Botón para predecir
if st.sidebar.button("Predecir"):

    # Validar las entradas

    if all(isinstance(val, (int)) and val >= 1 for val in [field_edad, field_nro_personas_trabajo,]) and all(isinstance(val, (int)) and val >= 0 for val in [field_horas_trabajo,]):
        # Crear un DataFrame con las entradas del usuario
        obs = pd.DataFrame({
            'sexo': [field_select_sexo],
            'edad': [field_edad],
            'nro_personas_trabajo': [field_nro_personas_trabajo],
            'horas_trabajo': [field_horas_trabajo],
            'NIVELI': [field_select_NIVELI],
            'Informal_P': [field_select_Informal_P]
        })

        # Mostrar el DataFrame de entradas para depuración
        st.write("DataFrame de Entradas:")
        st.write(obs)

        #----------------------Pipeline-------------------------
        # Predecir usando el modelo
        target = regressor.predict(obs)

        # Mostrar la predicción con un tamaño de fuente grande usando markdown
        st.markdown(f'<p style="font-size: 40px; color: green;">La predicción del Profit será: S/. {target[0]:,.2f}</p>', unsafe_allow_html=True)

    else:
        st.warning("Rellene todos los espacios en blanco")

# Colocar el botón "Resetear" debajo del botón "Predecir"
if st.sidebar.button("Resetear"):
    # Resetear inputs
    reset_inputs()
