import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.optimize import minimize

# --- Introducción ---
st.title("Análisis y Optimización de Portafolio de Inversión")
st.write("Este portafolio presenta un análisis detallado...")

# --- Sidebar para selección de fechas ---
st.sidebar.header("Seleccionar Rango de Fechas")
start_date = st.sidebar.date_input("Fecha de Inicio", pd.to_datetime('2019-01-01'))
end_date = st.sidebar.date_input("Fecha de Fin", pd.to_datetime('2023-01-01'))

# --- Descarga de datos ---
def cargar_datos(activos, start, end):
    data = yf.download(activos, start=start, end=end)['Close']
    return data

activos = ['IVV', 'SCHD', 'AAPL', 'NU', 'QCOM', 'INTC', 'AAL', 'RTX', 'GC=F', 'CL=F']
data = cargar_datos(activos, start_date, end_date)

# --- Visualización de precios de cierre ---
st.header("Evolución de Precios de Cierre")
selected_assets = st.multiselect("Seleccionar Activos", activos, default=['AAPL', 'IVV'])
if selected_assets:
    fig = px.line(data[selected_assets], x=data.index, y=selected_assets, title="Precios de Cierre")
    st.plotly_chart(fig)

# --- Análisis de Correlación ---
st.header("Análisis de Correlación")
corr_matrix = data.corr()
fig_corr = px.imshow(corr_matrix, x=activos, y=activos, color_continuous_scale='RdBu', title="Matriz de Correlación")
st.plotly_chart(fig_corr)

# --- Optimización de Portafolio (fragmento) ---
def port_ret(weights, data):
    # ... (cálculo del retorno del portafolio)
    return np.sum(data.mean() * weights) * 252

def port_vol(weights, data):
    # ... (cálculo de la volatilidad del portafolio)
    return np.sqrt(np.dot(weights.T, np.dot(data.cov() * 252, weights)))

def neg_sharpe(weights, data, rf_rate):
    # ... (cálculo del negativo del Ratio de Sharpe)
    return - (port_ret(weights, data) - rf_rate) / port_vol(weights, data)

# ... (código para la optimización)

# --- Visualización del Portafolio Final ---
st.header("Composición del Portafolio Optimizado")
pesos = [0.20, 0.15, 0.13, 0.08, 0.08, 0.08, 0.03, 0.10, 0.08, 0.05]  # Ejemplo
nombres = ['IVV', 'SCHD', 'AAPL', 'NU', 'QCOM', 'INTC', 'AAL', 'RTX', 'GC=F', 'CL=F']
inversiones = [p * 300000 for p in pesos]  # Asumiendo inversión total de $300,000

df_portafolio = pd.DataFrame({'Activo': nombres, 'Inversión': inversiones, '% del Portafolio': pesos})

fig_pie = px.pie(df_portafolio, values='% del Portafolio', names='Activo', title="Composición del Portafolio")
st.plotly_chart(fig_pie)

st.dataframe(df_portafolio)

