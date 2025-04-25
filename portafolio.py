import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import pandas_datareader.data as web
from scipy.stats import norm

@st.cache_data
def cargar_datos(archivo):
    data = pd.read_csv(archivo, header=[0, 1], index_col=0, parse_dates=True)
    return data

st.set_page_config(page_title="Inversiones Arriesgadas Internacional", layout="wide")

st.title("Inversiones Arriesgadas Internacional")
st.subheader("Portafolio de Inversión")

# Tabs principales
tabs = st.tabs(["Portafolio", "Análisis de Retornos", "Analisis de Riesgo", "CAPM", "Opciones", "Rentabilidad"])

# --- TAB 1: Portafolio ---
with tabs[0]:
    data = {
        'Activo': ['IVV','SCHD', 'AAPL', 'MSFT', 'QCOM', 'CRSP', 'GOOGL', 'RTX', 'GC=F', 'CL=F'],
        'Inversión': ['$36,000', '$36,000', '$75,000', '$39,000', '$18,000', '$18,000', '$24,000', '$24,000', '$15,000', '$15,000'],
        '% del Portafolio': ['12%', '12%', '25%', '13%', '6%', '6%', '8%', '8%', '5%', '5%'],
        'Justificación': [
            "ETF del S&P 500: base diversificada, bajo costo y riesgo.",
            "Generador de ingresos: empresas con dividendos sólidos, perfil defensivo, menor volatilidad, ideal para balancear el portafolio.",
            "Líder tecnológico: crecimiento estable, alta liquidez, sólida marca global y ecosistema cerrado que genera ingresos recurrentes.",
            "Líder tecnológico, crecimiento estable impulsado por su dominio en la nube (Azure), inteligencia artificial y software empresarial, con un modelo de ingresos recurrentes y fuerte posición financiera.",
            "Pilar tecnológico: protagonista en semiconductores y 5G, crecimiento sostenido en telecomunicaciones y dispositivos móviles.",
            "Innovación en edición genética, con un alto potencial de crecimiento en terapias personalizadas, posición estratégica en el desarrollo de tratamientos para enfermedades genéticas y fuerte pipeline de productos.",
            "Motor digital: liderazgo en publicidad online, inteligencia artificial y servicios cloud. Alta rentabilidad y resiliencia.",
            "Estabilidad defensiva: contratos gubernamentales y baja sensibilidad cíclica. Buen contrapeso frente a sectores volátiles.",
            "Oro: cobertura contra inflación y crisis.",
            "Petróleo: diversificación táctica, alta volatilidad."
        ]
    }
    df = pd.DataFrame(data)

    # Mostrar tabla de portafolio
    st.title("Portafolio de Inversión")
    st.subheader("Distribución y Justificación de Activos")
    st.dataframe(df)

    # Graficar distribución del portafolio (Gráfico de torta)
    df['% del Portafolio'] = df['% del Portafolio'].str.replace('%', '').astype(float)

    fig = px.pie(df, names='Activo', values='% del Portafolio', title='Distribución del Portafolio')
    st.plotly_chart(fig)


    
with tabs[1]:
    #selected_assets = st.multiselect("Seleccionar Activos", closing_prices_filtered.columns, default=closing_prices_filtered.columns[:1])  # Mostrar los primeros 5 por defecto
    data = cargar_datos("data_defi1.csv")  # Asegúrate que 'data_defi.csv' esté en la misma carpeta o ajusta la ruta
    closing_prices = data.xs('Close', axis=1, level=1)[:-90]
    returns = np.log(closing_prices).diff().dropna()  # Calcular los retornos aquí

    # --- Selección de fechas (FIJA) ---
    start_date = '2023-01-01'
    end_date = '2025-01-01'
    closing_prices_filtered = closing_prices.loc[start_date:end_date]
    returns_filtered = returns.loc[start_date:end_date]

    # --- Visualización de Precios de Cierre (Adaptado para Streamlit) ---
    st.header("Evolución de Precios de Cierre")

    # Selección de un solo activo (selectbox)
    selected_asset = st.selectbox("Seleccionar Activo", closing_prices_filtered.columns, index=0)

    # Crear las columnas para mostrar las gráficas lado a lado
    col1, col2 = st.columns(2)

    # Columna 1: Precios de Cierre del activo seleccionado
    with col1:
        if selected_asset:
            # Usar Plotly para la visualización interactiva
            fig_close = px.line(closing_prices_filtered[selected_asset], x=closing_prices_filtered.index, y=selected_asset,
                                title=f"Precio de Cierre de {selected_asset}", labels={'index': 'Fecha', 'value': 'Precio', 'variable': 'Activo'})
            st.plotly_chart(fig_close, use_container_width=True)
        else:
            st.warning("Por favor, selecciona un activo.")

    # Columna 2: Matriz de correlación de los retornos logarítmicos
    with col2:
        desired_ticker_order = ['CRSP', 'AAPL', 'CL=F', 'QCOM', 'GC=F', 'MSFT', 'SCHD', '^GSPC', 'RTX', 'IVV', 'GOOGL'] 
        closing_prices_filtered = closing_prices_filtered.reindex(columns=desired_ticker_order, level=0) 
        returns_filtered = np.log(closing_prices_filtered / closing_prices_filtered.shift(1)).dropna()
        tickers_ordered = closing_prices_filtered.columns.tolist()
        returns_filtered_ordered = returns_filtered[tickers_ordered]
        corr_matrix = returns_filtered_ordered.corr()

        # Crear máscara para ocultar la mitad superior de la matriz de correlación
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.mask(mask)

        # Graficar con Plotly
        fig_corr = px.imshow(
            corr_matrix_masked,
            color_continuous_scale='RdBu_r',
            title="Matriz de Correlación de los Retornos Logarítmicos",
            labels={'color': 'Correlación'},
            text_auto=".2f",
            aspect="equal",
            zmin=-1,
            zmax=1
        )

        fig_corr.update_traces(
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlación: %{z:.2f}<extra></extra>"
        )

        st.plotly_chart(fig_corr, use_container_width=True)


# --- TAB 3: Simulación GBM ---
with tabs[2]:
    data = cargar_datos("data_defi1.csv")

    selected_ticker = st.selectbox(
        "Seleccionar Activo para Simulación GBM",
        ['IVV', 'SCHD', 'AAPL', 'MSFT', 'QCOM', 'CRSP', 'GOOGL', 'RTX', 'GC=F', 'CL=F', '^GSPC'],
        index=0
    )

    n_simulaciones = 200
    n_dias = 100
    dt = 1
    num_simulations_var = 10000
    horizontes = {'Diario': 1, 'Semanal': 5, 'Mensual': 21}
    niveles_confianza = {'VaR 95%': 0.05, 'VaR 99%': 0.01}

    if selected_ticker and data is not None and not data.empty:
        precios_cierre = data.xs('Close', axis=1, level=1)
        returns = np.log(precios_cierre / precios_cierre.shift(1)).dropna()

        try:
            mu = returns[selected_ticker].mean()
            sigma = returns[selected_ticker].std()
            S0 = precios_cierre[selected_ticker].dropna().iloc[-1]

            # Simulación de trayectorias GBM
            simulations = np.zeros((n_dias, n_simulaciones))
            for j in range(n_simulaciones):
                prices_sim = [S0]
                for t in range(1, n_dias):
                    drift = (mu - 0.5 * sigma**2) * dt
                    shock = sigma * np.random.normal()
                    S_t = prices_sim[-1] * np.exp(drift + shock)
                    prices_sim.append(S_t)
                simulations[:, j] = prices_sim

            dias = list(range(n_dias))
            p5 = np.percentile(simulations, 5, axis=1)
            p95 = np.percentile(simulations, 95, axis=1)
            mean = np.mean(simulations, axis=1)

            # Gráfico de simulación
            fig_gbm = go.Figure()
            fig_gbm.add_trace(go.Scatter(x=dias, y=mean, mode='lines', name='Media', line=dict(color='blue')))
            fig_gbm.add_trace(go.Scatter(x=dias, y=p5, mode='lines', name='P5', line=dict(color='green')))
            fig_gbm.add_trace(go.Scatter(x=dias, y=p95, mode='lines', name='P95', line=dict(color='orange')))
            fig_gbm.update_layout(title=f"Simulación GBM para {selected_ticker}",
                                  xaxis_title="Días", yaxis_title="Precio", template="plotly_white")

            # VaR Simulado
            fila_resultado = {}
            for nombre_h, T in horizontes.items():
                simulaciones_var = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.randn(num_simulations_var))
                perdidas = S0 - simulaciones_var
                for nombre_var, alpha in niveles_confianza.items():
                    var_value = np.percentile(perdidas, 100 * (1 - alpha))
                    fila_resultado[f'{nombre_var} {nombre_h}'] = round(var_value, 2)
            df_var = pd.DataFrame(fila_resultado, index=[selected_ticker]).T
            df_var.columns = ["VaR (USD)"]

            # Mostrar en tres columnas
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.subheader("Simulación de Movimiento Browniano Geométrico")
                st.plotly_chart(fig_gbm, use_container_width=True)

            with col2:
                st.subheader("VaR Simulado")
                st.dataframe(df_var.style.format({"VaR (USD)": "${:,.2f}"}), use_container_width=True)

            with col3:
                st.subheader("VaR Paramétrico")
                try:
                    investment = 300000
                    z_score = -1.65  # 95% nivel de confianza
                    ticker_returns = returns[selected_ticker]

                    mean_return = ticker_returns.mean()
                    volatility = ticker_returns.std()
                    daily_var = mean_return + z_score * volatility
                    monthly_var = daily_var * np.sqrt(20)
                    var_investment = monthly_var * investment

                    df_var_param = pd.DataFrame({
                        "Media Retornos": [mean_return],
                        "Volatilidad": [volatility],
                        "VaR Diario (%)": [daily_var],
                        "VaR Mensual (%)": [monthly_var],
                        "VaR sobre $300K (USD)": [var_investment]
                    }, index=[selected_ticker]).T
                    df_var_param.columns = ['Valor']

                    st.dataframe(df_var_param.style.format({
                        "Valor": lambda x: (
                            f"${x:,.2f}" if "USD" in df_var_param.index[df_var_param["Valor"] == x][0]
                            else f"{x:.4%}" if "%" in df_var_param.index[df_var_param["Valor"] == x][0]
                            else f"{x:.4f}"
                        )
                    }), use_container_width=True)

                except Exception as e:
                    st.error(f"Error al calcular VaR paramétrico: {e}")

        except KeyError as e:
            st.error(f"Error al simular {selected_ticker}: Ticker '{e}' no encontrado en los datos.")
        except Exception as e:
            st.error(f"Error al simular {selected_ticker}: {e}")
   

    
# --- TAB 5: CAPM ---
with tabs[3]:
    st.header("Análisis de Riesgo Sistemático - CAPM")

    # Descargar la tasa libre de riesgo (T-bill 3 meses)
    try:
        tbill_data = web.DataReader('TB3MS', 'fred', start_date, end_date)
        tbill_rate = tbill_data.dropna() / 100 / 360  # Tasa diaria
        tbill_rate = tbill_rate.reindex(returns.index, method='ffill')
    except Exception as e:
        st.error(f"No se pudo descargar la tasa libre de riesgo: {e}")

    try:
        capm_results = pd.DataFrame(columns=['Alpha', 'Beta', 'R-squared'])
        tickers = ['IVV', 'SCHD', 'AAPL', 'MSFT', 'QCOM', 'CRSP', 'GOOGL', 'RTX', 'GC=F', 'CL=F']
        for ticker in tickers:
            if ticker == '^GSPC':  # Excluir benchmark
                continue
            try:
                Y = returns[ticker] - tbill_rate['TB3MS']
                X = returns['^GSPC'] - tbill_rate['TB3MS']
                df = pd.concat([Y, X], axis=1).dropna()
                df.columns = ['Y', 'X']
                df = sm.add_constant(df)
                model = sm.OLS(df['Y'], df[['const', 'X']]).fit()
                capm_results.loc[ticker] = [model.params['const'], model.params['X'], model.rsquared]
            except Exception as e:
                st.warning(f"No se pudo calcular CAPM para {ticker}: {e}")

        # Mostrar tabla y gráfica en columnas
        st.subheader("Alpha, Beta y R² de los Activos")
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(capm_results.style.format({
                "Alpha": "{:.4f}",
                "Beta": "{:.4f}",
                "R-squared": "{:.4f}"
            }), use_container_width=True)

        with col2:
            fig_beta = px.bar(
                capm_results.reset_index(),
                x="index", y="Beta",
                title="Beta de los Activos respecto al S&P 500",
                labels={"index": "Activo", "Beta": "Valor Beta"},
                color_discrete_sequence=["#00BFC4"]
            )
            fig_beta.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_beta, use_container_width=True)

        # Análisis puntual con scatter
        with st.expander("Análisis detallado por activo (CAPM Scatter)"):
            selected_capm = st.selectbox("Seleccionar activo", capm_results.index.tolist())
            Y = returns[selected_capm] - tbill_rate['TB3MS']
            X = returns['^GSPC'] - tbill_rate['TB3MS']
            df_capm = pd.concat([Y, X], axis=1).dropna()
            df_capm.columns = ['Exceso Retorno Activo', 'Exceso Retorno Mercado']
            fig_scatter = px.scatter(
                df_capm,
                x='Exceso Retorno Mercado',
                y='Exceso Retorno Activo',
                trendline='ols',
                title=f"CAPM: {selected_capm} vs. S&P 500"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    except Exception as e:
        st.error(f"Error general en el análisis CAPM: {e}")



with tabs[4]:
    st.header("Cobertura del Portafolio con Opciones (Black-Scholes)")

    # --- Parámetros
    valor_portafolio = 300_000
    S = 100       # Precio actual (puedes hacer selectbox dinámico más adelante)
    K = 100
    T = 0.5
    r = 0.01
    sigma = 0.20

    # --- Beta de cada activo e inversión
    activos = [
        {"activo": "IVV", "inversion": 36000, "beta": 0.994557},
        {"activo": "SCHD", "inversion": 36000, "beta": 0.693977},
        {"activo": "AAPL", "inversion": 75000, "beta": 1.033159},
        {"activo": "MSFT", "inversion": 39000, "beta": 1.184513},
        {"activo": "QCOM", "inversion": 18000, "beta": 1.711365},
        {"activo": "CRSP", "inversion": 18000, "beta": 1.821759},
        {"activo": "GOOGL", "inversion": 24000, "beta": 1.318200},
        {"activo": "RTX", "inversion": 24000, "beta": 0.397839},
        {"activo": "GC=F", "inversion": 15000, "beta": 0.097596},
        {"activo": "CL=F", "inversion": 15000, "beta": 0.165583}
    ]
    beta_portafolio = sum([a["inversion"] / valor_portafolio * a["beta"] for a in activos])

    # --- Función Black-Scholes
    def black_scholes_option(S, K, T, r, sigma, option_type='put'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # --- Cálculo cobertura
    n_contratos = int(np.ceil((valor_portafolio * beta_portafolio) / (S * 100)))
    put_price = black_scholes_option(S, K, T, r, sigma, option_type='put')
    costo_total_cobertura = put_price * 100 * n_contratos

    # --- Mostrar métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Precio Opción Put', f"${put_price:.2f}")
    col2.metric("Beta Portafolio", f"{beta_portafolio:.4f}")
    col3.metric("Contratos necesarios", f"{n_contratos}")
    col4.metric("Costo Total Cobertura", f"${costo_total_cobertura:,.2f}")

    # --- Simulación de Payoff
    S_T = np.linspace(50, 150, 100)
    portafolio_sin_cobertura = beta_portafolio * valor_portafolio * (S_T / S - 1)
    portafolio_put = np.maximum(K - S_T, 0) * 100 * n_contratos
    portafolio_cubierto = portafolio_sin_cobertura + portafolio_put - costo_total_cobertura

    # --- Gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(S_T, portafolio_sin_cobertura, '--', label='Sin Cobertura')
    ax.plot(S_T, portafolio_cubierto, label='Con Cobertura (Put)', linewidth=2)
    ax.axhline(0, color='black', linestyle=':')
    ax.set_xlabel("Precio subyacente al vencimiento (IVV)")
    ax.set_ylabel("Payoff neto del portafolio")
    ax.set_title("Cobertura con Opción Put Europea (Black-Scholes)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # --- Análisis Técnico Opcional ---
    with st.expander("Velas Japonesas con Bandas de Bollinger"):
        tickers_simular = ['IVV','SCHD', 'AAPL', 'MSFT', 'QCOM', 'CRSP', 'GOOGL', 'RTX', 'GC=F', 'CL=F']
        data_boll = yf.download(tickers_simular, period="6mo", interval="1d", group_by='ticker', auto_adjust=True)

        selected_boll = st.selectbox("Seleccionar activo", tickers_simular, key="activo_seleccionado")



        def plot_bollinger_candlestick(ticker, df):
            df = df.dropna()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['Upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
            df['Lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
            ax.bar(df.index, df['Close'] - df['Open'], bottom=df['Open'], color=colors, width=0.8)
            ax.vlines(df.index, df['Low'], df['High'], color=colors, linewidth=1)
            ax.plot(df['MA20'], color='blue', linewidth=2)
            ax.plot(df['Upper'], color='red', linewidth=2)
            ax.plot(df['Lower'], color='red', linewidth=2)
            ax.set_title(f"{ticker} - Velas + Bandas de Bollinger")
            ax.set_ylabel("Precio")
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig

        try:
            df_boll = data_boll[selected_boll].copy()
            fig_boll = plot_bollinger_candlestick(selected_boll, df_boll)
            st.pyplot(fig_boll)
        except Exception as e:
            st.warning(f"No se pudo procesar el activo seleccionado. Error: {str(e)}")

# --- TAB 7: Conclusiones ---
with tabs[5]:
    st.header("Análisis de Portafolio y Rentabilidad")

    # Cargar el archivo con precios diarios (ajustar la ruta según sea necesario)
    try:
        df = pd.read_csv("data_defi1.csv", header=[0, 1], index_col=0, parse_dates=True)

        # Extraer precios de cierre
        df_closes = df.xs('Close', level=1, axis=1)

        # Definir los porcentajes del portafolio
        porcentajes = {
            'IVV': 0.12,
            'SCHD': 0.12,
            'AAPL': 0.25,
            'MSFT': 0.13,
            'QCOM': 0.06,
            'CRSP': 0.06,
            'GOOGL': 0.08,
            'RTX': 0.08,
            'GC=F': 0.05,
            'CL=F': 0.05
        }

        # Filtrar los activos según los porcentajes definidos
        activos = list(porcentajes.keys())
        df_closes = df_closes[activos]

        # Multiplicar cada columna por su peso en el portafolio
        df_ponderado = df_closes.copy()
        for activo in activos:
            df_ponderado[activo] = df_closes[activo] * porcentajes[activo]

        # Calcular el valor total ponderado del portafolio por días
        df_ponderado['Portafolio'] = df_ponderado.sum(axis=1)

        # Calcular los log-retornos diarios del portafolio
        log_retornos_portafolio = np.log(df_ponderado['Portafolio'] / df_ponderado['Portafolio'].shift(1)).dropna()

        # Rentabilidad total (compuesta) del portafolio en el periodo
        rentabilidad_total = np.mean(log_retornos_portafolio) * 120  # Multiplicado por 120 para anualizar
        st.write(f"Rentabilidad total del portafolio: {rentabilidad_total:.2%}")

        # Mostrar las primeras filas del portafolio ponderado
        st.subheader("Portafolio Ponderado")
        st.write(df_ponderado.head(10))

       

        # Mostrar gráfico de log-retornos del portafolio
        st.subheader("Log-Retornos del Portafolio")
        fig_retornos, ax_retornos = plt.subplots(figsize=(10, 5))
        ax_retornos.plot(log_retornos_portafolio, label='Log-Retornos del Portafolio', color='orange')
        ax_retornos.set_title('Log-Retornos Diarios del Portafolio')
        ax_retornos.set_xlabel('Fecha')
        ax_retornos.set_ylabel('Log-Retorno')
        ax_retornos.grid(True)
        ax_retornos.legend()
        st.pyplot(fig_retornos)

    except Exception as e:
        st.error(f"Error al cargar o procesar los datos: {e}")
