import streamlit as st
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Proyección LTE-A Paita 2025",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos visuales
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=0.9)

# --- FUNCIÓN SEGURA PARA CARGAR IMÁGENES ---
def cargar_imagen_segura(nombre_archivo, width=None):
    """Intenta cargar imagen de varias formas para evitar errores"""
    if os.path.exists(nombre_archivo):
        try:
            img = Image.open(nombre_archivo)
            if width:
                st.image(img, width=width)
            else:
                st.image(img, use_column_width=True)
            return True
        except:
            return False
    return False

# ==========================================
# BARRA LATERAL (SIDEBAR) - CON INTEGRANTES
# ==========================================
with st.sidebar:
    # 1. Logo
    if not cargar_imagen_segura("logo-visualizate.png", width=220):
        st.image("https://www.utp.edu.pe/sites/default/files/logo_utp_0.png", width=220)
    
    st.markdown("---")
    
    # 2. INTEGRANTES (AÑADIDOS)
    st.markdown("### 🎓 Integrantes")
    st.markdown("""
    **Nureña Torres, Carlos**
    *U20306681*
    
    **Lloclla Manuel, Oscar**
    *U20101084*
    
    **Diaz Delgadillo, Mary**
    *U21216162*
    
    **Alburqueque Nicola, Mario**
    *U21209590*
    """)
    
    st.markdown("---")
    st.header("🎛️ Panel de Control")

# ==========================================
# GESTIÓN DE MEMORIA
# ==========================================
if 'prediccion_realizada' not in st.session_state:
    st.session_state['prediccion_realizada'] = False

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
@st.cache_data
def cargar_datos():
    archivos = glob.glob('dataset*.csv')
    lista_dfs = []
    for f in archivos:
        try:
            df = pd.read_csv(f, sep=';', encoding='latin-1')
            lista_dfs.append(df)
        except:
            continue
    
    if lista_dfs:
        df_concat = pd.concat(lista_dfs, ignore_index=True)
        # Filtro Paita
        df_paita = df_concat[df_concat['ADM_LEVEL_2_NAME'] == 'Paita'].copy()
        
        cols_numericas = ['MES', 'AVERAGE_THROUGHPUT_DOWNLOAD_4G', 
                'AVERAGE_LATENCY_4G', 'PACKET_LOSS_4G', 'TIME_PERCENTAGE_4G']
        for col in cols_numericas:
            df_paita[col] = pd.to_numeric(df_paita[col], errors='coerce')
        
        df_paita.dropna(subset=['AVERAGE_THROUGHPUT_DOWNLOAD_4G'], inplace=True)
        df_paita.sort_values(by='MES', inplace=True)
        return df_paita
    return None

df = cargar_datos()

if df is None:
    st.error("⚠️ Error: No se encontraron archivos 'dataset_*.csv'.")
    st.stop()

st.sidebar.success(f"✅ Datos Históricos: {len(df)} registros")

# ==========================================
# 2. MOTOR DE IA (Validado ~0.88 R2)
# ==========================================
@st.cache_resource(ttl="2h")
def entrenar_modelo_final(data):
    le = LabelEncoder()
    data_ml = data.copy()
    data_ml['OPERADOR_CODE'] = le.fit_transform(data_ml['NETWORK_CARRIER'])
    
    features = ['MES', 'AVERAGE_LATENCY_4G', 'PACKET_LOSS_4G', 'TIME_PERCENTAGE_4G', 'OPERADOR_CODE']
    X = data_ml[features].fillna(0)
    y = data_ml['AVERAGE_THROUGHPUT_DOWNLOAD_4G']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo Validación
    modelo_test = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=9, random_state=42)
    modelo_test.fit(X_train, y_train)
    score_real = modelo_test.score(X_test, y_test)
    
    # Modelo Final
    modelo_final = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=9, random_state=42)
    modelo_final.fit(X, y)
    
    importances = pd.DataFrame({
        'Factor': ['Mes', 'Latencia (ms)', 'Packet Loss (%)', 'Cobertura (%)', 'Operador'],
        'Importance': modelo_final.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return modelo_final, importances, le, score_real

modelo_rf, df_importancias, encoder_operador, precision = entrenar_modelo_final(df)
st.sidebar.info(f"🤖 Precisión del Modelo (R²): {precision:.2f}")

# ==========================================
# 3. INTERFAZ GRÁFICA
# ==========================================

st.markdown("""
    <h1 style='text-align: center; color: #1f449c; font-size: 26px;'>
    Diseño de una Red LTE-Advanced con técnicas de Machine Learning para mejorar los indicadores de cobertura y calidad del servicio en el Puerto de Paita, Región Piura, Perú
    </h1>
""", unsafe_allow_html=True)

# --- BANNER PRINCIPAL ---
# Intentamos cargar paita.jpg
if not cargar_imagen_segura("paita.jpg"):
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Plaza_de_Armas_de_Paita.jpg/1024px-Plaza_de_Armas_de_Paita.jpg", 
        use_column_width=True, 
        caption="Puerto de Paita (Imagen Web)"
    )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Diagnóstico", 
    "📈 Tendencias", 
    "🧠 Causa Raíz", 
    "📅 Proyección Cierre 2025", 
    "🔮 Simulador Manual"
])

# --- TAB 1: DIAGNÓSTICO ---
with tab1:
    st.subheader("Situación a Octubre 2025")
    ultimo_mes = df['MES'].max()
    df_last = df[df['MES'] == ultimo_mes]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Velocidad Promedio", f"{df_last['AVERAGE_THROUGHPUT_DOWNLOAD_4G'].mean():.2f} Mbps")
    col2.metric("Latencia", f"{df_last['AVERAGE_LATENCY_4G'].mean():.1f} ms", delta_color="inverse")
    col3.metric("Packet Loss", f"{df_last['PACKET_LOSS_4G'].mean():.2f} %", delta_color="inverse")
    
    st.dataframe(df_last.groupby('NETWORK_CARRIER')[['AVERAGE_THROUGHPUT_DOWNLOAD_4G', 'AVERAGE_LATENCY_4G']].mean().style.highlight_max(axis=0))

# --- TAB 2: TENDENCIAS (CON TABLA DETALLADA) ---
with tab2:
    st.subheader("Histórico Enero - Octubre")
    trend_data = df.groupby(['MES', 'NETWORK_CARRIER'])[['AVERAGE_THROUGHPUT_DOWNLOAD_4G', 'AVERAGE_LATENCY_4G']].mean().reset_index()
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=trend_data, x='MES', y='AVERAGE_THROUGHPUT_DOWNLOAD_4G', hue='NETWORK_CARRIER', marker='o', ax=ax1)
    ax1.set_title("Evolución de Velocidad (Mbps)")
    ax1.set_xticks(range(1, 11))
    ax1.legend(title='Operador', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small', title_fontsize='small')
    st.pyplot(fig1)
    
    st.subheader("📋 Detalle de Datos Históricos (Ene - Oct)")
    tabla_hist = trend_data.pivot(index='NETWORK_CARRIER', columns='MES', values='AVERAGE_THROUGHPUT_DOWNLOAD_4G')
    nombres_meses = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Set', 10:'Oct'}
    tabla_hist = tabla_hist.rename(columns=nombres_meses)
    st.dataframe(tabla_hist.style.format("{:.2f} Mbps").highlight_max(axis=0, color='#d4edda').highlight_min(axis=0, color='#f8d7da'))

# --- TAB 3: CAUSA RAÍZ ---
with tab3:
    st.subheader("Factores Críticos")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Factor', data=df_importancias, palette='viridis', ax=ax3)
    st.pyplot(fig3)
    st.caption("Factores que más limitan la velocidad en Paita según el modelo Random Forest.")

# --- TAB 4: PROYECCIÓN (TABLA COMPLETA) ---
with tab4:
    st.markdown("### 📅 Proyección de Cierre de Año")
    st.write("Proyección continua Enero-Diciembre.")
    
    operadores = encoder_operador.classes_
    trend_hist = df.groupby(['MES', 'NETWORK_CARRIER'])['AVERAGE_THROUGHPUT_DOWNLOAD_4G'].mean().reset_index()
    trend_hist['TIPO'] = 'Histórico'
    
    df_proyecciones_list = []

    for op in operadores:
        datos_op = df[df['NETWORK_CARRIER'] == op].sort_values(by='MES')
        val_oct = trend_hist[(trend_hist['NETWORK_CARRIER'] == op) & (trend_hist['MES'] == 10)]['AVERAGE_THROUGHPUT_DOWNLOAD_4G'].values[0]
        max_hist = datos_op['AVERAGE_THROUGHPUT_DOWNLOAD_4G'].max()
        
        base_lat = datos_op['AVERAGE_LATENCY_4G'].tail(3).mean()
        base_loss = datos_op['PACKET_LOSS_4G'].tail(3).mean()
        base_cob = datos_op['TIME_PERCENTAGE_4G'].tail(3).mean()
        op_code = encoder_operador.transform([op])[0]
        
        predicciones = []
        for m in [11, 12]:
            if m == 11:
                sim_lat = base_lat
                sim_loss = base_loss
            else:
                sim_lat = base_lat * 1.05 
                sim_loss = base_loss * 1.05
            
            entrada = pd.DataFrame({
                'MES': [m],
                'AVERAGE_LATENCY_4G': [sim_lat],
                'PACKET_LOSS_4G': [sim_loss],
                'TIME_PERCENTAGE_4G': [base_cob],
                'OPERADOR_CODE': [op_code]
            })
            
            pred = modelo_rf.predict(entrada)[0]
            techo = max_hist * 1.15
            pred = min(pred, techo)
            predicciones.append(pred)

        df_proyecciones_list.append({'MES': 10, 'NETWORK_CARRIER': op, 'AVERAGE_THROUGHPUT_DOWNLOAD_4G': val_oct, 'TIPO': 'Proyección'})
        df_proyecciones_list.append({'MES': 11, 'NETWORK_CARRIER': op, 'AVERAGE_THROUGHPUT_DOWNLOAD_4G': predicciones[0], 'TIPO': 'Proyección'})
        df_proyecciones_list.append({'MES': 12, 'NETWORK_CARRIER': op, 'AVERAGE_THROUGHPUT_DOWNLOAD_4G': predicciones[1], 'TIPO': 'Proyección'})

    df_proj = pd.DataFrame(df_proyecciones_list)
    df_master = pd.concat([trend_hist, df_proj], ignore_index=True)

    fig_proj, ax_proj = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_master, x='MES', y='AVERAGE_THROUGHPUT_DOWNLOAD_4G', hue='NETWORK_CARRIER', style='TIPO', markers=True, dashes={'Histórico': (None, None), 'Proyección': (2, 2)}, ax=ax_proj)
    ax_proj.set_title("Proyección Cierre 2025")
    ax_proj.set_xticks(range(1, 13))
    ax_proj.set_xticklabels(['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Set','Oct','NOV*','DIC*'])
    ax_proj.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
    ax_proj.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small', title_fontsize='small')
    st.pyplot(fig_proj)
    
    st.write("#### 📊 Reporte Anual Completo (Real + Proyectado)")
    real_part = trend_hist[trend_hist['MES'] < 10][['NETWORK_CARRIER', 'MES', 'AVERAGE_THROUGHPUT_DOWNLOAD_4G']]
    proj_part = df_proj[['NETWORK_CARRIER', 'MES', 'AVERAGE_THROUGHPUT_DOWNLOAD_4G']]
    full_year_data = pd.concat([real_part, proj_part], ignore_index=True)
    pivot_full = full_year_data.pivot(index='NETWORK_CARRIER', columns='MES', values='AVERAGE_THROUGHPUT_DOWNLOAD_4G')
    nombres_meses_full = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Set', 10:'Oct', 11:'Nov*', 12:'Dic*'}
    pivot_full = pivot_full.rename(columns=nombres_meses_full)
    st.dataframe(pivot_full.style.format("{:.2f}").highlight_max(axis=0))
    st.caption("* Meses proyectados por el modelo de IA.")

# --- TAB 5: SIMULADOR MANUAL ---
with tab5:
    st.markdown("### 🔮 Simulador Manual 'What-If'")
    col_sim1, col_sim2 = st.columns([1, 2])
    with col_sim1:
        with st.form(key='sim_form'):
            sim_operador = st.selectbox("Operador:", encoder_operador.classes_)
            def_lat = float(df['AVERAGE_LATENCY_4G'].mean())
            def_loss = float(df['PACKET_LOSS_4G'].mean())
            sim_latencia = st.slider("Latencia (ms):", 10.0, 150.0, def_lat)
            sim_loss = st.slider("Packet Loss (%):", 0.0, 10.0, def_loss)
            sim_cobertura = st.slider("Cobertura (%):", 50.0, 100.0, 95.0)
            sim_mes = st.slider("Mes Futuro:", 1, 12, 11)
            submit_val = st.form_submit_button("🚀 Simular Escenario")
            if submit_val:
                op_code = encoder_operador.transform([sim_operador])[0]
                input_data = pd.DataFrame({
                    'MES': [sim_mes],
                    'AVERAGE_LATENCY_4G': [sim_latencia],
                    'PACKET_LOSS_4G': [sim_loss],
                    'TIME_PERCENTAGE_4G': [sim_cobertura],
                    'OPERADOR_CODE': [op_code]
                })
                pred = modelo_rf.predict(input_data)[0]
                hist_avg = df[df['NETWORK_CARRIER'] == sim_operador]['AVERAGE_THROUGHPUT_DOWNLOAD_4G'].mean()
                delta = pred - hist_avg
                st.session_state['prediccion_realizada'] = True
                st.session_state['resultado_velocidad'] = pred
                st.session_state['delta_comparativa'] = delta
                st.session_state['sim_operador_mem'] = sim_operador

    with col_sim2:
        if st.session_state['prediccion_realizada']:
            vel = st.session_state['resultado_velocidad']
            dlt = st.session_state['delta_comparativa']
            op = st.session_state['sim_operador_mem']
            st.metric(f"Velocidad Simulada ({op})", f"{vel:.2f} Mbps", f"{dlt:.2f} Mbps")
            st.progress(min(vel / 50.0, 1.0))
        else:
            st.info("Configura y simula un escenario específico.")

st.markdown("---")
st.caption("Machine Learning para mejorar los indicadores de cobertura y calidad del servicio en el Puerto de Paita")