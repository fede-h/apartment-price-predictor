import streamlit as st
import pandas as pd
import pickle
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# Configuración de la página
st.set_page_config(
    page_title="Predictor de precios",
    page_icon="🏠",
    layout="centered"
)

# Título y descripción
st.title("Predictor de precios de propiedades")
st.markdown("#### Ingresá los detalles de la propiedad para obtener una estimación precisa del precio")

# Cargar el modelo
@st.cache_resource
def load_model():
    try:
        with open('randomforest.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ No se encontró el modelo. Asegurate de que 'random_forest_model.pkl' esté en el mismo directorio.")
        return None

model = load_model()

# Lista de comunas de Buenos Aires
COMUNAS = [
    'Comuna 1', 'Comuna 2', 'Comuna 3', 'Comuna 4', 'Comuna 5',
    'Comuna 6', 'Comuna 7', 'Comuna 8', 'Comuna 9', 'Comuna 10',
    'Comuna 11', 'Comuna 12', 'Comuna 13', 'Comuna 14', 'Comuna 15'
]

# Inicializar session state
if "lat" not in st.session_state:
    st.session_state.lat = -34.6037
if "lon" not in st.session_state:
    st.session_state.lon = -58.3816

st.header("Ubicación de la Propiedad")
st.markdown("**Hacé click en el mapa para seleccionar la ubicación exacta**")

# ---- Crear el mapa ----
m = folium.Map(
    location=[st.session_state.lat, st.session_state.lon],
    zoom_start=12,
    tiles="OpenStreetMap"
)

# Agregar marcador
folium.Marker(
    [st.session_state.lat, st.session_state.lon],
    popup="Ubicación seleccionada",
    icon=folium.Icon(color="red", icon="home")
).add_to(m)

# Mostrar mapa e interactuar
map_data = st_folium(
    m,
    width=700,
    height=400,
    key="property_map",
    returned_objects=["last_clicked"]
)

# ---- Capturar click en mapa ----
if map_data and map_data.get("last_clicked") is not None:
    click_lat = map_data["last_clicked"]["lat"]
    click_lon = map_data["last_clicked"]["lng"]

    # Solo actualizar si cambió
    if (click_lat != st.session_state.lat) or (click_lon != st.session_state.lon):
        st.session_state.lat = click_lat
        st.session_state.lon = click_lon

# ---- Inputs manuales ----
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    new_lat = st.number_input(
        "Latitud",
        min_value=-90.0,
        max_value=90.0,
        value=st.session_state.lat,
        step=0.0001,
        format="%.4f"
    )
with col2:
    new_lon = st.number_input(
        "Longitud",
        min_value=-180.0,
        max_value=180.0,
        value=st.session_state.lon,
        step=0.0001,
        format="%.4f"
    )

with col3:
    if st.button("Actualizar Mapa", use_container_width=True):
        st.session_state.lat = new_lat
        st.session_state.lon = new_lon

comuna = st.selectbox(
    "Comuna",
    options=COMUNAS,
    help="Seleccioná la comuna donde se encuentra la propiedad"
)

# Sección: Información de la Propiedad
st.header("Información de la Propiedad")

col1, col2, col3 = st.columns(3)

with col1:
    dormitorios = st.number_input(
        "Dormitorios",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )

with col2:
    banos = st.number_input(
        "Baños",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

with col3:
    ambientes = st.number_input(
        "Ambientes",
        min_value=0,
        max_value=20,
        value=3,
        step=1
    )

sup_m2 = st.number_input(
    "Superficie (m²)",
    min_value=0.0,
    max_value=10000.0,
    value=50.0,
    step=1.0,
    format="%.1f"
)

# Espaciador
st.markdown("---")

# Botón de predicción
if st.button("Calcular Precio Estimado", type="primary", use_container_width=True):
    if model is not None:
        # Preparar los datos para la predicción
        comuna_num = int(comuna.split()[-1])
        
        # Crear DataFrame con las features en el orden correcto
        input_data = pd.DataFrame({
            'sup_m2': [sup_m2],
            'lat': [st.session_state.lat],
            'lon': [st.session_state.lon],
            'banos': [banos],
            'comuna_num': [comuna_num],
            'ambientes': [ambientes],
            'dormitorios': [dormitorios]
            

        })

        # Hacer la predicción
        try:
            prediction = model.predict(input_data)[0]
            
            # Mostrar resultado
            st.success("✅ Predicción completada")
            
            # Mostrar precio estimado en grande
            st.markdown("### Precio Estimado")
            st.markdown(f"# ${prediction:,.0f}")
            
            # Información adicional
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precio por m²", f"${prediction/sup_m2:,.0f}")
            with col2:
                st.metric("Superficie", f"{sup_m2:.0f} m²")
            with col3:
                st.metric("Ambientes", f"{ambientes}")
            
            # Mostrar detalles de la ubicación
            st.markdown("---")
            st.markdown("###  Detalles de Ubicación")
            st.write(f"**Coordenadas:** {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
            st.write(f"**Comuna:** {comuna}")
            
        except Exception as e:
            st.error(f"Error al hacer la predicción: {str(e)}")
    else:
        st.warning("No se puede hacer la predicción sin el modelo cargado.")

# Información adicional al pie
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>Esta es una estimación basada en datos históricos y puede variar según las condiciones del mercado.</p>
</div>
""", unsafe_allow_html=True)