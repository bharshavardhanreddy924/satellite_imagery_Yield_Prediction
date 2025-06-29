# CRITICAL: st.set_page_config() MUST be the first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Agricultural Intelligence Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now import everything else
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ee
import geemap.foliumap as geemap 
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import branca.colormap as cm
import calendar
from scipy import stats
import statsmodels.api as sm
from shapely.geometry import Polygon
from geopy.geocoders import Nominatim
import backend
# Import your backend (this should work now with fixed NumPy)
try:
    import backend
    BACKEND_AVAILABLE = True
    st.sidebar.success("✅ Backend loaded successfully!")
except ImportError as e:
    BACKEND_AVAILABLE = False
    st.sidebar.error(f"❌ Backend import failed: {e}")
except Exception as e:
    BACKEND_AVAILABLE = False
    st.sidebar.error(f"❌ Backend initialization failed: {e}")

# =================================================================================================
# STYLING
# =================================================================================================
st.markdown("""
<style>
    .reportview-container { background: #F0F2F6; }
    .sidebar .sidebar-content { background: #F0F2F6; }

    .stButton>button {
        color: #FFFFFF;
        background-color: #28a745;
        border-radius: 8px;
        border: 1px solid #28a745;
        padding: 10px 24px;
        font-size: 16px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        color: #28a745;
        background-color: #FFFFFF;
        border: 1px solid #28a745;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    div[data-testid="stMetric"] {
        background-color: #E8F5E9;
        border: 1px solid #A5D6A7;
        border-radius: 10px;
        padding: 15px;
        color: #1B5E20 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] > div > div { color: #1B5E20 !important; }
    div[data-testid="stMetric"] label { color: #388E3C !important; }
</style>
""", unsafe_allow_html=True)

# =================================================================================================
# CONSTANTS AND CONFIGURATIONS
# =================================================================================================
LOCATIONS = {
    "Bengaluru, India": {"latitude": 12.97, "longitude": 77.59},
    "Mysuru, India": {"latitude": 12.30, "longitude": 76.65},
    "Hubli, India": {"latitude": 15.36, "longitude": 75.13},
    "Mangaluru, India": {"latitude": 12.87, "longitude": 74.85},
    "Kalaburagi, India": {"latitude": 17.33, "longitude": 76.83},
    "Belagavi, India": {"latitude": 15.85, "longitude": 74.50},
    "Udupi, India": {"latitude": 13.33, "longitude": 74.75},
    "Chikkamagaluru, India": {"latitude": 13.32, "longitude": 75.78},
    "Shimoga, India": {"latitude": 13.94, "longitude": 75.56},
    "Dharwad, India": {"latitude": 15.4589, "longitude": 75.0078},
}

PARAMETERS = {
    "T2M": "Temperature at 2 Meters (°C)",
    "T2M_MAX": "Maximum Temperature at 2 Meters (°C)",
    "T2M_MIN": "Minimum Temperature at 2 Meters (°C)",
    "PRECTOTCORR": "Precipitation Corrected (mm/day)",
    "RH2M": "Relative Humidity at 2 Meters (%)",
    "WS2M": "Wind Speed at 2 Meters (m/s)",
    "ALLSKY_SFC_SW_DWN": "All Sky Surface Shortwave Downward Irradiance (W/m²)",
    "T2MDEW": "Dew Point at 2 Meters (°C)",
    "PS": "Surface Pressure (kPa)",
    "CLRSKY_SFC_SW_DWN": "Clear Sky Surface Shortwave Downward Irradiance (W/m²)",
    "ALLSKY_SFC_LW_DWN": "All Sky Surface Longwave Downward Irradiance (W/m²)",
    "TS": "Earth Skin Temperature (°C)", 
    "QV2M": "Specific Humidity at 2 Meters (g/kg)",
    "WS10M": "Wind Speed at 10 Meters (m/s)",
    "WD2M": "Wind Direction at 2 Meters (degrees)", 
    "WD10M": "Wind Direction at 10 Meters (degrees)",
}

NASA_POWER_SUPPORTED_PARAMS = [
    "T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR",    
    "RH2M", "QV2M", "WS2M", "WS10M", "WD2M", "WD10M",
    "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN", "CLRSKY_SFC_SW_DWN",
    "PS", "T2MDEW", "TS"
]

GEE_DATASETS = {
    "NASA/GLDAS/V021/NOAH/G025/T3H": "GLDAS Noah Land Surface Model (Soil Moisture)",
    "NASA/GPM_L3/IMERG_V06": "GPM: Precipitation (IMERG)",
    "ECMWF/ERA5/DAILY": "ERA5 Daily Aggregates (Temperature, Precipitation)",
    "MODIS/006/MOD11A1": "MODIS Land Surface Temperature (Daily)",
    "COPERNICUS/S5P/NRTI/L3_AER_AI": "Sentinel-5P Aerosol Index",
    "COPERNICUS/S2_SR": "Sentinel-2 Surface Reflectance (NDVI, Bands)",
    "COPERNICUS/S1_GRD": "Sentinel-1 Ground Range Detected (VV for Soil Moisture Proxy)",
    "MODIS/006/MOD13A2": "MODIS 16-Day NDVI" 
}

EE_BANDS = {
    "NASA/GLDAS/V021/NOAH/G025/T3H": {"SoilMoi0_10cm_inst": "Soil Moisture 0-10cm (kg/m^2)"},
    "NASA/GPM_L3/IMERG_V06": {"precipitationCal": "Precipitation (mm/hr)"},
    "ECMWF/ERA5/DAILY": {
        "mean_2m_air_temperature": "Mean Air Temperature (K)", 
        "total_precipitation": "Total Precipitation (m)" 
    },
    "MODIS/006/MOD11A1": {"LST_Day_1km": "Land Surface Temp (Day, Scaled K)"}, 
    "COPERNICUS/S5P/NRTI/L3_AER_AI": {"absorbing_aerosol_index": "Aerosol Index"},
    "COPERNICUS/S2_SR": {
        "NDVI": "NDVI (Calculated)", 
        "B4": "Red Band (Surface Reflectance)",
        "B8": "NIR Band (Surface Reflectance)"
    },
    "COPERNICUS/S1_GRD": {"VV": "VV Polarization (dB)"},
    "MODIS/006/MOD13A2": {"NDVI": "NDVI (Scaled x10000)"} 
}

# =================================================================================================
# INITIALIZATION FUNCTIONS
# =================================================================================================
def initialize_gee():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize()
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
            return True
        except Exception as e:
            st.sidebar.warning(f"Earth Engine setup failed: {e}")
            return False

def initialize_backend():
    """Initialize backend for yield prediction"""
    if BACKEND_AVAILABLE:
        try:
            backend.initialize_gee()
            crop_data = backend.load_crop_data()
            return True, crop_data
        except Exception as e:
            st.sidebar.error(f"Backend initialization failed: {e}")
            return False, None
    return False, None

def initialize_session_state():
    """Initialize all session state variables"""
    # Climate analysis states
    if 'nasa_df' not in st.session_state: 
        st.session_state.nasa_df = pd.DataFrame()
    if 'location_info' not in st.session_state: 
        st.session_state.location_info = {}
    if 'ee_map' not in st.session_state: 
        st.session_state.ee_map = None
    if 'ee_time_series_plot' not in st.session_state: 
        st.session_state.ee_time_series_plot = None
    if 'ee_time_series_data' not in st.session_state: 
        st.session_state.ee_time_series_data = pd.DataFrame()
    if 'ee_stl_plot' not in st.session_state: 
        st.session_state.ee_stl_plot = None
    if 'current_ee_dataset' not in st.session_state: 
        st.session_state.current_ee_dataset = None
    if 'current_ee_band' not in st.session_state: 
        st.session_state.current_ee_band = None
    
    # Yield prediction states
    if 'map_center' not in st.session_state: 
        st.session_state.map_center = [15.3173, 75.7139]
    if 'map_zoom' not in st.session_state: 
        st.session_state.map_zoom = 7
    if 'field_details' not in st.session_state: 
        st.session_state.field_details = {}
    if 'regional_prediction_result' not in st.session_state: 
        st.session_state.regional_prediction_result = None
    if 'gee_initialized' not in st.session_state: 
        st.session_state.gee_initialized = False
    if 'backend_available' not in st.session_state: 
        st.session_state.backend_available = False
    if 'crop_data' not in st.session_state: 
        st.session_state.crop_data = None

# =================================================================================================
# NASA POWER API FUNCTIONS
# =================================================================================================
def fetch_nasa_power_data(lat: float, lon: float, start_date: datetime, end_date: datetime, parameters: list[str]) -> dict | None:
    """Fetch climate data from NASA POWER API"""
    supported = [p for p in parameters if p in NASA_POWER_SUPPORTED_PARAMS]
    if not supported:
        st.error("No valid NASA POWER parameters selected for fetching.")
        return None

    params_str = ",".join(supported)
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={params_str}"
        "&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start_date.strftime('%Y%m%d')}"
        f"&end={end_date.strftime('%Y%m%d')}"
        "&format=JSON"
    )

    with st.spinner(f"Fetching NASA POWER data for {len(supported)} parameters..."):
        try:
            resp = requests.get(url, timeout=60) 
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            st.error("Timeout error fetching data from NASA POWER.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching NASA POWER data: {e}")
            return None

def process_nasa_data(data_json):
    """Process NASA POWER API response into DataFrame"""
    if not data_json or "properties" not in data_json or "parameter" not in data_json["properties"]:
        st.warning("NASA POWER data is missing expected structure.")
        return pd.DataFrame() 

    time_series = data_json["properties"]["parameter"]
    if not time_series:
        st.warning("No time series data found in NASA POWER response.")
        return pd.DataFrame()

    first_param_key = list(time_series.keys())[0]
    dates_str = list(time_series[first_param_key].keys())
    
    df_data = []
    for date_str in dates_str:
        try:
            row_date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError: 
            continue 
            
        row = {"Date": row_date}
        for param_key, param_values in time_series.items():
            value = param_values.get(date_str)
            if value is None or value == -999: 
                row[param_key] = np.nan
            else:
                row[param_key] = float(value) 
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    if 'Date' in df.columns and not df.empty:
         df = df.sort_values(by='Date').reset_index(drop=True)
    return df

def create_monthly_summary(df):
    """Create monthly summary statistics"""
    if df.empty or 'Date' not in df.columns:
        return {}
    df_copy = df.copy()
    df_copy['Month_Year'] = df_copy['Date'].dt.strftime('%Y-%m')
    
    monthly_data_agg = {}
    numeric_cols = df_copy.select_dtypes(include=np.number).columns
    for param in numeric_cols:
        summary = df_copy.groupby('Month_Year')[param].agg(['mean', 'min', 'max', 'sum']).reset_index()
        monthly_data_agg[param] = summary
    return monthly_data_agg

# =================================================================================================
# GOOGLE EARTH ENGINE FUNCTIONS
# =================================================================================================
def calculate_ndvi_s2(image):
    """Calculate NDVI for Sentinel-2 images"""
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

def get_gee_image_collection(dataset_id, start_date_dt, end_date_dt, point_geom):
    """Get Google Earth Engine image collection"""
    collection = ee.ImageCollection(dataset_id).filterBounds(point_geom).filterDate(
        start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d'))
    
    if dataset_id == 'COPERNICUS/S2_SR':
        collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    return collection

def get_gee_image_and_region(dataset_id, band_id, start_date_dt, end_date_dt, region_coords):
    """Get GEE image and region for visualization"""
    point = ee.Geometry.Point([region_coords["longitude"], region_coords["latitude"]])
    display_region_geom = point.buffer(20000) 

    collection = get_gee_image_collection(dataset_id, start_date_dt, end_date_dt, point)

    image_for_map = None
    if dataset_id == 'COPERNICUS/S2_SR' and band_id == 'NDVI':
        collection_ndvi = collection.map(calculate_ndvi_s2)
        image_for_map = collection_ndvi.select('NDVI').mean()
    elif dataset_id == 'MODIS/006/MOD13A2' and band_id == 'NDVI': 
        image_for_map = collection.select('NDVI').mean().multiply(0.0001) 
    elif dataset_id == 'MODIS/006/MOD11A1' and band_id == 'LST_Day_1km': 
        image_for_map = collection.select('LST_Day_1km').mean().multiply(0.02).subtract(273.15)
    elif dataset_id == 'ECMWF/ERA5/DAILY' and band_id == 'mean_2m_air_temperature':
        image_for_map = collection.select(band_id).mean().subtract(273.15) 
    elif dataset_id == 'ECMWF/ERA5/DAILY' and band_id == 'total_precipitation':
        image_for_map = collection.select(band_id).mean().multiply(1000) 
    else:
        if collection.size().getInfo() > 0:
             image_for_map = collection.select(band_id).mean()
        else:
            st.warning(f"No images found in collection for {dataset_id} / {band_id}")
            return None, display_region_geom, point
        
    return image_for_map, display_region_geom, point

def get_visualization_params(dataset_id, band_id):
    """Get visualization parameters for GEE datasets"""
    if dataset_id == "NASA/GLDAS/V021/NOAH/G025/T3H" and band_id == "SoilMoi0_10cm_inst":
        return {"min": 0, "max": 50, "palette": ['#d2b48c', '#ffeb3b', '#4caf50', '#81d4fa', '#0277bd']} 
    elif dataset_id == "NASA/GPM_L3/IMERG_V06" and band_id == "precipitationCal":
        return {"min": 0, "max": 10, "palette": ['#ffffff', '#0000ff', '#800080']} 
    elif dataset_id == "ECMWF/ERA5/DAILY" and band_id == "mean_2m_air_temperature": 
        return {"min": -10, "max": 45, "palette": ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FFA500', '#FF0000', '#800000']} 
    elif dataset_id == "ECMWF/ERA5/DAILY" and band_id == "total_precipitation": 
        return {"min": 0, "max": 50, "palette": ['#ffffff', '#2196f3', '#0d47a1']} 
    elif dataset_id == "MODIS/006/MOD11A1" and band_id == "LST_Day_1km": 
        return {"min": 0, "max": 50, "palette": ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FFA500', '#FF0000', '#800000']}
    elif dataset_id == "COPERNICUS/S5P/NRTI/L3_AER_AI" and band_id == "absorbing_aerosol_index":
        return {"min": -1, "max": 2, "palette": ['#000000', '#0000FF', '#FF00FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']}
    elif (dataset_id == 'COPERNICUS/S2_SR' or dataset_id == 'MODIS/006/MOD13A2') and band_id == 'NDVI':
        return {"min": -0.2, "max": 1, "palette": ['#8B4513', '#FFFFE0', '#006400', '#00FF00']} 
    elif dataset_id == "COPERNICUS/S2_SR" and band_id == "B4": 
        return {"min": 0, "max": 3000, "gamma": 1.4} 
    elif dataset_id == "COPERNICUS/S2_SR" and band_id == "B8": 
        return {"min": 0, "max": 5000, "gamma": 1.4}
    elif dataset_id == "COPERNICUS/S1_GRD" and band_id == "VV": 
        return {"min": -25, "max": 0, "palette": ['#000000', '#FFFFFF']} 
    else: 
        return {"min": 0, "max": 1000, "palette": ['#0000FF', '#00FF00', '#FF0000']}

def create_ee_map_display(dataset_id, band_id, start_date_dt, end_date_dt, region_coords, location_name):
    """Create Earth Engine map display"""
    try:
        ee_image, display_region, point_geom = get_gee_image_and_region(dataset_id, band_id, start_date_dt, end_date_dt, region_coords)
        
        if ee_image is None:
            st.warning(f"Could not generate mean GEE image for {EE_BANDS[dataset_id].get(band_id, band_id)}.")
            return None

        m = geemap.Map(plugin_LatLngPopup=False, plugin_Fullscreen=False, add_google_map=False) 
        m.set_center(region_coords["longitude"], region_coords["latitude"], 8) 

        vis_params = get_visualization_params(dataset_id, band_id)
        band_title = EE_BANDS[dataset_id].get(band_id, band_id)
        
        try:
            ee_image.bandNames().getInfo() 
        except ee.EEException as img_err:
            st.warning(f"Could not obtain band information for GEE image ({band_title}): {img_err}")

        m.addLayer(ee_image.clip(display_region), vis_params, f"{band_title}")
        try:
            m.add_colorbar(vis_params, label=band_title, layer_name=f"{band_title}")
        except Exception as cb_err:
            st.warning(f"Could not add colorbar for {band_title}: {cb_err}")

        folium.Marker(
            [region_coords["latitude"], region_coords["longitude"]],
            popup=f"{location_name}<br>Lat: {region_coords['latitude']:.2f}, Lon: {region_coords['longitude']:.2f}",
            tooltip=location_name,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        m.add_layer_control()
        return m

    except ee.EEException as e:
        st.error(f"Google Earth Engine Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating Earth Engine map: {type(e).__name__}: {e}")
        return None

def create_ee_time_series_plot(dataset_id, band_id, start_date_dt, end_date_dt, point_geom, scale=30):
    """Create Earth Engine time series plot"""
    try:
        collection = get_gee_image_collection(dataset_id, start_date_dt, end_date_dt, point_geom)

        actual_band_id = band_id 
        
        if dataset_id == 'COPERNICUS/S2_SR' and band_id == 'NDVI':
            collection = collection.map(calculate_ndvi_s2)
        elif dataset_id == 'MODIS/006/MOD13A2' and band_id == 'NDVI':
            def scale_modis_ndvi(image):
                return image.select('NDVI').multiply(0.0001).rename('NDVI').copyProperties(image, ['system:time_start'])
            collection = collection.map(scale_modis_ndvi)
        elif dataset_id == 'MODIS/006/MOD11A1' and band_id == 'LST_Day_1km':
            def scale_convert_lst(image):
                lst_c = image.select('LST_Day_1km').multiply(0.02).subtract(273.15) 
                return lst_c.rename('LST_Day_1km').copyProperties(image, ['system:time_start'])
            collection = collection.map(scale_convert_lst)
            scale = 1000 
        elif dataset_id == 'ECMWF/ERA5/DAILY' and band_id == 'mean_2m_air_temperature':
            def convert_temp(image):
                temp_c = image.select(band_id).subtract(273.15) 
                return temp_c.rename(band_id).copyProperties(image, ['system:time_start'])
            collection = collection.map(convert_temp)
            scale = 27830 
        elif dataset_id == 'ECMWF/ERA5/DAILY' and band_id == 'total_precipitation':
            def convert_precip(image):
                precip_mm = image.select(band_id).multiply(1000) 
                return precip_mm.rename(band_id).copyProperties(image, ['system:time_start'])
            collection = collection.map(convert_precip)
            scale = 27830

        if collection.size().getInfo() == 0:
            st.info(f"No images found for GEE time series: {EE_BANDS[dataset_id].get(band_id, band_id)}.")
            return None, pd.DataFrame()

        def extract_value_at_point(image):
            value = image.select(actual_band_id).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point_geom, 
                scale=scale,
                maxPixels=1e9,
                bestEffort=True 
            ).get(actual_band_id) 
            return ee.Feature(None, {'date': image.date().format('YYYY-MM-dd'), 'value': value})

        features = collection.map(extract_value_at_point).filter(ee.Filter.notNull(['value'])).getInfo()['features']
        
        data = [{'date': pd.to_datetime(f['properties']['date']), 'value': f['properties']['value']} for f in features if f['properties']['value'] is not None] 
        df_ts = pd.DataFrame(data)

        if df_ts.empty:
            st.info(f"No valid data points for GEE time series: {EE_BANDS[dataset_id].get(band_id, band_id)}.")
            return None, pd.DataFrame()
        
        df_ts = df_ts.sort_values(by='date').reset_index(drop=True)

        plot_title_band_name = EE_BANDS[dataset_id].get(band_id, band_id) 
        fig = px.line(df_ts, x='date', y='value', 
                      title=f"Time Series: {plot_title_band_name}",
                      labels={'date': 'Date', 'value': plot_title_band_name})
        fig.update_layout(height=400)
        return fig, df_ts

    except ee.EEException as e:
        st.error(f"Google Earth Engine Error during time series generation: {e}")
        return None, pd.DataFrame()
    except Exception as e:
        st.error(f"Error creating GEE time series plot: {type(e).__name__}: {e}")
        return None, pd.DataFrame()

# =================================================================================================
# PLOTTING FUNCTIONS
# =================================================================================================
def create_temperature_plot(df):
    """Create temperature trend plot"""
    fig = go.Figure()
    if 'T2M_MAX' in df.columns and not df['T2M_MAX'].isnull().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['T2M_MAX'], name='Max Temp', line=dict(color='red', width=1)))
    if 'T2M' in df.columns and not df['T2M'].isnull().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['T2M'], name='Avg Temp', line=dict(color='orange', width=2)))
    if 'T2M_MIN' in df.columns and not df['T2M_MIN'].isnull().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['T2M_MIN'], name='Min Temp', line=dict(color='blue', width=1)))
    
    if not fig.data: return None 
    fig.update_layout(title='Temperature Trends', xaxis_title='Date', yaxis_title='Temperature (°C)', height=500)
    return fig

def create_precipitation_plot(df):
    """Create precipitation plot"""
    if 'PRECTOTCORR' not in df.columns or df['PRECTOTCORR'].isnull().all(): return None
    fig = px.bar(df, x='Date', y='PRECTOTCORR', labels={'PRECTOTCORR': 'Precipitation (mm/day)'}, title='Daily Precipitation')
    fig.update_layout(height=400)
    return fig

def create_humidity_wind_plot(df):
    """Create humidity and wind speed plot"""
    has_rh = 'RH2M' in df.columns and not df['RH2M'].isnull().all()
    has_ws = 'WS2M' in df.columns and not df['WS2M'].isnull().all()
    if not (has_rh or has_ws): return None

    fig = go.Figure()
    if has_rh:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RH2M'], name='Relative Humidity', line=dict(color='blue')))
    if has_ws:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['WS2M'], name='Wind Speed', line=dict(color='green'), yaxis='y2' if has_rh else 'y1')) 
    
    fig.update_layout(
        title='Humidity and Wind Speed',
        xaxis_title='Date',
        yaxis=dict(title='Relative Humidity (%)' if has_rh else 'Wind Speed (m/s)'),
        yaxis2=dict(title='Wind Speed (m/s)', overlaying='y', side='right', showgrid=False) if has_rh and has_ws else None,
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def create_solar_radiation_plot(df):
    """Create solar radiation plot"""
    if 'ALLSKY_SFC_SW_DWN' not in df.columns or df['ALLSKY_SFC_SW_DWN'].isnull().all(): return None
    fig = px.line(df, x='Date', y='ALLSKY_SFC_SW_DWN', labels={'ALLSKY_SFC_SW_DWN': 'Solar Radiation (W/m²)'}, title='Solar Radiation')
    fig.update_layout(height=400)
    return fig

# =================================================================================================
# ANALYSIS FUNCTIONS
# =================================================================================================
def perform_climate_anomaly_analysis(df, param='T2M'):
    """Perform climate anomaly analysis"""
    if param not in df.columns or df[param].isnull().all(): return None, pd.DataFrame()
    df_copy = df.copy().dropna(subset=[param])
    if df_copy.empty: return None, pd.DataFrame()

    df_copy['Month'] = df_copy['Date'].dt.month
    monthly_avg = df_copy.groupby('Month')[param].mean().reset_index().rename(columns={param: 'Monthly_Avg'})
    
    df_merged = pd.merge(df_copy, monthly_avg, on='Month', how='left')
    df_merged['Anomaly'] = df_merged[param] - df_merged['Monthly_Avg']
    
    fig = px.scatter(df_merged, x='Date', y='Anomaly', color='Anomaly',
                     color_continuous_scale='RdBu_r', title=f'{PARAMETERS.get(param, param)} Anomalies',
                     labels={'Anomaly': f'Anomaly ({PARAMETERS.get(param, param)})'}) 
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(height=400)
    return fig, df_merged[['Date', param, 'Monthly_Avg', 'Anomaly']]

def perform_trend_analysis(df, param='T2M'):
    """Perform trend analysis on climate parameter"""
    if param not in df.columns or df[param].isnull().all(): return None, {}
    df_copy = df.copy().dropna(subset=[param])
    if len(df_copy) < 2: return None, {} 

    df_copy['DateOrdinal'] = df_copy['Date'].apply(lambda x: x.toordinal())
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_copy['DateOrdinal'], df_copy[param])
    
    df_copy['Trend'] = intercept + slope * df_copy['DateOrdinal']
    annual_change = slope * 365.25
    
    fig = go.Figure()
    param_desc = PARAMETERS.get(param, param)
    fig.add_trace(go.Scatter(x=df_copy['Date'], y=df_copy[param], name=param_desc, mode='markers', marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=df_copy['Date'], y=df_copy['Trend'], name=f'Trend (slope: {slope:.4e}/day)', line=dict(color='red')))
    
    fig.update_layout(title=f'Trend for {param_desc} (Annual: {annual_change:.2f})', xaxis_title='Date', yaxis_title=param_desc, height=400)
    
    trend_stats = {
        'Parameter': param_desc, 'Slope_per_day': slope, 'Intercept': intercept, 'R_squared': r_value**2, 
        'P_value': p_value, 'Std_Err': std_err, 'Annual_Change': annual_change,
        'Significance': "Statistically significant (p<0.05)" if p_value < 0.05 else "Not statistically significant (p>=0.05)"
    }
    return fig, trend_stats

def create_correlation_matrix(df):
    """Create correlation matrix for climate variables"""
    numeric_df = df.select_dtypes(include=[np.number])
    if 'DateOrdinal' in numeric_df.columns: 
        numeric_df = numeric_df.drop('DateOrdinal', axis=1)
    
    rename_map = {col: PARAMETERS.get(col, col) for col in numeric_df.columns}
    display_df = numeric_df.rename(columns=rename_map)

    if display_df.shape[1] < 2: return None, pd.DataFrame() 
        
    corr_matrix = display_df.corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title='Correlation Matrix of Climate Variables')
    fig.update_layout(height=max(400, display_df.shape[1]*40) ) 
    return fig, corr_matrix

def create_climate_classification(df):
    """Create climate classification analysis"""
    if not ({'T2M', 'PRECTOTCORR'} <= set(df.columns)) or df[['T2M', 'PRECTOTCORR']].isnull().all().any():
        return None, pd.DataFrame()
    
    df_copy = df.copy().dropna(subset=['T2M', 'PRECTOTCORR'])
    if df_copy.empty: return None, pd.DataFrame()
    df_copy['Month'] = df_copy['Date'].dt.month
    
    monthly_summary = df_copy.groupby('Month').agg(Avg_T=('T2M', 'mean'), Total_P=('PRECTOTCORR', 'sum')).reset_index() 
    monthly_summary['Month_Name'] = monthly_summary['Month'].apply(lambda x: calendar.month_abbr[x])

    conditions = [
        (monthly_summary['Avg_T'] >= 18) & (monthly_summary['Total_P'] >= 60), 
        (monthly_summary['Avg_T'] >= 18), 
        (monthly_summary['Avg_T'] < 18) & (monthly_summary['Avg_T'] >= 0), 
        (monthly_summary['Avg_T'] < 0)  
    ]
    values = ['Tropical Wet', 'Tropical Dry/Monsoon', 'Temperate', 'Polar']
    monthly_summary['Climate_Type'] = np.select(conditions, values, default='Undefined')
    
    fig = px.scatter(monthly_summary, x='Avg_T', y='Total_P', color='Climate_Type', text='Month_Name',
                     title='Monthly Climate Character (Illustrative)',
                     labels={'Avg_T': 'Avg Monthly Temp (°C)', 'Total_P': 'Total Monthly Precip (mm)'})
    fig.update_traces(textposition='top center', marker=dict(size=10))
    fig.update_layout(height=500)
    return fig, monthly_summary

def show_monthly_summaries_ui(monthly_data_agg, params_dict):
    """Show monthly summaries UI"""
    st.subheader("Monthly Climate Summaries")
    if not monthly_data_agg:
        st.info("No monthly summary data to display.")
        return

    param_names = [params_dict.get(p, p) for p in monthly_data_agg.keys()]
    if not param_names:
        st.info("No parameters found for monthly summary.")
        return
        
    tabs = st.tabs(param_names)
    
    for i, (param_key, tab_content) in enumerate(zip(monthly_data_agg.keys(), tabs)):
        with tab_content:
            df_monthly_param = monthly_data_agg[param_key]
            numeric_cols_in_summary = ['mean', 'min', 'max', 'sum']
            cols_to_format = [col for col in numeric_cols_in_summary if col in df_monthly_param.columns]
            if cols_to_format:
                 st.dataframe(df_monthly_param.style.format("{:.2f}", subset=pd.IndexSlice[:, cols_to_format]))
            else:
                 st.dataframe(df_monthly_param)
            
            fig = go.Figure()
            if 'mean' in df_monthly_param.columns:
                fig.add_trace(go.Scatter(x=df_monthly_param['Month_Year'], y=df_monthly_param['mean'], name='Mean', mode='lines+markers'))
            if 'min' in df_monthly_param.columns:
                fig.add_trace(go.Scatter(x=df_monthly_param['Month_Year'], y=df_monthly_param['min'], name='Min', line=dict(dash='dot')))
            if 'max' in df_monthly_param.columns:
                fig.add_trace(go.Scatter(x=df_monthly_param['Month_Year'], y=df_monthly_param['max'], name='Max', line=dict(dash='dot')))
            if 'sum' in df_monthly_param.columns and param_key == 'PRECTOTCORR':
                 fig.add_trace(go.Bar(x=df_monthly_param['Month_Year'], y=df_monthly_param['sum'], name='Total Sum', opacity=0.6))

            if fig.data: 
                fig.update_layout(title=f'Monthly Statistics for {params_dict.get(param_key, param_key)}',
                                xaxis_title='Month-Year', yaxis_title=params_dict.get(param_key, param_key))
                st.plotly_chart(fig, use_container_width=True)

def create_seasonal_analysis(df, param='T2M'):
    """Create seasonal analysis for climate parameter"""
    if param not in df.columns or df[param].isnull().all(): return None, pd.DataFrame()
    df_copy = df.copy().dropna(subset=[param])
    if df_copy.empty: return None, pd.DataFrame()

    df_copy['Month'] = df_copy['Date'].dt.month
    def get_season(month): 
        if month in [12, 1, 2]: return 'Winter (DJF)'
        if month in [3, 4, 5]: return 'Spring (MAM)'
        if month in [6, 7, 8]: return 'Summer (JJA)'
        return 'Autumn (SON)' 
    df_copy['Season'] = df_copy['Month'].apply(get_season)
    
    seasonal_stats = df_copy.groupby('Season')[param].agg(['mean', 'min', 'max', 'std']).reset_index()
    season_order = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]
    
    param_desc = PARAMETERS.get(param, param)
    fig = px.bar(seasonal_stats, x='Season', y='mean', error_y='std', color='Season',
                 title=f'Seasonal Analysis of {param_desc}',
                 labels={'mean': f'Average {param_desc}', 'Season': 'Season'},
                 category_orders={"Season": season_order})
    fig.update_layout(height=400)
    return fig, seasonal_stats

def perform_stl_decomposition(df_ts, value_col='value', period=365): 
    """Perform STL decomposition on time series"""
    if df_ts.empty or value_col not in df_ts.columns or df_ts[value_col].isnull().all():
        return None
    
    series = df_ts.set_index('date')[value_col].dropna()
    if len(series) < 2 * period: 
        st.warning(f"Not enough data points ({len(series)}) for STL decomposition with period {period}.")
        return None
    if series.index.nunique() != len(series): 
        series = series.groupby(series.index).mean() 
    
    try:
        result = sm.tsa.seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
        
        decomp_df = pd.DataFrame({
            'Observed': result.observed, 'Trend': result.trend,
            'Seasonal': result.seasonal, 'Residual': result.resid
        }).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decomp_df['date'], y=decomp_df['Observed'], name='Observed', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=decomp_df['date'], y=decomp_df['Trend'], name='Trend', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=decomp_df['date'], y=decomp_df['Seasonal'], name='Seasonal', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=decomp_df['date'], y=decomp_df['Residual'], name='Residual', line=dict(color='green', dash='dot')))
        fig.update_layout(title=f'STL Decomposition of {value_col}', height=700, xaxis_title='Date') 
        return fig
    except Exception as e:
        st.error(f"Error during STL decomposition: {type(e).__name__}: {e}")
        return None

# =================================================================================================
# EXTREME EVENT ANALYSIS FUNCTIONS
# =================================================================================================
def detect_heatwaves(df, temp_col='T2M_MAX', threshold=35, min_days=3):
    """Detect heatwave events"""
    if temp_col not in df.columns or df[temp_col].isnull().all(): return pd.DataFrame()
    df_c = df.copy().dropna(subset=[temp_col, 'Date']) 
    df_c['Heatwave_Candidate'] = df_c[temp_col] > threshold
    df_c['Group'] = (df_c['Heatwave_Candidate'] != df_c['Heatwave_Candidate'].shift()).cumsum()
    heatwave_events = []
    for _, group_df in df_c.groupby('Group'):
        if group_df['Heatwave_Candidate'].all() and len(group_df) >= min_days:
            heatwave_events.append({
                'Start_Date': group_df['Date'].min().strftime('%Y-%m-%d'),
                'End_Date': group_df['Date'].max().strftime('%Y-%m-%d'),
                'Duration_Days': len(group_df),
                f'Avg_Max_Temp': group_df[temp_col].mean()
            })
    return pd.DataFrame(heatwave_events)

def detect_coldwaves(df, temp_col='T2M_MIN', threshold=5, min_days=3):
    """Detect coldwave events"""
    if temp_col not in df.columns or df[temp_col].isnull().all(): return pd.DataFrame()
    df_c = df.copy().dropna(subset=[temp_col, 'Date'])
    df_c['Coldwave_Candidate'] = df_c[temp_col] < threshold
    df_c['Group'] = (df_c['Coldwave_Candidate'] != df_c['Coldwave_Candidate'].shift()).cumsum()
    coldwave_events = []
    for _, group_df in df_c.groupby('Group'):
        if group_df['Coldwave_Candidate'].all() and len(group_df) >= min_days:
            coldwave_events.append({
                'Start_Date': group_df['Date'].min().strftime('%Y-%m-%d'),
                'End_Date': group_df['Date'].max().strftime('%Y-%m-%d'),
                'Duration_Days': len(group_df),
                f'Avg_Min_Temp': group_df[temp_col].mean()
            })
    return pd.DataFrame(coldwave_events)

def simple_drought_index(df, precip_col='PRECTOTCORR', window=30): 
    """Calculate simple drought index"""
    if precip_col not in df.columns or df[precip_col].isnull().all(): return pd.DataFrame()
    df_c = df.copy().dropna(subset=[precip_col, 'Date'])
    df_c['Drought_Index_Value'] = df_c[precip_col].rolling(window=window, min_periods=max(1, window//2)).sum()
    return df_c[['Date', 'Drought_Index_Value']].dropna()

def diurnal_temperature_range_calc(df): 
    """Calculate diurnal temperature range"""
    if {'T2M_MAX', 'T2M_MIN'} <= set(df.columns) and \
       not df['T2M_MAX'].isnull().all() and \
       not df['T2M_MIN'].isnull().all():
        df_c = df.copy().dropna(subset=['T2M_MAX', 'T2M_MIN', 'Date'])
        if df_c.empty: return pd.DataFrame()
        df_c['DTR'] = df_c['T2M_MAX'] - df_c['T2M_MIN']
        return df_c[['Date', 'DTR']]
    return pd.DataFrame()

# =================================================================================================
# MAIN APPLICATION
# =================================================================================================
def main():
    # Initialize session state
    initialize_session_state()
    
    # Initialize services
    if not st.session_state.gee_initialized:
        st.session_state.gee_initialized = initialize_gee()
    
    if not st.session_state.backend_available:
        st.session_state.backend_available, st.session_state.crop_data = initialize_backend()

    # App title and description
    st.title("🌾 Agricultural Intelligence Dashboard")
    st.markdown("""
    Comprehensive platform combining climate analysis and crop yield prediction for precision agriculture.
    Powered by NASA POWER API, Google Earth Engine, and AI-driven yield models.
    """)

    # Create main tabs
    tabs = st.tabs([
        "🌦️ Climate Analysis", 
        "📍 Field Analysis & Yield Prediction", 
        "💰 Crop Profitability", 
        "🌍 Regional Overview"
    ])

    # =================================================================================================
    # TAB 1: CLIMATE ANALYSIS
    # =================================================================================================
    with tabs[0]:
        st.header("🌦️ Climate Analysis Dashboard")
        st.markdown("Analyze historical and current climate patterns using NASA POWER data and satellite imagery.")

        # Sidebar controls for climate analysis
        with st.sidebar:
            st.header("📍 Location Selection")
            selected_loc_name = st.selectbox("Select Predefined Location", options=list(LOCATIONS.keys()), index=0)
            
            lat_default = LOCATIONS[selected_loc_name]["latitude"]
            lon_default = LOCATIONS[selected_loc_name]["longitude"]

            use_custom_coords = st.checkbox("Use Custom Coordinates")
            if use_custom_coords:
                custom_lat = st.number_input("Latitude", value=lat_default, min_value=-90.0, max_value=90.0, format="%.4f")
                custom_lon = st.number_input("Longitude", value=lon_default, min_value=-180.0, max_value=180.0, format="%.4f")
                lat, lon = custom_lat, custom_lon
                current_loc_name = "Custom Location"
            else:
                lat, lon = lat_default, lon_default
                current_loc_name = selected_loc_name
            
            current_location_info = {'name': current_loc_name, 'latitude': lat, 'longitude': lon}

            st.header("📅 Date Range")
            api_lag_days = 7 
            max_end_date = datetime.now().date() - timedelta(days=api_lag_days)
            default_start_date = max_end_date - timedelta(days=365) 

            date_tuple = st.date_input("Select Date Range", 
                                       value=(default_start_date, max_end_date),
                                       min_value=datetime(1981, 1, 1).date(), 
                                       max_value=max_end_date)
            start_date_dt, end_date_dt = datetime.combine(date_tuple[0], datetime.min.time()), \
                                         datetime.combine(date_tuple[1], datetime.max.time())

            st.header("📊 Climate Parameters")
            default_params = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "WS2M"]
            selected_nasa_params = []
            with st.expander("Select Parameters", expanded=False):
                for param_code, param_desc in PARAMETERS.items():
                    if param_code in NASA_POWER_SUPPORTED_PARAMS: 
                        if st.checkbox(param_desc, value=(param_code in default_params), key=f"cb_{param_code}"):
                            selected_nasa_params.append(param_code)

            if st.button("🚀 Analyze Climate Data", type="primary"):
                if not selected_nasa_params:
                    st.warning("Please select at least one climate parameter.")
                elif start_date_dt >= end_date_dt:
                    st.error("Start date must be before end date.")
                else:
                    st.session_state.location_info = current_location_info 
                    nasa_data_json = fetch_nasa_power_data(lat, lon, start_date_dt, end_date_dt, selected_nasa_params)
                    if nasa_data_json:
                        st.session_state.nasa_df = process_nasa_data(nasa_data_json)
                        if st.session_state.nasa_df.empty:
                            st.warning(f"No data returned for {current_location_info['name']}.")
                        else:
                            st.success(f"Climate data fetched for {current_location_info['name']}!")
                    else:
                        st.session_state.nasa_df = pd.DataFrame()
                    
                    # Reset GEE states
                    st.session_state.ee_map = None
                    st.session_state.ee_time_series_plot = None
                    st.session_state.ee_time_series_data = pd.DataFrame()
                    st.session_state.ee_stl_plot = None
                    st.session_state.current_ee_dataset = None
                    st.session_state.current_ee_band = None
                    st.rerun()

        # Main climate analysis content
        display_loc_info = st.session_state.location_info if st.session_state.location_info else current_location_info
        
        # Location map
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"🌍 Location: {display_loc_info['name']}")
            simple_map = geemap.Map(center=[display_loc_info['latitude'], display_loc_info['longitude']], 
                                  zoom=7, basemap="OpenStreetMap", add_google_map=False)
            simple_map.add_marker(location=[display_loc_info['latitude'], display_loc_info['longitude']], 
                                popup=f"{display_loc_info['name']}<br>Lat: {display_loc_info['latitude']:.2f}, Lon: {display_loc_info['longitude']:.2f}",
                                icon=folium.Icon(color='blue', icon='map-marker'))
            st_folium(simple_map, width=None, height=300, returned_objects=[])

        with col2:
            if not st.session_state.nasa_df.empty:
                df = st.session_state.nasa_df
                st.metric("📅 Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
                st.metric("📊 Data Points", f"{len(df)} days")
                st.metric("📍 Coordinates", f"{display_loc_info['latitude']:.2f}, {display_loc_info['longitude']:.2f}")

        # Climate analysis results
        if not st.session_state.nasa_df.empty:
            df = st.session_state.nasa_df
            loc_info = st.session_state.location_info 
            
            st.header(f"Climate Analysis Results for {loc_info['name']}")
            
            climate_tabs = st.tabs(["Basic Plots", "Advanced Analysis", "🛰️ GEE Map & Time Series", "Seasonal Patterns", "Extreme Events", "Data Export"])

            with climate_tabs[0]: 
                st.subheader("Basic Climate Variable Plots")
                col1a, col1b = st.columns(2)
                with col1a:
                    temp_plot = create_temperature_plot(df)
                    if temp_plot: st.plotly_chart(temp_plot, use_container_width=True)
                    else: st.info("Temperature data not available or not selected.")
                with col1b:
                    precip_plot = create_precipitation_plot(df)
                    if precip_plot: st.plotly_chart(precip_plot, use_container_width=True)
                    else: st.info("Precipitation data not available or not selected.")
                
                humidity_wind_plot = create_humidity_wind_plot(df)
                if humidity_wind_plot: st.plotly_chart(humidity_wind_plot, use_container_width=True)
                else: st.info("Humidity or Wind data not available or not selected.")

                solar_plot = create_solar_radiation_plot(df)
                if solar_plot: st.plotly_chart(solar_plot, use_container_width=True)
                else: st.info("Solar Radiation data not available or not selected.")

            with climate_tabs[1]: 
                st.subheader("Advanced Climate Analysis")
                if {'T2M', 'PRECTOTCORR'} <= set(df.columns):
                    st.markdown("#### Climate Classification (Illustrative)")
                    class_fig, class_data = create_climate_classification(df)
                    if class_fig: st.plotly_chart(class_fig, use_container_width=True)
                
                st.markdown("#### Correlation Matrix")
                corr_fig, _ = create_correlation_matrix(df) 
                if corr_fig: st.plotly_chart(corr_fig, use_container_width=True)

                st.markdown("#### Trend Analysis")
                trend_param_options = [p for p in df.columns if p not in ['Date', 'DateOrdinal'] and pd.api.types.is_numeric_dtype(df[p])]
                if trend_param_options:
                    default_trend_idx = trend_param_options.index('T2M') if 'T2M' in trend_param_options else 0
                    trend_param = st.selectbox("Select parameter for Trend Analysis:", trend_param_options, index=default_trend_idx, key="trend_param_select")
                    trend_fig, trend_stats_data = perform_trend_analysis(df, trend_param)
                    if trend_fig: 
                        st.plotly_chart(trend_fig, use_container_width=True)
                        st.json(trend_stats_data, expanded=False) 
                else: st.info("No suitable numeric parameters for trend analysis.")

                st.markdown("#### Anomaly Analysis")
                anomaly_param_options = trend_param_options 
                if anomaly_param_options:
                    default_anomaly_idx = anomaly_param_options.index('T2M') if 'T2M' in anomaly_param_options else 0
                    anomaly_param = st.selectbox("Select parameter for Anomaly Analysis:", anomaly_param_options, index=default_anomaly_idx, key="anomaly_param_select")
                    anomaly_fig, _ = perform_climate_anomaly_analysis(df, anomaly_param) 
                    if anomaly_fig: st.plotly_chart(anomaly_fig, use_container_width=True)
                else: st.info("No suitable numeric parameters for anomaly analysis.")

            with climate_tabs[2]: 
                st.subheader("Google Earth Engine: Spatial Map and Time Series")
                st.write(f"Using location: {loc_info['name']} (Lat: {loc_info['latitude']:.2f}, Lon: {loc_info['longitude']:.2f})")
                
                if st.session_state.gee_initialized:
                    gee_start_dt_display = st.session_state.nasa_df['Date'].min() 
                    gee_end_dt_display = st.session_state.nasa_df['Date'].max()
                    st.write(f"GEE analysis date range: {gee_start_dt_display.strftime('%Y-%m-%d')} to {gee_end_dt_display.strftime('%Y-%m-%d')}")

                    gee_col1, gee_col2 = st.columns([1,2]) 
                    with gee_col1:
                        dataset_keys = list(GEE_DATASETS.keys())
                        selected_ee_dataset_id = st.selectbox("Select GEE Dataset:", dataset_keys, 
                                                           format_func=lambda x: GEE_DATASETS[x], key="gee_dataset_select")
                        
                        available_bands = EE_BANDS.get(selected_ee_dataset_id, {})
                        if not available_bands:
                            st.error(f"No bands defined for dataset: {selected_ee_dataset_id}")
                        else:
                            selected_ee_band_id = st.selectbox("Select GEE Variable/Band:", list(available_bands.keys()),
                                                            format_func=lambda x: available_bands[x], key="gee_band_select")

                            if st.button("🌍 Generate GEE Map & Time Series", key="generate_gee"):
                                with st.spinner("Processing GEE data... This may take a moment."):
                                    st.session_state.current_ee_dataset = selected_ee_dataset_id
                                    st.session_state.current_ee_band = selected_ee_band_id
                                    
                                    st.session_state.ee_map = create_ee_map_display(selected_ee_dataset_id, selected_ee_band_id, 
                                                                                 gee_start_dt_display, gee_end_dt_display, loc_info, loc_info['name'])
                                    
                                    point_geometry = ee.Geometry.Point([loc_info["longitude"], loc_info["latitude"]])
                                    gee_scale = 30 
                                    if selected_ee_dataset_id and selected_ee_dataset_id.startswith("MODIS"): gee_scale = 250 
                                    if selected_ee_dataset_id and (selected_ee_dataset_id.startswith("ECMWF") or selected_ee_dataset_id.startswith("NASA/GLDAS")): gee_scale = 5000 

                                    ts_plot, ts_data = create_ee_time_series_plot(selected_ee_dataset_id, selected_ee_band_id,
                                                                          gee_start_dt_display, gee_end_dt_display, point_geometry, scale=gee_scale)
                                    st.session_state.ee_time_series_plot = ts_plot
                                    st.session_state.ee_time_series_data = ts_data if ts_data is not None else pd.DataFrame()

                                    if not st.session_state.ee_time_series_data.empty:
                                        stl_period = 365 
                                        if selected_ee_dataset_id and "MODIS/006/MOD13A2" in selected_ee_dataset_id: stl_period = int(365.25 / 16)
                                        elif selected_ee_dataset_id and "COPERNICUS/S2_SR" in selected_ee_dataset_id: stl_period = int(365.25 / 5) 

                                        st.session_state.ee_stl_plot = perform_stl_decomposition(st.session_state.ee_time_series_data, 'value', period=stl_period)
                                    else:
                                        st.session_state.ee_stl_plot = None
                                
                                if st.session_state.ee_map or st.session_state.ee_time_series_plot:
                                     st.success("GEE analysis complete.")
                                else:
                                     st.warning("GEE analysis did not produce results.")
                                st.rerun() 

                    with gee_col2: 
                        if st.session_state.ee_map:
                            band_desc = EE_BANDS.get(st.session_state.current_ee_dataset, {}).get(st.session_state.current_ee_band, 'N/A')
                            st.markdown(f"#### Map: {band_desc}")
                            st_folium(st.session_state.ee_map, width=None, height=450, returned_objects=[])
                        
                        if st.session_state.ee_time_series_plot:
                            band_desc = EE_BANDS.get(st.session_state.current_ee_dataset, {}).get(st.session_state.current_ee_band, 'N/A')
                            st.markdown(f"#### Time Series Plot: {band_desc}")
                            st.plotly_chart(st.session_state.ee_time_series_plot, use_container_width=True)
                            if not st.session_state.ee_time_series_data.empty and 'value' in st.session_state.ee_time_series_data:
                                stats_ts = st.session_state.ee_time_series_data['value'].agg(['mean', 'min', 'max', 'std']).to_frame().T
                                st.write("Time Series Statistics:")
                                st.dataframe(stats_ts.style.format("{:.2f}", subset=[col for col in ['mean', 'min', 'max', 'std'] if col in stats_ts.columns]))

                        if st.session_state.ee_stl_plot:
                            band_desc = EE_BANDS.get(st.session_state.current_ee_dataset, {}).get(st.session_state.current_ee_band, 'N/A')
                            st.markdown(f"#### STL Decomposition: {band_desc}")
                            st.plotly_chart(st.session_state.ee_stl_plot, use_container_width=True)
                else:
                    st.warning("Google Earth Engine not initialized. Satellite analysis unavailable.")
        
            with climate_tabs[3]: 
                st.subheader("Seasonal Patterns")
                seasonal_param_options = [p for p in df.columns if p not in ['Date', 'DateOrdinal'] and pd.api.types.is_numeric_dtype(df[p])]
                if seasonal_param_options:
                    default_seasonal_idx = seasonal_param_options.index('T2M') if 'T2M' in seasonal_param_options else 0
                    seasonal_param = st.selectbox("Select parameter for Seasonal Analysis:", seasonal_param_options, key="seasonal_select", index=default_seasonal_idx)
                    season_fig, season_stats_df = create_seasonal_analysis(df, seasonal_param)
                    if season_fig: 
                        st.plotly_chart(season_fig, use_container_width=True)
                        st.dataframe(season_stats_df.style.format("{:.2f}", subset=[col for col in ['mean', 'min', 'max', 'std'] if col in season_stats_df.columns]))
                else: st.info("No suitable parameters for seasonal analysis.")
                
                monthly_summary_data = create_monthly_summary(df)
                show_monthly_summaries_ui(monthly_summary_data, PARAMETERS)

            with climate_tabs[4]: 
                st.subheader("Extreme Event Analysis")
                
                dtr_data = diurnal_temperature_range_calc(df) 
                if not dtr_data.empty:
                    st.markdown("#### Diurnal Temperature Range (DTR)")
                    fig_dtr = px.line(dtr_data, x='Date', y='DTR', title='Daily DTR')
                    st.plotly_chart(fig_dtr, use_container_width=True)
                    st.metric("Average DTR", f"{dtr_data['DTR'].mean():.2f} °C")

                heatwaves_df = detect_heatwaves(df)
                coldwaves_df = detect_coldwaves(df)
                
                ext_col1, ext_col2 = st.columns(2)
                with ext_col1:
                    st.markdown("#### Heatwave Events (Tmax > 35°C for 3+ days)")
                    if not heatwaves_df.empty: 
                        st.dataframe(heatwaves_df.style.format({'Avg_Max_Temp': "{:.2f}", 'Duration_Days':'{:.0f}'}))
                    else: st.info("No heatwaves detected based on criteria.")
                with ext_col2:
                    st.markdown("#### Coldwave Events (Tmin < 5°C for 3+ days)")
                    if not coldwaves_df.empty: 
                        st.dataframe(coldwaves_df.style.format({'Avg_Min_Temp': "{:.2f}", 'Duration_Days':'{:.0f}'}))
                    else: st.info("No coldwaves detected based on criteria.")

                if 'PRECTOTCORR' in df.columns:
                    drought_df = simple_drought_index(df)
                    if not drought_df.empty:
                        st.markdown("#### Simple Drought Index (30-day rolling precip sum)")
                        fig_drought = px.line(drought_df, x='Date', y='Drought_Index_Value', title='Drought Index')
                        st.plotly_chart(fig_drought, use_container_width=True)

            with climate_tabs[5]: 
                st.subheader("Data Export Options")
                st.markdown("#### NASA POWER Data")
                if st.checkbox("Show Raw NASA POWER Data Table", key="show_raw_nasa"):
                    st.dataframe(df)
                
                csv_nasa = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download NASA POWER Data as CSV",
                    data=csv_nasa,
                    file_name=f"nasa_power_data_{loc_info['name'].replace(' ', '_').replace(',', '')}.csv",
                    mime='text/csv',
                    key="download_nasa_csv"
                )

                st.markdown("#### GEE Time Series Data")
                if not st.session_state.ee_time_series_data.empty:
                    if st.checkbox("Show GEE Time Series Data Table", key="show_raw_gee"):
                        st.dataframe(st.session_state.ee_time_series_data)
                    
                    csv_gee = st.session_state.ee_time_series_data.to_csv(index=False).encode('utf-8')
                    ds_name_safe = st.session_state.current_ee_dataset.replace('/', '_') if st.session_state.current_ee_dataset else "gee"
                    band_name_safe = st.session_state.current_ee_band if st.session_state.current_ee_band else "data"
                    loc_name_safe = loc_info['name'].replace(' ', '_').replace(',', '')
                    st.download_button(
                        label=f"Download GEE Time Series ({band_name_safe}) as CSV",
                        data=csv_gee,
                        file_name=f"gee_timeseries_{ds_name_safe}_{band_name_safe}_{loc_name_safe}.csv",
                        mime='text/csv',
                        key="download_gee_csv"
                    )
                else:
                    st.info("No GEE time series data available to export.")
        else:
            st.info("⬅️ Select location, date range, and parameters in the sidebar, then click 'Analyze Climate Data'.")

    # =================================================================================================
    # TAB 2: FIELD ANALYSIS & YIELD PREDICTION
    # =================================================================================================
    with tabs[1]:
        st.header("📍 Field Analysis & Yield Prediction")
        st.markdown("Draw your field boundaries and get AI-powered yield predictions using satellite imagery.")

        if not st.session_state.backend_available:
            st.warning("⚠️ Yield prediction backend not available. Field mapping still works.")

        col1, col2 = st.columns([0.65, 0.35])

        with col1:
            # Location search
            search_query = st.text_input("Search for a location (e.g., 'Dharwad, Karnataka')", 
                                       placeholder="Enter location name...")
            if st.button("🔍 Search Location"):
                if search_query:
                    geolocator = Nominatim(user_agent="agri_intelligence_dashboard")
                    try:
                        location = geolocator.geocode(search_query, addressdetails=True, language="en")
                        if location:
                            st.session_state.map_center = [location.latitude, location.longitude]
                            st.session_state.map_zoom = 15
                            st.success(f"Location found: {location.address}")
                        else: 
                            st.warning("Location not found. Please try a different search term.")
                    except Exception as e: 
                        st.error(f"Geocoding service failed: {e}")

            # Interactive map for field drawing
            m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, 
                          tiles="Esri.WorldImagery")
            Draw(export=False, position='topleft', 
                 draw_options={'polygon': True, 'rectangle': True, 'circle': False, 'marker': False, 
                              'circlemarker': False, 'polyline': False}).add_to(m)
            
            map_output = st_folium(m, key="field_map", width='100%', height=500)

            # Process drawn field
            if map_output and map_output.get("last_active_drawing"):
                new_drawing = map_output["last_active_drawing"]
                if st.session_state.field_details.get("geometry_json") != new_drawing['geometry']:
                    try:
                        polygon = Polygon(new_drawing['geometry']['coordinates'][0])
                        # Use backend function if available, otherwise estimate
                        if BACKEND_AVAILABLE:
                            area_ha = backend.get_polygon_area(polygon) / 10000
                            district = backend.get_district_from_coords(polygon.centroid.y, polygon.centroid.x)
                        else:
                            # Simple area estimation
                            area_sq_m = polygon.area * 111000 * 111000 * np.cos(np.radians(polygon.centroid.y))
                            area_ha = area_sq_m / 10000
                            district = "Unknown"
                        
                        st.session_state.field_details = {
                            "geometry_json": new_drawing['geometry'],
                            "area_ha": area_ha,
                            "district": district,
                            "centroid": {"lat": polygon.centroid.y, "lon": polygon.centroid.x}
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing field geometry: {e}")

        with col2:
            st.subheader("📋 Field Information")
            details = st.session_state.field_details
            
            if details:
                st.metric("🏞️ Field Area", f"{details.get('area_ha', 0):.2f} hectares")
                st.metric("📍 District", details.get('district', 'Unknown'))
                
                if details.get('centroid'):
                    centroid = details['centroid']
                    st.metric("🎯 Center Coordinates", f"{centroid['lat']:.4f}, {centroid['lon']:.4f}")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🗑️ Clear Field", use_container_width=True):
                        st.session_state.field_details = {}
                        st.rerun()
                
                with col_b:
                    if st.button("🌡️ Get Climate Data", use_container_width=True):
                        if details.get('centroid'):
                            # Auto-populate climate analysis with field location
                            st.session_state.location_info = {
                                'name': f"Field Location ({details.get('district', 'Unknown')})",
                                'latitude': details['centroid']['lat'],
                                'longitude': details['centroid']['lon']
                            }
                            st.info("Field location set for climate analysis. Switch to Climate Analysis tab.")

                # Yield Prediction Section
                if st.session_state.backend_available and details.get('district'):
                    st.markdown("---")
                    st.subheader("🌾 Yield Prediction")
                    
                    try:
                        available_crops = backend.get_available_crops()
                        if not available_crops:
                            st.error("No trained models found.")
                        else:
                            selected_crop = st.selectbox("Select Crop", available_crops)
                            prediction_year = st.selectbox("Prediction Year", range(pd.Timestamp.now().year + 1, 2018, -1))
                            
                            col_baseline, col_refined = st.columns(2)
                            
                            with col_baseline:
                                st.markdown("**⚡ Baseline Prediction**")
                                if st.button("Get Baseline", use_container_width=True):
                                    baseline_yield, total_yield, unit = backend.get_baseline_prediction(
                                        st.session_state.crop_data, details['district'], selected_crop, prediction_year, details['area_ha']
                                    )
                                    details['baseline_prediction'] = {"yield": baseline_yield, "total": total_yield, "unit": unit, "crop": selected_crop}
                            
                            with col_refined:
                                st.markdown("**🛰️ Satellite-Refined**")
                                if st.button("Get Refined", use_container_width=True):
                                    with st.spinner(f"Analyzing satellite data for {selected_crop}..."):
                                        field_geom_ee = backend.geojson_to_ee_geometry(details['geometry_json'])
                                        refined_result = backend.predict_yield_for_field(
                                            crop_name=selected_crop, district_name=details['district'], 
                                            year=prediction_year, field_geom=field_geom_ee
                                        )
                                        if refined_result:
                                            details['refined_prediction'] = {
                                                "yield": refined_result['calibrated_yield_per_hectare'],
                                                "total": refined_result['calibrated_yield_per_hectare'] * details['area_ha'],
                                                "unit": backend.get_unit_for_crop(selected_crop),
                                                "crop": selected_crop
                                            }
                                        else: 
                                            st.error("Refined prediction failed.")
                            
                            # Display predictions
                            if details.get('baseline_prediction'):
                                bp = details['baseline_prediction']
                                st.success(f"**Baseline for {bp['crop']}:**")
                                st.metric(f"Avg. Yield ({bp['unit']}/ha)", f"{bp['yield']:.2f}")
                                st.metric(f"Total Production ({bp['unit']})", f"{bp['total']:.2f}")
                            
                            if details.get('refined_prediction'):
                                rp = details['refined_prediction']
                                st.success(f"**Refined Prediction for {rp['crop']}:**")
                                st.metric(f"Avg. Yield ({rp['unit']}/ha)", f"{rp['yield']:.2f}")
                                st.metric(f"Total Production ({rp['unit']})", f"{rp['total']:.2f}")
                    except Exception as e:
                        st.error(f"Yield prediction error: {e}")
                elif not st.session_state.backend_available:
                    st.info("Yield prediction requires backend initialization.")
            else:
                st.info("👆 Draw your field on the map to begin analysis")
                st.markdown("""
                **Instructions:**
                1. Search for your location
                2. Use the polygon tool to draw field boundaries
                3. View field details and predictions
                """)

    # =================================================================================================
    # TAB 3: CROP PROFITABILITY
    # =================================================================================================
    with tabs[2]:
        st.header("💰 Crop Profitability Analysis")
        st.markdown("Analyze potential profits for different crops based on historical yields and market prices.")

        details = st.session_state.field_details
        if not details or not details.get('area_ha'):
            st.warning("⚠️ Please define your field in the **Field Analysis** tab first.")
            st.info("Field area is required to calculate total profitability.")
        else:
            st.info(f"📊 Analyzing profitability for {details['area_ha']:.2f} hectare field in {details.get('district', 'Unknown')} district")
            
            if st.session_state.backend_available and details.get('district'):
                # Profitability analysis controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    analysis_year = st.selectbox("Analysis Year", range(pd.Timestamp.now().year + 1, 2018, -1))
                with col2:
                    price_scenario = st.selectbox("Price Scenario", ["Current Market", "Conservative", "Optimistic"])
                with col3:
                    if st.button("📈 Analyze Profitability", use_container_width=True):
                        with st.spinner("Calculating crop profitability..."):
                            try:
                                profit_data = backend.analyze_profitability_baseline(
                                    st.session_state.crop_data, details['district'], analysis_year
                                )
                                
                                if not profit_data.empty:
                                    profit_data['Total Est. Revenue for Field (INR)'] = profit_data['Est. Revenue (INR/Hectare)'] * details.get('area_ha', 0)
                                    
                                    profit_display = profit_data.copy()
                                    profit_display['Avg. Yield'] = profit_display.apply(lambda r: f"{r['Avg. Yield']:.2f} {r['Unit']}/Hectare", axis=1)
                                    profit_display['Market Price'] = profit_display.apply(lambda r: f"₹ {r['Market Price']:,.2f}/{r['Price Unit']}", axis=1)
                                    profit_display['Est. Revenue (INR/Hectare)'] = profit_display['Est. Revenue (INR/Hectare)'].apply(lambda x: f"₹ {x:,.0f}")
                                    profit_display['Total Est. Revenue for Field (INR)'] = profit_display['Total Est. Revenue for Field (INR)'].apply(lambda x: f"₹ {x:,.0f}")

                                    st.success("Profitability Analysis Complete!")
                                    st.dataframe(profit_display[['Crop', 'Avg. Yield', 'Market Price', 'Est. Revenue (INR/Hectare)', 'Total Est. Revenue for Field (INR)']], use_container_width=True, hide_index=True)
                                    
                                    st.subheader("Revenue Comparison Chart")
                                    chart_data = profit_data[['Crop', 'Est. Revenue (INR/Hectare)']].copy()
                                    chart_data = chart_data[chart_data['Est. Revenue (INR/Hectare)'] > 0]
                                    
                                    if not chart_data.empty:
                                        chart_data = chart_data.set_index('Crop')
                                        st.bar_chart(chart_data)
                                    else:
                                        st.write("No profitable crops found to display.")
                                else: 
                                    st.error("Could not perform profitability analysis.")
                            except Exception as e:
                                st.error(f"Profitability analysis failed: {e}")
            else:
                st.info("Profitability analysis requires backend functionality and field location.")

    # =================================================================================================
    # TAB 4: REGIONAL OVERVIEW
    # =================================================================================================
    with tabs[3]:
        st.header("🌍 Regional Agricultural Overview")
        st.markdown("District-level analysis and yield predictions using satellite data and AI models.")

        if st.session_state.backend_available:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🗺️ District Selection")
                
                try:
                    districts = backend.get_karnataka_districts()
                    if districts:
                        selected_district = st.selectbox("Select District", districts, 
                                                       index=districts.index("Dharwad") if "Dharwad" in districts else 0)
                        
                        available_crops_regional = backend.get_available_crops()
                        if available_crops_regional:
                            selected_crop = st.selectbox("Select Crop", available_crops_regional, 
                                                       index=available_crops_regional.index('Onion') if 'Onion' in available_crops_regional else 0, 
                                                       key="regional_crop")
                            analysis_year = st.selectbox("Analysis Year", range(pd.Timestamp.now().year + 1, 2018, -1), key="regional_year")
                            
                            if st.button("🔍 Analyze District", type="primary", use_container_width=True):
                                with st.spinner(f"Running AI prediction for {selected_crop} in {selected_district}..."):
                                    try:
                                        yield_pred, centroid = backend.predict_yield_for_district(selected_crop, selected_district, analysis_year)
                                        
                                        if yield_pred and centroid:
                                            st.session_state.regional_prediction_result = {
                                                "yield": yield_pred, "centroid": centroid, "district": selected_district,
                                                "crop": selected_crop, "unit": backend.get_unit_for_crop(selected_crop)
                                            }
                                            st.success("Regional analysis complete!")
                                        else:
                                            st.session_state.regional_prediction_result = None
                                            st.error("Could not generate prediction.")
                                    except Exception as e:
                                        st.error(f"Regional analysis failed: {e}")
                except Exception as e:
                    st.error(f"Backend error: {e}")

            with col2:
                st.subheader("📊 Regional Analysis Results")
                
                if st.session_state.regional_prediction_result:
                    res = st.session_state.regional_prediction_result
                    st.metric(f"**AI-Predicted Yield in {res['district']}**", f"{res['yield']:.2f} {res['unit']}/Hectare")
                    
                    map_center = [res['centroid']['latitude'], res['centroid']['longitude']]
                    regional_map = folium.Map(location=map_center, zoom_start=9, tiles="CartoDB positron")
                    
                    try:
                        district_geojson = backend.get_district_geojson(res['district'])
                        if district_geojson:
                            folium.GeoJson(district_geojson, style_function=lambda x: {
                                'color': '#28a745', 'fillColor': '#8fbc8f', 'weight': 2.5, 'fillOpacity': 0.6
                            }).add_to(regional_map)
                    except:
                        pass
                    
                    folium.Marker(map_center, popup=f"{res['district']}: {res['yield']:.2f} {res['unit']}/ha",
                                icon=folium.Icon(color='green', icon='leaf')).add_to(regional_map)
                    
                    st_folium(regional_map, width='100%', height=400, key="regional_map_display")
                else:
                    st.info("Select a district and crop, then click 'Analyze District Yield' to see results.")
        else:
            st.info("Regional overview requires backend functionality.")

    # Sidebar information
    with st.sidebar:
        st.markdown("---")
        st.info(
            """
            **🌾 Agricultural Intelligence Dashboard**
            
            Integrating:
            - NASA POWER Climate Data
            - Google Earth Engine Satellite Imagery  
            - AI-Powered Yield Prediction
            - Market Analysis & Profitability
            
            For precision agriculture and informed decision making.
            """
        )
        
        with st.expander("🔧 System Status"):
            st.write("**Services:**")
            st.write(f"✅ Climate Analysis: Active")
            st.write(f"{'✅' if st.session_state.gee_initialized else '❌'} Google Earth Engine: {'Active' if st.session_state.gee_initialized else 'Inactive'}")
            st.write(f"{'✅' if st.session_state.backend_available else '❌'} Yield Prediction: {'Active' if st.session_state.backend_available else 'Inactive'}")

        with st.expander("📚 Data Sources & Methods"):
            st.markdown("""
            **NASA POWER**: Daily climate data (temperature, precipitation, humidity, etc.)
            
            **Google Earth Engine**: Satellite imagery analysis (NDVI, LST, soil moisture)
            
            **AI Models**: Machine learning for crop yield prediction using historical data and satellite features
            
            **Market Data**: Crop prices and profitability analysis
            """)

if __name__ == "__main__":
    main()