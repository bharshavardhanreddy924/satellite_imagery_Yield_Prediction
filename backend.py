import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ee
import pandas as pd
import numpy as np
import time
import calendar
import requests
import joblib
import tensorflow as tf
from shapely.geometry import Polygon
import json

# --- Configuration ---
GEE_PROJECT_ID = "positive-tempo-456012-d8"

CROP_DATA_CSV = "final.csv"
SAVED_MODEL_DIR = "saved_crop_models_v3"
MODEL_SAVE_PATH_TEMPLATE = os.path.join(SAVED_MODEL_DIR, "{crop_name}_yield_model.keras")
SCALERS_SAVE_PATH_TEMPLATE = os.path.join(SAVED_MODEL_DIR, "{crop_name}_scalers.joblib")
KARNATAKA_DISTRICTS_FC_PATH = "FAO/GAUL/2015/level2"
KARNATAKA_STATE_NAME_IN_GEE = 'Karnataka'
FALLBACK_DISTRICT_COORDS = {
    'Bagalkot': [75.7, 16.18], 'Bangalore rural': [77.68, 13.28], 'Belgaum': [74.5, 15.85], 'Bellary': [76.92, 15.14], 'Bengaluru urban': [77.59, 12.97], 'Bidar': [77.51, 17.91],
    'Bijapur': [75.71, 16.83], 'Chamarajanagar': [76.94, 11.92], 'Chikballapur': [77.73, 13.43], 'Chikmagalur': [75.77, 13.31], 'Chitradurga': [76.39, 14.22], 'Dakshin kannad': [75.25, 12.85],
    'Davangere': [75.92, 14.46], 'Dharwad': [75.00, 15.46], 'Gadag': [75.63, 15.43], 'Gulbarga': [76.83, 17.32], 'Hassan': [76.09, 13.00], 'Haveri': [75.40, 14.79],
    'Kodagu': [75.73, 12.33], 'Kolar': [78.13, 13.13], 'Koppal': [76.15, 15.35], 'Mandya': [76.89, 12.52], 'Mysore': [76.63, 12.29], 'Raichur': [77.34, 16.20],
    'Ramanagara': [77.28, 12.72], 'Shimoga': [75.56, 13.92], 'Tumkur': [77.11, 13.34], 'Udupi': [74.74, 13.34], 'Uttar kannad': [74.48, 14.79], 'Yadgir': [77.13, 16.76],
    'Vijayanagar': [76.46, 15.23]
}

# --- Caches ---
GEE_VI_CACHE, NASA_POWER_CACHE, DISTRICT_GEOMETRY_CACHE = {}, {}, {}

# --- Constants ---
GROWING_SEASON_START_MONTH, GROWING_SEASON_END_MONTH = 6, 11
NUM_MONTHLY_TIMESTEPS = (GROWING_SEASON_END_MONTH - GROWING_SEASON_START_MONTH) + 1
VI_FEATURE_SUPERSET = ['NDVI', 'EVI', 'SAVI']
SENTINEL2_START_YEAR = 2016
MODIS_VI_COLLECTION, MODIS_NDVI_BAND, MODIS_EVI_BAND, MODIS_VI_SCALE_FACTOR = 'MODIS/061/MOD13A1', 'NDVI', 'EVI', 0.0001
VI_LIST_MODIS, VI_LIST_S2 = ['NDVI', 'EVI'], ['NDVI', 'EVI', 'SAVI']
S2_QA_BAND, S2_CLOUD_BIT_MASK, S2_CIRRUS_BIT_MASK = 'QA60', 1 << 10, 1 << 11
NASA_POWER_PARAMS_DICT = {
    "T2M": "Temp_2m_C", "T2M_MAX": "Temp_Max_2m_C", "T2M_MIN": "Temp_Min_2m_C",
    "PRECTOTCORR": "Precip_mm_day", "RH2M": "RelHum_2m_percent", "WS2M": "WindSpeed_2m_mps",
    "ALLSKY_SFC_SW_DWN": "SolarRad_AllSky_WM2",
}
NASA_POWER_PARAMS_LIST = list(NASA_POWER_PARAMS_DICT.keys())


def initialize_gee():
    try:
        if not GEE_PROJECT_ID or GEE_PROJECT_ID == "your-google-earth-engine-project-id":
            raise ValueError("Please set your GEE_PROJECT_ID in backend.py")
        ee.Initialize(project=GEE_PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')

def load_crop_data():
    if not os.path.exists(CROP_DATA_CSV):
        raise FileNotFoundError(f"The data file '{CROP_DATA_CSV}' was not found.")
    df = pd.read_csv(CROP_DATA_CSV, on_bad_lines='skip')
    df['Year_temp'] = pd.to_numeric(df['Year'].astype(str).str.slice(0, 4), errors='coerce')
    df.dropna(subset=['Year_temp', 'District'], inplace=True)
    df['Year'] = df['Year_temp'].astype(int)
    df.drop(columns=['Year_temp'], inplace=True)
    return df

# --- Fast Baseline Predictions ---

def get_baseline_prediction(crop_df, district, crop, year, field_area_ha):
    unit = get_unit_for_crop(crop)
    actual_yield_col = next((col for col in crop_df.columns if col.startswith(f"{crop} - Yield")), None)
    if not actual_yield_col: return 0, 0, unit
    
    filtered_df = crop_df[(crop_df['District'] == district) & (crop_df['Year'] == year)]
    if filtered_df.empty or filtered_df[actual_yield_col].isnull().all():
        fallback_df = crop_df[crop_df['District'] == district].sort_values('Year', ascending=False)
        if not fallback_df.empty: filtered_df = fallback_df
    
    if filtered_df.empty: return 0, 0, unit
    
    avg_yield = filtered_df[actual_yield_col].mean()
    total_yield = (avg_yield * field_area_ha) if pd.notna(avg_yield) else 0
    return avg_yield if pd.notna(avg_yield) else 0, total_yield, unit

def analyze_profitability_baseline(crop_df, district, year):
    results = []
    # Use the corrected function that checks for model files
    for crop in get_available_crops():
        avg_yield, _, unit = get_baseline_prediction(crop_df, district, crop, year, 1)
        if avg_yield > 0:
            price = get_hardcoded_crop_price(crop)
            if price > 0:
                results.append({
                    "Crop": crop, "Avg. Yield": avg_yield, "Unit": unit,
                    "Market Price": price, "Price Unit": get_price_unit_for_crop(crop),
                    "Est. Revenue (INR/Hectare)": avg_yield * price
                })
    return pd.DataFrame(results).sort_values('Est. Revenue (INR/Hectare)', ascending=False) if results else pd.DataFrame()


# --- Slow, Refined Predictions (Satellite-based) ---
def fetch_monthly_nasa_power_data(lat, lon, year, start_month, end_month, parameters, retries=3, delay=5):
    cache_key = (lat, lon, year, tuple(parameters))
    if cache_key in NASA_POWER_CACHE: return NASA_POWER_CACHE[cache_key]
    start_date_str = f"{year}{start_month:02d}01"
    end_day = calendar.monthrange(year, end_month)[1]
    end_date_str = f"{year}{end_month:02d}{end_day:02d}"
    params_str = ",".join(parameters)
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={params_str}&community=AG&longitude={lon}&latitude={lat}&start={start_date_str}&end={end_date_str}&format=JSON"
    for _ in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data_json = resp.json()
            if "properties" not in data_json or "parameter" not in data_json["properties"]: return {}
            daily_df = pd.DataFrame(data_json['properties']['parameter'])
            daily_df.index = pd.to_datetime(daily_df.index, format='%Y%m%d')
            daily_df[daily_df == -999] = np.nan
            daily_df['Month'] = daily_df.index.month
            monthly_features = {}
            for p_code, p_label in NASA_POWER_PARAMS_DICT.items():
                monthly_mean = daily_df.groupby('Month')[p_code].mean()
                monthly_features[f'{p_label}_mean_monthly'] = [monthly_mean.get(m, np.nan) for m in range(start_month, end_month + 1)]
                if 'PREC' in p_code.upper() or 'SOLAR' in p_code.upper() or 'ALLSKY' in p_code.upper():
                    monthly_sum = daily_df.groupby('Month')[p_code].sum()
                    monthly_features[f'{p_label}_sum_monthly'] = [monthly_sum.get(m, np.nan) for m in range(start_month, end_month + 1)]
            NASA_POWER_CACHE[cache_key] = monthly_features
            return monthly_features
        except requests.exceptions.RequestException: time.sleep(delay)
    return {}

def mask_s2_clouds_gee(image):
    qa = image.select(S2_QA_BAND)
    cloud_mask = qa.bitwiseAnd(S2_CLOUD_BIT_MASK).eq(0)
    cirrus_mask = qa.bitwiseAnd(S2_CIRRUS_BIT_MASK).eq(0)
    return image.updateMask(cloud_mask).updateMask(cirrus_mask).select(['B2', 'B3', 'B4', 'B8']).divide(10000.0)

def calculate_s2_vis_gee(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1.0))', {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}).rename('EVI')
    savi = image.expression('((NIR - RED) / (NIR + RED + 0.5)) * (1.5)', {'NIR': image.select('B8'), 'RED': image.select('B4')}).rename('SAVI')
    return image.addBands([ndvi, evi, savi])

def get_monthly_gee_satellite_features(aoi_geometry, year, start_month, end_month):
    try:
        geojson_str = json.dumps(aoi_geometry.getInfo())
        cache_key = (geojson_str, year)
    except Exception: cache_key = None
    if cache_key and cache_key in GEE_VI_CACHE: return GEE_VI_CACHE[cache_key]
    use_modis = year < SENTINEL2_START_YEAR
    vi_list_for_sensor = VI_LIST_MODIS if use_modis else VI_LIST_S2
    monthly_vi_features = {f'{vi}_monthly_series': [np.nan] * NUM_MONTHLY_TIMESTEPS for vi in VI_FEATURE_SUPERSET}
    for i, month in enumerate(range(start_month, end_month + 1)):
        start_date, end_date = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
        try:
            if use_modis:
                collection = ee.ImageCollection(MODIS_VI_COLLECTION).filterDate(start_date, end_date)
                processed_collection = collection.select(vi_list_for_sensor).map(lambda img: img.multiply(MODIS_VI_SCALE_FACTOR))
            else:
                collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(start_date, end_date).filterBounds(aoi_geometry)
                processed_collection = collection.map(mask_s2_clouds_gee).map(calculate_s2_vis_gee).select(vi_list_for_sensor)
            if processed_collection.size().getInfo() > 0:
                stats = processed_collection.mean().reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi_geometry, scale=100, bestEffort=True, maxPixels=1e9).getInfo()
                for vi in vi_list_for_sensor: monthly_vi_features[f'{vi}_monthly_series'][i] = stats.get(vi)
        except Exception: continue
    for vi in VI_FEATURE_SUPERSET:
        valid_points = [x for x in monthly_vi_features[f'{vi}_monthly_series'] if x is not None and not np.isnan(x)]
        monthly_vi_features[f'{vi}_peak'] = max(valid_points) if valid_points else 0
        monthly_vi_features[f'{vi}_auc'] = np.trapz(valid_points) if len(valid_points) > 1 else (valid_points[0] if valid_points else 0)
    if cache_key: GEE_VI_CACHE[cache_key] = monthly_vi_features
    return monthly_vi_features

def get_district_geometry_and_centroid(district_name_str, state_name_str=KARNATAKA_STATE_NAME_IN_GEE):
    cache_key = (district_name_str, state_name_str)
    if cache_key in DISTRICT_GEOMETRY_CACHE: return DISTRICT_GEOMETRY_CACHE[cache_key]
    try:
        district_feature = ee.FeatureCollection(KARNATAKA_DISTRICTS_FC_PATH).filter(ee.Filter.And(ee.Filter.eq('ADM1_NAME', state_name_str), ee.Filter.eq('ADM2_NAME', district_name_str))).first()
        if district_feature.getInfo():
            geometry = district_feature.geometry()
            centroid_coords = geometry.centroid(maxError=1).coordinates().getInfo()
            result = (geometry, {'longitude': centroid_coords[0], 'latitude': centroid_coords[1]})
            DISTRICT_GEOMETRY_CACHE[cache_key] = result
            return result
    except Exception: pass
    if district_name_str in FALLBACK_DISTRICT_COORDS:
        coords = FALLBACK_DISTRICT_COORDS[district_name_str]
        lon, lat = coords[0], coords[1]
        result = (ee.Geometry.Point(lon, lat).buffer(10000), {'longitude': lon, 'latitude': lat})
        DISTRICT_GEOMETRY_CACHE[cache_key] = result
        return result
    return None, None

def get_karnataka_districts():
    try:
        fc = ee.FeatureCollection(KARNATAKA_DISTRICTS_FC_PATH).filter(ee.Filter.eq('ADM1_NAME', KARNATAKA_STATE_NAME_IN_GEE))
        return sorted(fc.aggregate_array('ADM2_NAME').getInfo())
    except Exception: return sorted(list(FALLBACK_DISTRICT_COORDS.keys()))

def get_district_geojson(district_name):
    geom, _ = get_district_geometry_and_centroid(district_name)
    return geom.getInfo() if geom else None

def get_polygon_area(polygon_shapely):
    try: return ee.Geometry.Polygon(list(polygon_shapely.exterior.coords)).area().getInfo()
    except Exception: return 0

def geojson_to_ee_geometry(geojson_data):
    try: return ee.Geometry(geojson_data)
    except Exception: return None

def get_district_from_coords(lat, lon):
    try:
        point = ee.Geometry.Point(lon, lat)
        district_feature = ee.FeatureCollection(KARNATAKA_DISTRICTS_FC_PATH).filter(ee.Filter.eq('ADM1_NAME', KARNATAKA_STATE_NAME_IN_GEE)).filterBounds(point).first()
        return district_feature.get('ADM2_NAME').getInfo() if district_feature.getInfo() else None
    except Exception: return None

# --- TAB 3 FIX IS HERE ---
def get_available_crops():
    """Gets a list of crops ONLY if a model file exists for it."""
    if not os.path.exists(SAVED_MODEL_DIR): return []
    available_models = []
    for f in os.listdir(SAVED_MODEL_DIR):
        if f.endswith("_yield_model.keras"):
            crop_name = f.replace("_yield_model.keras", "")
            available_models.append(crop_name)
    return sorted(available_models)

def get_unit_for_crop(crop_name):
    unit_map = {'Coconut': 'Nuts', 'Cotton(lint)': 'Bales', 'Mesta': 'Bales'}
    return unit_map.get(crop_name, 'Tonnes')

def get_price_unit_for_crop(crop_name):
    unit = get_unit_for_crop(crop_name)
    return "Tonne" if unit == "Tonnes" else unit[:-1]

def get_hardcoded_crop_price(crop_name):
    price_map = {
        'Arecanut': 450000, 'Coconut': 30, 'Coriander': 80000, 'Dry chillies': 250000, 'Garlic': 175000,
        'Turmeric': 120000, 'Banana': 25000, 'Onion': 30000, 'Sugarcane': 3200, 'Sweet potato': 25000,
        'Sannhamp': 4500, 'Brinjal': 20000, 'Citrus Fruit': 45000, 'Grapes': 80000, 'Mango': 65000,
        'Papaya': 15000, 'Pome Fruit': 120000, 'Tomato': 40000, 'Cotton(lint)': 12000, 'Mesta': 5000,
        'Cashewnut Raw': 110000, 'Tobacco': 180000, 'Black pepper': 550000, 'Ginger': 150000,
        'Tapioca': 35000, 'Cardamom': 1500000, 'Dry Ginger': 250000, 'Potato': 22000, 'Peas & beans (Pulses)': 75000
    }
    return price_map.get(crop_name, 0)

def load_model_and_scalers(crop_name):
    model_path = MODEL_SAVE_PATH_TEMPLATE.format(crop_name=crop_name)
    scalers_path = SCALERS_SAVE_PATH_TEMPLATE.format(crop_name=crop_name)
    if not os.path.exists(model_path) or not os.path.exists(scalers_path): return None, None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'rmse'])
        return model, joblib.load(scalers_path)
    except Exception: return None, None

def _run_prediction_logic(model, scalers, lat, lon, aoi_geom, year):
    try:
        weather_data = fetch_monthly_nasa_power_data(lat, lon, year, GROWING_SEASON_START_MONTH, GROWING_SEASON_END_MONTH, NASA_POWER_PARAMS_LIST)
        year_for_gee = min(year, pd.Timestamp.now().year - 1) if year >= pd.Timestamp.now().year else year
        satellite_data = get_monthly_gee_satellite_features(aoi_geom, year_for_gee, GROWING_SEASON_START_MONTH, GROWING_SEASON_END_MONTH)
        mvi_series_list, vi_scalar_list, mw_series_list = [], [], []
        for vi in VI_FEATURE_SUPERSET:
            mvi_series_list.append(satellite_data.get(f'{vi}_monthly_series', [0.0]*NUM_MONTHLY_TIMESTEPS))
            vi_scalar_list.extend([satellite_data.get(f'{vi}_peak', 0.0), satellite_data.get(f'{vi}_auc', 0.0)])
        weather_feature_count = 0
        for p_code, p_label in NASA_POWER_PARAMS_DICT.items():
            mw_series_list.append(weather_data.get(f'{p_label}_mean_monthly', [0.0]*NUM_MONTHLY_TIMESTEPS))
            weather_feature_count += 1
            if 'PREC' in p_code.upper() or 'SOLAR' in p_code.upper() or 'ALLSKY' in p_code.upper():
                mw_series_list.append(weather_data.get(f'{p_label}_sum_monthly', [0.0]*NUM_MONTHLY_TIMESTEPS))
                weather_feature_count += 1
        X_mvi_s = np.nan_to_num(np.array(mvi_series_list).T).reshape(1, NUM_MONTHLY_TIMESTEPS, len(VI_FEATURE_SUPERSET))
        X_vi_sc = np.nan_to_num(np.array(vi_scalar_list)).reshape(1, -1)
        X_mw_s = np.nan_to_num(np.array(mw_series_list).T).reshape(1, NUM_MONTHLY_TIMESTEPS, weather_feature_count)
        X_mvi_s_scaled = scalers['mvi_s'].transform(X_mvi_s.reshape(-1, X_mvi_s.shape[-1])).reshape(X_mvi_s.shape)
        X_vi_sc_scaled = scalers['vi_sc'].transform(X_vi_sc)
        X_mw_s_scaled = scalers['mw_s'].transform(X_mw_s.reshape(-1, X_mw_s.shape[-1])).reshape(X_mw_s.shape)
        pred_scaled = model.predict([X_mvi_s_scaled, X_vi_sc_scaled, X_mw_s_scaled], verbose=0)
        return scalers['y'].inverse_transform(pred_scaled)[0][0]
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None

def predict_yield_for_district(crop_name, district_name, year):
    model, scalers = load_model_and_scalers(crop_name)
    if not model or not scalers: return None, None
    district_geom, district_centroid = get_district_geometry_and_centroid(district_name)
    if not district_geom or not district_centroid: return None, None
    pred = _run_prediction_logic(model, scalers, district_centroid['latitude'], district_centroid['longitude'], district_geom, year)
    return pred, district_centroid

def get_field_and_district_ndvi(field_geom, district_geom, year):
    def get_avg_ndvi(geom, year_for_ndvi):
        try:
            use_s2 = year_for_ndvi >= SENTINEL2_START_YEAR
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED' if use_s2 else MODIS_VI_COLLECTION).filterDate(f'{year_for_ndvi}-06-01', f'{year_for_ndvi}-11-30').filterBounds(geom)
            if use_s2:
                ndvi = collection.map(mask_s2_clouds_gee).map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')).select('NDVI')
            else:
                ndvi = collection.select(MODIS_NDVI_BAND).map(lambda img: img.multiply(MODIS_VI_SCALE_FACTOR))
            return ndvi.mean().reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=100, bestEffort=True, maxPixels=1e9).get('NDVI').getInfo()
        except Exception: return 0
    year_for_ndvi = min(year, pd.Timestamp.now().year - 1)
    return get_avg_ndvi(field_geom, year_for_ndvi), get_avg_ndvi(district_geom, year_for_ndvi)

def predict_yield_for_field(crop_name, district_name, year, field_geom):
    district_yield, _ = predict_yield_for_district(crop_name, district_name, year)
    if district_yield is None: return None
    district_geom, _ = get_district_geometry_and_centroid(district_name)
    if not district_geom: return None
    field_ndvi, district_ndvi = get_field_and_district_ndvi(field_geom, district_geom, year)
    calibrated_yield = district_yield * (field_ndvi / district_ndvi) if district_ndvi > 0.1 and field_ndvi > 0.1 else district_yield
    return {"district_yield_per_hectare": district_yield, "calibrated_yield_per_hectare": calibrated_yield, "field_ndvi": field_ndvi, "district_ndvi": district_ndvi}