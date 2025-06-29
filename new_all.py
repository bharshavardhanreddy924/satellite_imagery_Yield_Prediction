# Suppress TensorFlow INFO messages (needs to be at_very_top)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ee
# import geemap # Not strictly used for core logic here, but often useful with GEE
import time
import calendar
# import json # Not directly used
# import csv # Not directly used
import joblib
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.integrate import trapezoid # np.trapezoid is generally preferred if available

# --- Configuration & Constants ---
try:
    # !!! IMPORTANT: SET YOUR GEE PROJECT ID HERE !!!
    GEE_PROJECT_ID = 'positive-tempo-456012-d8' # User provided GEE Project ID
    
    if GEE_PROJECT_ID:
        ee.Initialize(project=GEE_PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
        print(f"Earth Engine Initialized with project: {GEE_PROJECT_ID} and high-volume endpoint.")
    else:
        print("Attempting generic GEE initialization (project ID not explicitly set). Consider setting GEE_PROJECT_ID environment variable.")
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com') # Generic initialization with high-volume
        print("Earth Engine Initialized with high-volume endpoint (no specific project).")
except Exception as e_init:
    print(f"Earth Engine not initialized ({e_init}). Attempting authentication...")
    try:
        ee.Authenticate()
        if GEE_PROJECT_ID:
            ee.Initialize(project=GEE_PROJECT_ID, opt_url='https://earthengine-highvolume.googleapis.com')
            print(f"Earth Engine Authenticated and Initialized with project: {GEE_PROJECT_ID} and high-volume endpoint.")
        else:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            print("Earth Engine Authenticated and Initialized with high-volume endpoint (no specific project).")
    except Exception as e_auth:
        print(f"CRITICAL: GEE initialization/authentication failed: {e_auth}. Original error: {e_init}.")
        print("Please ensure you have authenticated GEE and, if needed, set a valid project ID.")
        exit()

CROP_DATA_CSV = "final.csv" # Path to your crop data CSV

# MODIFIED: Use the user-provided directory for saved models
SAVED_MODEL_DIR = r"D:/YieldPrediction/saved_crop_models_v3"

MODEL_SAVE_PATH_TEMPLATE = os.path.join(SAVED_MODEL_DIR, "{crop_name}_yield_model.keras")
SCALERS_SAVE_PATH_TEMPLATE = os.path.join(SAVED_MODEL_DIR, "{crop_name}_scalers.joblib")

KARNATAKA_DISTRICTS_FC_PATH = "FAO/GAUL/2015/level2"
KARNATAKA_STATE_NAME_IN_GEE = 'Karnataka'

NASA_POWER_PARAMS_DICT = {
    "T2M": "Temp_2m_C", "T2M_MAX": "Temp_Max_2m_C", "T2M_MIN": "Temp_Min_2m_C",
    "PRECTOTCORR": "Precip_mm_day", "RH2M": "RelHum_2m_percent", "WS2M": "WindSpeed_2m_mps",
    "ALLSKY_SFC_SW_DWN": "SolarRad_AllSky_WM2",
}
NASA_POWER_PARAMS_LIST = list(NASA_POWER_PARAMS_DICT.keys())

S2_BANDS_FOR_VIS = ['B2', 'B3', 'B4', 'B8']
S2_QA_BAND = 'QA60'
S2_CLOUD_BIT_MASK = 1 << 10
S2_CIRRUS_BIT_MASK = 1 << 11

MODIS_VI_COLLECTION = 'MODIS/006/MOD13A1' #MODIS/061/MOD13A1
MODIS_NDVI_BAND = 'NDVI'
MODIS_EVI_BAND = 'EVI'
MODIS_VI_SCALE_FACTOR = 0.0001

SENTINEL2_START_YEAR = 2016 #Sentinel available from late 2015, practically 2016

GROWING_SEASON_START_MONTH = 6
GROWING_SEASON_END_MONTH = 11 # Inclusive
NUM_MONTHLY_TIMESTEPS = (GROWING_SEASON_END_MONTH - GROWING_SEASON_START_MONTH) + 1

# Define the superset of VIs you might encounter or want to use as features
VI_LIST_S2 = ['NDVI', 'EVI', 'SAVI'] # VIs derivable from Sentinel-2
VI_LIST_MODIS = ['NDVI', 'EVI']      # VIs available from MODIS
VI_FEATURE_SUPERSET = ['NDVI', 'EVI', 'SAVI'] # The full list of VIs the model expects input for (MODIS years will have NaN for SAVI)

MIN_SAMPLES_FOR_TRAINING = 15 # Minimum samples to attempt K-fold CV
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY_SECONDS = 15

# Fallback coordinates provided by the user [lon, lat]
FALLBACK_DISTRICT_COORDS = {
    'Bagalkot': [75.7010, 16.1691], 'Bangalore rural': [77.6832, 13.2854],
    'Belgaum': [74.4977, 15.8497], 'Bellary': [76.9214, 15.1394],
    'Bengaluru urban': [77.5946, 12.9716], 'Bidar': [77.5500, 17.9144],
    'Bijapur': [75.7154, 16.8302], 'Chamarajanagar': [76.9390, 11.9262],
    'Chikballapur': [77.7315, 13.4356], 'Chikmagalur': [75.7804, 13.3161],
    'Chitradurga': [76.4010, 14.2250], 'Dakshin kannad': [75.0337, 12.8698],
    'Davangere': [75.9200, 14.4644], 'Dharwad': [75.0080, 15.4589],
    'Gadag': [75.6200, 15.4315], 'Gulbarga': [76.8343, 17.3297],
    'Hassan': [76.0962, 13.0068], 'Haveri': [75.4000, 14.7949],
    'Kodagu': [75.7339, 12.3375], 'Kolar': [78.1324, 13.1357],
    'Koppal': [76.1562, 15.3442], 'Mandya': [76.8956, 12.5218],
    'Mysore': [76.6394, 12.2958], 'Raichur': [77.3566, 16.2120],
    'Ramanagara': [77.2807, 12.7223], 'Shimoga': [75.5681, 13.9299],
    'Tumkur': [77.1025, 13.3379], 'Udupi': [74.7421, 13.3409],
    'Uttar kannad': [74.4858, 14.7935], 'Vijayanagar': [76.3820, 15.2289], # Assuming this might be a newer district name
    'Yadgir': [77.1315, 16.7689]
}
FALLBACK_DISTRICT_AOI_RADIUS_M = 10000 # 10km radius for fallback AOI

# --- Caches for API calls ---
NASA_POWER_CACHE = {}
GEE_VI_CACHE = {}
DISTRICT_GEOMETRY_CACHE = {}

# --- 1. CSV Parsing ---
def parse_crop_data_from_file(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Crop data CSV not found at {csv_path}")
        return pd.DataFrame()
    try:
        # Try to detect encoding, common ones are utf-8, latin1, cp1252
        encodings_to_try = ['utf-8', 'latin1', 'cp1252']
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(csv_path, quotechar='"', encoding=enc, on_bad_lines='warn')
                print(f"Successfully read CSV with encoding: {enc}")
                break
            except UnicodeDecodeError:
                print(f"Failed to read CSV with encoding: {enc}")
            except Exception as e_read: # Catch other potential read errors
                print(f"Error reading CSV with encoding {enc}: {e_read}")
        
        if df is None:
            print(f"Error: Could not read CSV from path '{csv_path}' with tried encodings.")
            return pd.DataFrame()

        # Standardize column names (example: "Yield (Tonnes/ Hectare)" -> "Yield_TonnesperHectare")
        new_column_names = {}
        for original_col_name in df.columns:
            clean_name = original_col_name.replace(" - ", "_") # "State - District" -> "State_District"
            clean_name = clean_name.replace(" (", "_")     # "Yield (Unit)" -> "Yield_Unit)"
            clean_name = clean_name.replace(")", "")       # "Yield_Unit)" -> "Yield_Unit"
            clean_name = clean_name.replace("/", "per")    # "Tonnes/Hectare" -> "TonnesperHectare"
            clean_name = clean_name.replace(" ", "")       # "Crop Name" -> "CropName"
            new_column_names[original_col_name] = clean_name
        df.rename(columns=new_column_names, inplace=True)

        # Parse Year: Handles "YYYY - YY" format and plain "YYYY"
        def parse_year(year_str):
            if isinstance(year_str, str) and ' - ' in year_str:
                return int(year_str.split(' - ')[0]) # Takes the start year
            try: return int(year_str)
            except (ValueError, TypeError):
                if pd.isna(year_str): return np.nan
                try: return int(float(year_str)) # Handles cases like "2015.0"
                except ValueError: return np.nan # If truly unparseable
        if 'Year' in df.columns:
            df['Year'] = df['Year'].apply(parse_year)
        else:
            print("Warning: 'Year' column not found in CSV. Year-based operations might fail.")
            return pd.DataFrame() # Or handle as appropriate

        # Convert numeric columns to numeric, coercing errors
        for col in df.columns:
            if col not in ['State', 'District', 'Year']: # Assuming these are categorical/identifier
                 # Try to convert, replace non-numeric with NaN, then fill NaN if needed or handle later
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['Year'], inplace=True) # Remove rows where Year could not be parsed
        df['Year'] = df['Year'].astype(int)

        # Clean District and State names (remove leading numbers like "1. Bangalore")
        if 'District' in df.columns:
            df['District'] = df['District'].astype(str).str.replace(r'^\d+\.\s*', '', regex=True).str.strip().fillna('')
        if 'State' in df.columns:
            df['State'] = df['State'].astype(str).str.replace(r'^\d+\.\s*', '', regex=True).str.strip().fillna('')
        
        # Filter for Karnataka if State column exists
        if 'State' in df.columns:
            df = df[df['State'].str.lower() == KARNATAKA_STATE_NAME_IN_GEE.lower()]

        print(f"Successfully parsed CSV. Shape after filtering for '{KARNATAKA_STATE_NAME_IN_GEE}': {df.shape}. Columns example: {df.columns.tolist()[:5]}...")
        return df
    except Exception as e:
        print(f"Error parsing CSV from file '{csv_path}': {type(e).__name__} - {e}")
        return pd.DataFrame()

# --- 2. Helper to get Cleaned Crop Names ---
def get_all_crop_names_cleaned_from_df(df):
    crop_names = set()
    # Iterate through columns to find potential yield columns
    for col in df.columns:
        # Example: "Arecanut_Yield_TonnesperHectare"
        if "_Yield_" in col: # A common pattern for yield columns
            parts = col.split("_Yield_")
            if len(parts) > 0:
                crop_name_candidate = parts[0]
                # Basic check to avoid adding 'State', 'District', 'Year' if they somehow match
                if crop_name_candidate and crop_name_candidate.lower() not in ["state", "district", "year", ""]:
                    crop_names.add(crop_name_candidate)
    
    sorted_crops = sorted(list(crop_names))
    print(f"Identified {len(sorted_crops)} unique crop identifiers (cleaned): {sorted_crops if len(sorted_crops) < 10 else str(sorted_crops[:10])[:-1] + '...' + str(sorted_crops[-1])[1:] }")
    return sorted_crops

# --- 3. Data Acquisition ---
def fetch_monthly_nasa_power_data(lat, lon, year, start_month, end_month, parameters,
                                  retries=API_RETRY_ATTEMPTS, delay=API_RETRY_DELAY_SECONDS):
    start_date_dt = datetime(year, start_month, 1)
    end_day_of_end_month = calendar.monthrange(year, end_month)[1]
    end_date_dt = datetime(year, end_month, end_day_of_end_month)

    params_str = ",".join(parameters)
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    url = (f"{base_url}?parameters={params_str}&community=AG&longitude={lon}&latitude={lat}"
           f"&start={start_date_dt.strftime('%Y%m%d')}&end={end_date_dt.strftime('%Y%m%d')}&format=JSON")

    # Initialize features with NaNs for all months in the season
    monthly_features = {}
    for p_code in parameters:
        p_label = NASA_POWER_PARAMS_DICT.get(p_code, p_code) # Get user-friendly name
        monthly_features[f'{p_label}_mean_monthly'] = [np.nan] * NUM_MONTHLY_TIMESTEPS
        if 'PREC' in p_code.upper() or 'SOLAR' in p_code.upper() or 'ALLSKY' in p_code.upper(): # Sum for precip/solar
            monthly_features[f'{p_label}_sum_monthly'] = [np.nan] * NUM_MONTHLY_TIMESTEPS
            
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=90) # Increased timeout
            resp.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            data_json = resp.json()

            if "properties" not in data_json or "parameter" not in data_json["properties"]:
                # print(f"    NASA POWER: Unexpected JSON structure for ({lat:.2f},{lon:.2f}) year {year}.")
                return monthly_features # Return NaNs

            time_series = data_json["properties"]["parameter"]
            # Check if any of the requested parameters are even present
            if not time_series or not parameters or parameters[0] not in time_series or not time_series[parameters[0]]:
                # print(f"    NASA POWER: No data for requested parameters for ({lat:.2f},{lon:.2f}) year {year}.")
                return monthly_features

            dates_str = list(time_series[parameters[0]].keys()) # Get dates from the first parameter
            
            df_data = []
            for date_str_item in dates_str:
                row = {"Date": datetime.strptime(date_str_item, "%Y%m%d")}
                for pk, pv in time_series.items(): # pk=param_code, pv=dict_of_dates_for_param
                    val = pv.get(date_str_item, np.nan) # Get value for this date for this param
                    row[pk] = float(val) if val != -999 else np.nan # NASA POWER uses -999 for missing
                df_data.append(row)

            daily_df = pd.DataFrame(df_data)
            if daily_df.empty: return monthly_features

            daily_df['Month'] = daily_df['Date'].dt.month
            
            for i, month_num in enumerate(range(start_month, end_month + 1)):
                month_df = daily_df[daily_df['Month'] == month_num]
                if not month_df.empty:
                    for p_code in parameters:
                        p_label = NASA_POWER_PARAMS_DICT.get(p_code, p_code)
                        mean_col_name = f'{p_label}_mean_monthly'
                        sum_col_name = f'{p_label}_sum_monthly'
                        
                        if p_code in month_df.columns and not month_df[p_code].isnull().all():
                            monthly_features[mean_col_name][i] = month_df[p_code].mean()
                            if sum_col_name in monthly_features: # Check if sum feature exists for this param
                                monthly_features[sum_col_name][i] = month_df[p_code].sum()
            return monthly_features # Success
        
        except requests.exceptions.RequestException as e:
            print(f"    NASA POWER Error (Attempt {attempt+1}/{retries}) for ({lat:.2f},{lon:.2f}) yr {year}: {e}")
            if attempt < retries - 1: time.sleep(delay)
            # else: print(f"    Max retries for NASA POWER API for ({lat:.2f},{lon:.2f}) yr {year}.")
        except (KeyError, TypeError, ValueError) as e_proc: # Errors during JSON parsing or data processing
            print(f"    Error processing NASA POWER response for yr {year} ({lat:.2f},{lon:.2f}): {type(e_proc).__name__} - {e_proc}.")
            # This might indicate bad data from API not caught by initial checks, return NaNs
            break # Don't retry if data format is the issue
        except Exception as e_gen: # Catch any other unexpected errors
            print(f"    General error in NASA POWER fetch for yr {year} ({lat:.2f},{lon:.2f}): {type(e_gen).__name__} - {e_gen}")
            if attempt < retries - 1: time.sleep(delay) # Retry for general errors
    return monthly_features # Return NaNs if all retries fail or non-retryable error

def mask_s2_clouds_gee(image):
    qa = image.select(S2_QA_BAND)
    cloud_mask = qa.bitwiseAnd(S2_CLOUD_BIT_MASK).eq(0)
    cirrus_mask = qa.bitwiseAnd(S2_CIRRUS_BIT_MASK).eq(0)
    return image.updateMask(cloud_mask).updateMask(cirrus_mask).select(S2_BANDS_FOR_VIS).divide(10000.0)

def calculate_s2_vis_gee(image):
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # EVI = 2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)) --- L is 1 for S2
    evi_expr = '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1.0))'
    evi = image.expression(
        evi_expr,
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}
    ).rename('EVI')
    
    # SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L) --- L is 0.5 for S2 typically
    savi_expr = '((NIR - RED) / (NIR + RED + 0.5)) * (1.5)'
    savi = image.expression(
        savi_expr,
        {'NIR': image.select('B8'), 'RED': image.select('B4')}
    ).rename('SAVI')
    
    return image.addBands([ndvi, evi, savi])

def get_monthly_gee_satellite_features(aoi_geometry, year, start_month, end_month):
    use_modis = year < SENTINEL2_START_YEAR
    vi_list_for_sensor = VI_LIST_MODIS if use_modis else VI_LIST_S2
    
    # Initialize features for all VIs in the superset
    monthly_vi_features = {f'{vi}_monthly_series': [np.nan] * NUM_MONTHLY_TIMESTEPS for vi in VI_FEATURE_SUPERSET}
    for vi_super in VI_FEATURE_SUPERSET:
        monthly_vi_features[f'{vi_super}_peak'] = np.nan
        monthly_vi_features[f'{vi_super}_auc'] = np.nan

    sensor_name = "MODIS" if use_modis else "Sentinel-2"
    # print(f"      Fetching VIs using {sensor_name} for year {year} ({', '.join(vi_list_for_sensor)}). AOI type: {aoi_geometry.type().getInfo() if aoi_geometry else 'N/A'}")
    current_month_num_for_error_reporting = "N/A" # For more specific error messages

    try:
        for i, month_num in enumerate(range(start_month, end_month + 1)):
            current_month_num_for_error_reporting = month_num
            month_start_str = f"{year}-{str(month_num).zfill(2)}-01"
            month_end_day = calendar.monthrange(year, month_num)[1]
            month_end_str = f"{year}-{str(month_num).zfill(2)}-{str(month_end_day).zfill(2)}"
            
            raw_image_collection = None
            processed_image_collection = None

            # print(f"        Processing {sensor_name} for {year}-{str(month_num).zfill(2)}...")
            if use_modis:
                raw_image_collection = ee.ImageCollection(MODIS_VI_COLLECTION)\
                                   .filterBounds(aoi_geometry)\
                                   .filterDate(ee.Date(month_start_str), ee.Date(month_end_str).advance(1, 'day'))
                
                num_raw_images = raw_image_collection.size().getInfo()
                # print(f"          Found {num_raw_images} raw MODIS images.")
                
                def scale_modis_vis(image): # MODIS VIs need scaling
                    return image.multiply(MODIS_VI_SCALE_FACTOR).copyProperties(image, ['system:time_start'])

                # Select and rename bands to match S2 VI names for consistency if possible
                # MODIS/006/MOD13A1 band names are 'NDVI', 'EVI'
                processed_image_collection = raw_image_collection.select([MODIS_NDVI_BAND, MODIS_EVI_BAND], vi_list_for_sensor).map(scale_modis_vis)

            else: # Use Sentinel-2
                raw_image_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
                                   .filterBounds(aoi_geometry)\
                                   .filterDate(ee.Date(month_start_str), ee.Date(month_end_str).advance(1, 'day'))
                
                num_raw_images = raw_image_collection.size().getInfo()
                # print(f"          Found {num_raw_images} raw Sentinel-2 images (before cloud mask).")
                
                processed_image_collection = raw_image_collection.map(mask_s2_clouds_gee).map(calculate_s2_vis_gee).select(vi_list_for_sensor)

            # Now check size of *processed* collection if it's not empty from raw
            if num_raw_images > 0 and processed_image_collection.size().getInfo() > 0:
                mean_monthly_image = processed_image_collection.mean() # Mosaic and take mean
                
                # Reduce region to get mean VI values for the AOI
                vi_means_dict = mean_monthly_image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi_geometry,
                    scale=500, # MODIS is 500m, S2 is finer but upsampling for consistency or downsampling S2.
                    maxPixels=1e12, # Increased
                    bestEffort=True
                ).getInfo()

                # Store the VI values for the VIs available from THIS sensor
                for vi_name_from_sensor in vi_list_for_sensor: # e.g. ['NDVI', 'EVI'] for MODIS
                    if vi_means_dict and vi_means_dict.get(vi_name_from_sensor) is not None:
                        monthly_vi_features[f'{vi_name_from_sensor}_monthly_series'][i] = vi_means_dict.get(vi_name_from_sensor)
            # else:
                # print(f"          No usable {sensor_name} images after processing for {year}-{str(month_num).zfill(2)} in AOI.")
        
        # After collecting all monthly series, calculate peak and AUC for each VI in the SUPERSET
        for vi_super in VI_FEATURE_SUPERSET:
            series = np.array(monthly_vi_features[f'{vi_super}_monthly_series'])
            valid_points_indices = ~np.isnan(series) # Find where data is not NaN
            valid_points = series[valid_points_indices]
            
            if len(valid_points) > 0:
                monthly_vi_features[f'{vi_super}_peak'] = np.nanmax(series) # Max of the (potentially gappy) series
                # For AUC, only integrate over the valid (non-NaN) parts if more than one point
                if len(valid_points) > 1:
                    # Use np.trapz, which handles unevenly spaced data if x-coordinates were provided.
                    # Here, we assume monthly timesteps are effectively unit-spaced for simplicity.
                    monthly_vi_features[f'{vi_super}_auc'] = np.trapz(valid_points)
                elif len(valid_points) == 1: # If only one valid point, AUC is just that point
                    monthly_vi_features[f'{vi_super}_auc'] = valid_points[0] 
                # If no valid points, AUC remains NaN
        
        return monthly_vi_features

    except ee.EEException as e:
        print(f"    GEE Error (get_monthly_gee_satellite_features) for {year}-{str(current_month_num_for_error_reporting).zfill(2)}: {e}. AOI Type: {aoi_geometry.type().getInfo() if aoi_geometry else 'N/A'}")
    except Exception as e_gen: # Catch any other unexpected errors during GEE processing
        print(f"    General Error (get_monthly_gee_satellite_features) processing {year}-{current_month_num_for_error_reporting}: {type(e_gen).__name__} - {e_gen}")

    # If any error occurs, return the initialized (mostly NaN) feature dict
    return monthly_vi_features

# --- 4. District Mapping & AOI ---
def get_district_geometry_and_centroid(district_name_str, state_name_str=KARNATAKA_STATE_NAME_IN_GEE):
    # Clean the district name: remove "NUMBER. " prefix if present, trim whitespace
    clean_district_name = district_name_str.split('. ')[-1].strip() if '. ' in district_name_str else district_name_str.strip()
    
    cache_key = (clean_district_name, state_name_str)
    if cache_key in DISTRICT_GEOMETRY_CACHE:
        return DISTRICT_GEOMETRY_CACHE[cache_key]

    try:
        # GEE Feature Collection for districts
        fc = ee.FeatureCollection(KARNATAKA_DISTRICTS_FC_PATH)\
               .filter(ee.Filter.eq('ADM1_NAME', state_name_str)) # Filter by state (e.g., 'Karnataka')

        # Attempt to find the district by its cleaned name
        district_feature = fc.filter(ee.Filter.eq('ADM2_NAME', clean_district_name)).first()

        # If not found, try title case (e.g., "Bagalkot" -> "Bagalkot", "bangalore rural" -> "Bangalore Rural")
        if not district_feature.getInfo() or not district_feature.getInfo().get('geometry'):
            title_case_name = clean_district_name.title()
            if title_case_name != clean_district_name: # Only retry if title case is different
                # print(f"    Retrying GEE lookup for '{clean_district_name}' as '{title_case_name}'...")
                district_feature = fc.filter(ee.Filter.eq('ADM2_NAME', title_case_name)).first()
        
        feature_info = district_feature.getInfo() # GetInfo once
        if feature_info and feature_info.get('geometry'): # Check if feature and its geometry exist
            geometry = district_feature.geometry()
            try:
                # Simplify geometry before centroid calculation for complex shapes if needed, then get centroid
                centroid_coords = geometry.centroid(maxError=10).coordinates().getInfo() # Relaxed maxError for centroid
            except ee.EEException: # If centroid fails, try simplifying geometry first
                # print(f"    Centroid calculation failed for {clean_district_name}, simplifying geometry...")
                centroid_coords = geometry.simplify(maxError=1000).centroid(maxError=100).coordinates().getInfo()

            result = (geometry, {'longitude': centroid_coords[0], 'latitude': centroid_coords[1]})
            DISTRICT_GEOMETRY_CACHE[cache_key] = result
            return result
        else:
            # print(f"    District '{clean_district_name}' (or title-cased) not found via GEE in {state_name_str}. Attempting fallback coordinates.")
            if clean_district_name in FALLBACK_DISTRICT_COORDS:
                coords = FALLBACK_DISTRICT_COORDS[clean_district_name]
                lon, lat = coords[0], coords[1]
                # print(f"      Using fallback coordinates for {clean_district_name}: Lon={lon}, Lat={lat}")
                point_geom = ee.Geometry.Point(lon, lat)
                aoi_geom = point_geom.buffer(FALLBACK_DISTRICT_AOI_RADIUS_M) # Create a circular AOI
                result = (aoi_geom, {'longitude': lon, 'latitude': lat})
                DISTRICT_GEOMETRY_CACHE[cache_key] = result
                return result
            else:
                print(f"    Warning: District '{clean_district_name}' not found in GEE for {state_name_str} and not in fallback coordinate list.")
                DISTRICT_GEOMETRY_CACHE[cache_key] = (None, None)
                return None, None

    except Exception as e:
        print(f"    Error in get_district_geometry_and_centroid for '{district_name_str}': {type(e).__name__} - {e}")
        DISTRICT_GEOMETRY_CACHE[cache_key] = (None, None)
        return None, None

# --- 5. Data Preparation for Training ---
def prepare_training_data_for_crop_all_years(full_crop_df, crop_name_cleaned, max_districts_per_year=None):
    master_data = []
    
    # Construct the expected yield column name
    yield_col_name = f"{crop_name_cleaned}_Yield_Tonne_per_Hectare" # Example, adjust if your unit naming is different
    
    # Fallback: if the exact name isn't found, try to find a yield column for the crop
    if yield_col_name not in full_crop_df.columns:
        potential_yield_cols = [col for col in full_crop_df.columns if col.startswith(crop_name_cleaned + "_Yield_")]
        if not potential_yield_cols:
            print(f"  Error: No yield column found for crop '{crop_name_cleaned}' (expected pattern: {crop_name_cleaned}_Yield_UNIT).")
            return pd.DataFrame()
        yield_col_name = potential_yield_cols[0] # Take the first one found
        print(f"  Info: Using yield column: {yield_col_name} for crop {crop_name_cleaned} (fallback match).")

    # Ensure all necessary base columns are present
    required_base_cols = ['State', 'District', 'Year']
    if not all(col in full_crop_df.columns for col in required_base_cols + [yield_col_name]):
        print(f"  Error: One or more required columns ({required_base_cols + [yield_col_name]}) missing for crop {crop_name_cleaned}.")
        return pd.DataFrame()

    # Select relevant data for the specific crop
    crop_specific_df = full_crop_df[required_base_cols + [yield_col_name]].copy()
    crop_specific_df.dropna(subset=[yield_col_name], inplace=True) # Remove rows with no yield data
    crop_specific_df = crop_specific_df[crop_specific_df[yield_col_name] > 0] # Assuming yield must be positive

    if crop_specific_df.empty:
        print(f"  No valid yield data for crop {crop_name_cleaned} after initial filtering.")
        return pd.DataFrame()

    unique_years_for_crop = sorted(crop_specific_df['Year'].unique())
    total_processed_samples = 0

    for year_loop in unique_years_for_crop:
        year_filtered_df = crop_specific_df[crop_specific_df['Year'] == year_loop]
        unique_districts_in_year = year_filtered_df['District'].unique()
        
        districts_to_process_this_year = unique_districts_in_year
        if max_districts_per_year and len(unique_districts_in_year) > max_districts_per_year:
            # Potentially sample, or just take the first N for speed if testing
            districts_to_process_this_year = unique_districts_in_year[:max_districts_per_year]
            # print(f"    Limiting to {max_districts_per_year} districts for {year_loop} (testing).")

        processed_in_year_count = 0
        for i_dist, district_name_loop in enumerate(districts_to_process_this_year):
            if not district_name_loop or pd.isna(district_name_loop): continue # Skip if district name is missing

            # print(f"    Processing: {crop_name_cleaned} - Year: {year_loop} - District: {district_name_loop} ({i_dist+1}/{len(districts_to_process_this_year)})")

            # --- Get AOI and Centroid (with caching) ---
            cache_key_geom = (district_name_loop, KARNATAKA_STATE_NAME_IN_GEE) # Assuming always Karnataka for this project context
            if cache_key_geom not in DISTRICT_GEOMETRY_CACHE:
                # print(f"      District geom not in cache for {district_name_loop}, fetching...")
                geom, centroid = get_district_geometry_and_centroid(district_name_loop, KARNATAKA_STATE_NAME_IN_GEE)
                DISTRICT_GEOMETRY_CACHE[cache_key_geom] = (geom, centroid)
            else:
                geom, centroid = DISTRICT_GEOMETRY_CACHE[cache_key_geom]

            if not geom or not centroid:
                # print(f"      Skipping {district_name_loop} in {year_loop}: No geometry or centroid found.")
                continue
            
            # --- Get Yield Value ---
            district_year_data_row = year_filtered_df[year_filtered_df['District'] == district_name_loop]
            if district_year_data_row.empty: continue # Should not happen if logic is correct
            yield_value = district_year_data_row[yield_col_name].iloc[0]
            
            # --- Fetch Weather Data (with caching) ---
            weather_cache_key = (district_name_loop, year_loop, "weather") # More specific cache key
            if weather_cache_key in NASA_POWER_CACHE:
                monthly_weather_features = NASA_POWER_CACHE[weather_cache_key]
                # print(f"      Weather data for {district_name_loop}, {year_loop} loaded from cache.")
            else:
                # print(f"      Fetching NASA POWER for {district_name_loop}, {year_loop}...")
                monthly_weather_features = fetch_monthly_nasa_power_data(
                    centroid['latitude'], centroid['longitude'], year_loop,
                    GROWING_SEASON_START_MONTH, GROWING_SEASON_END_MONTH, NASA_POWER_PARAMS_LIST
                )
                # Cache if data is reasonably valid (e.g., not all NaNs for primary params)
                if monthly_weather_features and not all(all(np.isnan(v_list) for v_list in val_list) if isinstance(val_list, list) else np.isnan(val_list) for val_list in monthly_weather_features.values()):
                     NASA_POWER_CACHE[weather_cache_key] = monthly_weather_features
                time.sleep(0.05) # Small delay to be kind to APIs, was 0.1
            
            # --- Fetch Satellite VI Data (with caching) ---
            vi_cache_key = (district_name_loop, year_loop, "vi") # More specific cache key
            if vi_cache_key in GEE_VI_CACHE:
                monthly_vi_features = GEE_VI_CACHE[vi_cache_key]
                # print(f"      Satellite VI data for {district_name_loop}, {year_loop} loaded from cache.")
            else:
                # print(f"      Fetching GEE VI features for {district_name_loop}, {year_loop}...")
                monthly_vi_features = get_monthly_gee_satellite_features(
                    geom, year_loop, GROWING_SEASON_START_MONTH, GROWING_SEASON_END_MONTH
                )
                # Cache if data is reasonably valid (check a primary VI series)
                primary_vi_check_key = f"{(VI_LIST_MODIS[0] if year_loop < SENTINEL2_START_YEAR else VI_LIST_S2[0])}_monthly_series"
                if monthly_vi_features and monthly_vi_features.get(primary_vi_check_key) and not np.all(np.isnan(monthly_vi_features[primary_vi_check_key])):
                    GEE_VI_CACHE[vi_cache_key] = monthly_vi_features
                time.sleep(0.05) # Small delay, was 0.1

            # --- Validate fetched data ---
            if not monthly_weather_features or not monthly_vi_features:
                # print(f"      Skipping {district_name_loop}, {year_loop}: Missing weather or VI data.")
                continue

            # Check if essential VI data is present (e.g., at least one VI series for the sensor used)
            # This check ensures that the _monthly_series for the expected VIs are lists of correct length and not all NaN
            check_vi_list_for_sensor = VI_LIST_MODIS if year_loop < SENTINEL2_START_YEAR else VI_LIST_S2
            valid_satellite_input = True
            for vi_to_check in check_vi_list_for_sensor:
                series_key_to_check = f'{vi_to_check}_monthly_series'
                current_series = monthly_vi_features.get(series_key_to_check)
                if not isinstance(current_series, list) or \
                   len(current_series) != NUM_MONTHLY_TIMESTEPS or \
                   np.all(np.isnan(current_series)):
                    # print(f"      Warning: Invalid or all-NaN for critical VI '{series_key_to_check}' for {district_name_loop}, {year_loop}. Sensor: {'MODIS' if year_loop < SENTINEL2_START_YEAR else 'Sentinel-2'}.")
                    if vi_to_check == check_vi_list_for_sensor[0]: # If the *first* expected VI is bad, mark as invalid
                        valid_satellite_input = False
                    # Still fill with NaNs for consistency in structure, but this sample might be problematic
                    monthly_vi_features[series_key_to_check] = [np.nan] * NUM_MONTHLY_TIMESTEPS 
            
            if not valid_satellite_input:
                # print(f"      Skipping {district_name_loop}, {year_loop} due to critical missing satellite VI data after fetch.")
                continue
            
            # Combine all features for this sample
            combined_sample_features = {
                'District': district_name_loop,
                'Year': year_loop,
                'Yield': yield_value,
                **monthly_weather_features,
                **monthly_vi_features # This now includes _monthly_series, _peak, _auc for all in VI_FEATURE_SUPERSET
            }
            master_data.append(combined_sample_features)
            processed_in_year_count += 1
            total_processed_samples +=1
        
        if processed_in_year_count > 0:
            print(f"    Data collected for {processed_in_year_count}/{len(districts_to_process_this_year)} districts for {crop_name_cleaned} in {year_loop}.")

    if total_processed_samples > 0:
        print(f"  Total valid samples collected for {crop_name_cleaned}: {total_processed_samples}")
    else:
        print(f"  No valid samples collected for {crop_name_cleaned} across all years.")

    return pd.DataFrame(master_data)

# --- 6. Model Definition ---
def build_monthly_yield_prediction_model(monthly_vi_series_shape, # (num_timesteps, num_vi_features)
                                         vi_scalar_shape,         # (num_scalar_vi_features,)
                                         monthly_weather_series_shape, # (num_timesteps, num_weather_features)
                                         l2_reg=0.005): # Added L2 regularization

    # Input for monthly VI time series (e.g., NDVI, EVI, SAVI month by month)
    monthly_vi_input = tf.keras.Input(shape=monthly_vi_series_shape, name='monthly_vi_input') 
    # LSTM branch for VIs
    lstm_vi = tf.keras.layers.LSTM(48, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(monthly_vi_input)
    lstm_vi = tf.keras.layers.BatchNormalization()(lstm_vi) # Batch Norm
    lstm_vi = tf.keras.layers.Dropout(0.3)(lstm_vi) # Dropout
    lstm_vi = tf.keras.layers.LSTM(24, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(lstm_vi)
    lstm_vi = tf.keras.layers.BatchNormalization()(lstm_vi)

    # Input for scalar VI features (e.g., peak NDVI, AUC of EVI)
    vi_scalar_input = tf.keras.Input(shape=vi_scalar_shape, name='vi_scalar_input') 
    # Dense branch for scalar VIs
    dense_vi_scalar = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(vi_scalar_input)
    dense_vi_scalar = tf.keras.layers.BatchNormalization()(dense_vi_scalar)

    # Input for monthly weather time series (e.g., Temp, Precip month by month)
    monthly_weather_input = tf.keras.Input(shape=monthly_weather_series_shape, name='monthly_weather_input')
    # LSTM branch for weather
    lstm_weather = tf.keras.layers.LSTM(48, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(monthly_weather_input)
    lstm_weather = tf.keras.layers.BatchNormalization()(lstm_weather)
    lstm_weather = tf.keras.layers.Dropout(0.3)(lstm_weather)
    lstm_weather = tf.keras.layers.LSTM(24, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(lstm_weather)
    lstm_weather = tf.keras.layers.BatchNormalization()(lstm_weather)

    # Merge all processed features
    merged = tf.keras.layers.Concatenate()([lstm_vi, dense_vi_scalar, lstm_weather])

    # Final dense layers for prediction
    final_dense = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(merged)
    final_dense = tf.keras.layers.BatchNormalization()(final_dense)
    final_dense = tf.keras.layers.Dropout(0.4)(final_dense) # Higher dropout before output
    
    output = tf.keras.layers.Dense(1, activation='linear', name='yield_output')(final_dense) # Linear for regression

    model = tf.keras.Model(
        inputs=[monthly_vi_input, vi_scalar_input, monthly_weather_input],
        outputs=output
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adam optimizer
                  loss='mse',  # Mean Squared Error for regression
                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]) # MAE and RMSE
    return model

# --- 7. Training and Evaluation ---
def train_and_evaluate_monthly_model(model_builder_fn, 
                                     X_mvi_s, X_vi_sc, X_mw_s, y, 
                                     n_splits=5, epochs=50, batch_size=4):
    
    if len(y) < 2 : # Need at least 2 samples for any kind of split or training
        print("  Not enough data (<2 samples) for training. Skipping model training.")
        return None, {}, pd.DataFrame()

    # Adjust n_splits if dataset is too small for the requested number of splits
    if len(y) < n_splits :
        n_splits = max(2, len(y)) # Ensure n_splits is at least 2 if len(y) >= 2, or equals len(y)
        print(f"  Warning: Dataset size ({len(y)}) is less than n_splits ({n_splits if n_splits != max(2,len(y)) else 'original ' + str(n_splits)}). Adjusting n_splits to {max(2, len(y))}.")
        if max(2, len(y)) < 2: # This case means len(y) is 0 or 1, handled by the first check.
             print("  Dataset too small for K-Fold. Training on all data without validation split (if possible).")
             # Fallback: Train on all data if it's extremely small (e.g. 1 sample, though this is unusual)
             # This part is more of a safeguard, usually the len(y) < MIN_SAMPLES_FOR_TRAINING check catches this.
             if len(y) == 0: return None, {}, pd.DataFrame() # Cannot train on 0 samples

             # Simple train if only 1 sample (not ideal, but to prevent crash) - this won't validate.
             # For such small data, consider if training is meaningful at all.
             # Scaling (fit_transform on single sample is tricky, usually fit on train, transform on test)
             # Here, we'll just create scalers and "train" briefly. Prediction will be poor.
             scaler_mvi_s = MinMaxScaler().fit(X_mvi_s.reshape(-1, X_mvi_s.shape[-1]))
             X_mvi_s_sc = scaler_mvi_s.transform(X_mvi_s.reshape(-1, X_mvi_s.shape[-1])).reshape(X_mvi_s.shape)
             scaler_vi_sc = MinMaxScaler().fit(X_vi_sc); X_vi_sc_sc = scaler_vi_sc.transform(X_vi_sc)
             scaler_mw_s = StandardScaler().fit(X_mw_s.reshape(-1, X_mw_s.shape[-1])) # StandardScaler often better for weather
             X_mw_s_sc = scaler_mw_s.transform(X_mw_s.reshape(-1, X_mw_s.shape[-1])).reshape(X_mw_s.shape)
             scaler_y = MinMaxScaler().fit(y.reshape(-1,1)); y_sc = scaler_y.transform(y.reshape(-1,1))
             
             model = model_builder_fn(
                (X_mvi_s_sc.shape[1], X_mvi_s_sc.shape[2]), # (timesteps, features)
                (X_vi_sc_sc.shape[1],),                     # (features,)
                (X_mw_s_sc.shape[1], X_mw_s_sc.shape[2])  # (timesteps, features)
             )
             model.fit([X_mvi_s_sc, X_vi_sc_sc, X_mw_s_sc], y_sc, epochs=max(1, epochs//10), batch_size=1, verbose=0) # Minimal training
             all_scalers = {'mvi_s':scaler_mvi_s, 'vi_sc':scaler_vi_sc, 'mw_s':scaler_mw_s, 'y':scaler_y}
             return model, all_scalers, pd.DataFrame([{'fold':1, 'rmse': np.nan, 'mae':np.nan, 'r2':np.nan}]) # No meaningful metrics


    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    trained_models = [] # Store models from each fold

    # --- Pre-scale ALL data using scalers fitted on the ENTIRE dataset ---
    # This is a common practice for K-fold when you intend to train a final model on all data later,
    # or if you want consistent scaling across folds for comparison.
    # Alternatively, fit scalers ONLY on training data within each fold to prevent data leakage into validation.
    # For simplicity here, scaling on all data first.
    # IMPORTANT: For rigorous evaluation, fit scalers *inside* the loop on X_train_fold only.
    # However, since we will retrain a final model on all data, pre-scaling all helps there.
    
    # Scaler for Monthly VI Series (MinMaxScaler is common for VIs which are often bounded 0-1 or -1 to 1)
    # Reshape to 2D for scaler, then back to 3D
    scaler_mvi_s = MinMaxScaler()
    X_mvi_s_sc = scaler_mvi_s.fit_transform(X_mvi_s.reshape(-1, X_mvi_s.shape[-1])).reshape(X_mvi_s.shape)

    # Scaler for VI Scalar features (MinMaxScaler)
    scaler_vi_sc = MinMaxScaler()
    X_vi_sc_sc = scaler_vi_sc.fit_transform(X_vi_sc)

    # Scaler for Monthly Weather Series (StandardScaler for weather which can have wider, varying ranges)
    scaler_mw_s = StandardScaler()
    X_mw_s_sc = scaler_mw_s.fit_transform(X_mw_s.reshape(-1, X_mw_s.shape[-1])).reshape(X_mw_s.shape)
    
    # Scaler for Target (Yield) (MinMaxScaler often good if yield has a known positive range)
    scaler_y = MinMaxScaler()
    y_sc = scaler_y.fit_transform(y.reshape(-1,1)) # Scaler expects 2D

    for fold, (train_idx, val_idx) in enumerate(kf.split(y_sc)): # Split on scaled y
        print(f"\n    --- Fold {fold+1}/{n_splits} ---")
        
        # Split pre-scaled data into training and validation for this fold
        X_tr_mvi_s, X_val_mvi_s = X_mvi_s_sc[train_idx], X_mvi_s_sc[val_idx]
        X_tr_vi_sc, X_val_vi_sc = X_vi_sc_sc[train_idx], X_vi_sc_sc[val_idx]
        X_tr_mw_s, X_val_mw_s = X_mw_s_sc[train_idx], X_mw_s_sc[val_idx]
        y_tr_sc, y_val_sc = y_sc[train_idx], y_sc[val_idx]

        # Build a new model instance for each fold
        model = model_builder_fn(
            (X_tr_mvi_s.shape[1], X_tr_mvi_s.shape[2]), # (timesteps, features_mvi_s)
            (X_tr_vi_sc.shape[1],),                     # (features_vi_sc,)
            (X_tr_mw_s.shape[1], X_tr_mw_s.shape[2])    # (timesteps, features_mw_s)
        )
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6, verbose=0)
        
        history = model.fit(
            [X_tr_mvi_s, X_tr_vi_sc, X_tr_mw_s], y_tr_sc,
            validation_data=([X_val_mvi_s, X_val_vi_sc, X_val_mw_s], y_val_sc),
            epochs=epochs,
            batch_size=batch_size, # Ensure batch size is not larger than smallest dataset split
            callbacks=[early_stopping, reduce_lr],
            verbose=1 # Show training progress
        )
        trained_models.append(model) # Store the trained model for this fold

        # Evaluate on validation set (scaled predictions, then inverse_transform)
        y_pred_scaled_val = model.predict([X_val_mvi_s, X_val_vi_sc, X_val_mw_s], verbose=0)
        y_pred_original_val = scaler_y.inverse_transform(y_pred_scaled_val)
        y_true_original_val = scaler_y.inverse_transform(y_val_sc)

        rmse_val = np.sqrt(mean_squared_error(y_true_original_val, y_pred_original_val))
        mae_val = mean_absolute_error(y_true_original_val, y_pred_original_val)
        r2_val = r2_score(y_true_original_val, y_pred_original_val) if len(y_true_original_val) >= 2 else np.nan
        
        r2_str = f"{r2_val:.3f}" if not np.isnan(r2_val) else "N/A (too few samples)"
        print(f"    Fold {fold+1} Validation Metrics - RMSE: {rmse_val:.3f}, MAE: {mae_val:.3f}, R²: {r2_str}")
        fold_metrics.append({'fold': fold+1, 'rmse': rmse_val, 'mae': mae_val, 'r2': r2_val})

    metrics_df = pd.DataFrame(fold_metrics)
    if not metrics_df.empty:
        print("\n    --- Cross-Validation Metrics Summary ---")
        print(metrics_df)
        # Calculate and print average metrics (excluding 'fold' column)
        print("\n    Average CV Metrics:")
        print(metrics_df.drop(columns=['fold']).mean().rename('Average'))
        
        # Select the best model based on lowest validation RMSE (or other metric)
        # Ensure 'rmse' column exists and is not all NaN before finding idxmin
        if 'rmse' in metrics_df.columns and not metrics_df['rmse'].isnull().all():
            best_fold_idx = metrics_df['rmse'].idxmin()
        else: # Default to the first model if RMSE is not available or all NaN
            best_fold_idx = 0 
    else: # If metrics_df is empty (e.g., training was skipped or failed for all folds)
        best_fold_idx = 0 # Default, though trained_models might also be empty

    # The "final" model can be the one from the best fold, or a new model retrained on all data.
    # Here, we pick the best from CV. For production, retraining on all data is often preferred.
    final_model_from_cv = trained_models[best_fold_idx] if trained_models else None
    
    # Store all scalers (fitted on the entire dataset initially)
    all_fitted_scalers = {
        'mvi_s': scaler_mvi_s, 
        'vi_sc': scaler_vi_sc, 
        'mw_s': scaler_mw_s, 
        'y': scaler_y
    }
    
    return final_model_from_cv, all_fitted_scalers, metrics_df

# --- 8. Prediction Function ---
def predict_yield_monthly_features(model, scalers, 
                                   pred_lat, pred_lon, pred_aoi_geom, # Location for prediction
                                   prediction_year,                  # Target year for prediction
                                   crop_name_for_pred=""):           # For logging

    if model is None or not scalers:
        print(f"  Error: Model or scalers missing for crop {crop_name_for_pred}. Cannot make prediction.")
        return None

    current_sys_year = datetime.now().year
    current_sys_month = datetime.now().month
    
    # Determine the year to use for GEE VI data fetching
    # If predicting for a future year, or current year before/during growing season when VIs aren't complete,
    # use last year's (or older) VIs as a proxy.
    year_for_gee_vi_fetch = prediction_year 
    gee_vi_data_source_info = f"GEE VI data for year: {year_for_gee_vi_fetch}"

    if prediction_year > current_sys_year: # Future prediction
        year_for_gee_vi_fetch = current_sys_year -1 # Use last fully completed year's VIs
        if year_for_gee_vi_fetch < 2000 : year_for_gee_vi_fetch = 2000 # Ensure not too far back for MODIS
        gee_vi_data_source_info = f"GEE VI data for year (proxy for future prediction): {year_for_gee_vi_fetch}"
        # print(f"  Note: Prediction year {prediction_year} is in the future. Using GEE VI data from {year_for_gee_vi_fetch} as proxy.")
    elif prediction_year == current_sys_year and current_sys_month < GROWING_SEASON_END_MONTH :
        # Current year, but growing season might not be complete for VI data
        year_for_gee_vi_fetch = current_sys_year - 1 # Default to last year
        if current_sys_month <= GROWING_SEASON_START_MONTH : # If very early in season, maybe even year before last
            year_for_gee_vi_fetch = max(2000, current_sys_year - 2) # Cap at 2000 for MODIS lower limit
        gee_vi_data_source_info = f"GEE VI data for year (proxy for ongoing season): {year_for_gee_vi_fetch}"
        # print(f"  Note: Prediction year {prediction_year} is current, season ongoing. Using GEE VI data from {year_for_gee_vi_fetch} as proxy.")

    # print(f"\n  --- Predicting Yield for Crop: {crop_name_for_pred} ---")
    # print(f"  Target Location: Lat={pred_lat:.2f}, Lon={pred_lon:.2f}")
    # print(f"  Prediction Year: {prediction_year}")
    # print(f"  {gee_vi_data_source_info}")

    # 1. Fetch NASA POWER weather data for the *prediction_year*
    # print(f"    Fetching Monthly NASA POWER data for prediction year {prediction_year} at ({pred_lat:.2f},{pred_lon:.2f})...")
    weather_data_for_prediction = fetch_monthly_nasa_power_data(
        pred_lat, pred_lon, prediction_year,
        GROWING_SEASON_START_MONTH, GROWING_SEASON_END_MONTH, NASA_POWER_PARAMS_LIST
    )
    time.sleep(0.05) # API kindness

    # Prepare weather series input (X_mw_s)
    user_input_monthly_weather_series_list = []
    for p_code in NASA_POWER_PARAMS_LIST:
        p_friendly_name = NASA_POWER_PARAMS_DICT.get(p_code, p_code)
        user_input_monthly_weather_series_list.append(weather_data_for_prediction.get(f'{p_friendly_name}_mean_monthly', [np.nan]*NUM_MONTHLY_TIMESTEPS))
        if 'PREC' in p_code.upper() or 'SOLAR' in p_code.upper() or 'ALLSKY' in p_code.upper():
            user_input_monthly_weather_series_list.append(weather_data_for_prediction.get(f'{p_friendly_name}_sum_monthly', [np.nan]*NUM_MONTHLY_TIMESTEPS))
    
    X_mw_s_input_user = np.array(user_input_monthly_weather_series_list).T # Transpose to (timesteps, features)
    X_mw_s_input_user = X_mw_s_input_user.reshape(1, NUM_MONTHLY_TIMESTEPS, X_mw_s_input_user.shape[1]) # Reshape to (1, timesteps, features)

    # 2. Fetch GEE satellite VI features for the *year_for_gee_vi_fetch*
    # print(f"    Fetching Monthly GEE VI features for GEE data year {year_for_gee_vi_fetch} using AOI (type: {pred_aoi_geom.type().getInfo() if pred_aoi_geom else 'N/A'})...")
    satellite_vi_data_for_prediction = get_monthly_gee_satellite_features(
        pred_aoi_geom, year_for_gee_vi_fetch, 
        GROWING_SEASON_START_MONTH, GROWING_SEASON_END_MONTH
    )
    time.sleep(0.05)

    # Prepare VI series (X_mvi_s) and VI scalar (X_vi_sc) inputs
    user_input_monthly_vi_series_list = []
    user_input_vi_scalar_list = []
    
    # Determine which VIs would have been available from GEE for year_for_gee_vi_fetch
    vi_list_expected_from_gee = VI_LIST_MODIS if year_for_gee_vi_fetch < SENTINEL2_START_YEAR else VI_LIST_S2
    
    is_critical_vi_data_missing = False # Flag
    if not satellite_vi_data_for_prediction: # If GEE fetch completely failed
        is_critical_vi_data_missing = True
        # print(f"    Warning: GEE VI data fetch failed completely for year {year_for_gee_vi_fetch}. VI features will be zero.")
        # Create empty/NaN structures for all VI features
        satellite_vi_data_for_prediction = {f'{vi}_monthly_series': [np.nan] * NUM_MONTHLY_TIMESTEPS for vi in VI_FEATURE_SUPERSET}
        for vi_super in VI_FEATURE_SUPERSET:
            satellite_vi_data_for_prediction[f'{vi_super}_peak'] = np.nan
            satellite_vi_data_for_prediction[f'{vi_super}_auc'] = np.nan

    # Populate features based on VI_FEATURE_SUPERSET, filling with NaNs if a VI wasn't available (e.g. SAVI in MODIS era)
    temp_vi_series_data_from_this_sensor = {} # Holds actual series data for VIs from this sensor
    temp_vi_scalar_data_from_this_sensor = [] # Holds actual scalar data for VIs from this sensor

    # First, extract what was actually fetched for the sensor used
    for vi_name_from_sensor in vi_list_expected_from_gee:
        series = satellite_vi_data_for_prediction.get(f'{vi_name_from_sensor}_monthly_series', [np.nan]*NUM_MONTHLY_TIMESTEPS)
        # Basic validation for fetched series
        if not isinstance(series, list) or len(series) != NUM_MONTHLY_TIMESTEPS or np.all(np.isnan(series)):
            series = [np.nan] * NUM_MONTHLY_TIMESTEPS # Standardize to NaN list if problematic
            if vi_name_from_sensor == vi_list_expected_from_gee[0]: # If primary VI for sensor is bad
                 is_critical_vi_data_missing = True
                 # print(f"    Warning: Critical VI series '{vi_name_from_sensor}' missing/invalid for GEE year {year_for_gee_vi_fetch}.")
        temp_vi_series_data_from_this_sensor[vi_name_from_sensor] = series
        temp_vi_scalar_data_from_this_sensor.extend([
            satellite_vi_data_for_prediction.get(f'{vi_name_from_sensor}_peak', np.nan),
            satellite_vi_data_for_prediction.get(f'{vi_name_from_sensor}_auc', np.nan)
        ])
    
    if is_critical_vi_data_missing:
        print(f"    Warning: Critical satellite VI data appears missing or invalid for GEE year {year_for_gee_vi_fetch}. Prediction quality may be affected as VI features might be zeroed.")

    # Now, construct the final feature lists based on the VI_FEATURE_SUPERSET order
    for vi_s2_superset_name in VI_FEATURE_SUPERSET: # Iterate in the defined superset order
        # For monthly series:
        user_input_monthly_vi_series_list.append(temp_vi_series_data_from_this_sensor.get(vi_s2_superset_name, [np.nan]*NUM_MONTHLY_TIMESTEPS))
        
        # For scalar features:
        if vi_s2_superset_name in vi_list_expected_from_gee: # If this VI was expected from the sensor for that year
            try:
                idx_in_sensor_specific_list = vi_list_expected_from_gee.index(vi_s2_superset_name) * 2 # Each VI has 2 scalar features (peak, auc)
                user_input_vi_scalar_list.extend(temp_vi_scalar_data_from_this_sensor[idx_in_sensor_specific_list : idx_in_sensor_specific_list+2])
            except ValueError: # Should not happen if logic is correct
                user_input_vi_scalar_list.extend([np.nan, np.nan])
        else: # This VI (e.g. SAVI for MODIS) was not available
            user_input_vi_scalar_list.extend([np.nan, np.nan])


    X_mvi_s_input_user = np.array(user_input_monthly_vi_series_list).T # Transpose to (timesteps, features)
    X_mvi_s_input_user = X_mvi_s_input_user.reshape(1, NUM_MONTHLY_TIMESTEPS, len(VI_FEATURE_SUPERSET)) # Reshape to (1, timesteps, num_vi_superset)
    X_vi_sc_input_user = np.array(user_input_vi_scalar_list).reshape(1, -1) # Reshape to (1, num_scalar_features_superset)

    # 3. Preprocess inputs: Handle NaNs (e.g., convert to 0 or mean) and scale
    # Convert NaNs to 0.0 (or consider imputation if more sophisticated handling is needed)
    X_mvi_s_input_user = np.nan_to_num(X_mvi_s_input_user, nan=0.0)
    X_vi_sc_input_user = np.nan_to_num(X_vi_sc_input_user, nan=0.0)
    X_mw_s_input_user  = np.nan_to_num(X_mw_s_input_user, nan=0.0)

    # Apply scaling using the loaded scalers
    # Reshape series data for scalers (2D), then reshape back
    X_mvi_s_input_user_flat = X_mvi_s_input_user.reshape(-1, X_mvi_s_input_user.shape[-1])
    X_mvi_s_input_user_scaled = scalers['mvi_s'].transform(X_mvi_s_input_user_flat).reshape(X_mvi_s_input_user.shape)
    
    X_vi_sc_input_user_scaled = scalers['vi_sc'].transform(X_vi_sc_input_user)
    
    X_mw_s_input_user_flat = X_mw_s_input_user.reshape(-1, X_mw_s_input_user.shape[-1])
    X_mw_s_input_user_scaled = scalers['mw_s'].transform(X_mw_s_input_user_flat).reshape(X_mw_s_input_user.shape)

    # 4. Make prediction
    # print("    Making prediction with the model...")
    predicted_yield_scaled = model.predict([X_mvi_s_input_user_scaled, X_vi_sc_input_user_scaled, X_mw_s_input_user_scaled], verbose=0)
    
    # 5. Inverse transform the prediction to original scale
    predicted_yield_original = scalers['y'].inverse_transform(predicted_yield_scaled)
    
    return predicted_yield_original[0][0] # Return the single predicted value


# --- 9. Main Execution ---
if __name__ == "__main__":
    # --- MODIFIED: Configuration for Prediction/Testing ---
    TRAIN_MODELS_FLAG = False  # Set to False to skip training and only do prediction
    
    # Prediction settings:
    PREDICTION_MODE = "district"  # Options: "point" or "district"
    
    # For "point" mode:
    USER_LATITUDE_PREDICT = 12.3375  # Example: Kodagu region (often grows black pepper)
    USER_LONGITUDE_PREDICT = 75.7339 # Example: Kodagu region
    USER_AOI_RADIUS_METERS_PREDICT = 5000 # 5km radius AOI for point-based prediction

    # For "district" mode:
    USER_DISTRICT_PREDICT = "Kodagu" # Example district known for pepper, ensure it's in your FALLBACK_DISTRICT_COORDS or GEE
    
    YEAR_TO_PREDICT_FOR = datetime.now().year # Predict for the current year (or set a specific year)
    
    # CROP_TO_PREDICT_FOR: CHOOSE THE CROP YOU WANT TO PREDICT.
    # This MUST match the naming convention of your saved model files.
    CROP_TO_PREDICT_FOR = 'Blackpepper' # <<< SET THIS TO THE DESIRED CROP: BLACKPEPPER

    # Training specific (not used if TRAIN_MODELS_FLAG is False)
    MAX_DISTRICTS_PER_YEAR_FOR_TRAINING = None # None for all, or e.g., 3 for faster testing during training
    EPOCHS_FOR_TRAINING = 50 
    BATCH_SIZE_FOR_TRAINING = 4
    # --- End of Modified Configuration ---

    print("--- Multi-Crop Yield Prediction Model (Monthly Features) ---")
    if not os.path.exists(SAVED_MODEL_DIR):
        # This check is more relevant for training. For prediction, we expect the dir to exist.
        print(f"Warning: SAVED_MODEL_DIR '{SAVED_MODEL_DIR}' does not exist. Models cannot be loaded.")
        # os.makedirs(SAVED_MODEL_DIR) # Not creating if it's for loading
        # print(f"Created directory: {SAVED_MODEL_DIR}")

    print("\nStep 1: Loading and Parsing Crop Data (for crop name list)...")
    full_crop_df = parse_crop_data_from_file(CROP_DATA_CSV)
    if full_crop_df.empty and TRAIN_MODELS_FLAG: # If training, CSV is essential
        print("Exiting: Crop data CSV could not be parsed or is empty. Training cannot proceed.")
        exit()
    elif full_crop_df.empty and not TRAIN_MODELS_FLAG:
        print(f"Warning: Crop data CSV ('{CROP_DATA_CSV}') could not be parsed or is empty. Will rely on hardcoded CROP_TO_PREDICT_FOR.")
        all_cleaned_crop_names = [CROP_TO_PREDICT_FOR] # Use the hardcoded one directly
    else:
        all_cleaned_crop_names = get_all_crop_names_cleaned_from_df(full_crop_df)

    if not all_cleaned_crop_names and TRAIN_MODELS_FLAG:
        print("No crop yield columns identified from CSV. Please check CSV column naming. Training cannot proceed.")
        exit()
        
    if TRAIN_MODELS_FLAG:
        print(f"\nStep 2: Starting Training Loop for {len(all_cleaned_crop_names)} Crops...")
        for crop_idx, crop_name_cleaned_loop in enumerate(all_cleaned_crop_names):
            print(f"\n--- [{crop_idx+1}/{len(all_cleaned_crop_names)}] Processing Crop: {crop_name_cleaned_loop} ---")
            model_path = MODEL_SAVE_PATH_TEMPLATE.format(crop_name=crop_name_cleaned_loop)
            scalers_path = SCALERS_SAVE_PATH_TEMPLATE.format(crop_name=crop_name_cleaned_loop)
            
            # Optional: Skip if already trained (uncomment if needed)
            # if os.path.exists(model_path) and os.path.exists(scalers_path):
            #     print(f"  Model for {crop_name_cleaned_loop} already exists. Skipping training.")
            #     continue

            print(f"  Preparing training data for {crop_name_cleaned_loop} (all available years)...")
            training_data_df = prepare_training_data_for_crop_all_years(full_crop_df, crop_name_cleaned_loop, max_districts_per_year=MAX_DISTRICTS_PER_YEAR_FOR_TRAINING)
            
            if training_data_df.empty or len(training_data_df) < MIN_SAMPLES_FOR_TRAINING:
                print(f"  Insufficient training data ({len(training_data_df)} samples) for {crop_name_cleaned_loop}. Minimum {MIN_SAMPLES_FOR_TRAINING} required. Skipping model training.")
                continue
            
            print("  Step 2a: Engineering Features for Model Input...")
            # --- Monthly VI Series (X_mvi_s) ---
            # Shape: (num_samples, NUM_MONTHLY_TIMESTEPS, len(VI_FEATURE_SUPERSET))
            mvi_s_cols = [f'{vi}_monthly_series' for vi in VI_FEATURE_SUPERSET]
            X_mvi_s_list_for_crop = []
            for _, row in training_data_df.iterrows():
                sample_series_data = []
                for col_name in mvi_s_cols:
                    series = row.get(col_name, [np.nan]*NUM_MONTHLY_TIMESTEPS) # Default to NaNs
                    if not isinstance(series, list) or len(series) != NUM_MONTHLY_TIMESTEPS:
                        series = [np.nan] * NUM_MONTHLY_TIMESTEPS # Ensure correct shape
                    sample_series_data.append(series)
                X_mvi_s_list_for_crop.append(np.array(sample_series_data).T) # Transpose to (timesteps, features)
            X_mvi_s_np = np.array(X_mvi_s_list_for_crop)

            # --- VI Scalar Features (X_vi_sc) ---
            # Shape: (num_samples, len(VI_FEATURE_SUPERSET) * 2) (for peak and auc for each VI)
            vi_sc_cols = [item for vi_super in VI_FEATURE_SUPERSET for item in (f'{vi_super}_peak', f'{vi_super}_auc')]
            X_vi_sc_np = training_data_df[vi_sc_cols].values

            # --- Monthly Weather Series (X_mw_s) ---
            # Shape: (num_samples, NUM_MONTHLY_TIMESTEPS, num_weather_metrics)
            mw_series_cols_ordered = [] # Ensure consistent order of weather features
            for p_code in NASA_POWER_PARAMS_LIST:
                p_name = NASA_POWER_PARAMS_DICT.get(p_code, p_code)
                mw_series_cols_ordered.append(f'{p_name}_mean_monthly')
                if 'PREC' in p_code.upper() or 'SOLAR' in p_code.upper() or 'ALLSKY' in p_code.upper():
                    mw_series_cols_ordered.append(f'{p_name}_sum_monthly')
            
            X_mw_s_list_for_crop = []
            for _, row in training_data_df.iterrows():
                current_sample_mw_series_data = []
                for col_name in mw_series_cols_ordered:
                    series = row.get(col_name, [np.nan]*NUM_MONTHLY_TIMESTEPS)
                    if not isinstance(series, list) or len(series) != NUM_MONTHLY_TIMESTEPS:
                        series = [np.nan] * NUM_MONTHLY_TIMESTEPS
                    current_sample_mw_series_data.append(series)
                if current_sample_mw_series_data: # Should always be true if columns are present
                    X_mw_s_list_for_crop.append(np.array(current_sample_mw_series_data).T) 
            
            if X_mw_s_list_for_crop:
                X_mw_s_np = np.array(X_mw_s_list_for_crop)
            else: # Fallback if something went wrong, create zero array
                X_mw_s_np = np.zeros((len(training_data_df), NUM_MONTHLY_TIMESTEPS, len(mw_series_cols_ordered)))

            # --- Target Variable (y) ---
            y_np = training_data_df['Yield'].values

            # Handle NaNs in features (e.g., convert to 0.0 or mean of column)
            # Model cannot handle NaNs directly.
            X_mvi_s_np = np.nan_to_num(X_mvi_s_np, nan=0.0) # Replace NaNs with 0
            X_vi_sc_np = np.nan_to_num(X_vi_sc_np, nan=0.0)
            X_mw_s_np  = np.nan_to_num(X_mw_s_np, nan=0.0)
            if np.isnan(y_np).any(): # If NaNs in target, fill with mean (or drop rows earlier)
                y_np = np.nan_to_num(y_np, nan=np.nanmean(y_np)) 

            print(f"    Input Shapes for {crop_name_cleaned_loop}: X_mvi_series: {X_mvi_s_np.shape}, X_vi_scalar: {X_vi_sc_np.shape}, X_mw_series: {X_mw_s_np.shape}, y_yield: {y_np.shape}")
            
            print(f"  Step 2b: Training Model for {crop_name_cleaned_loop}...")
            # Determine n_splits for KFold, ensuring it's at least 2 and not more than num_samples
            num_samples_for_crop = len(training_data_df)
            n_kf_splits = max(2, min(5, num_samples_for_crop // max(1,MIN_SAMPLES_FOR_TRAINING // 3) )) # Heuristic for splits
            if num_samples_for_crop < 4 : n_kf_splits = max(2,num_samples_for_crop) # If very few samples, use leave-one-out or 2-fold
            if n_kf_splits > num_samples_for_crop : n_kf_splits = num_samples_for_crop # Cap at num_samples


            final_model_crop, data_scalers_crop, cv_summary_crop = train_and_evaluate_monthly_model(
                build_monthly_yield_prediction_model, 
                X_mvi_s_np, X_vi_sc_np, X_mw_s_np, y_np, 
                n_splits=n_kf_splits, 
                epochs=EPOCHS_FOR_TRAINING, 
                batch_size=BATCH_SIZE_FOR_TRAINING
            )

            if final_model_crop and data_scalers_crop:
                print(f"  Saving model for {crop_name_cleaned_loop} to {model_path}")
                final_model_crop.save(model_path)
                print(f"  Saving scalers for {crop_name_cleaned_loop} to {scalers_path}")
                joblib.dump(data_scalers_crop, scalers_path)
                print(f"  Model and scalers for {crop_name_cleaned_loop} saved successfully.")
            else:
                print(f"  Model training failed or produced no model for {crop_name_cleaned_loop}.")
    else:
        print("\nTRAIN_MODELS_FLAG is False. Skipping training.")

    # --- Prediction Section ---
    print(f"\nStep 3: Prediction Example ---")
    print(f"  Attempting prediction for CROP: '{CROP_TO_PREDICT_FOR}', YEAR: {YEAR_TO_PREDICT_FOR}")

    # Validate if the chosen crop for prediction is among the known/processed ones (if CSV was loaded)
    if CROP_TO_PREDICT_FOR not in all_cleaned_crop_names and not full_crop_df.empty:
        print(f"  Error: Crop '{CROP_TO_PREDICT_FOR}' not recognized from CSV data. Available crops: {all_cleaned_crop_names}")
        print("  Please ensure CROP_TO_PREDICT_FOR matches a crop name derived from your CSV or a saved model file prefix.")
        exit()
    
    model_path_pred = MODEL_SAVE_PATH_TEMPLATE.format(crop_name=CROP_TO_PREDICT_FOR)
    scalers_path_pred = SCALERS_SAVE_PATH_TEMPLATE.format(crop_name=CROP_TO_PREDICT_FOR)
    
    loaded_model_for_pred = None
    loaded_scalers_for_pred = None

    if os.path.exists(model_path_pred) and os.path.exists(scalers_path_pred):
        try:
            print(f"  Loading pre-trained model from: {model_path_pred}")
            loaded_model_for_pred = tf.keras.models.load_model(model_path_pred, compile=False) # Compile=False initially
            # Re-compile if needed (e.g. if it wasn't saved with optimizer state or you want to change it)
            # For prediction only, often not strictly necessary but good practice if metrics were part of compile.
            loaded_model_for_pred.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                                          loss='mse', 
                                          metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
            
            print(f"  Loading scalers from: {scalers_path_pred}")
            loaded_scalers_for_pred = joblib.load(scalers_path_pred)
            print("  Model and scalers for prediction loaded successfully.")
        except Exception as e:
            print(f"  Error loading model or scalers for '{CROP_TO_PREDICT_FOR}': {type(e).__name__} - {e}")
            print("  Ensure the model and scaler files are not corrupted and compatible.")
    else:
        print(f"  Model file ('{model_path_pred}') or scalers file ('{scalers_path_pred}') not found for crop '{CROP_TO_PREDICT_FOR}'.")
        print(f"  Cannot proceed with prediction for this crop. Please check paths and file names in '{SAVED_MODEL_DIR}'.")

    if loaded_model_for_pred and loaded_scalers_for_pred:
        pred_target_lat, pred_target_lon, pred_target_aoi, prediction_location_description = None, None, None, ""
        
        if PREDICTION_MODE == "point":
            pred_target_lat, pred_target_lon = USER_LATITUDE_PREDICT, USER_LONGITUDE_PREDICT
            print(f"  Prediction mode: 'point'. Using Lat: {pred_target_lat}, Lon: {pred_target_lon}")
            # Create a GEE geometry object for the point, buffered to an AOI
            pred_target_aoi = ee.Geometry.Point(pred_target_lon, pred_target_lat).buffer(USER_AOI_RADIUS_METERS_PREDICT)
            prediction_location_description = f"point ({USER_LATITUDE_PREDICT:.2f}, {USER_LONGITUDE_PREDICT:.2f}) with {USER_AOI_RADIUS_METERS_PREDICT}m radius"
        
        elif PREDICTION_MODE == "district":
            print(f"  Prediction mode: 'district'. Using district: {USER_DISTRICT_PREDICT}")
            # Get GEE geometry for the district
            district_geom_pred, district_centroid_pred = get_district_geometry_and_centroid(USER_DISTRICT_PREDICT)
            if district_geom_pred and district_centroid_pred:
                pred_target_lat, pred_target_lon = district_centroid_pred['latitude'], district_centroid_pred['longitude']
                pred_target_aoi = district_geom_pred
                prediction_location_description = f"district '{USER_DISTRICT_PREDICT}' (centroid Lat: {pred_target_lat:.2f}, Lon: {pred_target_lon:.2f})"
                print(f"    Successfully obtained geometry for district '{USER_DISTRICT_PREDICT}'.")
            else:
                print(f"  Error: Could not obtain geometry for district '{USER_DISTRICT_PREDICT}'. Cannot make prediction for this district.")
                pred_target_aoi = None # Ensure it's None if lookup failed
        else:
            print(f"  Error: Unknown PREDICTION_MODE '{PREDICTION_MODE}'. Choose 'point' or 'district'.")

        if pred_target_aoi: # Proceed only if AOI is valid
            print(f"  Calling prediction function for {CROP_TO_PREDICT_FOR} at {prediction_location_description} for year {YEAR_TO_PREDICT_FOR}...")
            predicted_yield = predict_yield_monthly_features(
                loaded_model_for_pred, loaded_scalers_for_pred, 
                pred_target_lat, pred_target_lon, pred_target_aoi, 
                YEAR_TO_PREDICT_FOR, 
                crop_name_for_pred=CROP_TO_PREDICT_FOR
            )
            
            if predicted_yield is not None:
                print(f"\n  >>> FINAL PREDICTED YIELD for {CROP_TO_PREDICT_FOR.upper()} <<<")
                print(f"  Location: {prediction_location_description}")
                print(f"  Year: {YEAR_TO_PREDICT_FOR}")
                print(f"  Predicted Yield: {predicted_yield:.2f} (units as per training data, likely Tonne/Hectare)")
            else:
                print(f"  Prediction failed for {CROP_TO_PREDICT_FOR} at {prediction_location_description} for year {YEAR_TO_PREDICT_FOR}.")
        else:
            if PREDICTION_MODE == "district": # Only print this if district mode failed to get AOI
                 print(f"  Cannot make prediction as AOI for {USER_DISTRICT_PREDICT} was not determined.")
    
    else:
        print(f"  Prediction skipped as model or scalers for '{CROP_TO_PREDICT_FOR}' could not be loaded.")

    print("\n--- Script Finished ---")