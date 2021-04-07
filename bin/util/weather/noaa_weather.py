# Copyright: This script is revised from https://github.com/tsbyq/NOAA_weather_downloader.git

import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import gzip

from ftplib import FTP
import geocoder
from ish_parser import ish_report, ish_reportException


DIR_THIS_SCRIPT = os.path.dirname(os.path.realpath(__file__))


# Function to download raw
def ftp_to_raw_entry_list(file_name_noaa, file_name_local, ftp):
    from ftplib import FTP
    '''This function download raw weather data entries from noaa
    '''
    # Retrieve binary gzip file from NOAA
    try:
        file = open(file_name_local, 'wb')
        ftp.retrbinary('RETR '+ file_name_noaa, file.write)
        file.close()
    except:
        os.remove(file_name_local)
        print(f"Failed to download the file: {file_name_noaa}")
        return

    # Read binary gzip file into a list
    v_noaa_raw_entries = []
    gzip_file = gzip.open(file_name_local, 'rb')

    for i, line in enumerate(gzip_file):
        temp_bytes = line[0:-1]
        temp_str = str(temp_bytes)[2:-1]
        v_noaa_raw_entries.append(temp_str)

    return v_noaa_raw_entries

# Functions to get data from parsed ish report.
get_datetime_from_rpt = lambda x: ish_report().loads(x).datetime
get_T_C_from_rpt = lambda x: ish_report().loads(x).air_temperature.get_numeric()
get_T_F_from_rpt = lambda x: ish_report().loads(x).air_temperature.get_fahrenheit().get_numeric()

def get_vars(rpt):
    x = ish_report().loads(rpt)
    ls = x.formatted().split('\n')[1:-1]
    keys = [ele.split(': ')[0] for ele in ls]
    values = [ele.split(': ')[1] for ele in ls]
    i_p = keys.index('Precipitation')
    i_cc = keys.index('Cloud Coverage')
    i_cc_2 = keys.index('Cloud Summation')
    values[i_p] = ls[i_p].split('Precipitation: ')[1]
    values[i_cc] = ls[i_cc].split('Cloud Coverage: ')[1]
    values[i_cc_2] = ls[i_cc_2].split('Cloud Summation: ')[1]
    dict_vals = dict(zip(keys, values))
    return dict_vals

def download_noaa_weather_element(station_ID, year, ftp_instance):
    ftp_path = '/pub/data/noaa/' + str(year)
    file_name_noaa = station_ID + '-' + str(year) + '.gz'
    raw_file_out_dir = ''
    file_name_local = os.path.join(raw_file_out_dir, station_ID + '-' + str(year) + '.gz')
    ftp_instance.cwd(ftp_path)
    v_noaa_raw_elements = ftp_to_raw_entry_list(file_name_noaa, file_name_local, ftp_instance)
    v_noaa_datetime = list(map(get_datetime_from_rpt, v_noaa_raw_elements))
    v_noaa_T_C = list(map(get_T_C_from_rpt, v_noaa_raw_elements))
    df_out = pd.DataFrame(OrderedDict({
        'Datetime': v_noaa_datetime,
        'Temperature': pd.to_numeric(v_noaa_T_C, errors='coerce'),
    }))
    # Clean the raw zip file
    os.remove(file_name_local)
    return(df_out)

def download_noaa_weather_element_detailed(station_ID, year, ftp_instance):
    ftp_path = '/pub/data/noaa/' + str(year)
    file_name_noaa = station_ID + '-' + str(year) + '.gz'
    raw_file_out_dir = ''
    file_name_local = os.path.join(raw_file_out_dir, station_ID + '-' + str(year) + '.gz')
    ftp_instance.cwd(ftp_path)
    v_noaa_raw_elements = ftp_to_raw_entry_list(file_name_noaa, file_name_local, ftp_instance)
    arr_noaa_raw_elements = pd.Series(v_noaa_raw_elements)
    df_out = pd.DataFrame(list(arr_noaa_raw_elements.apply(get_vars)))
    v_noaa_datetime = list(map(get_datetime_from_rpt, v_noaa_raw_elements))
    v_noaa_T_C = list(map(get_T_C_from_rpt, v_noaa_raw_elements))
    df_out['DateTime'] = v_noaa_datetime
    df_out['Temperature'] = v_noaa_T_C
    df_out = df_out.drop(['Weather Station', 'Elevation', 'Time', 'Air Temperature'], axis=1)
    # Clean the raw zip file
    os.remove(file_name_local)
    return(df_out)

# Function to wrap all
def download_noaa_weather(station_list_csv_path, years=[2019], work_dir='./'):
    """
    { item_description }
    """


    from ftplib import FTP
    try:
        df_stations = pd.read_csv(station_list_csv_path)
    except:
        print(f"Fail to read station list from {station_list_csv_path}")
        return

    # Log in to NOAA FTP
    ftp=FTP('ftp.ncdc.noaa.gov')
    ftp.login()
    print('NOAA FTP login succeeded.')

    n_total_stations = len(df_stations['StationID'].index)

    for year in years:
        # Create the year sub-folder if not exists.
        year_sub_dir = os.path.join(work_dir, str(year))
        v_missing = []
        if not os.path.exists(year_sub_dir):
            os.mkdir(year_sub_dir)
        for i, station_ID in enumerate(df_stations['StationID']):
            try:
                print(f"Downloading {year} weather data for station: {station_ID} -- ({i}/{n_total_stations}) -- {round(i/n_total_stations*100, 2)}%")
                df_out = download_noaa_weather_element(station_ID, year, ftp)
                df_out.to_csv(os.path.join(year_sub_dir, str(year) + "_" + station_ID + ".csv"), index = False)
            except:
                print(f"Failed to download weather data for station: {station_ID}")
                v_missing.append(station_ID)
        df_missing = pd.DataFrame({'Missing Station ID': v_missing})
        df_missing.to_csv(str(year) + "_" + 'missing.csv', index=False)

    ftp.quit()
    print('Weather download finished. Logout FTP.')


def add_weather_to_ts(df_ts, noaa_station_id, dir_all_weather=None, new_cols=None, ftp_instance=None):
    '''
    This function add weather data to a dataframe of time-series data. The weather data is from the specified NOAA weather station.
    If the directory of pre-downloaded weather files is not provided, it will download the weather data from NOAA on the fly.
    '''
    if dir_all_weather==None:
        print('Will download from NOAA')
        if ftp_instance == None:
            from ftplib import FTP
            ftp_instance=FTP('ftp.ncdc.noaa.gov')
            ftp_instance.login()
            print('NOAA FTP login succeeded.')
        for i, year in enumerate(set(pd.to_datetime(df_ts['Datetime']).dt.year)):
            df_weather_temp = download_noaa_weather_element(noaa_station_id, year, ftp_instance)
            df_weather_temp['Datetime'] = pd.to_datetime(df_weather_temp['Datetime'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)
            if i == 0:
                df_weather = df_weather_temp
            else:
                df_weather = df_weather.append(df_weather_temp)
        ftp_instance.quit()
        print('---> NOAA FTP Logout succeeded.')
    else:
        print('Use pre-downloaded weather data')
        # Get the raw weather data
        for i, year in enumerate(set(pd.to_datetime(df_ts['Datetime']).dt.year)):
            dir_weather_csv = os.path.join(dir_all_weather, str(year), f"{year}_{noaa_station_id}.csv")
            df_weather_temp = pd.read_csv(dir_weather_csv, parse_dates=True)
            df_weather_temp['Datetime'] = pd.to_datetime(df_weather_temp['Datetime'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)
            if i == 0:
                df_weather = df_weather_temp
            else:
                df_weather = df_weather.append(df_weather_temp)

    # Merge the weather data to the time-series data
    df_ts_weather = pd.merge_asof(df_ts, df_weather.dropna().sort_values('Datetime'), on='Datetime', tolerance=pd.Timedelta('3600s'))
    df_ts_weather = df_ts_weather.interpolate(method='linear', limit_direction='both', limit=6, order=2)  # Interpolate to fill missing values
    return df_ts_weather if new_cols==None else df_ts_weather[new_cols]


################################################################################

# Some utility functions
################################################################################
def geocode_address(input_address):
    """
    Geocode an address to (lat, lon)
    """
    try:
        latlng = geocoder.arcgis(input_address).latlng
        return latlng
    except:
        pass

    try:
        latlng = geocoder.osm(input_address).latlng
        return latlng
    except:
        pass

    try:
        latlng = geocoder.ottawa(input_address).latlng
        return latlng
    except:
        pass

    try:
        latlng = [float(s) for s in geocoder.yandex(input_address).latlng]
        return latlng
    except:
        pass

def haversine_distance(lat1, lon1, lat2, lon2):
    # Get radians from decimals
    r_lat1, r_lon1, r_lat2, r_lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate the distance between the two locations
    temp = np.sin((r_lat2 - r_lat1) / 2) ** 2 + np.cos(r_lat1) * np.cos(r_lat2) * np.sin((r_lon2 - r_lon1) / 2) ** 2
    distance = 2 * 6371 * np.arcsin(np.sqrt(temp))
    return (distance)

def find_closest_weather_station(tuple_lat_lon,
    df_weather_station_list=pd.read_csv(os.path.join(DIR_THIS_SCRIPT, 'valid_stations_20200303.csv'))
    ):
    """
    Finds a closest weather station.
    Return a list of weather station IDs and names ordered by the distance to the location.

    :param      tuple_lat_lon:            The tuple (lat, lon)
    :type       tuple_lat_lon:            { type_description }
    :param      df_weather_station_list:  The df weather station list
    :type       df_weather_station_list:  { pandas dataframe }
    """
    latitude, longitude = float(tuple_lat_lon[0]), float(tuple_lat_lon[1])

    v_coord = np.asarray(df_weather_station_list[['LAT', 'LON']].values)
    # Find the closest and second closest weather station (backup if the closest doesn't work)
    v_distance = [haversine_distance(latitude, longitude, coord[0], coord[1]) for coord in v_coord]
    closest_index = np.argmin(v_distance)
    second_closest_index = np.argpartition(v_distance, 2)[2]
    third_closest_index = np.argpartition(v_distance, 3)[3]

    closest_weather_station_ID = df_weather_station_list.loc[closest_index, 'StationID']
    closest_weather_station_name = df_weather_station_list.loc[closest_index, 'STATION NAME']
    second_closest_weather_station_ID = df_weather_station_list.loc[second_closest_index, 'StationID']
    second_closest_weather_station_name = df_weather_station_list.loc[second_closest_index, 'STATION NAME']
    third_closest_weather_station_ID = df_weather_station_list.loc[third_closest_index, 'StationID']
    third_closest_weather_station_name = df_weather_station_list.loc[third_closest_index, 'STATION NAME']

    station_IDs = [closest_weather_station_ID, second_closest_weather_station_ID, third_closest_weather_station_ID]
    station_names = [closest_weather_station_name, second_closest_weather_station_name, third_closest_weather_station_name]

    return station_IDs, station_names

def parse_cloud(x): 
    cloud_cover_str = x.split(', ')[0][-2:]
    try:
        cloud_cover = int(cloud_cover_str)
    except:
        cloud_cover = np.nan
    
    return cloud_cover

def download_weather(station_ID, year):
    ftp=FTP('ftp.ncdc.noaa.gov')
    ftp.login()
    weather_data = download_noaa_weather_element_detailed(station_ID, year, ftp)
    ftp.quit()
    # Index by time
    weather_data.index = pd.to_datetime(weather_data['DateTime'])
    # Parse cloud cover
    if 'Cloud Coverage' in weather_data.columns:
        weather_data['CloudCoverage'] = weather_data['Cloud Coverage'].apply(parse_cloud)
        weather_data = weather_data[['Temperature', 'CloudCoverage']]
    else:
        weather_data = weather_data[['Temperature']]
        print('Cloud coverness data is not available in weather station {}'.format(station_ID))

    return weather_data

if __name__ == '__main__':
    address = 'Berkeley, CA'
    year = 2018
    station_ids, station_names = find_closest_weather_station(geocode_address(address))
    weather = download_weather(station_ids[2], year)
    # print(weather)

