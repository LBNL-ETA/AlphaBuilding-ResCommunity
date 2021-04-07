### NOAA Weather Downloader
This is a routine to download weather data from National Oceanic and Atmospheric Administration (NOAA).

``noaa_weather.py`` contains:
* the code to download sub-hourly data from NOAA in batch for a list of weather stations
* an example to find the closest NOAA weather station for a (latitude, longitude) coordinate.

``isd-history.csv`` contains the raw NOAA weather stations data. 

``station_list.csv`` contains the filtered NOAA weather stations that have valid data from 2010 to 2020. User can create their own ``station_list.csv`` to specify stations. 

