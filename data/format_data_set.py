import glob
import os
from datetime import datetime
import zipfile
import pickle
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap

import utils
from config_trAISformer import Config

# This script expects at least one AIS download zip in the train, test, and valid, folders
# When changing the data set, delete everything from these folders except the new zip files
# After running a csv will be extracted from each zip and a pickel file will be generated per folder under the parent

bm = Basemap()   # default: projection='cyl'
cf = Config()

MINIMUM_HRS_BETWEEN_POINTS = 2
MINIMUM_TRACK_LENGTH = 20
MAXIMUM_SAMPLING_RATE_MINUTES = 10
MINIMUM_SPEED = 0

path = 'ais_downloads'
folders = ['train', 'test', 'valid']

def over_water(lat, lon):
    boolean_series = pd.Series()
    for i, value in lat.items():
        boolean_series.at[i] = (bm.is_land(lat[i], lon[i]) is False)

    return boolean_series

for folder in folders:
    for file in os.listdir(path + '/' + folder):
        if file.endswith('.zip') and not os.path.isfile(path + '/' + folder + '/' + file.replace('.zip', '.csv')):
            print('Unzipping ' + path + '/' + folder + '/' + file)
            with zipfile.ZipFile(path + '/' + folder + '/' + file, 'r') as zip_ref:
                zip_ref.extractall(path + '/' + folder)

    print('Reading ' + path + '/' + folder + '/*.csv')
    df = pd.concat(map(pd.read_csv, glob.glob(path + '/' + folder + '/*.csv')))

    print('Applying filters')
    df = df[
         (df['LAT'] >= cf.lat_min) &  # Within bounding box
         (df['LAT'] <= cf.lat_max) &
         (df['LON'] >= cf.lon_min) &
         (df['LON'] <= cf.lon_max) &
         (df['SOG'] > MINIMUM_SPEED) & # Moving
        (df['SOG'] <= cf.sog_size)  # Realistic speed
    ]

    df = df[['MMSI', 'LAT', 'LON', 'SOG', 'COG', 'VesselType', 'BaseDateTime']]  # Feature selection
    df['BaseDateTime'] = df['BaseDateTime'].apply(lambda date_string:datetime.fromisoformat(date_string).timestamp()) # Transform datetime
    grouped_df = df.sort_values(['BaseDateTime'],ascending=True).groupby('MMSI')  # group by vessel and sort by time

    data_set = []
    traj = []
    DATE_TIME_INDEX_IN_DF = 5  # [lat, lon, sog, cog, vesseltype, basedatetime]
    for group_name, df_group in grouped_df:
        if len(traj) >= MINIMUM_TRACK_LENGTH:
            data_set.append({'mmsi': group_name, 'traj': np.array(traj)})

        traj = []  # start new trajectory
        for row in df_group.iterrows():

            # Filter out points over land
            if bm.is_land(row[1]['LAT'], row[1]['LON']):
                print("Filtering: " + row[1]['LAT'] + ", " + row[1]['LON'])
                continue

            # Normalize
            row_values = [
                utils.nomalize_value(row[1]['LAT'], cf.lat_min, cf.lat_max),
                utils.nomalize_value(row[1]['LON'], cf.lon_min, cf.lon_max),
                utils.nomalize_value(row[1]['SOG'], 0, cf.sog_size),
                utils.nomalize_value(row[1]['COG'], 0, cf.cog_size),
                utils.nomalize_value(row[1]['VesselType'], 0, cf.type_size),
                row[1]['BaseDateTime'],
                row[1]['MMSI']
            ]

            if traj:  # not empty
                if (row[1]['BaseDateTime'] - traj[-1][DATE_TIME_INDEX_IN_DF])/(60) >= MAXIMUM_SAMPLING_RATE_MINUTES:  # Enforce sampling rate
                    if (row[1]['BaseDateTime'] - traj[-1][DATE_TIME_INDEX_IN_DF])/(60*60) <= MINIMUM_HRS_BETWEEN_POINTS:  # Split trajectories on gaps
                        traj.append(row_values)
                    else:
                        if len(traj) >= MINIMUM_TRACK_LENGTH:
                            data_set.append({'mmsi': group_name, 'traj': np.array(traj)})

                        traj = []  # start new trajectory
            else:  # empty
                traj.append(row_values)

    print('Creating ' + path + '/' + folder + '.pkl')
    with open(path + '/' + folder + '.pkl', 'wb') as handle:
        pickle.dump(data_set, handle)