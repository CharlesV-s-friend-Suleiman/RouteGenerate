import pickle
import numpy as np
import pandas as pd
#from artificiaMapGenerate import GRIDSIZE_ROW, GRIDSIZE_COL
#from trajGenerate import GridMap
"""
this file is used to extract features from the generated-data using kinda RF method from 
'A Hybrid Method for Intercity Transport Mode Identification Based on Mobility Features and Sequential Relations Mined from Cellular Signaling Data'
"""

# prepare the test data

#  with route name to get ['mode']
"""raw_df = pd.read_csv('data/artificial_traj.csv')
with open('data/artificial_network.pkl', 'rb') as f:
    artificial_net = pickle.load(f)
artificial_map = GridMap(GRIDSIZE_ROW, GRIDSIZE_COL, artificial_net)"""
# without route name,in real-world data
raw_df = pd.read_csv('data/artificial_traj3.csv')



# calculate durations between two trajectory records using time,if the traj record is the first record of a traj, the duration is 0
raw_df['duration'] = raw_df.groupby('traj_id')['time'].diff().fillna(0)
# calculate the distance between two trajectory records using locx and locy, if the traj record is the first record of a traj, the distance is 0

raw_df['dx'] = raw_df.groupby('traj_id')['locx'].diff().fillna(0)
raw_df['dy'] = raw_df.groupby('traj_id')['locy'].diff().fillna(0)
raw_df['distance'] = np.sqrt(raw_df['dx'] ** 2 + raw_df['dy'] ** 2)

raw_df['speed'] = 60 *raw_df['distance'] / raw_df['duration'].replace(0, 1)
# calculate the acceleration between two trajectory records using speed, if the speed is 0, the acceleration is 0
raw_df['acc'] = raw_df['speed'].diff().fillna(0) / raw_df['duration'].replace(0, 1)
# calculate cosine  between 3 trajectory records using locx and locy,handling the first and last elements by setting their cosine values to 1
raw_df['cosine'] = raw_df.groupby('traj_id').apply(lambda x: (x['locx'].diff().fillna(0) * x['locx'].shift(-1).fillna(0) + x['locy'].diff().fillna(0) * x['locy'].shift(-1).fillna(0)) / (np.sqrt(x['locx'].diff().fillna(0) ** 2 + x['locy'].diff().fillna(0) ** 2) * np.sqrt(x['locx'].shift(-1).fillna(0) ** 2 + x['locy'].shift(-1).fillna(0) ** 2)).replace(0, 1)).reset_index(drop=True)


# method with route name to get ['mode']
"""H = []
O = []
R = []
for each_road in artificial_net:
    if each_road['name'].startswith('H'): H.append(each_road['id'])
    elif each_road['name'].startswith('O'): O.append(each_road['id'])
    else: R.append(each_road['id'])
raw_df['mode'] = raw_df['route_id'].apply(lambda x: 1 if x in H else (2 if x in O else 3))

def isClose(gridmap,x,y,mode):
    return gridmap.is_close(x,y,mode)

raw_df['GG'] = raw_df.apply(lambda x: isClose(artificial_map,int(x['locx']),int(x['locy']),1),axis=1)
raw_df['GSD'] = raw_df.apply(lambda x: isClose(artificial_map,int(x['locx']),int(x['locy']),2),axis=1)
raw_df['TG'] = raw_df.apply(lambda x: isClose(artificial_map,int(x['locx']),int(x['locy']),3),axis=1)
raw_df.to_csv('data/artificial_traj_features_2.csv', index=False)"""

# method without route name:
raw_df['mode'] = raw_df['mode'].apply(lambda x : 1 if x=='GG' else (2 if x== 'GSD' else 3))
raw_df.to_csv('data/realWorldFeature.csv')


