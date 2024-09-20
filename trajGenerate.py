import random
import pickle
import pandas as pd

GRIDSIZE_ROW = 100
GRIDSIZE_COL = 100


class GridMap():
    """
    GridMap class contains the information of grid-map of a given area.
    each gridMap includes a type of travel mode (e.g. expressway, railway, highway, etc.)
    """
    def __init__(self, row: int, col: int,  infos: list):
        self.row = row
        self.col = col

        self.route_infomap = [[set() for j in range(self.col)] for i in range(self.row)]
        for info in infos:
            for cell in info['route']:
                if info['id'] not in self.route_infomap[cell[0]][cell[1]]:
                    self.route_infomap[cell[0]][cell[1]].add(info['id'])

        self.mode_infomap = [[set() for j in range(self.col)] for i in range(self.row)] # 0: unknown, 1: highway, 2: GSD, 3: TG
        for info in infos:
            for cell in info['route']:
                if info['name'].startswith('H'):
                    self.mode_infomap[cell[0]][cell[1]].add(1)
                elif info['name'].startswith('O'):
                    self.mode_infomap[cell[0]][cell[1]].add(2)
                else:
                    self.mode_infomap[cell[0]][cell[1]].add(3)

    def is_close(self, x:int, y:int, mode:int)->int:
        """
        check if the given cell is close to the given mode,0: not close, 1: close
        'close' means the cell and its 8 neighbors have the same mode,if the cell is on the edge of the map, the neighbors are less than 8
        """
        neighbors = [
            (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
            (x, y - 1), (x, y + 1),
            (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
        ]

        for nx, ny in neighbors:
            if 0 <= nx < self.row and 0 <= ny < self.col:
                if mode in self.mode_infomap[nx][ny]:
                    return 1
        return 0


def _bias(method: str, para1,para2):
    """
    bias function
    :param method: 'uniform' or 'normal'
    :param para1:  if method is 'uniform', para1 is the lower bound; if method is 'normal', para1 is the mean
    :param para2:  if method is 'uniform', para2 is the upper bound; if method is 'normal', para2 is the standard deviation
    :return: return bias value
    """
    if method == 'uniform':
        return random.uniform(para1, para2)
    if method == 'normal':
        return random.normalvariate(para1, para2)

def _change_route(prob:float,current_grid_xy:tuple, current_route):  # TODO UNFINISHED
    """
    change the route according to the probability
    :param prob: the probability of changing route
    :param current_grid: the current grid
    :param current_route: the current route
    :return: the new route
    """
    if random.random() < prob:
        return current_route
    else:
        return current_route

def _allow(neighbor: int, mode: str) -> bool:
    """
    check if the neighbor is allowed to travel of the given mode: static, TG, GG, GSD, TS
    :param neighbor: int, element of a list of 9 elements, the 4-th element is the grid itself, the other 8 elements are the neighbors
    :param mode: str, 'TG', 'GG', 'GSD', 'static'
    :return: bool, True if the neighbor is allowed to travel of the given mode, False otherwise
    """
    if mode == 'TG' or mode == 'static':
        return neighbor>>1 & 1 == 1
    elif mode == 'GG':
        return neighbor>>3 & 1 == 1
    elif mode =='GSD':
        return neighbor>>2 & 1 == 1 or neighbor>>5 & 1 == 1
    elif mode == 'TS':
        return neighbor>>6 & 1 == 1 or neighbor>>1 & 1 == 1
    elif mode == 'static':
        return True

def _allow_change(current_gird: int, mode: str) -> bool:
    """
    check if the mode can be changed at the current grid
    :param current_grid: bit-int, the current grid information
    :param mode: str, 'TG-GSD', 'GSD-GG', 'TS-TG'
    :return: bool, True if the mode can be changed at the current grid, False otherwise
    """
    if mode == 'TG-GSD' or mode == 'GSD-TG':
        return current_gird & 1 == 1 and (current_gird>>2 & 1 == 1 or current_gird>>5 & 1 == 1)
    elif mode == 'GSD-GG' or mode == 'GG-GSD':
        return current_gird >> 4 & 1 == 1 and (current_gird>>2 & 1 == 1 or current_gird>>5 & 1 == 1)
    elif mode == 'TS-TG' or mode == 'TG-TS':
        return current_gird & 1 == 1 and (current_gird>>1 & 1 == 1 and current_gird>>6 & 1 == 1)
    else :
        return False

def _get_new_coordinates(x, y, position)->tuple:
    offsets = [
        (-1, 1), (0, 1), (1, 1),
        (-1, 0), (0, 0), (1, 0),
        (-1, -1), (0, -1), (1, -1)
    ]

    if 0 <= position < len(offsets):
        dx, dy = offsets[position]
        return (x + dx, y + dy)
    else:
        raise ValueError("Invalid position")

def generate_traj(gridmap:GridMap ,from_map_records: list, size: int,time_interval=10) -> pd.DataFrame:
    """
    Generate a cell-trajectory from a given map
    """
    df = pd.DataFrame(columns = ["ID", "time", "locx", "locy", "route_id"])
    # the trajectory information includes route_id which can indicate travel mode

    for i in range(size):
        traj_len = random.randint(5, 15)
        traj_id = str(i + 1)
        pre_x,pre_y = -1,-1
        pre_loc_idx = -1
        start_route = random.choice(from_map_records)

        for j in range(traj_len):
            if j == 0:
                time = 0
                locx, locy = start_route['route'][0]
                loc_idx = 0
                route_id = start_route['id']

            else:
                time = j * time_interval
                speed = start_route['speed']
                position_interval = int(speed * time_interval / 60)
                rts_tuple_list = start_route['route']

                loc_idx = pre_loc_idx

                while position_interval > 0 and loc_idx < len(rts_tuple_list) - 1:
                    current_point = rts_tuple_list[loc_idx]
                    next_point = rts_tuple_list[loc_idx + 1]
                    distance = ((current_point[0] - next_point[0]) ** 2 + (
                                current_point[1] - next_point[1]) ** 2) ** 0.5
                    position_interval -= distance
                    loc_idx += 1
                locx,locy = rts_tuple_list[loc_idx]
                route_id = start_route['id']

            pre_x, pre_y, pre_loc_idx = locx, locy, loc_idx
            newpoint = pd.DataFrame([[traj_id,
                                      time,
                                      locx+int(_bias('normal',-1.5,1.5)),
                                      locy+int(_bias('normal',-1.5,1.5)),
                                      route_id]], columns=["ID", "time", "locx", "locy", "route_id"])
            df = df._append(newpoint)

    return df

def generate_traj_single(real_map, size:int, mode:str, time_interval = 6) -> pd.DataFrame:
    """
    Generate a cell-trajectory from a  real-world given map
    1. choice the travel mode and start grid
    2. dfs to generate the full travel trajectory
    3. sample the simulated-trajectory with time_interval from the full trajectory
    4. return the simulated-trajectory as pd.DataFrame and real-world routes as pd.DataFrame

    :param real_map: from GIS, structur: { grid[x,y]: [binary * 9] # the neighbor information}]}
    :param size: the size of the dataset
    :param mode: the travel mode, 'TG', 'GG', 'GSD', 'TS'
    :param time_interval: init==10 min
    :return: pandas.DataFrame with columns = ["traj_id", "time", "locx", "locy", "travel_mode"]
    """
    df = pd.DataFrame(columns = ["ID", "time", "locx", "locy", "mode"])
    # the trajectory information includes route_id which can indicate travel mode

    i = 0
    while i < size: # for each trajectory
        traj_id = mode + str(i + 1)
        if mode == "TG": # highspeed rail, must start from station
            start , _ = random.choice([(k,v) for k,v in real_map.items() if v[4] & 1 == 1])
        elif mode == "GG": #highway, must start from highway zoll
            start , _ = random.choice([(k,v) for k,v in real_map.items() if (v[4]) >>4 & 1 == 1 ])
        elif mode == 'GSD': # normal road
            start , _ = random.choice([(k,v) for k,v in real_map.items() if (v[4]) >>2 & 1 == 1 or (v[4]) >> 5 & 1 == 1])
        elif mode == 'TS': # normal railway, must start from station
            start , _ = random.choice([(k,v) for k,v in real_map.items() if (v[4]) & 1 == 1])

        # generate the total-trajectory timestamp record by record using dfs, the traj includes sequence of (x,y) with len==100
        mode_rtslen_dict = {'TG': 300, 'GG': 200, 'GSD': 200, 'TS': 200}
        expected_sample_len = mode_rtslen_dict[mode] # can be modified if necessary, this is dfs depth
        cnt = 0
        hashmap = set() # hashmap of (x,y)
        route_list = [] # list of (x,y) in order

        current_pos = start
        while cnt < expected_sample_len:
            current_x, current_y = current_pos[0], current_pos[1]
            route_list.append(current_pos)
            hashmap.add(current_pos)
            cnt += 1

            random_shuffle = []
            for idx, neighbor in enumerate(real_map[current_pos]):
                if _get_new_coordinates(current_x, current_y, idx) not in hashmap and _allow(neighbor ,mode):
                    random_shuffle.append(idx)
            if not random_shuffle: # no neighbor available, break to next trajectory
                break
            delta = random.choice(random_shuffle)
            current_pos = _get_new_coordinates(current_x, current_y, delta)

        # sample the simulated-trajectory with time_interval from the full trajectory
        mode_v_dict = {'TG': 300, 'GG': 120, 'GSD': 60, 'TS': 150}
        if mode in mode_v_dict:
            v = mode_v_dict[mode]
        else: v = 0 # station/static
        mode_len_dict = {'TG': 3, 'GG': 4, 'GSD': 5, 'TS': 4}
        mode_vbias_dict = {'TG': 50, 'GG': 20, 'GSD': 20, 'TS': 30}

        j = 0
        timestamp = 0
        traj_len = 0

        # with the hypothesis that the route length in each grid is 1 km, this can be proved by real-data distribution
        sub_df = pd.DataFrame(columns=["ID", "time", "locx", "locy", "mode"])
        while j < len(route_list):
            timestep = time_interval + _bias('normal',-1,1)
            timestamp += timestep
            spacestep = int(v + _bias('uniform', - mode_vbias_dict[mode], mode_vbias_dict[mode])) * timestep / 60
            traj_len += 1
            j = int(j+ spacestep)
            locx, locy = route_list[min(j, len(route_list)-1)][0], route_list[min(j, len(route_list)-1)][1]

            newpoint = pd.DataFrame([[traj_id,
                                      timestamp,
                                      locx + int(_bias('uniform',-1.5,1.5)),
                                      locy + int(_bias('uniform',-1.5,1.5)),
                                      mode]], columns=["ID", "time", "locx", "locy", "mode"])
            sub_df = sub_df._append(newpoint)

        # traj with enough len will be saved
        if traj_len >= mode_len_dict[mode]:
            df = df._append(sub_df)
            i += 1
        # else delete traj generated in this iteration
        else:
            del sub_df

    return df

def generate_traj_mixed(real_map, size:int, mode:str, time_interval=6)-> pd.DataFrame:
    """
    Generate mixed travel mode traj from a real-world given map,
    TG-GSD: the traj will start from TG and change to GSD if there exist stations near the grid
    GSD-TG: REVERSE
    GSD-GG: the traj will start from GSD and change to GG if there exist zolls near the grid
    GG-GSD: REVERSE
    TS-TG: the traj will start from TS and change to TG if there exist stations near the grid
    TG-TS: REVERSE


    near the station
    :param real_map: from GIS, structure: { grid[x,y]: [bit * 9] # the neighbor information}]}, bit zoll,GSD,GG,TG,station
    :param size:
    :param time_interval:
    :param mode: 'TG-GSD', 'GSD-GG', 'TS-TG'
    :return :
    """
    df = pd.DataFrame(columns = ["ID", "time", "locx", "locy", "mode"])
    # the trajectory information includes route_id which can indicate travel mode

    i = 0

    while i < size: # for each trajectory
        traj_id = mode + str(i + 1)
        if mode.startswith('TG'): # start from TG station
            start , _ = random.choice([(k,v) for k,v in real_map.items() if (v[4] & 1 == 1 and v[4] >>1 & 1 == 1)])
        elif mode.startswith("GSD"): # start from GSD
            start , _ = random.choice([(k,v) for k,v in real_map.items() if (v[4] >> 2 & 1 == 1 and v[4] >>5 & 1 == 1)])
        elif mode.startswith("GG"): # start from GG zoll
            start , _ = random.choice([(k,v) for k,v in real_map.items() if (v[4] >> 3 & 1 == 1 and v[4] >>4 & 1 == 1)])
        elif mode.startswith("TS"): # start from TS station
            start , _ = random.choice([(k,v) for k,v in real_map.items() if (v[4] & 1 == 1 and v[4] >>6 & 1 == 1)])

        # generate the total-trajectory timestamp record by record using dfs, the traj includes sequence of (x,y) with len==100
        expected_sample_len = 200
        cnt = 0
        hashmap = set() # hashmap of (x,y)
        route_list = [] # list of (x,y) in order

        current_pos = start
        changed = False # if the mode has been changed
        next_mode = mode.split('-')[1]
        former_mode = mode.split('-')[0]
        current_mode = mode.split('-')[0]
        change_position  = (-1, -1) # the position where the mode will be changed, if not changed, the position is (-1,-1)

        # to get a total route with expected_sample_len, including 1 mode change
        while cnt < expected_sample_len:
            current_x, current_y = current_pos[0], current_pos[1]
            route_list.append(current_pos)
            hashmap.add(current_pos)
            cnt += 1
            random_shuffle = []

            # if the mode has not been changed
            if not changed and _allow_change(real_map[current_pos][4], mode) and cnt>80 and random.random() < 0.8:
                changed = True
                current_mode = next_mode
                change_position = (current_x, current_y)

            for idx, neighbor in enumerate(real_map[current_pos]):
                # if neighbor is not visited and allowed to travel
                if _get_new_coordinates(current_x, current_y, idx) not in hashmap and _allow(neighbor ,current_mode):
                    random_shuffle.append(idx)
            if not random_shuffle:
                break
            delta = random.choice(random_shuffle)
            current_pos = _get_new_coordinates(current_x, current_y, delta)

        if change_position == (-1,-1): # no mode change, restart
            continue
        change_idx = route_list.index(change_position)

        # sample the simulated-trajectory with time_interval from the full trajectory
        mode_v_dict = {'TG': 300, 'GG': 120, 'GSD': 60, 'TS': 150}
        if current_mode in mode_v_dict:
            current_v = mode_v_dict[current_mode]
        else: current_v = 0 # station/static
        if next_mode in mode_v_dict:
            next_v = mode_v_dict[next_mode]
        else: next_v = 0 # station/static

        mode_vbias_dict = {'TG': 50, 'GG': 20, 'GSD': 20, 'TS': 30}

        j = 0
        timestamp = 0
        traj_len = 0

        # sample traj with time_interval from the full trajectory, former part with current_mode, latter part with next_mode
        sub_df = pd.DataFrame(columns=["ID", "time", "locx", "locy", "mode"])
        while j < change_idx:
            timestep = time_interval + _bias('normal',-1,1)
            timestamp += timestep
            spacestep = int(current_v + _bias('uniform', - mode_vbias_dict[former_mode], mode_vbias_dict[former_mode])) * timestep / 60
            traj_len += 1
            j = int(min(j + spacestep, change_idx))
            locx, locy = route_list[j][0], route_list[j][1]
            newpoint = pd.DataFrame([[traj_id,
                                      timestamp,
                                      locx + int(_bias('uniform',-1.5,1.5)),
                                      locy + int(_bias('uniform',-1.5,1.5)),
                                      former_mode]], columns=["ID", "time", "locx", "locy", "mode"])
            sub_df = sub_df._append(newpoint)
        # the record of station stop, 4 min for TG and 7 min for TS
        timestamp += 4 if former_mode == 'TG' else 7
        sub_df = sub_df._append(pd.DataFrame([[traj_id,
                                      timestamp,
                                      route_list[change_idx][0] + int(_bias('uniform',-1.5,1.5)),
                                      route_list[change_idx][1] + int(_bias('uniform',-1.5,1.5)),
                                      0]], columns=["ID", "time", "locx", "locy", "mode"]))
        # latter part
        while j < len(route_list)-1:
            timestep = time_interval + _bias('normal',-1,1)
            timestamp += timestep
            spacestep = int(next_v + _bias('uniform', - mode_vbias_dict[next_mode], mode_vbias_dict[next_mode])) * timestep / 60
            traj_len += 1
            j = int(min(j + spacestep, len(route_list)-1))
            locx, locy = route_list[j][0], route_list[j][1]
            newpoint = pd.DataFrame([[traj_id,
                                      timestamp,
                                      locx + int(_bias('uniform',-1.5,1.5)),
                                      locy + int(_bias('uniform',-1.5,1.5)),
                                      next_mode]], columns=["ID", "time", "locx", "locy", "mode"])
            sub_df = sub_df._append(newpoint)

        if traj_len >= 6:
            df = df._append(sub_df)
            i += 1
        else:
            del sub_df
    return df


# generate traj with route
#with open('data/artificial_network.pkl', 'rb') as f:
#    artificial_net = pickle.load(f)
#    artificial_map = GridMap(GRIDSIZE_ROW,GRIDSIZE_COL,artificial_net)
#    traj = generate_traj(artificial_map,artificial_net, size=400)
#    traj.to_csv('data/artificial_traj.csv', index=False)

# generate traj with grid
with open('data/GridModesAdjacentRes.pkl', 'rb') as f:
    real_map_data = pickle.load(f)
    traj_GG = generate_traj_single(real_map_data, size=100, mode='GG')
    traj_GSD = generate_traj_single(real_map_data, size=100, mode='GSD')
    traj_TS = generate_traj_single(real_map_data, size=100, mode='TS')
    traj_TG = generate_traj_single(real_map_data, size=100, mode='TG')

    traj_TG_GSD = generate_traj_mixed(real_map_data, size=100, mode='TG-GSD')
    traj_GSD_TG = generate_traj_mixed(real_map_data, size=100, mode='GSD-TG')
    traj_TS_TG = generate_traj_mixed(real_map_data, size=100, mode='TS-TG')
    traj_TG_TS = generate_traj_mixed(real_map_data, size=100, mode='TG-TS')
    traj_GSD_GG = generate_traj_mixed(real_map_data, size=100, mode='GSD-GG')
    traj_GG_GSD = generate_traj_mixed(real_map_data, size=100, mode='GG-GSD')

    traj = pd.concat([traj_GG, traj_GSD, traj_TS, traj_TG, traj_TG_GSD, traj_GSD_TG, traj_TS_TG, traj_TG_TS, traj_GSD_GG, traj_GG_GSD])
    traj.to_csv('data/artificial_traj_mixed.csv', index=False)
