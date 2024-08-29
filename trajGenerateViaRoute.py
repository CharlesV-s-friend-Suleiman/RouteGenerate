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
        self.info = infos

        self.infomap = [[set() for j in range(self.col)] for i in range(self.row)]
        for info in infos:
            for cell in info['route']:
                if info['id'] not in self.infomap[cell[0]][cell[1]]:
                    self.infomap[cell[0]][cell[1]].add(info['id'])


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


def generate_traj(gridmap:GridMap ,from_map_records: list, size: int,time_interval=10) -> pd.DataFrame:
    """
    Generate a cell-trajectory from a given map
    """
    df = pd.DataFrame([[-1,-1,-1,-1,-1]])
    # the trajectory information includes route_id which can indicate travel mode
    df.columns = ["traj_id", "time", "locx", "locy", "route_id"]

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
                                      route_id]], columns=["traj_id", "time", "locx", "locy", "route_id"])
            df = df._append(newpoint)

    return df


with open('data/artificial_records.pkl', 'rb') as f:
    artificial_net = pickle.load(f)
    artificial_map = GridMap(GRIDSIZE_ROW,GRIDSIZE_COL,artificial_net)
    traj = generate_traj(artificial_map,artificial_net, size=400)
    traj.to_csv('data/artificial_traj.csv', index=False)
