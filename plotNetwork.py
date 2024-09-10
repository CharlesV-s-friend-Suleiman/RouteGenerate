import matplotlib.pyplot as plt
import pickle

colors = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
    '#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17', '#666666', '#1b9e77', '#d95f02'
]

def plot_routes(routesset:list)->None:
    i=-1
    for r in routesset:
        i+=1
        x_coords, y_coords = zip(*(r['route']))
        plt.plot( [x for x in x_coords],[y for y in y_coords], 'o-', color=colors[i%len(colors)], markersize=0.5, linewidth=0.2, )
        mid_point_index = len(x_coords) // 2
    plt.grid(True)
    plt.size = (100, 100)
    plt.title("Grid Map", loc='center')

    plt.show(dpi=200)
    return None


#plot_routes(realdata)
with open('data/artificial_network.pkl', 'rb') as f:
    artificial_data = pickle.load(f)
plot_routes(artificial_data)