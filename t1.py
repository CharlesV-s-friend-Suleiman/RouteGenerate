import pickle

with open('data/GridModesAdjacentRes.pkl', 'rb') as f:
    gridModesAdjacentRes = pickle.load(f)

print(gridModesAdjacentRes[(285,105)])

print(gridModesAdjacentRes[(286,104)])