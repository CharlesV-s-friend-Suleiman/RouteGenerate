# this 1 file is to plot the map and route:
plotNetwork.py: A Python script to plot network data

# these 3 files below are used in sequence:
articificialMapGenrate.py: A Python script to generate artificial-grid map with multiple travel modes
trajGenerate.py: 2 methods to generate trajectories on the generated map: with route and without route
featureExtrac.py: A Python script to extract features (velocity, acceleration, cosine , etc.) from generated-traj data

# this 1 file use the real-world data to train a classifier and predict the travel mode of the generated traj data:
generateMethodValidRF.py: To validate the generated method using Random Forest classifier

#### 2. Data ####
artificial_network.pkl:  A m*n grid-map, each grid[x][y] contains the permitted travel mode of the grid
artificial_traj.csv: A list of trajectories, each trajectory is a list of points, each point is a tuple of (x, y, t, travel mode)
artificial_traj_feature.csv & trainRealData.csv: A list of features extracted from the trajectories used as train data & test data