import random
import pickle

GRIDSIZE_ROW = 100
GRIDSIZE_COL = 100


# artificial data generation
modellist = [['Railway', 300,'R'],
                ['Other', 60,'O'],
             ['Highway', 120,'H']]


def generate_route_bresenham(start: tuple, end: tuple):
    # Determine the number of intermediate points based on the Euclidean distance between start and end
    distance = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
    num_points = max(0, int((distance - 10) / 10))

    min_x = min(start[0], end[0])
    max_x = max(start[0], end[0])
    min_y = min(start[1], end[1])
    max_y = max(start[1], end[1])
    points = [(random.randint(min_x, max_x), random.randint(min_y, max_y)) for _ in range(num_points)]

    # Sort the points based on their Euclidean distance to the start point
    points.sort(key=lambda point: ((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2) ** 0.5)

    points = [start] + points + [end]
    # Generate path using Bresenham's line algorithm
    route = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                route.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                route.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        route.append((x, y))
    return route


def generate_records(size: int, storepath:str) -> list:
    records = []
    existing_points = {mode: [] for mode in [model[0] for model in modellist]}
    for i in range(size):
        randomseed = random.randint(0, 2)
        travelmode = modellist[randomseed][0]
        speed = modellist[randomseed][1]
        name = modellist[randomseed][2] + str(i)

        while True:
            start = (random.randint(0, GRIDSIZE_ROW), random.randint(0, GRIDSIZE_COL))
            end = (random.randint(0, GRIDSIZE_ROW), random.randint(0, GRIDSIZE_COL))
            if all(abs(start[0] - point[0]) + abs(start[1] - point[1]) > 3 and abs(end[0] - point[0]) + abs(end[1] - point[1]) > 3 for point in existing_points[travelmode]):
                break
        existing_points[travelmode].append(start)
        existing_points[travelmode].append(end)

        route = generate_route_bresenham(start, end)
        records.append({'name': name, 'id': i, 'speed': speed, 'route': route, 'travelmode': travelmode})

    with open(storepath, 'wb') as f:
        pickle.dump(records, f)

    return records


# artificial_records = generate_records(20, 'data/artificial_records.pkl')
with open('data/artificial_records.pkl', 'rb') as f:
    artificial_records = pickle.load(f)


