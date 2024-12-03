import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

def read_tsplib_file(filename):
    """
    Read and parse a TSPLIB format file.
    Returns coordinates and city names.
    """
    coordinates = []
    city_names = {}
    reading_coords = False
    reading_display = False
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
                continue
            elif line.startswith('DISPLAY_DATA_SECTION'):
                reading_coords = False
                reading_display = True
                continue
            elif line == 'EOF':
                reading_coords = False
                reading_display = False
                continue
                
            if reading_coords and line:
                # Parse coordinate lines
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        coordinates.append((node_id, x, y))
                    except ValueError:
                        continue
                        
            elif reading_display and line:
                # Parse display data lines
                parts = line.split('"')
                if len(parts) >= 2:
                    try:
                        node_id = int(parts[0].strip())
                        name = parts[1].strip()
                        city_names[node_id] = name
                    except ValueError:
                        continue
                        
    return coordinates, city_names

def calculate_distance(coord1, coord2):
    """
    Calculate Euclidean distance between two coordinates.
    """
    return np.sqrt((coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)

def solve_tsp_basic(coordinates):
    """
    Solve TSP using a simple brute force approach.
    Only suitable for small datasets (< 10 cities).
    Returns the best route and its total distance.
    """
    n = len(coordinates)
    best_distance = float('inf')
    best_route = None
    
    # Try all possible permutations
    for route in permutations(range(n)):
        distance = 0
        # Calculate total distance for this route
        for i in range(n):
            j = (i + 1) % n
            city1 = coordinates[route[i]]
            city2 = coordinates[route[j]]
            distance += calculate_distance(city1, city2)
            
        if distance < best_distance:
            best_distance = distance
            best_route = route
            
    return best_route, best_distance

def plot_route(coordinates, city_names, best_route):
    """
    Plot the cities and the optimal route.
    """
    plt.figure(figsize=(12, 8))
    
    # Extract coordinates for plotting
    xs = [coord[1] for coord in coordinates]
    ys = [coord[2] for coord in coordinates]
    
    # Plot cities
    plt.scatter(ys, xs, c='red', s=100)
    
    # Plot route
    for i in range(len(best_route)):
        j = (i + 1) % len(best_route)
        city1 = coordinates[best_route[i]]
        city2 = coordinates[best_route[j]]
        plt.plot([city2[2], city1[2]], [city2[1], city1[1]], 'b-', alpha=0.7)
    
    # Add city labels
    for i, coord in enumerate(coordinates):
        node_id = coord[0]
        name = city_names.get(node_id, f"City {node_id}")
        plt.annotate(name, (coord[2], coord[1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Calgary Neighborhoods - Optimal Route')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

def main():
    # Save the TSPLIB content to a file
    tsplib_content = """NAME: CALGARY_NEIGHBORHOODS
TYPE: TSP
COMMENT: Calgary neighborhood centers for TSP routing
DIMENSION: 8
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 51.0486 -114.0708
2 51.0540 -114.0953
3 51.0341 -114.0783
4 51.0486 -114.1247
5 51.0197 -114.0047
6 51.0831 -114.1310
7 51.0375 -114.1419
8 51.0647 -114.0890
EOF
DISPLAY_DATA_SECTION
1 "Downtown Core"
2 "Kensington"
3 "Beltline"
4 "Bowness"
5 "Inglewood"
6 "University Heights"
7 "Signal Hill"
8 "Bridgeland"
EOF"""
    
    with open('calgary_neighborhoods.tsp', 'w') as f:
        f.write(tsplib_content)
    
    # Read and solve the TSP
    coordinates, city_names = read_tsplib_file('calgary_neighborhoods.tsp')
    best_route, total_distance = solve_tsp_basic(coordinates)
    
    # Print the results
    print("\nOptimal Route:")
    print("-------------")
    for i, city_idx in enumerate(best_route):
        node_id = coordinates[city_idx][0]
        city_name = city_names.get(node_id, f"City {node_id}")
        print(f"{i+1}. {city_name}")
    
    print(f"\nTotal Distance: {total_distance:.2f} units")
    
    # Plot the route
    plot_route(coordinates, city_names, best_route)

if __name__ == "__main__":
    main()