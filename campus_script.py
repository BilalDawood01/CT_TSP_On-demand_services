# Imports
import osmnx as ox
import networkx as nx
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
import pandas as pd
from geopy.geocoders import Nominatim

# for bus functions
from shapely.geometry import Point
from shapely.ops import unary_union
import random

# COMMENT THIS OUT WHEN DEBUGGING
import warnings
warnings.filterwarnings("ignore")

# traccess 
# r5py
# qgis
# matbox
# python-tsp

# COMBINED FUNCTION IMPLEMENTATIONS - CURRENTLY THE ONLY ONE THAT WORKS

class CalgaryTSP:
    def __init__(self):
        self.locations_bbox = None
        self.geolocator = Nominatim(user_agent="calgary_tsp")
        # Define main transit hubs
        self.transit_hubs = {
            'C-Train Stations': [
                # South
                ("Fish Creek - Lacombe CTrain Station", 50.9233027, -114.0758714),
                ("Somerset-Bridlewood", 50.8986570, -114.0699560),
                
                # North
                ("Crowfoot - North and South", 51.1226635, -114.2070422),
                ("Tuscany Station - Rocky Ridge Terminal", 51.1350249, -114.2348192),
                
                # East-West
                ("Saddletowne CTrain Park and Ride", 51.1254114, -113.9487872),
                ("Sirocco Station", 51.0381335, -114.1695420),
                ("Marlborough Mall", 51.0552984, -113.9786681),
            ],
            'Major Bus Terminals': [
                # North Bus Terminal
                ("North Pointe", 51.1570621, -114.0680757)
            ]
        }


    def initialize_graph(self, locations, verbose=False):
        """Initialize graph based on locations"""
        # Get coordinates for all locations
        coords = []
        for loc in locations:
            location = self.geolocator.geocode(f"{loc}, Calgary, Alberta")
            if location:
                coords.append((location.latitude, location.longitude))
        # Add transit hub coordinates
        for hub_type in self.transit_hubs.values():
            for _, lat, lon in hub_type:
                coords.append((lat, lon))
        
        if not coords:
            raise ValueError("No valid locations found")
        
        # Calculate bounding box with padding
        min_lat = min(c[0] for c in coords)
        max_lat = max(c[0] for c in coords)
        min_lon = min(c[1] for c in coords)
        max_lon = max(c[1] for c in coords)
        
        # Add padding (about 20% of the range)
        lat_padding = (max_lat - min_lat) * 0.2
        lon_padding = (max_lon - min_lon) * 0.2
        self.locations_bbox = [
            min_lat - lat_padding,
            min_lon - lon_padding,
            max_lat + lat_padding,
            max_lon + lon_padding
        ]
        
        # Download graph for the specific area
        self.G = ox.graph_from_bbox(
            self.locations_bbox[0], self.locations_bbox[2],
            self.locations_bbox[1], self.locations_bbox[3],
            network_type='drive'
        )
        self.G_proj = ox.project_graph(self.G)
        
        print(f"Graph initialized with bounding box: {self.locations_bbox}")

    def get_node_coordinates(self, address, verbose = False):
        """Get the nearest node coordinates for a given address"""
        try:
            location = self.geolocator.geocode(f"{address}, Calgary, Alberta")
            if location:
                nearest_node = ox.nearest_nodes(self.G, location.longitude, location.latitude)
                return nearest_node, (location.latitude, location.longitude)
        except Exception as e:
            print(f"Error geocoding {address}: {e}")
        return None, None

    def calculate_route_distance(self, node1, node2, verbose=False):
        """Calculate the actual driving distance between two nodes"""
        try:
            route = nx.shortest_path(self.G_proj, node1, node2, weight='length')
            return sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route, 'length'))
        except nx.NetworkXNoPath:
            return float('inf')

    def solve_tsp(self, locations, nodes, verbose=False):
        """Solve TSP for given locations and nodes"""
        n = len(nodes)
        if n < 2:
            print("Not enough nodes to solve TSP.")
            return None, None, None

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        distances[i][j] = nx.shortest_path_length(self.G, nodes[i], nodes[j], weight='length')
                    except nx.NetworkXNoPath:
                        print(f"No path found between nodes {i} and {j}")
                        distances[i][j] = float('inf')

        if np.isinf(distances).all():
            print("No valid paths found between any nodes.")
            return None, None, None

        best_distance = float('inf')
        best_route = None
        for route in permutations(range(n)):
            distance = sum(distances[route[i]][route[(i+1)%n]] for i in range(n))
            if distance < best_distance:
                best_distance = distance
                best_route = route

        if best_route is None:
            print("Couldn't find a valid route.")
            return None, None, None

        return best_route, best_distance, nodes

    def plot_solution(self, locations, best_route, nodes, verbose=False):
        """Plot the solution with transit hubs"""
        plt.figure(figsize=(15, 10))
        ax = plt.axes([0.1, 0.1, 0.75, 0.8])

        # Plot base map
        ox.plot_graph(self.G, ax=ax, show=False, close=False,
                    edge_color='gray', edge_alpha=0.2, node_size=0)

        # Plot routes if we have a best_route
        if best_route is not None:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(best_route)))
            legend_elements = []
            for i in range(len(best_route)):
                start = best_route[i]
                end = best_route[(i+1) % len(best_route)]
                route = nx.shortest_path(self.G, nodes[start], nodes[end], weight='length')
                route_coords = []
                for node in route:
                    node_data = self.G.nodes[node]
                    route_coords.append((node_data['y'], node_data['x']))
                route_coords = np.array(route_coords)
                line = ax.plot(route_coords[:, 1], route_coords[:, 0],
                            color=colors[i], linewidth=2, alpha=0.7)[0]
                legend_elements.append((line, f'{locations[start]} → {locations[end]}'))

        # Plot location markers
        for i, loc in enumerate(locations):
            coord = self.geolocator.geocode(f"{loc}, Calgary, Alberta")
            if coord:
                ax.plot(coord.longitude, coord.latitude, 'o',
                        color='red', markersize=10, zorder=5)
                ax.annotate(f"{i+1}. {loc}",
                            xy=(coord.longitude, coord.latitude),
                            xytext=(5, 5), textcoords='offset points',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # Plot transit hubs
        for hub_type, hubs in self.transit_hubs.items():
            if hub_type == 'C-Train Stations':
                marker = '^'  # triangle for C-Train
                color = 'blue'
                size = 5
            else:
                marker = 's'  # square for bus terminals
                color = 'green'
                size = 5
            
            # Plot each hub
            for name, lat, lon in hubs:
                ax.plot(lon, lat, marker,
                        color=color, markersize=size, zorder=4,
                        label=hub_type if name == hubs[0][0] else "")
                # Add small labels for transit hubs
                ax.annotate(name,
                            xy=(lon, lat),
                            xytext=(3, 3),
                            textcoords='offset points',
                            fontsize=8,
                            bbox=dict(facecolor='white',
                                    edgecolor='none',
                                    alpha=0.7))

        # Create legend
        if best_route is not None:
            route_legend = [line for line, label in legend_elements]
            route_labels = [label for line, label in legend_elements]
            # Combine all legend elements
            ax.legend(route_legend +
                    [plt.Line2D([0], [0], marker='^', color='blue', linestyle='none'),
                    plt.Line2D([0], [0], marker='s', color='green', linestyle='none')],
                    route_labels + ['C-Train Stations', 'Bus Terminals'],
                    bbox_to_anchor=(1.05, 1), loc='upper left',
                    borderaxespad=0.)
        else:
            ax.legend([plt.Line2D([0], [0], marker='^', color='blue', linestyle='none'),
                    plt.Line2D([0], [0], marker='s', color='green', linestyle='none')],
                    ['C-Train Stations', 'Bus Terminals'],
                    bbox_to_anchor=(1.05, 1), loc='upper left',
                    borderaxespad=0.)

        plt.title("Calgary Transit Map with Pickup Locations")
        plt.tight_layout()
        plt.show()

    def map_pickups_to_hub(self, hub_name, num_pickups, pickup_radius, given_pickups=None,show_plot=False, verbose=False):
        """
        Map the best pickup and dropoff route to a transit hub.
        """
        # Find the hub coordinates
        hub_coords = None
        for hub_list in self.transit_hubs.values():
            for hub in hub_list:
                if hub[0] == hub_name:
                    hub_coords = (hub[1], hub[2])
                    break
            if hub_coords:
                break

        if not hub_coords:
            print(f"Transit hub {hub_name} not found.")
            return

        print(f"Hub coordinates: {hub_coords}")

        # Initialize the graph to set locations_bbox
        self.initialize_graph([f"{hub_coords[0]},{hub_coords[1]}"], verbose=verbose)

        # Generate or use given pickups
        if given_pickups is None:
            print("Generating random pickup locations...")
            pickups = self.generate_random_pickups(hub_coords, pickup_radius, num_pickups, verbose=verbose)
        else:
            pickups = given_pickups

        print(f"Generated pickup locations: {pickups}")

        locations = [f"Pickup {i+1}" for i in range(len(pickups))] + [hub_name]
        coords = pickups + [hub_coords]

        print("Calculating optimal route...")
        nodes = [ox.nearest_nodes(self.G, coord[1], coord[0]) for coord in coords]
        print(f"Valid nodes: {nodes}")

        if len(nodes) < 2:
            print("Not enough valid locations for a route.")
            return

        best_route, total_distance, _ = self.solve_tsp(locations, nodes, verbose=verbose)

        if best_route is None:
            print("Couldn't find valid route.")
        else:
            print("\nOptimal Route:")
            for idx in best_route:
                print(locations[idx])
            print(f"Total Distance: {total_distance / 1000:.2f} km")

        # Visualize the route
        if show_plot:
            self.plot_solution(locations, best_route, nodes)
        
        return total_distance/1000
        
    def generate_random_pickups(self, center, radius, num_pickups, verbose=False):
        """
        Generate random pickup locations on the street network within a given radius of a center point.
        """
        center_node = ox.nearest_nodes(self.G, center[1], center[0])
        pickup_nodes = []
        
        # Convert radius from km to meters
        radius_m = radius * 1000
        
        while len(pickup_nodes) < num_pickups:
            # Generate a random angle and distance
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, radius_m)
            
            # Calculate new point
            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)
            
            # Convert to lat/lon
            new_lon = center[1] + (dx / (111320 * np.cos(np.radians(center[0]))))
            new_lat = center[0] + (dy / 110540)
            
            # Find nearest node to this point
            nearest_node = ox.nearest_nodes(self.G, new_lon, new_lat)
            
            if nearest_node not in pickup_nodes:
                pickup_nodes.append(nearest_node)
        
        # Convert nodes to lat, lon coordinates
        pickups = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) for node in pickup_nodes]
        return pickups
    

def main():
    solver = CalgaryTSP()
    data = []

    hub_name = "Fish Creek - Lacombe CTrain Station"
    total_distance = solver.map_pickups_to_hub(hub_name=hub_name, num_pickups=8, pickup_radius=4, show_plot=True)

    # Collect data for DataFrame
    data.append({
        'Park_and_Ride_Station': hub_name,
        'Passengers': 8,
        'Radius': 4,
        'Tour_Length': total_distance
    })

    df = pd.DataFrame(data, columns=['Park_and_Ride_Station', 'Passengers', 'Radius', 'Tour_Length'])
    # Optionally, save the DataFrame to a CSV file
    df.to_csv('trip_data_campus.csv', index=False)

    for passengers in range(4, 10):  
                for radius in range(2, 6):  
                    print("Computing Test 1 for: [Passengers, Radius] = [" + str(passengers) + ", " + str(radius) + "]")
                    total_distance = solver.map_pickups_to_hub(hub_name=hub_name, num_pickups=passengers, pickup_radius=radius)
                    print("-----------------------------------------------------------------")
                    
                    # Collect data for DataFrame
                    data.append({
                        'Park_and_Ride_Station': hub_name,
                        'Passengers': passengers,
                        'Radius': radius,
                        'Tour_Length': total_distance
                    })
                    
                    print("Computing Test 2 for: [Passengers, Radius] = [" + str(passengers) + ", " + str(radius) + "]")
                    total_distance = solver.map_pickups_to_hub(hub_name=hub_name, num_pickups=passengers, pickup_radius=radius)
                    print("-----------------------------------------------------------------")
                    
                    # Collect data for DataFrame
                    data.append({
                        'Park_and_Ride_Station': hub_name,
                        'Passengers': passengers,
                        'Radius': radius,
                        'Tour_Length': total_distance
                    })
                    
                    print("Computing Test 3 for: [Passengers, Radius] = [" + str(passengers) + ", " + str(radius) + "]")
                    total_distance = solver.map_pickups_to_hub(hub_name=hub_name, num_pickups=passengers, pickup_radius=radius)
                    print("-----------------------------------------------------------------")
                    
                    # Collect data for DataFrame
                    data.append({
                        'Park_and_Ride_Station': hub_name,
                        'Passengers': passengers,
                        'Radius': radius,
                        'Tour_Length': total_distance
                    })

    df = pd.DataFrame(data, columns=['Park_and_Ride_Station', 'Passengers', 'Radius', 'Tour_Length'])
    # Optionally, save the DataFrame to a CSV file
    df.to_csv('trip_data_campus.csv', index=False)    


if __name__ == "__main__":
    main()