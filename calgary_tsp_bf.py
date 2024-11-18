import osmnx as ox
import networkx as nx
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import random

class CalgaryTSP:
    def __init__(self):
        self.locations_bbox = None
        self.geolocator = Nominatim(user_agent="calgary_tsp")  # Geocoding service
        
        # Define main transit hubs
        self.transit_hubs = {
            'C-Train Stations': [
                ("Downtown West/Kerby", 51.0461, -114.0833),
                ("8th Street SW", 51.0461, -114.0789),
                ("7th Street SW", 51.0461, -114.0744),
                ("6th Street SW", 51.0461, -114.0700),
                ("3rd Street SW", 51.0461, -114.0567),
                ("City Hall", 51.0461, -114.0522),
                ("Erlton/Stampede", 51.0328, -114.0517),
                ("39th Avenue", 51.0219, -114.0517),
                ("Chinook", 50.9967, -114.0517),
                ("Heritage", 50.9883, -114.0517),
                ("Southland", 50.9633, -114.0517),
                ("Anderson", 50.9567, -114.0517),
                ("SAIT/ACAD/Jubilee", 51.0644, -114.0892),
                ("Lions Park", 51.0711, -114.0892),
                ("University", 51.0794, -114.1231),
                ("Brentwood", 51.0867, -114.1231),
                ("Dalhousie", 51.1042, -114.1231),
                ("Crowfoot", 51.1217, -114.1231),
                ("Tuscany", 51.1269, -114.2247)
            ],
            'Major Bus Terminals': [
                ("North Pointe Terminal", 51.1642, -114.0694),
                ("Brentwood Station", 51.0867, -114.1231),
                ("Lions Park Station", 51.0711, -114.0892),
                ("Anderson Station", 50.9567, -114.0517),
                ("Heritage Station", 50.9883, -114.0517),
                ("South Campus Hospital", 51.0794, -114.1231),
                ("Chinook Station", 50.9967, -114.0517)
            ]
        }

    def initialize_graph(self, locations):
        """
        Initialize the graph using OSM for a bounding box around the provided locations.
        """
        # Gather coordinates for locations
        coords = []
        for loc in locations:
            location = self.geolocator.geocode(f"{loc}, Calgary, Alberta")
            if location:
                coords.append((location.latitude, location.longitude))
        
        # Add transit hub coordinates to ensure coverage
        for hub_type in self.transit_hubs.values():
            for _, lat, lon in hub_type:
                coords.append((lat, lon))
        
        if not coords:
            raise ValueError("No valid locations found for graph initialization")
        
        # Calculate bounding box with padding
        min_lat = min(c[0] for c in coords)
        max_lat = max(c[0] for c in coords)
        min_lon = min(c[1] for c in coords)
        max_lon = max(c[1] for c in coords)
        
        lat_padding = (max_lat - min_lat) * 0.2
        lon_padding = (max_lon - min_lon) * 0.2
        
        self.locations_bbox = [
            min_lat - lat_padding,
            min_lon - lon_padding,
            max_lat + lat_padding,
            max_lon + lon_padding
        ]
        
        # Load OSM graph for the area
        self.G = ox.graph_from_bbox(
            self.locations_bbox[0], self.locations_bbox[2],
            self.locations_bbox[1], self.locations_bbox[3],
            network_type='drive'
        )
        self.G_proj = ox.project_graph(self.G)  # Project graph for accurate distance calculations
        
        print(f"Graph initialized with bounding box: {self.locations_bbox}")

    def get_node_coordinates(self, address_or_coords):
        """
        Get the nearest graph node for an address or lat/lon coordinates.
        """
        try:
            # Detect lat/lon input more robustly
            parts = address_or_coords.split(',')
            if len(parts) == 2:
                try:
                    lat, lon = map(float, parts)
                except ValueError:
                    # Fallback to geocoding if not valid numbers
                    raise ValueError(f"Invalid coordinate format: {address_or_coords}")
            else:
                # Perform geocoding if input is not lat/lon
                location = self.geolocator.geocode(f"{address_or_coords}, Calgary, Alberta")
                if location:
                    lat, lon = location.latitude, location.longitude
                else:
                    print(f"Geocoding failed for address: {address_or_coords}")
                    return None, None
            
            # Check if lat/lon is within graph bounds
            if not (self.locations_bbox[0] <= lat <= self.locations_bbox[2] and
                    self.locations_bbox[1] <= lon <= self.locations_bbox[3]):
                print(f"Coordinates {lat},{lon} are outside the graph bounding box.")
                return None, None
            
            # Find nearest node
            nearest_node = ox.nearest_nodes(self.G, lon, lat)
            return nearest_node, (lat, lon)
        except Exception as e:
            print(f"Error processing {address_or_coords}: {e}")
        return None, None


    def calculate_route_distance(self, node1, node2):
        """
        Calculate the actual driving distance between two nodes.
        """
        try:
            route = nx.shortest_path(self.G_proj, node1, node2, weight='length')
            return sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route, 'length'))
        except nx.NetworkXNoPath:
            return float('inf')

    def solve_tsp(self, locations):
        """
        Solve TSP for the given locations and return the optimal route and distance.
        """
        self.initialize_graph(locations)
        
        nodes = []
        coords = []
        
        for loc in locations:
            node, coord = self.get_node_coordinates(loc)
            if node:
                nodes.append(node)
                coords.append(coord)
        
        if len(nodes) < 2:
            print("Not enough valid nodes for solving TSP.")
            return None, None, None
        
        n = len(nodes)
        distances = np.zeros((n, n))
        
        # Compute pairwise distances
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i][j] = self.calculate_route_distance(nodes[i], nodes[j])
        
        # Solve TSP using brute force (for small datasets)
        best_distance = float('inf')
        best_route = None
        for route in permutations(range(n)):
            distance = sum(distances[route[i]][route[(i+1)%n]] for i in range(n))
            if distance < best_distance:
                best_distance = distance
                best_route = route
        
        return best_route, best_distance, nodes

    def generate_random_pickups(self, center, radius, num_points):
        """
        Generate random pickup points around a central location within a given radius.
        """
        random_points = []
        for _ in range(num_points):
            distance = random.uniform(0, radius)
            angle = random.uniform(0, 2 * np.pi)
            new_lat = center[0] + (distance / 111) * np.cos(angle)
            new_lon = center[1] + (distance / (111 * np.cos(np.radians(center[0])))) * np.sin(angle)
            
            # Log and validate against bounding box
            print(f"Generated random point: {new_lat}, {new_lon}")
            if not (self.locations_bbox[0] <= new_lat <= self.locations_bbox[2] and
                    self.locations_bbox[1] <= new_lon <= self.locations_bbox[3]):
                print(f"Point {new_lat},{new_lon} is outside the graph bounding box.")
                continue
            random_points.append((new_lat, new_lon))
        return random_points

def initialize_graph(self, locations):
    """
    Initialize the graph using OSM for a bounding box around the provided locations.
    """
    # Existing logic...

    self.locations_bbox = [
        min_lat - lat_padding,
        min_lon - lon_padding,
        max_lat + lat_padding,
        max_lon + lon_padding
    ]
    
    print(f"Graph initialized with bounding box: {self.locations_bbox}")

def map_pickups_to_hub(hub_name, num_pickups, pickup_radius, given_pickups=None):
    """
    Map the best pickup and dropoff route to a transit hub.
    """
    solver = CalgaryTSP()
    
    # Find the hub coordinates
    hub_coords = None
    for hub_list in solver.transit_hubs.values():
        for hub in hub_list:
            if hub[0] == hub_name:
                hub_coords = (hub[1], hub[2])
                break
        if hub_coords:
            break
    
    if not hub_coords:
        print(f"Transit hub {hub_name} not found.")
        return

    # Initialize the graph to set locations_bbox
    solver.initialize_graph([f"{hub_coords[0]},{hub_coords[1]}"])

    # Generate or use given pickups
    if given_pickups is None:
        print("Generating random pickup locations...")
        pickups = solver.generate_random_pickups(hub_coords, pickup_radius, num_pickups)
    else:
        pickups = given_pickups

    locations = [f"Pickup {i+1}" for i in range(len(pickups))] + [hub_name]
    coords = pickups + [hub_coords]
    
    print("Calculating optimal route...")
    
    nodes = [solver.get_node_coordinates(f"{coord[0]},{coord[1]}") for coord in coords]
    nodes = [n[0] for n in nodes if n is not None and n[0] is not None]

    if len(nodes) < 2:
        print("Not enough valid locations for a route.")
        return
    
    best_route, total_distance, _ = solver.solve_tsp(locations)
    if best_route is None:
        print("Couldn't find valid route.")
        return
    
    print("\nOptimal Route:")
    for idx in best_route:
        print(locations[idx])
    print(f"Total Distance: {total_distance / 1000:.2f} km")


def main():
    map_pickups_to_hub("Chinook", 3, 2)  # 5 random pickups within 5 km of Chinook hub

    # locations = [
    #     "Downtown Core, Calgary",
    #     "Kensington, Calgary",
    #     "Beltline, Calgary",
    #     "Bowness, Calgary",
    #     "Inglewood, Calgary",
    #     "University Heights, Calgary",
    #     "Signal Hill, Calgary",
    #     "Bridgeland, Calgary"
    # ]
    
    # solver = CalgaryTSP()
    
    # print("Calculating optimal route...")
    # best_route, total_distance, nodes = solver.solve_tsp(locations)
    
    # if best_route is None:
    #     print("Couldn't find valid route")
    #     return
        
    # print("\nOptimal Route:")
    # print("-------------")
    # for i, idx in enumerate(best_route):
    #     print(f"{i+1}. {locations[idx]}")
    # print(f"\nTotal Distance: {total_distance/1000:.2f} km")
    
    # print("\nGenerating map visualization...")
    # solver.plot_solution(locations, best_route, nodes)

if __name__ == "__main__":
    main()
