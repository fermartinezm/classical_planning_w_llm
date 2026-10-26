# Import necessary modules
from collections import deque

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function for BFS
def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all reachable nodes.
    graph: adjacency list representation {node: [neighbor1, neighbor2, ...]}
    Returns: dictionary {node: distance}
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    # Ensure start_node is in graph keys, even if it has no neighbors (isolated)
    if start_node not in graph:
        graph[start_node] = [] # Add as isolated node if not already present

    while queue:
        current_node = queue.popleft()

        # current_node is guaranteed to be in graph keys here
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

# Assume Heuristic base class exists and provides the structure
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass

class transportHeuristic: # Inherits from Heuristic base class implicitly as per examples
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions needed to move each package to its goal location
    independently, considering pick-up, drive, and drop actions. Drive cost is
    estimated by the shortest path distance in the road network. Capacity and
    vehicle availability are relaxed.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network graph, and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are always (at package location)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                 package, location = parts[1], parts[2]
                 self.goal_locations[package] = location

        # Build the road network graph and collect all relevant locations
        self.road_graph = {}
        all_locations = set()

        # Add locations from road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                if l2 not in self.road_graph:
                     self.road_graph[l2] = []
                self.road_graph[l1].append(l2)
                all_locations.add(l1)
                all_locations.add(l2)

        # Add locations from initial state 'at' facts
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3: # (at locatable location)
                 # Assume the second argument of 'at' is always a location.
                 loc = parts[2]
                 all_locations.add(loc)
                 if loc not in self.road_graph: # Add as isolated node if not in road facts
                     self.road_graph[loc] = []

        # Add locations from goal 'at' facts
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "at" and len(parts) == 3:
                 loc = parts[2]
                 all_locations.add(loc)
                 if loc not in self.road_graph: # Add as isolated node if not in road facts
                     self.road_graph[loc] = []

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in all_locations: # Use all collected locations as start nodes
            self.distances[start_loc] = bfs(self.road_graph, start_loc)

        # Penalty for unreachable goals or invalid states
        self.unreachable_penalty = 1000


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached
        # A state is a goal state if all goals are a subset of the state facts.
        if self.goals <= state:
             return 0

        # Track current status of packages and vehicles
        current_status = {} # {object_name: ('at', location) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle_name: location}

        # Populate current_status and vehicle_locations by parsing the state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_status[obj] = ('at', loc)
                # Assuming objects starting with 'v' are vehicles
                if obj.startswith('v'): # Domain-specific assumption based on examples
                    vehicle_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 current_status[package] = ('in', vehicle)

        total_cost = 0

        # Calculate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            # Find package's current status
            if package not in current_status:
                 # Package is not mentioned in 'at' or 'in' facts. Invalid state?
                 # Add a large penalty.
                 total_cost += self.unreachable_penalty
                 continue # Skip this package

            status_type, status_obj = current_status[package]

            # Check if package is already at goal
            if status_type == 'at' and status_obj == goal_location:
                continue # Package is already at goal

            # Package needs to be moved
            if status_type == 'at':
                current_l = status_obj
                # Cost = pick-up (1) + drive (distance) + drop (1)
                # Need distance from current_l to goal_location
                if current_l in self.distances and goal_location in self.distances.get(current_l, {}):
                    drive_cost = self.distances[current_l][goal_location]
                    total_cost += 1 + drive_cost + 1
                else:
                    # Unreachable path from current_l to goal_location
                    total_cost += self.unreachable_penalty

            elif status_type == 'in':
                vehicle = status_obj
                # Package is in a vehicle. Find vehicle's location.
                if vehicle in vehicle_locations:
                    vehicle_l = vehicle_locations[vehicle]
                    # Cost = drive (distance) + drop (1)
                    # Need distance from vehicle_l to goal_location
                    if vehicle_l in self.distances and goal_location in self.distances.get(vehicle_l, {}):
                         drive_cost = self.distances[vehicle_l][goal_location]
                         total_cost += drive_cost + 1
                    else:
                         # Unreachable path from vehicle_l to goal_location
                         total_cost += self.unreachable_penalty

                else:
                    # Vehicle location not found. Should not happen in a valid state.
                    # Add a penalty.
                    total_cost += self.unreachable_penalty # Penalty for package in vehicle with unknown location

        return total_cost
