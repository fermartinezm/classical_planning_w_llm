from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts gracefully
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         return [] # Return empty list for malformed facts
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, ignoring vehicle capacity constraints and the need for a
    vehicle to reach a package's location for pickup.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - The cost of picking up a package is 1.
    - The cost of dropping a package is 1.
    - The cost of driving a vehicle between adjacent locations is 1.
    - The cost of driving a vehicle between non-adjacent locations is the
      shortest path distance in the road network.
    - A suitable vehicle is always available for pickup at the package's
      current location if needed.
    - Vehicle capacity is ignored when estimating costs for individual packages.
    - Goal conditions only involve packages being at specific locations.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Identifies all locations and vehicles present in the initial state and static facts.
    - Builds a graph of the road network from static facts.
    - Computes all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Initialize the total heuristic cost to 0.
    3. Determine the current location or container (vehicle) for every package
       and vehicle in the state by parsing 'at' and 'in' facts.
    4. For each package that has a goal location:
       a. If the package is currently at its goal location on the ground, the
          cost for this package is 0. Continue to the next package.
       b. If the package is on the ground at a location different from its goal:
          - It needs to be picked up (cost 1).
          - It needs to be transported by a vehicle from its current location
            to its goal location. The estimated cost is the shortest path
            distance between these locations.
          - It needs to be dropped at the goal location (cost 1).
          - The total estimated cost for this package is 1 (pick) + distance (drives) + 1 (drop).
       c. If the package is inside a vehicle:
          - Find the current location of the vehicle.
          - If the vehicle is at the package's goal location:
            - It needs to be dropped (cost 1).
            - The total estimated cost for this package is 1.
          - If the vehicle is at a location different from the package's goal:
            - It needs to be transported by the vehicle from the vehicle's
              current location to the package's goal location. The estimated
              cost is the shortest path distance between these locations.
            - It needs to be dropped at the goal location (cost 1).
            - The total estimated cost for this package is distance (drives) + 1 (drop).
       d. If the package's status cannot be determined (e.g., not 'at' a location, not 'in' a vehicle),
          or if a required location is unreachable, return infinity.
    5. Sum the estimated costs for all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest paths.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                # Assuming goal is always (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Identify all locations and vehicles
        self.locations = set()
        self.vehicles = set()
        self.road_graph = defaultdict(list)

        # Collect locations and build road graph from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "road" and len(parts) == 3:
                _, loc1, loc2 = parts[1], parts[2]
                self.road_graph[loc1].append(loc2)
                self.locations.add(loc1)
                self.locations.add(loc2)
            elif predicate == "capacity" and len(parts) == 3:
                 # (capacity ?v - vehicle ?s1 - size)
                 self.vehicles.add(parts[1])

        # Collect locations and vehicles from initial state
        # This ensures we include objects/locations present in init but maybe not static/goals
        for fact in task.initial_state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                # (at ?x - locatable ?v - location)
                obj, loc = parts[1], parts[2]
                self.locations.add(loc) # Add location
                # If obj is not a package with a goal, assume it's a vehicle
                if obj not in self.goal_locations:
                     self.vehicles.add(obj)
            elif predicate == "in" and len(parts) == 3:
                 # (in ?x - package ?v - vehicle)
                 self.vehicles.add(parts[2]) # Add vehicle

        # Compute all-pairs shortest paths using BFS from each location.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search starting from start_node to find shortest
        distances to all other nodes in the road graph.
        Returns a dictionary mapping reachable locations to their distance from start_node.
        Unreachable locations are not included or implicitly have infinite distance.
        """
        distances = {start_node: 0}
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()
            dist = distances[current_loc]

            # Check if current_loc has outgoing roads
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if neighbor not in distances: # Check if visited
                        distances[neighbor] = dist + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is already reached
        if self.goals <= state:
             return 0

        # Track package locations and vehicle contents/locations
        package_locations = {} # package -> location (if on ground)
        package_in_vehicle = {} # package -> vehicle (if in vehicle)
        vehicle_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # It's a package with a goal
                    package_locations[obj] = loc
                elif obj in self.vehicles: # It's a vehicle
                    vehicle_locations[obj] = loc
                # Other 'at' facts (e.g., at size loc) are ignored

            elif predicate == "in" and len(parts) == 3:
                 pkg, veh = parts[1], parts[2]
                 if pkg in self.goal_locations and veh in self.vehicles: # Ensure they are known types
                    package_in_vehicle[pkg] = veh
                 # Other 'in' facts are ignored

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Determine package's current status
            is_at_location = package in package_locations
            is_in_vehicle = package in package_in_vehicle

            # Case 1: Package is already at its goal location on the ground
            if is_at_location and package_locations[package] == goal_location:
                continue # Cost is 0 for this package

            # Case 2: Package is on the ground at a different location
            elif is_at_location:
                current_location = package_locations[package]

                # Needs pick-up (1), transport (distance), drop (1).
                # Check if current_location is in our distance map (should be if collected correctly)
                if current_location not in self.distances:
                     # This indicates an issue with initial state parsing or unreachable location
                     return float('inf')

                dist = self.distances[current_location].get(goal_location, float('inf'))

                if dist == float('inf'):
                    return float('inf') # Goal location unreachable from package's current location

                total_cost += 1  # pick-up
                total_cost += dist # drive actions
                total_cost += 1  # drop

            # Case 3: Package is inside a vehicle
            elif is_in_vehicle:
                 vehicle = package_in_vehicle[package]

                 # Find vehicle's location
                 if vehicle not in vehicle_locations:
                     # Vehicle location unknown - should not happen in valid states
                     return float('inf')

                 vehicle_location = vehicle_locations[vehicle]

                 # Check if vehicle_location is in our distance map
                 if vehicle_location not in self.distances:
                      # This indicates an issue with initial state parsing or unreachable location
                      return float('inf')

                 # Needs transport (distance) and drop (1).
                 dist = self.distances[vehicle_location].get(goal_location, float('inf'))

                 if dist == float('inf'):
                     return float('inf') # Goal location unreachable from vehicle's current location

                 # If the vehicle is already at the goal location, only drop is needed.
                 if vehicle_location == goal_location:
                     total_cost += 1 # drop
                 else:
                     total_cost += dist # drive actions
                     total_cost += 1 # drop
            else:
                # Package exists in goal_locations but is neither at a location nor in a vehicle in the state.
                # This indicates an invalid or unreachable state.
                return float('inf')

        return total_cost
