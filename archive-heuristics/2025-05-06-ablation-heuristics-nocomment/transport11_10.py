from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Helper to parse a PDDL fact string into a list of parts."""
    # Remove surrounding parentheses and split by space
    # Handle cases with empty facts or malformed strings defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
         return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal by summing the estimated
    costs for each package that is not yet at its goal location. It considers
    the actions needed to move a package: pick-up, drive, and drop. It uses
    precomputed shortest path distances on the road network to estimate drive costs.
    Capacity constraints and vehicle availability/assignment are ignored for
    simplicity and efficiency, making the heuristic non-admissible but potentially
    effective for greedy best-first search.

    Assumptions:
    - The input state is valid according to the domain rules (e.g., a package is
      either 'at' a location or 'in' a vehicle, but not both).
    - All packages mentioned in the goal are present in the initial state and
      subsequent states, either 'at' a location or 'in' a vehicle.
    - The road network defined by 'road' facts is static.
    - The goal is defined by 'at' predicates for packages.
    - Objects appearing in 'at' facts that are not goal packages are vehicles.

    Heuristic Initialization:
    The heuristic's constructor (`__init__`) performs the following steps:
    1. Extracts the goal location for each package from the task's goal facts.
    2. Builds a graph representing the road network based on the static 'road' facts.
       It collects all locations mentioned in static 'road' facts, initial state 'at'
       facts, and goal 'at' facts to ensure all relevant locations are included.
    3. Computes the shortest path distance between every pair of locations in the
       road network using Breadth-First Search (BFS). These distances are stored
       for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic function (`__call__`) computes the estimated cost for a given state:
    1. It parses the current state to determine the current status and location
       of each package (either 'at' a location or 'in' a vehicle) and the current
       location of each vehicle. Packages are identified based on whether they
       appear in the task's goals. Objects in 'at' facts that are not goal packages
       are assumed to be vehicles.
    2. It initializes a total estimated cost to 0.
    3. For each package that has a goal location:
       a. It retrieves the package's current status and location (or vehicle and vehicle's location).
       b. If the package is currently 'at' a location different from its goal location:
          The estimated cost for this package includes:
          - 1 action for 'pick-up'.
          - The shortest path distance (number of drive actions) from its current location
            to its goal location.
          - 1 action for 'drop'.
          These costs are added to the total.
       c. If the package is currently 'in' a vehicle:
          It finds the current location of the vehicle.
          If the vehicle's location is different from the package's goal location:
             The estimated cost for this package includes:
             - The shortest path distance (number of drive actions) from the vehicle's
               current location to the package's goal location.
             - 1 action for 'drop'.
             These costs are added to the total.
          If the vehicle's location is the same as the package's goal location:
             The estimated cost for this package includes:
             - 1 action for 'drop'.
             This cost is added to the total.
4. The final total estimated cost is returned. This value is 0 if and only if
   all packages are currently 'at' their respective goal locations.
"""
    def __init__(self, task):
        """
        Initializes the heuristic. Precomputes goal locations and road network distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state # Need initial state to find all locations

        # 1. Extract goal locations for packages
        self.package_goal_location = {}
        # Collect all goal objects (packages) for easy lookup
        self.all_goal_packages = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal is typically (at package location)
            if parts and parts[0] == 'at' and len(parts) == 3:
                package = parts[1]
                location = parts[2]
                self.package_goal_location[package] = location
                self.all_goal_packages.add(package)

        # 2. Build the road graph and collect all relevant locations
        self.road_graph = {} # {location: [reachable_location1, ...]}
        all_locations = set()

        # Add locations from road facts
        for fact_str in static_facts:
            parts = get_parts(fact_str)
            if parts and parts[0] == 'road' and len(parts) == 3:
                loc1 = parts[1]
                loc2 = parts[2]
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = []
                self.road_graph[loc1].append(loc2)
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Add locations from initial state (at facts)
        for fact_str in initial_state_facts:
             parts = get_parts(fact_str)
             if parts and parts[0] == 'at' and len(parts) == 3:
                  all_locations.add(parts[2]) # Add location from (at obj loc)

        # Add locations from goals (at facts)
        for goal_str in self.goals:
             parts = get_parts(goal_str)
             if parts and parts[0] == 'at' and len(parts) == 3:
                  all_locations.add(parts[2]) # Add location from (at package loc)

        # Ensure all collected locations are keys in the graph dictionary
        for loc in all_locations:
             if loc not in self.road_graph:
                  self.road_graph[loc] = []

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {} # {(loc1, loc2): distance}

        for start_node in self.road_graph.keys():
            q = deque([(start_node, 0)])
            visited = {start_node}

            while q:
                current_loc, current_dist = q.popleft()

                # Store distance
                self.distances[(start_node, current_loc)] = current_dist

                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, current_dist + 1))

        # Locations not reachable from a start_node will not have an entry in self.distances
        # The .get() method with float('inf') default handles this.


    def __call__(self, node):
        """
        Computes the heuristic value for a given state.
        Estimates the minimum number of actions (pick-up, drive, drop) required
        to move packages to their goal locations, ignoring capacity constraints
        and vehicle availability/assignment.
        """
        state = node.state

        # Extract dynamic information from the current state
        package_current_info = {} # Maps package -> ('at', location) or ('in', vehicle)
        vehicle_current_location = {} # Maps vehicle -> location

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3:
                obj = parts[1]
                loc = parts[2]
                # Check if the object is one of the packages we care about (i.e., in the goals)
                if obj in self.all_goal_packages:
                     package_current_info[obj] = ('at', loc)
                else:
                     # Assume other 'at' objects are vehicles
                     vehicle_current_location[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                package = parts[1]
                vehicle = parts[2]
                # Only track packages that are in the goals
                if package in self.all_goal_packages:
                    package_current_info[package] = ('in', vehicle)
            # Ignore other predicates like capacity, road, capacity-predecessor

        total_cost = 0

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_loc in self.package_goal_location.items():
            # If package is not mentioned in the state (e.g., invalid state), skip or assign high cost.
            # Assuming valid states where every goal package is either 'at' or 'in'.
            if package not in package_current_info:
                 # This case should ideally not happen in a reachable state from a valid initial state
                 # where all goal packages are initially placed. Assign a large cost.
                 total_cost += 1000 # Penalty for missing package info
                 continue

            current_info = package_current_info[package]
            status = current_info[0]

            if status == 'at':
                current_loc = current_info[1]
                if current_loc != goal_loc:
                    # Package is at a location, needs pickup, drive, drop
                    # Cost = 1 (pick-up) + dist(current_loc, goal_loc) (drive) + 1 (drop)
                    drive_cost = self.distances.get((current_loc, goal_loc), float('inf'))
                    total_cost += 2 + drive_cost
            elif status == 'in':
                vehicle = current_info[1]
                # Find the location of the vehicle holding the package
                if vehicle not in vehicle_current_location:
                    # Vehicle holding package is not at any location? Invalid state.
                    total_cost += 1000 # Penalty for invalid vehicle info
                    continue

                vehicle_loc = vehicle_current_location[vehicle]

                # Package is in a vehicle.
                # If vehicle is not at goal_loc, needs drive and drop.
                # If vehicle is at goal_loc, needs only drop.
                if vehicle_loc != goal_loc:
                    # Cost = dist(vehicle_loc, goal_loc) (drive) + 1 (drop)
                    drive_cost = self.distances.get((vehicle_loc, goal_loc), float('inf'))
                    total_cost += drive_cost + 1
                else: # vehicle_loc == goal_loc
                    # Cost = 1 (drop)
                    total_cost += 1

        return total_cost
