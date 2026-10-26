from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    all packages to their goal locations. It calculates the cost for each
    package independently, summing them up.

    # Assumptions:
    - Roads are bidirectional.
    - Vehicle capacity constraints are ignored.
    - Vehicle availability is ignored (a vehicle is assumed to be available
      at the package's location or the vehicle's location when needed).
    - Shared trips (carrying multiple packages in one vehicle) are not
      explicitly optimized; the cost is calculated per package as if it
      were transported individually.
    - The cost of each action (drive, pick-up, drop) is 1.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal state.
    - Builds a graph representing the road network from static facts.
    - Computes all-pairs shortest paths between locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every package
       that has a goal location.
    2. Identify the current location of every vehicle that might be carrying
       a package.
    3. Initialize the total heuristic cost to 0.
    4. For each package that needs to reach a specific goal location:
       a. Check if the package is already at its goal location. If yes,
          its cost contribution is 0.
       b. If the package is currently on the ground at a location `L`:
          - The estimated cost for this package is 1 (pick-up) +
            shortest_path(L, GoalLocation) + 1 (drop).
       c. If the package is currently inside a vehicle `V`, and vehicle `V`
          is at location `L_v`:
          - The estimated cost for this package is shortest_path(L_v, GoalLocation) +
            1 (drop). (Assumes the package was already picked up).
       d. If the goal location is unreachable from the package's current
          location (or vehicle's location), the state is likely unsolvable
          or requires actions not modeled by this simple path cost; return
          infinity.
    5. Sum the estimated costs for all packages that are not yet at their
       goal locations.
    6. Return the total sum as the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the
        road graph, and computing shortest paths.
        """
        # task object has attributes: name, facts, initial_state, goals, operators, static
        self.goals = task.goals  # Goal conditions (frozenset of facts)

        # Extract goal locations for each package
        self.goal_locations = {} # {package_name: goal_location_name}
        locations = set() # Collect all unique location names

        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3: # Ensure it's a valid (at ?obj ?loc) fact
                    package = parts[1]
                    location = parts[2]
                    self.goal_locations[package] = location
                    locations.add(location)

        # Build the road graph (adjacency list)
        road_graph = {} # {location_name: set(neighbor_location_names)}
        for fact in task.static:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's a valid (road ?l1 ?l2) fact
                    l1 = parts[1]
                    l2 = parts[2]
                    road_graph.setdefault(l1, set()).add(l2)
                    road_graph.setdefault(l2, set()).add(l1) # Assume bidirectional roads
                    locations.add(l1)
                    locations.add(l2)

        # Add locations from initial state 'at' facts, just in case some locations exist but aren't in roads or goals
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     locations.add(parts[2])


        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {} # {start_loc: {end_loc: distance}}
        for start_loc in locations:
            self.shortest_paths[start_loc] = self._bfs(road_graph, start_loc)

    def _bfs(self, graph, start_node):
        """
        Performs Breadth-First Search to find shortest paths from a start node.

        Args:
            graph: Adjacency list representation of the graph.
            start_node: The starting location for the BFS.

        Returns:
            A dictionary mapping reachable locations to their shortest distance
            from the start_node.
        """
        distances = {start_node: 0}
        queue = deque([start_node])

        # Ensure start_node is in the graph keys if it has no neighbors but exists
        if start_node not in graph:
             graph[start_node] = set()

        while queue:
            u = queue.popleft()
            # Check if u is a key in the graph before iterating neighbors
            if u in graph:
                for v in graph[u]:
                    if v not in distances:
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of facts)

        # Track current state of packages and vehicles
        package_current_info = {} # {package_name: {'type': 'at', 'location': loc} or {'type': 'in', 'vehicle': veh}}
        vehicle_locations = {} # {vehicle_name: loc}

        # Populate current state information by iterating through state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                obj = parts[1]
                loc = parts[2]
                # If the object is a package we care about (i.e., it's in our goal list)
                if obj in self.goal_locations:
                    package_current_info[obj] = {'type': 'at', 'location': loc}
                # Otherwise, assume it's a vehicle (or other locatable not in goal packages)
                # We only need vehicle locations if they carry packages
                # A simple way to identify vehicles is anything 'at' a location that isn't a goal package
                # This is an assumption, but works for typical transport problems
                # Also, a vehicle could be 'at' a location even if it's not carrying a goal package.
                # We need its location if a goal package is 'in' it.
                # Let's just collect all 'at' facts for non-goal-packages as potential vehicles.
                # This might include other locatables, but the heuristic only uses vehicle_locations
                # when a package is 'in' a vehicle, so it should be fine.
                # Avoid marking a package as a vehicle if it's already marked 'in'
                if obj not in package_current_info:
                     vehicle_locations[obj] = loc

            elif predicate == "in" and len(parts) == 3:
                package = parts[1]
                vehicle = parts[2]
                # If the package is one we care about (i.e., it's in our goal list)
                if package in self.goal_locations:
                    package_current_info[package] = {'type': 'in', 'vehicle': vehicle}
                # Else: It's a package not relevant to the goal, ignore.

            # Ignore other predicates like capacity, capacity-predecessor, road for heuristic calculation

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that needs to reach its goal
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at its goal location
            # We need to check if the package's current state is known and is 'at' the goal location
            current_info = package_current_info.get(package)

            if current_info and current_info['type'] == 'at' and current_info['location'] == goal_location:
                continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal, calculate its contribution to the heuristic
            if not current_info:
                 # This state is missing information about a goal package.
                 # This shouldn't happen in valid planning problems.
                 # Treat as unsolvable from here.
                 return float('inf')

            # Get the effective current location for path calculation
            effective_current_l = None
            if current_info['type'] == 'at':
                effective_current_l = current_info['location']
                # Cost includes pick-up (1)
                cost_contribution = 1
            elif current_info['type'] == 'in':
                vehicle = current_info['vehicle']
                # Need the location of the vehicle
                if vehicle not in vehicle_locations:
                    # Vehicle carrying the package is not located anywhere. Invalid state.
                    return float('inf')
                effective_current_l = vehicle_locations[vehicle]
                # Cost does NOT include pick-up (already picked up)
                cost_contribution = 0

            # Add drive cost (shortest path) and drop cost (1)
            if effective_current_l is None:
                 # Should not happen based on logic above, but as a safeguard
                 return float('inf')

            # Get shortest path distance
            # Ensure effective_current_l and goal_location are in our shortest_paths map
            if effective_current_l not in self.shortest_paths or goal_location not in self.shortest_paths.get(effective_current_l, {}):
                 # Goal location is unreachable from current effective location
                 return float('inf')

            drive_cost = self.shortest_paths[effective_current_l][goal_location]

            cost_contribution += drive_cost + 1 # Add drop cost (1)

            total_cost += cost_contribution

        return total_cost
