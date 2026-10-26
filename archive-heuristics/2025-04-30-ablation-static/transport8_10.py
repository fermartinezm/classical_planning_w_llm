from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
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
    This heuristic estimates the number of actions needed to move all packages
    to their goal locations. It considers the cost of picking up, dropping off,
    and the minimum driving distance required for each package.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity is ignored.
    - The availability of a vehicle at a package's location (if on the ground)
      is assumed, adding a pick-up cost.
    - The shortest path in the road network represents the minimum driving cost.
    - Roads are bidirectional.
    - The heuristic is 0 if and only if the state is a goal state.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph representation of the road network from static facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that has a goal location and is not yet at that goal location on the ground:
    1. Check if the package is currently on the ground or inside a vehicle.
    2. If the package is on the ground at location L_curr:
       - The cost contribution for this package is 1 (pick-up) + shortest_path_distance(L_curr, L_goal) + 1 (drop).
    3. If the package is inside a vehicle V, find V's current location L_v:
       - The cost contribution for this package is shortest_path_distance(L_v, L_goal) + 1 (drop).
    4. If the package is already on the ground at its goal location, its cost contribution is 0.
    5. The total heuristic value is the sum of the cost contributions for all packages whose goal is (at package goal_location).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all locations

        self.goal_locations = {}
        # Store goal locations for packages we care about
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        self.road_graph = {}
        all_locations = set()

        # Build road graph and collect all locations from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                all_locations.add(l1)
                all_locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming bidirectional roads

        # Also collect locations from initial state facts like (at obj loc)
        # This ensures locations mentioned in init but not in road facts are included
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at":
                 loc = parts[2]
                 all_locations.add(loc)
                 self.road_graph.setdefault(loc, set()) # Ensure all locations are keys in graph

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_loc in all_locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc, self.road_graph)

    def _bfs(self, start_loc, graph):
        """Performs BFS from start_loc to find distances to all reachable locations."""
        distances = {loc: float('inf') for loc in graph}
        if start_loc not in graph:
             # Should not happen if all_locations was populated correctly, but defensive
             return distances

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            curr_loc = queue.popleft()
            curr_dist = distances[curr_loc]

            # Check if curr_loc is a key in the graph before iterating neighbors
            if curr_loc in graph:
                for neighbor in graph[curr_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = curr_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        task = node.task # Access task to check goal_reached

        # Heuristic is 0 iff the state is a goal state
        if task.goal_reached(state):
             return 0

        current_locatables = {} # Maps locatable (package or vehicle) to its ground location
        package_in_vehicle = {} # Maps package to the vehicle it is in

        # Parse the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                current_locatables[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle

        total_cost = 0

        # Calculate cost for each package whose goal is (at package goal_location)
        for package, goal_location in self.goal_locations.items():
            # If the goal (at package goal_location) is already in the state, this package is done.
            if f"(at {package} {goal_location})" in state:
                 continue # Cost for this package is 0

            # Package is not yet on the ground at its goal location.
            # Calculate cost to get it there.

            current_effective_location = None
            if package in package_in_vehicle:
                # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle in current_locatables:
                    current_effective_location = current_locatables[vehicle]
                    # Cost is distance from vehicle location to goal + drop
                    if current_effective_location in self.shortest_paths and goal_location in self.shortest_paths[current_effective_location]:
                         dist = self.shortest_paths[current_effective_location][goal_location]
                         if dist == float('inf'):
                             # Goal location is unreachable from current effective location
                             return float('inf')
                         total_cost += dist + 1 # drive + drop
                    else:
                         # Should not happen if all locations are in graph, but handle defensively
                         return float('inf') # Unreachable goal
                else:
                    # Vehicle location unknown - indicates inconsistent state or parsing error
                    # Treat as unreachable for heuristic purposes
                    return float('inf')
            elif package in current_locatables:
                # Package is on the ground
                current_effective_location = current_locatables[package]
                # Cost is pick + distance from package location to goal + drop
                if current_effective_location in self.shortest_paths and goal_location in self.shortest_paths[current_effective_location]:
                     dist = self.shortest_paths[current_effective_location][goal_location]
                     if dist == float('inf'):
                         # Goal location is unreachable from current effective location
                         return float('inf')
                     total_cost += 1 + dist + 1 # pick + drive + drop
                else:
                     # Should not happen if all locations are in graph, but handle defensively
                     return float('inf')
            else:
                 # Package location unknown - indicates inconsistent state or parsing error
                 # Treat as unreachable for heuristic purposes
                 return float('inf')

        return total_cost
