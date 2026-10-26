from fnmatch import fnmatch
from collections import deque
import sys

# Assume Heuristic base class is available and provides the interface:
# class Heuristic:
#     def __init__(self, task): ...
#     def __call__(self, node): ...

# If the base class is not available, you might need to define a minimal one
# or adjust the class definition depending on the framework.
# For this response, we assume the Heuristic base class exists and provides
# the expected __init__ and __call__ methods, and the Task object structure.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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


class transportHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package that
    is not yet at its destination. The cost for a package is estimated based
    on whether it needs to be picked up, the shortest path distance the vehicle
    needs to drive, and dropping the package.

    # Assumptions
    - Each action (drive, pick-up, drop) costs 1.
    - Capacity constraints of vehicles are ignored.
    - Vehicle availability is ignored (assumes a suitable vehicle is available
      when a package needs to be picked up or transported).
    - Roads are bidirectional.
    - All locations mentioned in initial state, goals, or road facts are part
      of the transportation network.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a graph representing the road network based on `road` facts.
    - Computes all-pairs shortest path distances between all locations using BFS.
    - Identifies vehicles based on static `capacity` facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, the heuristic is 0.
    2. Identify the current location of every locatable object (`at` predicate)
       and which package is inside which vehicle (`in` predicate).
    3. Initialize the total heuristic cost to 0.
    4. For each package that has a specified goal location:
       a. Determine the package's current status: Is it on the ground at a
          location, or is it inside a vehicle?
       b. If the package is already at its goal location on the ground, it
          requires no further actions for this package. Continue to the next package.
       c. If the package is on the ground at a location *different* from its goal:
          - It needs to be picked up (1 action).
          - It needs to be transported by a vehicle from its current location
            to the goal location. The estimated cost is the shortest path
            distance between these locations (number of drive actions).
          - It needs to be dropped at the goal location (1 action).
          - Add 1 + shortest_distance + 1 to the total cost.
       d. If the package is inside a vehicle:
          - Find the current location of that vehicle.
          - If the vehicle is at the package's goal location:
            - The package only needs to be dropped (1 action).
            - Add 1 to the total cost.
          - If the vehicle is at a location *different* from the package's goal:
            - The vehicle needs to drive from its current location to the goal
              location. The estimated cost is the shortest path distance
              (number of drive actions).
            - The package needs to be dropped at the goal location (1 action).
            - Add shortest_distance + 1 to the total cost.
       e. If the package's current status or location is unknown or invalid,
          return infinity, indicating a likely unreachable state.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, computing shortest paths, and identifying vehicles.
        """
        self.goals = task.goals  # Goal conditions.
        self.static_facts = task.static  # Facts that are not affected by actions.

        # Extract goal locations for packages
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.package_goals[package] = location

        # Build the road network graph and collect all relevant locations
        self.road_graph = {}
        all_locations = set()

        # Add locations from road facts
        for fact in self.static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                all_locations.add(l1)
                all_locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Roads are bidirectional

        # Add locations from initial state and goal state 'at' facts
        for fact in task.initial_state | self.goals:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 all_locations.add(loc)

        # Ensure all collected locations are keys in the graph dictionary
        for loc in all_locations:
             self.road_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths using BFS from each location
        self.distances = {}
        for start_loc in all_locations:
            self.distances[start_loc] = self._bfs(start_loc, self.road_graph)

        # Identify vehicles based on static capacity facts
        self.vehicles = set()
        for fact in self.static_facts:
             if match(fact, "capacity", "*", "*"):
                 _, vehicle, _ = get_parts(fact)
                 self.vehicles.add(vehicle)

    def _bfs(self, start_node, graph):
        """Performs BFS to find shortest distances from start_node to all reachable nodes."""
        distances = {node: float('inf') for node in graph}
        if start_node in distances: # Ensure start_node is a known location
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current = queue.popleft()
                current_dist = distances[current]

                # Check if current node has neighbors in the graph
                if current in graph:
                    for neighbor in graph[current]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Retrieves the pre-calculated shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Check if both locations are in our pre-calculated distances map
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # This indicates one or both locations were not part of the
             # road network or initial/goal states, or they are disconnected.
             # Return infinity as they are unreachable from each other.
             return float('inf')

        return self.distances[loc1].get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Map current locations of packages and vehicles
        package_current_status = {} # {package: ('at', location) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle: location}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                obj, loc = args
                if obj in self.vehicles:
                    vehicle_locations[obj] = loc
                # Check if the object is a package we care about (i.e., has a goal)
                if obj in self.package_goals:
                     package_current_status[obj] = ('at', loc)
            elif predicate == "in" and len(args) == 2:
                package, vehicle = args
                # Ensure it's a package we care about
                if package in self.package_goals:
                     package_current_status[package] = ('in', vehicle)

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # Check if package is already at goal location on the ground
            if package in package_current_status and package_current_status[package] == ('at', goal_location):
                 continue # Package is at goal, cost is 0 for this package

            # Package is not at goal. Estimate cost.
            status = package_current_status.get(package)

            if status is None:
                 # Package is not found in the state facts (neither at nor in).
                 # This is likely an invalid state or the package doesn't exist.
                 # Assign a very high cost (infinity) as it indicates an unreachable goal state for this package.
                 return float('inf')

            status_type, current_pos = status

            if status_type == 'in':
                # Package is inside a vehicle
                vehicle = current_pos
                vehicle_loc = vehicle_locations.get(vehicle)

                if vehicle_loc is None:
                    # Vehicle location unknown? Invalid state? Assign infinity.
                    return float('inf')

                # Package is in vehicle at vehicle_loc
                if vehicle_loc == goal_location:
                    # Vehicle is at goal location, just need to drop
                    total_cost += 1 # drop action
                else:
                    # Vehicle needs to drive, then drop
                    dist = self.get_distance(vehicle_loc, goal_location)
                    if dist == float('inf'):
                         # Unreachable goal location for this package
                         return float('inf')
                    total_cost += dist # drive actions
                    total_cost += 1 # drop action

            elif status_type == 'at':
                # Package is on the ground at current_pos (which is a location)
                package_loc = current_pos

                # Needs pick-up, drive, drop
                dist = self.get_distance(package_loc, goal_location)
                if dist == float('inf'):
                     # Unreachable goal location for this package
                     return float('inf')

                total_cost += 1 # pick-up action
                total_cost += dist # drive actions
                total_cost += 1 # drop action
            else:
                 # Unknown status type - should not happen with current logic, but defensive
                 return float('inf')


        return total_cost

