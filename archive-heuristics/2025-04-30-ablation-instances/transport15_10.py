from fnmatch import fnmatch
from collections import deque
import sys

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If running standalone or without the specific planner structure,
# you might need a dummy Heuristic base class definition like this:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass
# class Task: # Dummy Task class if needed for standalone testing
#     def __init__(self, name, facts, initial_state, goals, operators, static):
#         self.name = name
#         self.facts = facts
#         self.initial_state = initial_state
#         self.goals = goals
#         self.operators = operators
#         self.static = static
#     def goal_reached(self, state):
#         return self.goals <= state
# class Node: # Dummy Node class if needed for standalone testing
#      def __init__(self, state):
#          self.state = state


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string case defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    # Ensure we have the same number of parts as args, or args is shorter and matches the beginning
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It calculates the minimum actions needed
    for each package independently, based on its current state (on the ground
    or in a vehicle) and the shortest path distance to its goal location.

    # Assumptions
    - Each package needs to reach a specific goal location specified in the task goals.
    - The cost of actions is 1 (drive, pick-up, drop).
    - The cost of driving between two locations is the shortest path distance
      in the road network.
    - The heuristic ignores vehicle capacity constraints and the coordination
      of multiple packages or vehicles. It assumes a vehicle is available
      when needed for a package on the ground.
    - All packages relevant to the goal are present in the initial state.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static facts.
    - Computes all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is the goal state. If yes, return 0.
    2. Identify the current location or containment (in a vehicle) for every package
       that has a goal location.
    3. Identify the current location for every vehicle.
    4. Initialize the total heuristic cost to 0.
    5. For each package whose goal location is known:
        a. Determine the package's current status: on the ground at `current_loc`
           or inside a vehicle `vehicle` which is at `current_loc`.
        b. If the package is on the ground at `current_loc`:
           - If `current_loc` is the goal location, cost for this package is 0.
           - If `current_loc` is different from the goal location:
             - It needs to be picked up, transported, and dropped.
             - Estimate cost: 1 (pick-up) + shortest_distance(current_location, goal_location) + 1 (drop).
             - Add this cost to the total. If the goal is unreachable, the state is likely unsolvable, return infinity.
        c. If the package is inside a vehicle `vehicle`:
           - Find the current location of the vehicle, `vehicle_loc`.
           - If `vehicle_loc` is the package's goal location:
             - It needs to be dropped.
             - Estimate cost: 1 (drop).
             - Add this cost to the total.
           - If `vehicle_loc` is different from the package's goal location:
             - It needs to be transported (by the vehicle) and dropped.
             - Estimate cost: shortest_distance(vehicle_loc, goal_location) + 1 (drop).
             - Add this cost to the total. If the goal is unreachable, return infinity.
    6. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the
        road network, and computing shortest path distances.
        """
        self.task = task # Store task to check goal_reached
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3:
                    _, package, location = parts
                    self.package_goals[package] = location

        # 2. Build the road network graph and collect all locations
        self.all_locations = set()
        roads = {} # Adjacency list: location -> [neighbor1, neighbor2, ...]

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, loc1, loc2 = parts
                    self.all_locations.add(loc1)
                    self.all_locations.add(loc2)
                    roads.setdefault(loc1, []).append(loc2)
                    roads.setdefault(loc2, []).append(loc1) # Roads are bidirectional

        # 3. Compute all-pairs shortest path distances using BFS
        self.distances = {} # (start_loc, end_loc) -> distance

        for start_loc in self.all_locations:
            # Run BFS from start_loc
            dist_from_start = {loc: float('inf') for loc in self.all_locations}
            dist_from_start[start_loc] = 0
            queue = deque([start_loc])

            while queue:
                curr_loc = queue.popleft()

                # Check if curr_loc is a valid key in roads
                if curr_loc in roads:
                    for neighbor in roads[curr_loc]:
                        if dist_from_start[neighbor] == float('inf'):
                            dist_from_start[neighbor] = dist_from_start[curr_loc] + 1
                            queue.append(neighbor)

            # Store distances from start_loc to all other locations
            for end_loc in self.all_locations:
                 self.distances[(start_loc, end_loc)] = dist_from_start[end_loc]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if goal is reached first
        if self.task.goal_reached(state):
             return 0

        # 2. & 3. Track where packages and vehicles are currently located or contained.
        package_current_status = {} # package -> ('at', loc) or ('in', vehicle)
        vehicle_current_loc = {} # vehicle -> loc

        # Infer object types based on predicates they appear in (simple heuristic assumption)
        # A more robust way would parse domain object types.
        # We assume objects starting with 'p' are packages, 'v' are vehicles.
        possible_packages = set(p for p, _ in self.package_goals.items())
        possible_vehicles = set() # Collect vehicles from state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1:]
                if obj in possible_packages or obj.startswith('p'): # Prioritize goal packages, then infer
                    package_current_status[obj] = ('at', loc)
                elif obj.startswith('v'): # Infer vehicles
                    possible_vehicles.add(obj)
                    vehicle_current_loc[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1:]
                possible_packages.add(package) # Add package if seen in 'in'
                possible_vehicles.add(vehicle) # Add vehicle if seen in 'in'
                package_current_status[package] = ('in', vehicle)
            # Ignore capacity facts for this heuristic's calculation logic

        total_cost = 0  # Initialize action cost counter.

        # 4. & 5. Calculate cost for each package that is not at its goal
        for package, goal_location in self.package_goals.items():
            # If a goal package is not found in the current state facts,
            # it implies an issue with the state representation or domain.
            # In a valid state, a goal package should always be locatable.
            if package not in package_current_status:
                 # This case indicates an unexpected state. Treat as unsolvable.
                 # print(f"Warning: Goal package {package} not found in state.") # Debugging
                 return float('inf')

            status, current_loc_or_vehicle = package_current_status[package]

            if status == 'at':
                current_location = current_loc_or_vehicle
                # Check if already at goal
                if current_location == goal_location:
                    continue # Package is at goal, cost is 0 for this package

                # Package is on the ground, not at goal. Needs pick-up, drive, drop.
                # Estimated cost: 1 (pick) + distance + 1 (drop)
                dist = self.distances.get((current_location, goal_location), float('inf'))
                if dist == float('inf'):
                    # Cannot reach goal location from current location
                    # print(f"Warning: Package {package} at {current_location} cannot reach goal {goal_location}.") # Debugging
                    return float('inf') # State is likely unsolvable

                total_cost += 2 + dist

            elif status == 'in':
                vehicle = current_loc_or_vehicle
                # Find the location of the vehicle carrying the package
                if vehicle not in vehicle_current_loc:
                    # Vehicle location unknown? Should not happen in valid states.
                    # Treat as unsolvable.
                    # print(f"Warning: Vehicle {vehicle} carrying {package} not found at any location.") # Debugging
                    return float('inf')

                current_location = vehicle_current_loc[vehicle]

                # Check if vehicle is at the goal location for the package
                if current_location == goal_location:
                    # Package is in vehicle at goal location. Needs drop.
                    # Estimated cost: 1 (drop)
                    total_cost += 1
                else:
                    # Package is in vehicle, vehicle not at goal location. Needs drive, drop.
                    # Estimated cost: distance + 1 (drop)
                    dist = self.distances.get((current_location, goal_location), float('inf'))
                    if dist == float('inf'):
                        # Cannot reach goal location from vehicle's current location
                        # print(f"Warning: Vehicle {vehicle} at {current_location} cannot reach package {package}'s goal {goal_location}.") # Debugging
                        return float('inf') # State is likely unsolvable
                    total_cost += dist + 1

        # 6. Return the total accumulated cost.
        return total_cost

