from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings.
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions to move all packages
    to their goal locations. It considers the package's current state (on ground
    or in vehicle) and the shortest path distance to the goal location.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity constraints are ignored.
    - Any package on the ground can be picked up by *some* vehicle (cost 1).
    - The cost of driving a vehicle is the shortest path distance between locations.
    - When a package is on the ground, the drive cost is calculated from the package's location to the goal, ignoring the cost for a vehicle to reach the package.

    # Heuristic Initialization
    - Extracts goal locations for each package from the goal state.
    - Builds a graph of locations based on `road` facts and locations mentioned in the initial state and goal state.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that has a goal location defined and is not yet at its goal location:
    1. Determine the package's current state: Is it on the ground at a location `L_p`, or is it inside a vehicle `V` which is currently at location `L_v`?
    2. Determine the package's goal location `L_goal`.
    3. Calculate the estimated cost for this package:
       - If the package is on the ground at `L_p` (`L_p != L_goal`):
         It needs to be picked up (1 action), transported by a vehicle from `L_p` to `L_goal` (`dist(L_p, L_goal)` drive actions), and dropped at `L_goal` (1 action).
         Estimated cost = 1 (pick-up) + `dist(L_p, L_goal)` (drive) + 1 (drop) = `dist(L_p, L_goal) + 2`.
       - If the package is inside a vehicle `V` which is at `L_v`:
         It needs to be transported by the vehicle from `L_v` to `L_goal` (`dist(L_v, L_goal)` drive actions), and dropped at `L_goal` (1 action).
         Estimated cost = `dist(L_v, L_goal)` (drive) + 1 (drop).
    4. If the package is already at its goal location (`(at package goal_location)` is in the state), the cost for this package is 0.
    5. The total heuristic value is the sum of the estimated costs for all packages not at their goal. If any required location is unreachable, the heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the location graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Need initial state to find all locations

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the location graph from road facts.
        self.location_graph = {}
        all_locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Roads are bidirectional
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Add locations from initial state and goals to ensure all relevant locations are included
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 all_locations.add(loc)
                 self.location_graph.setdefault(loc, []) # Ensure all locations are keys even if no roads

        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 all_locations.add(loc)
                 self.location_graph.setdefault(loc, []) # Ensure all locations are keys even if no roads


        # Compute all-pairs shortest paths using BFS.
        self.dist = {}
        for start_node in all_locations:
            self.dist[start_node] = self._bfs(start_node, all_locations)

    def _bfs(self, start_node, all_nodes):
        """
        Performs BFS starting from start_node to find shortest distances
        to all other nodes in the graph.
        """
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Check if u is in graph keys, handles isolated locations correctly
            if u in self.location_graph:
                for v in self.location_graph[u]:
                    if distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        # This dictionary will map locatable objects (packages, vehicles)
        # to their location string, or packages to the vehicle string they are in.
        current_locatables_state = {}
        for fact in state:
            # (at ?x - locatable ?v - location)
            if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 current_locatables_state[obj] = loc
            # (in ?x - package ?v - vehicle)
            elif match(fact, "in", "*", "*"):
                 _, package, vehicle = get_parts(fact)
                 current_locatables_state[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through all packages that have a goal location defined.
        for package, goal_location in self.goal_locations.items():
            # If the package is not currently in the state (neither at nor in),
            # it cannot be moved. This might indicate an unsolvable problem or
            # a package that doesn't exist. Treat as unreachable goal.
            if package not in current_locatables_state:
                 return float('inf') # Cannot reach goal if package is missing

            current_state_info = current_locatables_state[package]

            # Check if the fact (at package goal_location) is in the state.
            # This is the most reliable way to check if the package is AT its goal.
            if f"(at {package} {goal_location})" in state:
                 # Package is at goal, cost is 0 for this package.
                 continue

            # Package is not at goal. Calculate cost based on its current state.
            # Case 1: Package is on the ground at some location current_loc_p
            # Check if the current_state_info is a location string.
            # We can check if it's a key in our distance map (which contains all locations).
            if current_state_info in self.dist: # current_state_info is a location
                current_loc_p = current_state_info

                # Cost: pick-up (1) + drive (dist) + drop (1)
                # We need the distance from current_loc_p to goal_location
                drive_cost = self.dist.get(current_loc_p, {}).get(goal_location, float('inf'))

                # If goal location is unreachable from package's current location,
                # heuristic is infinity.
                if drive_cost == float('inf'):
                    return float('inf')

                total_cost += 1 # pick-up
                total_cost += drive_cost # drive
                total_cost += 1 # drop

            # Case 2: Package is inside a vehicle v
            # Check if the current_state_info is a vehicle string.
            # Vehicle names typically don't overlap with location names.
            # We can check if it's NOT a location string based on our distance map keys.
            elif current_state_info not in self.dist: # current_state_info is likely a vehicle name
                vehicle_v = current_state_info

                # Find the current location of the vehicle
                # The vehicle's location must be in the state as (at vehicle_v current_loc_v)
                current_loc_v = None
                for fact in state:
                    if match(fact, "at", vehicle_v, "*"):
                        _, _, loc = get_parts(fact)
                        current_loc_v = loc
                        break # Found vehicle location

                # If vehicle location not found (shouldn't happen in valid states),
                # this indicates an inconsistent state representation. Return infinity.
                if current_loc_v is None:
                     return float('inf')

                # Cost: drive (dist) + drop (1)
                # We need the distance from current_loc_v to goal_location
                drive_cost = self.dist.get(current_loc_v, {}).get(goal_location, float('inf'))

                # If goal location is unreachable from vehicle's current location,
                # heuristic is infinity.
                if drive_cost == float('inf'):
                    return float('inf')

                total_cost += drive_cost # drive
                total_cost += 1 # drop

            # Else: current_state_info is neither a known location nor a vehicle name
            # This indicates an unexpected state format. Return infinity.
            else:
                 return float('inf')

        # The heuristic is 0 only if all packages are at their goal locations.
        # The loop structure ensures this: if a package is at its goal, we 'continue'
        # and add 0 cost for it. If the loop finishes with total_cost == 0,
        # it means all packages were at their goals.
        # If total_cost is > 0, it means at least one package was not at its goal.
        # The minimum possible non-zero cost is 1 (a package is in a vehicle at the goal, needs drop).
        # So, total_cost == 0 implies goal state.

        return total_cost
