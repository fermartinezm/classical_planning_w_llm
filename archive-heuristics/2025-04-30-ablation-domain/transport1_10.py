# Need to import Heuristic from the correct path in the actual environment
# from heuristics.heuristic_base import Heuristic
# Assuming the base class is available in the 'heuristics' package

from fnmatch import fnmatch
from collections import deque
import sys # Import sys for float('inf')

# Helper functions (copied from Logistics example, slightly adapted)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
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


# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic
# If not, define a dummy one for testing:
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the base class is not found
    # This allows the code structure to be correct even outside the specific planner environment
    # In the actual planner environment, this dummy class will be replaced by the real one.
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, independently. It sums
    these individual costs. The cost for a package includes loading, driving,
    and unloading actions. Driving cost is estimated by the shortest path
    distance between locations in the road network.

    # Assumptions
    - The goal is to move specific packages to specific locations, specified by (at package location) facts in the task's goal conditions.
    - Any vehicle can be used to transport any package (capacity constraints are ignored in the heuristic calculation for simplicity).
    - A suitable vehicle is always available at the required location when a package needs to be loaded or transported.
    - The cost of load, unload, and drive actions is 1.
    - The road network is undirected (if there's a road from A to B, there's one from B to A). The provided examples show this symmetry.
    - All locations relevant to packages and vehicles are part of the road network graph defined by 'road' facts or mentioned in goal facts.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task's goal conditions.
    - Builds a graph representation of the road network from 'road' facts found in the static information.
    - Collects all unique locations mentioned in road facts and goal facts.
    - Computes and stores the shortest path distance between all pairs of these locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current position of every object (packages and vehicles) that is mentioned in an 'at' or 'in' fact in the state. A package can be on the ground at a location ('at') or inside a vehicle ('in'). A vehicle is always at a location ('at'). Store these positions in a dictionary mapping object name to its position (which is either a location name or a vehicle name).
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location defined in the task's goals:
        a. Get the package's current position from the dictionary created in step 1. If the package is not found in the state's position facts, something is wrong with the state; return infinity.
        b. Determine the package's current "effective" location. If its current position is a location name (and is a known location in the road graph or goal locations), that's its effective location. If its current position is a vehicle name, find the location of that vehicle from the dictionary created in step 1; that vehicle's location (if known) is the package's effective location for transport purposes. If the effective location cannot be determined (e.g., object/vehicle not located or location not in graph), return infinity.
        c. If the package's effective location is the same as its goal location:
           - If the package is on the ground at the goal location, it contributes 0 to the heuristic.
           - If the package is inside a vehicle that is at the goal location, it needs to be unloaded (1 action). Add 1 to the total cost.
        d. If the package's effective location is different from its goal location:
           - Get the shortest path distance from the effective location to the goal location using the pre-computed distances. If there is no path (distance is infinity), return infinity as the state is likely unsolvable.
           - If the package is on the ground at the effective location: It needs to be loaded (1 action), transported (drive_cost actions), and unloaded (1 action). Add 1 + drive_cost + 1 to the total cost.
           - If the package is inside a vehicle at the effective location: It needs to be transported (drive_cost actions) and unloaded (1 action). Add drive_cost + 1 to the total cost.
    4. Return the total accumulated cost. If any required shortest path was infinite or an invalid state was detected, infinity would have been returned earlier.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for shortest path calculations.
        """
        self.goals = task.goals
        static_facts = task.static # Static facts include 'road' and 'capacity-predecessor'

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # We only care about (at package location) goals for packages
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                # Basic check to filter out non-package 'at' goals if any exist (e.g., robot location goal)
                # Assuming packages start with 'p' based on examples, vehicles with 'v'
                # This is a heuristic assumption based on instance examples.
                if obj.startswith('p'):
                     self.goal_locations[obj] = location

        # Build the road network graph and collect all location names.
        self.road_graph = {}
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                all_locations.add(loc1)
                all_locations.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph.setdefault(loc2, set()).add(loc1)

        # Add locations mentioned in goals if they are not in road facts.
        # This ensures all relevant locations are included in the shortest path calculation.
        for goal_loc in self.goal_locations.values():
             all_locations.add(goal_loc)

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_loc in all_locations:
             # BFS returns distances to all reachable nodes from start_loc
            self.shortest_paths[start_loc] = self._bfs(start_loc, all_locations)


    def _bfs(self, start_node, all_locations):
        """
        Performs BFS from a start node to find shortest distances to all reachable nodes.
        Returns a dictionary mapping reachable location to distance.
        """
        distances = {loc: float('inf') for loc in all_locations}
        # If start_node is not a known location (e.g., isolated location not in any road fact or goal),
        # its distances to others are all infinity, which is handled by the initial dict.
        if start_node not in all_locations:
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Get neighbors from the graph, handle locations with no roads
            neighbors = self.road_graph.get(current_loc, set())

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Map objects (packages and vehicles) to their current location or container.
        current_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "at" and len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    current_positions[obj] = loc
                elif parts[0] == "in" and len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    current_positions[package] = vehicle # Package is inside a vehicle

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            current_pos = current_positions.get(package)

            # If package is not in the state's position facts, something is wrong.
            if current_pos is None:
                 # This package is not mentioned in 'at' or 'in' facts.
                 # This indicates an issue with the state representation or problem definition.
                 # Return infinity as the state is likely invalid or unsolvable.
                 return float('inf')

            # Determine the package's effective location for transport
            effective_location = None
            if current_pos in self.shortest_paths: # Check if current_pos is a known location from the graph
                effective_location = current_pos # Package is on the ground
            else: # current_pos is likely a vehicle name
                vehicle = current_pos
                vehicle_loc = current_positions.get(vehicle) # Get the vehicle's location
                if vehicle_loc in self.shortest_paths: # Check if vehicle's location is a known location
                     effective_location = vehicle_loc # Package is in a vehicle at vehicle_loc
                # else: vehicle_loc is not a known location or vehicle has no location, effective_location remains None

            # If we couldn't determine an effective location (e.g., object/vehicle not located in the graph)
            if effective_location is None:
                 return float('inf') # Cannot estimate cost, state is likely invalid or unsolvable

            # If the package is already at its goal location (either on ground or in vehicle at goal)
            if effective_location == goal_location:
                 # If the package is IN a vehicle AT the goal, it still needs to be unloaded.
                 # If the package is AT the goal on the ground, cost is 0.
                 if current_pos in self.shortest_paths: # Package is AT goal on ground
                     # Cost is 0 for this package
                     pass # total_cost remains unchanged
                 else: # Package is IN vehicle AT goal
                     # Needs unload (1 action)
                     total_cost += 1
            else: # effective_location != goal_location
                # Package needs to move from effective_location to goal_location

                # Get the shortest path cost from the effective location to the goal location
                # Use .get() with default float('inf') to handle cases where effective_location
                # is not a start node in shortest_paths or goal_location is not reachable.
                drive_cost = self.shortest_paths.get(effective_location, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                     # Goal is unreachable from effective location in the road network
                     return float('inf') # Problem is likely unsolvable from this state

                if current_pos in self.shortest_paths: # Package is on the ground at effective_location
                    # Needs load (1) + drive (drive_cost) + unload (1)
                    total_cost += 1 + drive_cost + 1
                else: # Package is in a vehicle at effective_location
                    # Needs drive (drive_cost) + unload (1)
                    total_cost += drive_cost + 1

        # The heuristic is 0 if and only if all packages are at their goal locations
        # AND are not inside a vehicle at the goal (which requires an unload).
        # The logic above correctly adds 1 for the unload action in that specific case.
        # So, h=0 iff all packages are (at p loc_goal) on the ground.
        # This correctly identifies the goal state (which only contains (at p loc) facts)
        # as having a heuristic value of 0.

        return total_cost
