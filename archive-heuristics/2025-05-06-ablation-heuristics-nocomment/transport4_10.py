# Required imports
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts represented as strings
def get_parts(fact):
    """Extract parts from a PDDL fact string like '(predicate arg1 arg2)'."""
    # Remove surrounding parentheses and split by spaces
    # Handle empty fact string or malformed facts gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we have enough parts to match against args
    if len(parts) < len(args):
        return False
    # Use fnmatch for flexible matching (e.g., '*' wildcard)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the domain-dependent heuristic class
# Uncomment the next line if inheriting from a Heuristic base class
# class transportHeuristic(Heuristic):
class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum required
        actions (pick-up, drop, drive) for each package that is not yet
        at its goal location. The drive cost is estimated by the shortest
        path distance in the road network. Capacity constraints and vehicle
        availability are ignored for simplicity and efficiency.

    Assumptions:
        - The road network defined by (road l1 l2) facts is static and
          bidirectional (implied by typical transport domains and examples).
        - All locations relevant to package movement are part of a connected
          component in the road network.
        - Packages mentioned in the goal are the only ones that need to
          reach specific locations.
        - Packages are always either at a location or in a vehicle in valid states.
        - Vehicle locations are always specified if they exist in valid states.

    Heuristic Initialization:
        - Parses the goal facts to identify the target location for each package
          that needs to be at a specific location. Stores this in `self.goal_locations`.
        - Parses the static (road l1 l2) facts to build an adjacency list
          representation of the road network graph. Identifies all unique locations.
        - Computes all-pairs shortest paths on the road network graph using
          BFS starting from each identified location. Stores distances in `self.dist`
          as a dictionary mapping `(start_loc, end_loc)` tuples to distance.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Iterate through each package and its required goal location stored in
           `self.goal_locations`. These are the packages that need to be delivered.
        3. For the current package `p` and its goal location `loc_p_goal`:
           a. Check if the fact `(at p loc_p_goal)` is present in the current state.
           b. If it is, the package is already at its goal location. It requires
              no further actions related to its final delivery point, so add 0 to `h`
              and proceed to the next package.
           c. If the package is not at its goal location, it needs to be moved.
              Find the package's current status (at a location or in a vehicle)
              and its effective current location by examining the state facts:
              - Initialize `current_loc = None`, `is_in_vehicle = False`, `vehicle = None`.
              - Iterate through the state facts.
              - If a fact `(at p loc_p_current)` is found where `parts[0] == 'at'` and `parts[1] == package`:
                The package `p` is currently at `loc_p_current`. Set `current_loc = parts[2]`
                and `is_in_vehicle = False`. Set `found_status = True`. Break the search.
              - If a fact `(in p v_current)` is found where `parts[0] == 'in'` and `parts[1] == package`:
                The package `p` is currently inside vehicle `v_current`. Set `is_in_vehicle = True`
                and `vehicle = parts[2]`. Set `found_status = True`. Break the search.
           d. If the package status was found and it's in a vehicle (`is_in_vehicle` is True):
              - Find the location of this vehicle by searching for a fact
                `(at vehicle loc_v_current)` in the state. Set `current_loc = loc_v_current`.
           e. If the package's effective current location (`current_loc`) was successfully determined:
              - This package needs to be dropped at the goal location. Add 1
                to `h` for the `drop` action.
              - If the package is currently at a location (not in a vehicle), it
                first needs to be picked up. Add 1 to `h` for the `pick-up` action.
              - If the package's effective current location (`current_loc`) is
                different from its goal location (`loc_p_goal`), it needs to be
                transported. The cost of this transport is estimated by the
                shortest path distance between `current_loc` and `loc_p_goal`
                in the road network. Look up this distance in the precomputed
                `self.dist` table and add it to `h`. If the locations are
                disconnected (distance is infinity), this package goal is
                unreachable from its current location, contributing infinity
                to the heuristic (handled by `.get(..., float('inf'))`).
           f. If the package's status or location could not be determined (e.g., package not in state, vehicle not at location), this indicates a potentially malformed state. The heuristic will not add cost for this package in this case, which might be acceptable depending on how invalid states are handled by the search.

        4. After processing all packages in `self.goal_locations`, the total
           value of `h` is the estimated cost. Return `h`.
    """
    def __init__(self, task):
        # Store goal locations for packages
        self.goal_locations = {}
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            # We only care about goal facts specifying package locations
            if parts and parts[0] == 'at' and len(parts) == 3: # Ensure it's (at package location)
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location
            # Ignore other potential goal types or malformed facts

        # Build road graph and compute distances
        self.road_graph = {}
        locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'road' and len(parts) == 3: # Ensure it's (road loc1 loc2)
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                # Add bidirectional roads
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1)

        self.dist = {}
        # Compute shortest paths from every location to every other location using BFS
        for start_loc in locations:
            q = deque([start_loc])
            visited = {start_loc}
            distance = {start_loc: 0}

            while q:
                u = q.popleft()
                # Store distance from start_loc to u
                self.dist[(start_loc, u)] = distance[u]

                # Explore neighbors
                for v in self.road_graph.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        distance[v] = distance[u] + 1
                        q.append(v)

    def __call__(self, node):
        state = node.state
        h = 0

        # Iterate through packages that have a goal location defined
        for package, goal_loc in self.goal_locations.items():

            # Check if package is already at its goal location
            if f'(at {package} {goal_loc})' in state:
                continue # Package is delivered, contributes 0 to heuristic

            # Package is not at goal. Find its current status and location.
            current_loc = None
            is_in_vehicle = False
            vehicle = None

            # Search for package's current location/status in the state
            # A package is either 'at' a location or 'in' a vehicle
            found_status = False
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] == package:
                    current_loc = parts[2]
                    is_in_vehicle = False
                    found_status = True
                    break
                elif parts and parts[0] == 'in' and len(parts) == 3 and parts[1] == package:
                    is_in_vehicle = True
                    vehicle = parts[2]
                    found_status = True
                    break

            # If package status was found and it's in a vehicle, find its effective location
            if found_status and is_in_vehicle:
                 # Search for vehicle's location in the state
                 vehicle_loc_found = False
                 for v_fact in state:
                     v_parts = get_parts(v_fact)
                     if v_parts and v_parts[0] == 'at' and len(v_parts) == 3 and v_parts[1] == vehicle:
                         current_loc = v_parts[2]
                         vehicle_loc_found = True
                         break # Found vehicle location
                 # If vehicle location wasn't found, current_loc remains None.
                 # This case is handled below.

            # If package's effective current location was successfully determined
            if current_loc is not None:
                # Package needs to be dropped at the goal location
                h += 1 # Cost for drop action

                # If package is currently at a location (not in a vehicle), it needs to be picked up first
                if not is_in_vehicle:
                    h += 1 # Cost for pick-up action

                # If the package's effective current location is different from its goal location, it needs driving
                if current_loc != goal_loc:
                    # Add the shortest path distance as the estimated drive cost
                    # Use .get with float('inf') to handle potential disconnected locations
                    h += self.dist.get((current_loc, goal_loc), float('inf'))
            # else: # Package status or location could not be determined - potentially malformed state
            #     pass # The heuristic adds 0 cost for this package in this case

        return h
