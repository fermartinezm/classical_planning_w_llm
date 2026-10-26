from fnmatch import fnmatch
from collections import deque
import sys # Used for a large number proxy for infinity

# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Define a large number to represent unreachable locations
UNREACHABLE_COST = sys.maxsize // 2 # Use a large integer

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match_fact(fact, predicate, *args):
    """
    Check if a PDDL fact matches a given predicate and arguments pattern.
    - `fact`: The complete fact string, e.g., "(at obj loc)".
    - `predicate`: The expected predicate name.
    - `args`: The expected arguments pattern (wildcards `*` allowed).
    Returns `True` if the fact matches, else `False`.
    """
    parts = get_parts(fact)
    if not parts:
        return False
    if parts[0] != predicate:
        return False
    # Check arguments
    fact_args = parts[1:]
    if len(fact_args) != len(args):
        return False
    return all(fnmatch(fact_arg, pattern_arg) for fact_arg, pattern_arg in zip(fact_args, args))


class transportHeuristic: # Inherit from Heuristic if base class is provided
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, summing the costs for all
    packages that are not yet at their destination. It calculates the cost for
    each package independently, ignoring vehicle capacity constraints and
    vehicle availability/coordination. The cost for a package includes pick-up,
    driving distance, and drop actions.

    # Assumptions:
    - Roads are bidirectional.
    - All locations mentioned in road facts are part of a single connected component
      or relevant locations (initial package/vehicle locations, goal locations)
      are within connected components that allow reaching goals. Unreachable goals
      are assigned a very high cost.
    - Vehicle capacity and availability are ignored; a vehicle is assumed to be
      available with sufficient capacity whenever a package needs to be picked up
      or dropped.
    - The cost of driving between two locations is the shortest path distance
      (number of road segments) in the road network. Each drive action covers
      one road segment.
    - Pick-up and drop actions each cost 1.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of the road network from static facts.
    - Computes all-pairs shortest path distances between all locations in the graph
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package. A package can be either
       on the ground at a location `(at package location)` or inside a vehicle
       `(in package vehicle)`, in which case its effective location is the
       location of the vehicle `(at vehicle location)`.
    2. For each package `p` whose goal location `goal_l` is known:
       a. Check if the package is already at its goal location (`(at p goal_l)` is true).
          If yes, the cost for this package is 0.
       b. If the package is not at its goal:
          i. Determine the package's current effective location (`current_l`).
             - Iterate through state facts to find `(at p current_l)` or `(in p v)`
               and then find `(at v vehicle_l)`.
          ii. Calculate the minimum actions required for this package:
              - If the package is on the ground at `current_l`:
                Cost = 1 (pick-up) + distance(`current_l`, `goal_l`) + 1 (drop).
              - If the package is inside a vehicle at `current_l`:
                Cost = distance(`current_l`, `goal_l`) + 1 (drop).
              - The distance is the precomputed shortest path distance in the road network.
              - If `goal_l` is unreachable from `current_l`, the distance is `UNREACHABLE_COST`.
    3. The total heuristic value is the sum of the costs calculated for each package
       that is not yet at its goal location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network graph for distance calculations.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are always (at package location)
            if match_fact(goal, "at", "*", "*"):
                 package, location = get_parts(goal)[1:]
                 self.goal_locations[package] = location

        # Build the road network graph and compute distances.
        self.road_graph = {}
        locations = set()

        for fact in static_facts:
            if match_fact(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming bidirectional roads

        self.locations = list(locations) # Store locations for easy iteration
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        in the road network graph using BFS.
        Returns a dictionary: {(loc1, loc2): distance}.
        Assigns UNREACHABLE_COST if no path exists.
        """
        distances = {}
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            distances[(start_node, start_node)] = 0

            while q:
                current_node, dist = q.popleft()

                if current_node in self.road_graph:
                    for neighbor in self.road_graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[(start_node, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        # Fill in unreachable pairs with UNREACHABLE_COST
        for l1 in self.locations:
            for l2 in self.locations:
                 if (l1, l2) not in distances:
                     distances[(l1, l2)] = UNREACHABLE_COST

        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        current_package_state = {} # package -> location or vehicle
        vehicle_locations = {}     # vehicle -> location

        # First pass to find vehicle locations
        for fact in state:
            if match_fact(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Assuming objects starting with 'v' are vehicles
                if obj.startswith('v'):
                    vehicle_locations[obj] = loc

        # Second pass to find package states (at location or in vehicle)
        for fact in state:
             if match_fact(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Assuming objects starting with 'p' are packages
                 if obj.startswith('p'):
                      current_package_state[obj] = loc # Package is on the ground

             elif match_fact(fact, "in", "*", "*"):
                 package, vehicle = get_parts(fact)[1:]
                 # Assuming 'in' applies to packages in vehicles
                 if package.startswith('p') and vehicle.startswith('v'):
                      current_package_state[package] = vehicle # Package is inside a vehicle


        total_cost = 0  # Initialize action cost counter.

        # Consider packages that need to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if package is in the current state representation
            if package not in current_package_state:
                 # This package's state is unknown, which is unexpected for a well-formed problem.
                 # Assign a large penalty to guide search away from such states.
                 total_cost += UNREACHABLE_COST
                 continue

            # Check if the package is already at its goal location on the ground
            # The goal is typically (at package location)
            is_at_goal_on_ground = False
            package_state_value = current_package_state[package]

            # If the package state value is a location and matches the goal location
            if package_state_value == goal_location:
                 # Need to confirm it's on the ground, not just that the vehicle it's in is at the goal location
                 # If the state value is a location string, it means it's on the ground.
                 if package_state_value in self.locations: # Check if the state value is a known location
                     is_at_goal_on_ground = True

            if is_at_goal_on_ground:
                # Package is already at the goal location on the ground. Cost is 0 for this package.
                continue

            # Package is not at the goal location on the ground. Calculate cost.
            cost_for_package = 0

            if package_state_value in self.locations:
                # Package is on the ground at package_state_value
                current_l = package_state_value
                # Cost = Pick-up + Drive + Drop
                # Need distance from current_l to goal_location
                drive_cost = self.distances.get((current_l, goal_location), UNREACHABLE_COST)

                cost_for_package = 1 + drive_cost + 1 # Pick-up + Drive + Drop

            elif package_state_value.startswith('v'):
                # Package is inside a vehicle
                vehicle = package_state_value
                if vehicle in vehicle_locations:
                    current_l = vehicle_locations[vehicle]
                    # Cost = Drive + Drop
                    # Need distance from current_l (vehicle location) to goal_location
                    drive_cost = self.distances.get((current_l, goal_location), UNREACHABLE_COST)

                    cost_for_package = drive_cost + 1 # Drive + Drop
                else:
                    # Vehicle containing package is not at any location? Unexpected state.
                    # Assign a large penalty.
                    cost_for_package = UNREACHABLE_COST

            else:
                 # Package state is something unexpected (not a location, not a vehicle)
                 # Assign a large penalty.
                 cost_for_package = UNREACHABLE_COST

            total_cost += cost_for_package

        # The heuristic should be 0 only for goal states.
        # The current sum is 0 if all packages are at their goal location on the ground.
        # This matches the typical goal structure (all packages at specific locations).
        # If the goal involved packages being *in* vehicles, this would need adjustment,
        # but based on the examples, the goal is packages on the ground.

        return total_cost

