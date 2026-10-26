# Required imports
import collections
# Assume Heuristic base class is available in heuristics/heuristic_base.py
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for malformed facts
        return []
    return fact[1:-1].split()

# BFS for shortest path on the road graph
def build_road_graph(static_facts):
    """Builds an adjacency list representation of the road network."""
    graph = collections.defaultdict(set)
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'road' and len(parts) == 3:
            l1, l2 = parts[1], parts[2]
            graph[l1].add(l2)
            graph[l2].add(l1) # Assuming roads are bidirectional
            locations.add(l1)
            locations.add(l2)
    return graph, list(locations)

def bfs_shortest_path(graph, start):
    """Computes shortest path distances from a start location using BFS."""
    # Use a large number for unreachable locations
    infinity = 1000000 # Sufficiently large for typical transport problems
    
    # Initialize distances for all locations known in the graph
    distances = {location: infinity for location in graph}
    
    # If the start location is not in the graph (e.g., isolated location not in road facts),
    # return distances dictionary where all are infinity.
    if start not in graph:
         return distances

    distances[start] = 0
    queue = collections.deque([start])

    while queue:
        current_loc = queue.popleft()
        
        # Explore neighbors
        for neighbor in graph.get(current_loc, []):
            if distances[neighbor] == infinity:
                distances[neighbor] = distances[current_loc] + 1
                queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(static_facts):
    """Computes shortest path distances between all pairs of locations."""
    graph, locations = build_road_graph(static_facts)
    all_distances = {}
    for start_loc in locations:
        all_distances[start_loc] = bfs_shortest_path(graph, start_loc)
    return all_distances

# Define the heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the individual costs. It considers whether a
    package is on the ground or inside a vehicle and uses precomputed shortest
    path distances between locations for drive actions.

    # Assumptions
    - Each package needs to reach a specific goal location specified by an `(at ?p ?l)` predicate in the goal.
    - Vehicle capacity is ignored.
    - The cost of getting a vehicle to a package's location (if on the ground)
      is implicitly included in the 'pick-up' action cost estimate, assuming a vehicle
      will eventually arrive.
    - The heuristic sums the estimated costs for each package independently,
      ignoring potential efficiencies from transporting multiple packages
      in one vehicle trip.
    - Roads are bidirectional.
    - All locations mentioned in state or goal facts are expected to be part of the road network defined by static facts, or reachable from it, for finite distance calculations in solvable problems.
    - Goal conditions are primarily `(at ?p ?l)` for packages. The heuristic returns 0 if all such package goals are met, even if other goal conditions exist.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations connected by roads from static facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Extract the current status (location on the ground or inside a vehicle) for every package by iterating through `(at ?p ?l)` and `(in ?p ?v)` facts in the state.
    2. Extract the current location for every vehicle by iterating through `(at ?v ?l)` facts in the state.
    3. Initialize the total heuristic cost to 0.
    4. For each package that has a goal location defined in the problem:
       a. Determine the package's goal location.
       b. Check if the package is currently located on the ground at its goal location (i.e., the fact `(at package goal_location)` is present in the state). If yes, the cost for this package is 0, and we move to the next package.
       c. If the package is not at its goal location on the ground:
          i. Find the package's current status from the extracted state information (`current_package_status`).
          ii. If the package's status is found and is 'at' a location `loc_p_current`:
              - The estimated cost for this package is the sum of:
                - 1 action for `pick-up`.
                - The shortest path distance (`dist`) between `loc_p_current` and `goal_location` for `drive` actions.
                - 1 action for `drop`.
              - Add `dist(loc_p_current, goal_location) + 2` to the total cost.
          iii. If the package's status is found and is 'in' a vehicle `v`:
              - Find the vehicle `v`'s current location, `loc_v_current`, from the extracted state information (`current_vehicle_locations`).
              - If the vehicle's location `loc_v_current` is the same as the package's `goal_location`:
                  - The estimated cost is 1 action for `drop`.
                  - Add 1 to the total cost.
              - If the vehicle's location `loc_v_current` is not the same as the package's `goal_location`:
                  - The estimated cost is the sum of:
                    - The shortest path distance (`dist`) between `loc_v_current` and `goal_location` for `drive` actions.
                    - 1 action for `drop`.
                  - Add `dist(loc_v_current, goal_location) + 1` to the total cost.
          iv. If the package's status is not found in the state (which indicates an issue with the state representation), add a large penalty.
    5. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and computing distances."""
        # The task object contains problem information like goals and static facts.
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        # Assuming goals are always of the form (at ?p ?l)
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if the goal fact is valid and is an 'at' predicate with 3 parts
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Note: This heuristic ignores other potential goal predicates if any exist.

        # Compute all-pairs shortest path distances between locations based on road facts.
        self.location_distances = compute_all_pairs_shortest_paths(static_facts)

        # Add any locations mentioned in goals but not in road facts to the distance map
        # so get_distance doesn't fail, although distances to/from them will be infinity.
        # This handles cases where a goal location might be isolated in the road graph.
        all_locations_in_graph = set(self.location_distances.keys())
        infinity = 1000000
        for loc in self.goal_locations.values():
             if loc not in all_locations_in_graph:
                  # Add the isolated location to the distance map, with infinite distance to/from others
                  self.location_distances[loc] = {other_loc: infinity for other_loc in all_locations_in_graph | {loc}}
                  # Also update existing locations to have infinite distance to this new location
                  for other_loc in all_locations_in_graph:
                       self.location_distances[other_loc][loc] = infinity


    def get_distance(self, loc1, loc2):
        """Safely get the precomputed distance between two locations."""
        # Return a large number if locations are not connected or not found
        infinity = 1000000
        
        # Check if both locations are in our distance map
        if loc1 not in self.location_distances or loc2 not in self.location_distances.get(loc1, {}):
             # This could happen if a location from state/goal wasn't in the road graph
             # or if loc1/loc2 are None/invalid.
             return infinity
             
        return self.location_distances[loc1].get(loc2, infinity)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # Extract current locations and package contents from the state.
        current_package_status = {} # package -> ('at', loc) or ('in', vehicle)
        current_vehicle_locations = {} # vehicle -> loc

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Simple check based on naming convention from examples
                # A more robust approach would use type information from task.objects if available
                if obj.startswith('p'): # Assuming objects starting with 'p' are packages
                     current_package_status[obj] = ('at', loc)
                elif obj.startswith('v'): # Assuming objects starting with 'v' are vehicles
                     current_vehicle_locations[obj] = loc
                # Add other locatable types if necessary based on domain definition if needed
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Assuming the first argument of 'in' is always a package
                current_package_status[package] = ('in', vehicle)
            # Ignore capacity facts and other predicates for this heuristic calculation

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package that has a goal location defined.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is currently satisfying its goal location predicate.
            # Assuming goal predicates are always (at ?p ?l).
            # A package satisfies (at p l) if the fact '(at p l)' is in the state.
            is_at_goal_location_on_ground = ('at', goal_location) in current_package_status.get(package, (None, None))

            if is_at_goal_location_on_ground:
                 # Package is already at the goal location on the ground. No cost for this package.
                 continue

            # If we reach here, the package is not yet at its goal location on the ground.
            # It might be elsewhere on the ground, or inside a vehicle.

            # Get the package's current status. If the package is not mentioned in state facts,
            # something is wrong with the state representation or problem definition.
            # Assuming valid states always include package locations/contents.
            if package not in current_package_status:
                 # This package is not in the state's 'at' or 'in' facts.
                 # This should not happen for locatable objects in a well-formed PDDL state.
                 # If it does, it might imply the package is unreachable or missing.
                 # Add a large penalty as it indicates an unexpected state.
                 # Using infinity from get_distance.
                 total_cost += self.get_distance(None, None) # Adds infinity
                 continue # Move to the next package

            status_type, status_obj = current_package_status[package]

            if status_type == 'at':
                # Package is on the ground at loc_p_current
                loc_p_current = status_obj
                
                # Cost estimate: pick-up (1) + drive (dist) + drop (1)
                # This assumes a vehicle can reach loc_p_current, pick up, drive to goal, and drop.
                drive_cost = self.get_distance(loc_p_current, goal_location)
                total_cost += drive_cost + 2

            elif status_type == 'in':
                # Package is inside a vehicle 'v'
                vehicle = status_obj
                
                # Find the vehicle's current location
                loc_v_current = current_vehicle_locations.get(vehicle)

                if loc_v_current is None:
                    # Vehicle location is unknown - shouldn't happen in a valid state
                    # Add a large penalty as it indicates an unexpected state.
                    total_cost += self.get_distance(None, None) # Adds infinity
                    continue # Move to the next package

                # Cost estimate: drive (dist) + drop (1)
                # This assumes the vehicle carrying the package will drive directly to the goal and drop it.
                drive_cost = self.get_distance(loc_v_current, goal_location)
                total_cost += drive_cost + 1

        # The heuristic is 0 if and only if total_cost is 0.
        # total_cost is 0 if and only if the loop finishes without adding any cost.
        # Cost is added only if a package is NOT at its goal location on the ground.
        # Therefore, total_cost is 0 iff all packages with goals are at their goal location on the ground.
        # This matches the typical goal structure (all packages at specific locations).
        # If the goal included other conditions, this heuristic could be 0 prematurely.
        # Assuming goals are solely package locations.

        return total_cost
