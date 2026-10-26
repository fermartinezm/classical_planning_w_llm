# from heuristics.heuristic_base import Heuristic # Assuming this is provided elsewhere
from fnmatch import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to compute shortest paths
def bfs(graph, start):
    """
    Computes shortest path distances from a start node in a graph.
    Args:
        graph: Adjacency list representation (dict: node -> list of neighbors).
        start: The starting node.
    Returns:
        A dictionary mapping each reachable node to its distance from the start node.
    """
    distances = {node: float('inf') for node in graph}
    if start not in graph:
        # Start node is not in the graph (e.g., an object name). Return empty distances.
        return {}

    distances[start] = 0
    queue = deque([start])

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic: # Inherit from Heuristic if base class is available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It sums the costs for each untightened goal nut, considering the travel needed
    to reach the nut's location, the cost to acquire a usable spanner (if not
    already carried), and the tightening action itself. Travel costs are estimated
    using shortest path distances on the location graph.

    # Assumptions
    - All nuts specified in the goal start in a 'loose' state.
    - A spanner becomes unusable after tightening one nut.
    - The man can carry multiple spanners.
    - The location graph defined by 'link' predicates is undirected (links are bidirectional).
    - The problem instance is solvable (enough spanners exist in total and locations are reachable).

    # Heuristic Initialization
    - Build the location graph from 'link' static facts.
    - Precompute shortest path distances between all pairs of locations using BFS.
    - Extract the set of goal nuts and their fixed locations from the goal and initial state.
    - Identify the name of the man object.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify which goal nuts are not yet tightened. Let this set be `UntightenedGoalNuts`.
    3. If `UntightenedGoalNuts` is empty, the heuristic is 0.
    4. Count the number of usable spanners the man is currently carrying (`N_carried`).
    5. Identify the usable spanners available at locations (`UsableSpannersAtLocs`) and store their locations.
    6. If the total number of available usable spanners (`N_carried + |UsableSpannersAtLocs|`) is less than the number of nuts needing tightening (`N_nuts = |UntightenedGoalNuts|`), the state is likely unsolvable; return a large value.
    7. Initialize the total heuristic cost to 0.
    8. Add the cost for the tightening actions: This is simply the number of nuts that need tightening (`N_nuts`).
    9. Add the estimated cost for man travel to reach the locations of the untightened nuts. This is estimated as the sum of shortest path distances from the man's current location to each *unique* location where an untightened nut is located.
    10. Add the estimated cost for acquiring additional usable spanners if needed.
        - Calculate how many additional spanners need to be picked up from locations: `N_acquire = max(0, N_nuts - N_carried)`.
        - If `N_acquire > 0`, calculate the shortest path distance from the man's current location to the location of each usable spanner on the ground.
        - Sort these distances and take the `N_acquire` smallest distances.
        - The cost to acquire these spanners is the sum of these `N_acquire` distances (travel) plus `N_acquire` (pickup actions). Add this to the total cost.
    11. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        object types, and precomputing shortest paths.
        """
        # Assuming task object has attributes: goals, static, initial_state, objects
        self.goals = task.goals
        self.static = task.static
        self.objects = task.objects

        # Find the man object name
        self.man_obj_name = None
        for obj_name, obj_type in self.objects:
            if obj_type == 'man':
                self.man_obj_name = obj_name
                break
        if self.man_obj_name is None:
             # This should not happen in valid PDDL problems for this domain
             print("Warning: Could not find object of type 'man' in task objects.")
             # Set a placeholder or handle error appropriately in a real system
             # For this heuristic, we'll assume a man object exists and is found.
             # If not found, subsequent steps relying on man_obj_name might fail.
             pass


        # Build the location graph from static 'link' facts
        self.location_graph = {}
        self.all_locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Assume links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Add any locations mentioned in initial state 'at' facts to ensure they are in the graph structure
        # This is important if the man, nuts, or spanners start at locations not connected by links
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 if loc not in self.all_locations:
                     self.all_locations.add(loc)
                     self.location_graph.setdefault(loc, []) # Add isolated location

        # Ensure all locations in the graph dict exist as keys, even if they have no links
        for loc in self.all_locations:
             self.location_graph.setdefault(loc, [])


        # Precompute shortest path distances from all locations
        self.all_pairs_distances = {}
        for start_loc in self.all_locations:
            self.all_pairs_distances[start_loc] = bfs(self.location_graph, start_loc)

        # Extract goal nuts and their locations (assuming nut locations are static)
        self.goal_nuts = set()
        self.nut_locations = {} # Map nut -> location

        # Find goal nuts
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

        # Find initial locations of all nuts (assuming they don't move)
        # We need the location of goal nuts.
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 obj_name, loc = get_parts(fact)[1:]
                 if obj_name in self.goal_nuts:
                     self.nut_locations[obj_name] = loc

        # Check if locations for all goal nuts were found
        if len(self.nut_locations) != len(self.goal_nuts):
             # This indicates an issue with the problem definition (goal nut location missing in init)
             # For heuristic purposes, we can't solve it if we don't know where the nut is.
             # A real planner might report an unsolvable problem. Heuristic returns large value.
             # We will check for missing nut locations in __call__ when accessing self.nut_locations.
             pass


    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, handling unreachable locations."""
        if loc1 not in self.all_pairs_distances or loc2 not in self.all_pairs_distances.get(loc1, {}):
             # loc1 is not a known location, or loc2 is not reachable from loc1
             return float('inf')
        return self.all_pairs_distances[loc1].get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify the man's current location.
        man_location = None
        carried_spanners = set()
        usable_spanners_at_locs = set() # Store as (spanner_name, location) tuples
        state_tightened_nuts = set()
        usable_spanners_in_state = set() # Track all usable spanners

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1:]
                if obj == self.man_obj_name:
                    man_location = loc
            elif parts[0] == "carrying":
                 carrier, spanner = parts[1:]
                 # Assuming carrier is the man
                 carried_spanners.add(spanner)
            elif parts[0] == "usable":
                 spanner = parts[1]
                 usable_spanners_in_state.add(spanner)
            elif parts[0] == "tightened":
                 nut = parts[1]
                 state_tightened_nuts.add(nut)

        if man_location is None:
             # Man's location not found - should not happen in valid states
             return 1000000 # Indicate problematic state

        # Identify usable spanners that are carried vs at locations
        usable_carried_spanners = {s for s in carried_spanners if s in usable_spanners_in_state}
        num_carried_usable = len(usable_carried_spanners)

        # Find usable spanners at locations
        for spanner in usable_spanners_in_state:
            if spanner not in carried_spanners: # If usable but not carried, it must be at a location
                 for fact in state:
                     if match(fact, "at", spanner, "*"):
                         usable_spanners_at_locs.add((spanner, get_parts(fact)[2]))
                         break # Found location, move to next usable spanner

        num_at_loc_usable = len(usable_spanners_at_locs)


        # 2. Identify untightened goal nuts.
        untightened_goal_nuts = {n for n in self.goal_nuts if n not in state_tightened_nuts}

        # 3. If empty, return 0.
        if not untightened_goal_nuts:
            return 0

        # 6. Check solvability based on spanner count.
        num_nuts_needed = len(untightened_goal_nuts)
        if num_carried_usable + num_at_loc_usable < num_nuts_needed:
             # Not enough usable spanners exist in the state to tighten all remaining nuts.
             return 1000000 # Use a large finite number

        # 7. Initialize cost.
        total_cost = 0

        # 8. Add cost for tightening actions.
        total_cost += num_nuts_needed

        # 9. Add estimated cost for man travel to nut locations.
        nut_locations_to_visit = set()
        for nut in untightened_goal_nuts:
            if nut not in self.nut_locations:
                 # Goal nut location not found in initial state - problem definition issue
                 return 1000000
            nut_locations_to_visit.add(self.nut_locations[nut])

        travel_to_nuts_cost = 0
        for loc in nut_locations_to_visit:
             dist = self.get_distance(man_location, loc)
             if dist == float('inf'):
                  # Cannot reach a nut location
                  return 1000000
             travel_to_nuts_cost += dist
        total_cost += travel_to_nuts_cost

        # 10. Add estimated cost for spanner acquisition travel and pickup.
        num_acquire = max(0, num_nuts_needed - num_carried_usable)

        if num_acquire > 0:
            spanner_loc_distances = []
            for spanner, loc in usable_spanners_at_locs:
                 dist = self.get_distance(man_location, loc)
                 if dist == float('inf'):
                      # Cannot reach a usable spanner location
                      return 1000000
                 spanner_loc_distances.append(dist)

            # We need num_acquire spanners from locations. We have num_at_loc_usable available.
            # The total count check (step 6) ensures num_carried_usable + num_at_loc_usable >= num_nuts_needed.
            # If num_acquire > 0, then num_nuts_needed > num_carried_usable.
            # This implies num_at_loc_usable >= num_nuts_needed - num_carried_usable = num_acquire.
            # So, there are always enough usable spanners at locations if num_acquire > 0 and step 6 passed.
            if len(spanner_loc_distances) < num_acquire:
                 # This indicates an inconsistency or error in logic/state representation
                 # Should not happen if total count check is correct
                 return 1000000 # Safety check

            spanner_loc_distances.sort()
            acquisition_travel_cost = sum(spanner_loc_distances[:num_acquire])
            acquisition_pickup_cost = num_acquire # 1 action per pickup

            total_cost += acquisition_travel_cost + acquisition_pickup_cost

        # 11. Return total estimated cost.
        return total_cost
