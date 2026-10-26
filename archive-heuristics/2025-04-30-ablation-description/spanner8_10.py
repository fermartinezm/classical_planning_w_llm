from fnmatch import fnmatch
from collections import deque
import sys # Used for float('inf')

# Assume Heuristic base class is available at heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to find shortest path distances
def bfs_shortest_path(graph, start, end=None):
    """
    Performs BFS to find shortest path distance from start.
    If end is None, returns dictionary of distances to all reachable nodes.
    If end is specified, returns distance to end, or float('inf') if unreachable.
    """
    # Handle case where start location is not in the graph keys (e.g., an object location not linked)
    # If start and end are the same, distance is 0. Otherwise, unreachable from this start.
    if start == end:
        return 0
    if start not in graph:
         return float('inf') # Cannot start BFS from a non-existent node in the graph

    distances = {start: 0}
    queue = deque([start])
    visited = {start}

    while queue:
        current_loc = queue.popleft()

        if end is not None and current_loc == end:
            return distances[current_loc]

        # Check if current_loc has neighbors in the graph
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)

    if end is not None:
        return float('inf') # End was not reached
    return distances # Return all distances

# Define the heuristic class, inheriting from Heuristic base if available
# class spannerHeuristic(Heuristic):
class spannerHeuristic: # Use this if Heuristic base is not provided in the execution environment
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the minimum number of actions required to tighten all
    loose nuts specified in the goal. It considers the cost to move the man to
    each nut's location and the cost to ensure the man has a usable spanner
    at that location, plus the cost of the tighten action itself.

    # Assumptions
    - Nut locations are static and can be found in the initial state.
    - There is exactly one man object in the domain, identifiable by name (e.g., 'bob')
      or by being the only locatable object that is not a spanner or nut based on naming convention.
    - Spanners and nuts are identifiable by name prefixes ('spanner', 'nut').
    - The 'link' predicates define an undirected graph of locations.
    - The heuristic assumes solvability if usable spanners exist somewhere.
      If a necessary location is unreachable or no usable spanners are available
      when needed, the heuristic returns a large value (infinity).

    # Heuristic Initialization
    - Extracts the set of goal nuts from the task goals.
    - Builds an undirected graph of locations based on 'link' predicates in static facts.
    - Records the initial locations of all nuts from the initial state, assuming
      these locations are static throughout the problem.
    - Identifies the name of the man object based on initial 'at' predicates and naming conventions.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of estimated costs for each loose nut
    that needs to be tightened according to the goal.

    For each nut `N` that is a goal and is currently loose:
    1.  Find the current location of the man (`M_loc`).
    2.  Find the location of the nut (`L_N`).
    3.  Determine if the man is currently carrying a usable spanner.
    4.  Identify the locations of all usable spanners that are currently on the ground.
    5.  Calculate the estimated cost to get the man to `L_N` *while* possessing a usable spanner.
        -   If the man is already carrying a usable spanner: The cost is the shortest path distance from `M_loc` to `L_N`.
        -   If the man is NOT carrying a usable spanner: He must acquire one. The most efficient way is to go to a usable spanner's location (`L_{S_u}`), pick it up, and then go to `L_N`. The cost for this sequence is `dist(M_loc, L_{S_u}) + 1 (pickup) + dist(L_{S_u}, L_N)`. The heuristic takes the minimum such cost over all available usable spanners on the ground. If no usable spanners are available, this part of the cost is considered infinite (or a large value), indicating an unsolvable state.
    6.  Add the calculated cost from step 5 to the total heuristic value.
    7.  Add 1 to the total heuristic value for the final `tighten_nut` action for this nut.

    The total heuristic is the sum of these costs for all loose goal nuts.
    If all goal nuts are already tightened, the heuristic is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # Assuming task object has attributes: goals, static, initial_state
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Extract goal nuts (objects that need to be tightened)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Build location graph from static links
        self.location_graph = {}
        all_locations_in_links = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1)
                all_locations_in_links.add(l1)
                all_locations_in_links.add(l2)

        # Store initial locations of nuts (assuming they are static)
        self.initial_nut_locations = {}
        # Identify all locations present in the initial state
        initial_locations_set = set()
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                initial_locations_set.add(loc)
                if obj.startswith('nut'): # Assuming nuts start with 'nut'
                     self.initial_nut_locations[obj] = loc

        # Add all locations from initial state to the graph keys, even if they have no links
        # This ensures BFS can be called with any location mentioned in the initial state
        for loc in initial_locations_set:
             self.location_graph.setdefault(loc, [])

        # Identify the man object name
        self.man_name = None
        # Look for an object that is 'at' a location and is not a spanner or nut by naming convention
        man_candidates = []
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Simple heuristic: Assume objects not starting with 'spanner' or 'nut' are potential men
                if not obj.startswith('spanner') and not obj.startswith('nut'):
                     man_candidates.append(obj)
        # Assume there is exactly one man object and it's the first candidate found
        if man_candidates:
            self.man_name = man_candidates[0]
        # else: self.man_name remains None. This case should ideally not happen in valid problems.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If the goal is already reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        # Find man's current location
        man_loc = None
        if self.man_name:
            for fact in state:
                if match(fact, "at", self.man_name, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        if man_loc is None:
             # Man's location not found in state (should not happen in valid states)
             # Or man_name was not identified in init
             return float('inf') # Indicate unsolvable or error state

        # Precompute distances from man's current location to all reachable locations
        # This BFS is done once per heuristic call
        distances_from_man = bfs_shortest_path(self.location_graph, man_loc)

        # Check if man is currently carrying a usable spanner
        carrying_usable_spanner = False
        carried_spanner_name = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner_name = get_parts(fact)[2]
                # Check if this carried spanner is usable in the current state
                if f"(usable {carried_spanner_name})" in state:
                    carrying_usable_spanner = True
                break # Assuming man carries at most one spanner

        # Find usable spanners on the ground and their locations
        usable_spanners_on_ground_locs = []
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Check if the object is a spanner, is usable, and is not the one the man is carrying (if any)
                if obj.startswith('spanner') and f"(usable {obj})" in state and obj != carried_spanner_name:
                    usable_spanners_on_ground_locs.append(loc)

        total_heuristic = 0
        large_value = float('inf') # Use infinity for unreachable/unsolvable

        # Iterate over goal nuts that are still loose
        for nut_name in self.goal_nuts:
            # Check if nut is already tightened
            if f"(tightened {nut_name})" in state:
                continue # This nut is already done

            # Nut is loose and needs tightening. Find its location.
            # Assuming nut locations are static and found in initial_state
            nut_loc = self.initial_nut_locations.get(nut_name)
            if nut_loc is None:
                 # Nut location not found (problem setup issue?)
                 return large_value # Indicate problem

            # Calculate cost to get the man to the nut's location with a usable spanner
            cost_to_reach_nut_with_spanner = 0

            if carrying_usable_spanner:
                # Man has a spanner, just needs to walk to the nut
                dist_to_nut = distances_from_man.get(nut_loc, large_value)
                if dist_to_nut == large_value:
                    return large_value # Cannot reach nut location
                cost_to_reach_nut_with_spanner = dist_to_nut
            else:
                # Man needs to get a usable spanner first
                if not usable_spanners_on_ground_locs:
                    # No usable spanners available on the ground
                    return large_value # Unsolvable state

                min_path_via_spanner = large_value
                for spanner_loc in usable_spanners_on_ground_locs:
                    # Cost is travel from man to spanner + pickup (1) + travel from spanner to nut
                    dist_m_to_spanner = distances_from_man.get(spanner_loc, large_value)
                    if dist_m_to_spanner == large_value:
                        continue # Cannot reach this spanner location from man's current location

                    # Need distance from spanner_loc to nut_loc. Run BFS from spanner_loc.
                    # This BFS is run for each candidate spanner location.
                    dist_spanner_to_nut = bfs_shortest_path(self.location_graph, spanner_loc, nut_loc)
                    if dist_spanner_to_nut == large_value:
                        continue # Cannot reach nut location from spanner location

                    path_cost = dist_m_to_spanner + 1 + dist_spanner_to_nut # Travel1 + Pickup + Travel2
                    min_path_via_spanner = min(min_path_via_spanner, path_cost)

                if min_path_via_spanner == large_value:
                    # Cannot reach any usable spanner and then the nut
                    return large_value

                cost_to_reach_nut_with_spanner = min_path_via_spanner

            # Add the cost to get man+spanner to nut location, plus the tighten action cost (1)
            total_heuristic += cost_to_reach_nut_with_spanner + 1

        return total_heuristic
