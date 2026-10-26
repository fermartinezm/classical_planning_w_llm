import collections
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists in the environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the base is not found
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


# Helper functions for parsing PDDL facts represented as strings
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
    # Check if the number of parts matches the number of args, considering trailing wildcards
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
         return False
    # Check if there are extra parts in the fact not covered by args (unless last arg is wildcard)
    if len(parts) > len(args) and args and args[-1] != '*':
         return False
    return True


# BFS implementation to find shortest paths in the location graph
def bfs_shortest_paths(graph, start_node):
    """Computes shortest path distances from start_node to all other nodes in the graph."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph
        if current_node not in graph:
             continue

        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It sums the estimated number of tighten actions, pickup actions, and walk actions.
    The walk cost is estimated as the sum of shortest path distances from the man's
    current location to each required location (nut locations and locations of
    needed ground spanners). This heuristic is non-admissible but aims to guide
    a greedy best-first search efficiently.

    # Assumptions:
    - The goal is to tighten a specific set of nuts, indicated by `(tightened nut)` facts in the goal.
    - A spanner is consumed (becomes unusable) after tightening one nut.
    - A man can carry multiple spanners simultaneously.
    - Links between locations defined by `(link l1 l2)` facts are bidirectional for walking.
    - All locations mentioned in the state or goals are part of the linked graph,
      or unreachable required locations result in an infinite heuristic value.
    - The object identified as the 'man' based on predicate patterns (`carrying` or `at` for non-spanner/nut objects) is indeed the man agent.

    # Heuristic Initialization
    - The constructor extracts the goal conditions to identify the nuts that need to be tightened.
    - It builds a graph of locations based on the `(link l1 l2)` static facts.
    - It computes all-pairs shortest paths between these locations using Breadth-First Search (BFS). This distance information is crucial for estimating walk costs.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1.  **Identify the Man:** Determine the object representing the man agent. This is done by looking for an object involved in a `(carrying ...)` fact or, if none exists, an object at a location that is not identified as a spanner or nut. Find the man's current location. If the man or his location cannot be determined or is not in the location graph, return infinity.
    2.  **Identify Loose Goal Nuts:** Find all nuts that are currently `(loose)` in the state and are also specified as `(tightened)` in the task's goal conditions. Get the location of each such nut. If any loose goal nut's location is unknown or not in the location graph, return infinity. If there are no loose goal nuts, the goal is considered reached from this state, so return 0.
    3.  **Count Usable Carried Spanners:** Count how many usable spanners the man is currently `(carrying)`.
    4.  **Identify Usable Ground Spanners:** Find all usable spanners that are currently `(at)` a location and are not being carried by the man. Record their locations. If any usable ground spanner's location is unknown or not in the location graph, return infinity.
    5.  **Calculate Needed Pickups:** Determine how many additional usable spanners the man needs to pick up from the ground. This is the maximum of 0 and the difference between the total number of loose goal nuts and the number of usable spanners the man is already carrying (`N_pickups_needed = max(0, N_loose_goals - N_usable_carried)`).
    6.  **Identify Required Locations:** Construct a set of locations the man *must* visit to achieve the goals from this state. This set includes:
        -   The location of each loose goal nut.
        -   If `N_pickups_needed` is greater than 0, it also includes the locations of the `N_pickups_needed` usable ground spanners that are closest to the man's current location.
    7.  **Estimate Walk Cost:** Calculate the estimated cost of walking. This is approximated as the sum of the shortest path distances from the man's current location to each distinct location in the set of required locations identified in step 6. If the man's location or any required location is unreachable in the graph, the walk cost is infinite.
    8.  **Calculate Total Heuristic:** The total heuristic value is the sum of three components:
        -   The number of loose goal nuts (`N_loose_goals`), representing the minimum number of `tighten_nut` actions required.
        -   The number of needed ground spanners (`N_pickups_needed`), representing the minimum number of `pickup_spanner` actions required.
        -   The estimated walk cost calculated in step 7.

    This sum provides a non-admissible estimate that combines the cost of required actions (tighten, pickup) with an estimate of the necessary travel, prioritizing states where fewer actions are needed and required items/locations are closer.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the location graph."""
        self.goals = task.goals
        self.static_facts = task.static

        # Identify all locations mentioned in static links
        self.locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Build location graph (adjacency list) from link facts
        self.location_graph = {loc: set() for loc in self.locations}
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Assuming bidirectional links

        # Compute all-pairs shortest paths
        self.distances = {}
        # Ensure all locations in the graph are keys in the distances map
        all_graph_locations = set(self.location_graph.keys())
        for start_loc in all_graph_locations:
             self.distances[start_loc] = bfs_shortest_paths(self.location_graph, start_loc)

        # Identify goal nuts from goal conditions
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

    def get_location(self, state, obj):
        """Find the current location of an object in the state."""
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]
        return None # Object not found at any location (e.g., carried)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify the man object and his current location
        man_obj = None
        # Try finding man from 'carrying' fact (men are the only objects that carry)
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                man_obj = get_parts(fact)[1]
                break
        # If not carrying, find an object at a location that is not a spanner or nut
        # This relies on identifying spanners and nuts first
        if man_obj is None:
            # Find objects that are spanners or nuts based on predicates
            spanners_in_state = {get_parts(f)[1] for f in state if match(f, "usable", "*")} # Usable implies spanner
            nuts_in_state = {get_parts(f)[1] for f in state if match(f, "loose", "*") or match(f, "tightened", "*")} # Loose/tightened implies nut
            for fact in state:
                if match(fact, "at", "*", "*"):
                    obj_name = get_parts(fact)[1]
                    # If an object is at a location and is not a known spanner or nut, assume it's the man
                    if obj_name not in spanners_in_state and obj_name not in nuts_in_state:
                        man_obj = obj_name
                        break

        # If man object still not found, state is likely invalid or unexpected
        if man_obj is None:
             return float('inf') # Cannot solve without a man

        man_location = self.get_location(state, man_obj)
        # Man must be at a location in a valid state
        if man_location is None:
             return float('inf') # Cannot solve if man's location is unknown or not 'at' a location

        # Ensure man's location is a known location in our graph
        if man_location not in self.locations:
             return float('inf') # Cannot calculate distances


        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts_with_loc = []
        for nut in self.goal_nuts:
            # Check if nut is loose in current state by checking for the exact fact string
            if f"(loose {nut})" in state:
                nut_location = self.get_location(state, nut)
                if nut_location:
                    # Ensure nut location is a known location in our graph
                    if nut_location not in self.locations:
                         return float('inf') # Cannot calculate distances
                    loose_goal_nuts_with_loc.append((nut, nut_location))
                else:
                     # Nut location unknown? Should not happen if nut is 'at' a location.
                     return float('inf') # Cannot solve if nut location is unknown

        # If no loose goal nuts, goal is reached
        if not loose_goal_nuts_with_loc:
            return 0

        N_loose_goals = len(loose_goal_nuts_with_loc)

        # 3. Identify usable spanners the man is carrying
        usable_carried_spanners = set()
        for fact in state:
            if match(fact, "carrying", man_obj, "*"):
                spanner_name = get_parts(fact)[2]
                # Check if the carried spanner is usable by checking for the exact fact string
                if f"(usable {spanner_name})" in state:
                    usable_carried_spanners.add(spanner_name)
        N_usable_carried = len(usable_carried_spanners)

        # 4. Identify usable spanners on the ground and their locations
        usable_ground_spanners_with_loc = []
        # Find all usable spanners
        usable_spanners_in_state = {get_parts(f)[1] for f in state if match(f, "usable", "*")}

        for spanner_name in usable_spanners_in_state:
            # Check if this usable spanner is carried by the man
            if f"(carrying {man_obj} {spanner_name})" not in state:
                # If not carried, it must be on the ground. Get its location.
                spanner_location = self.get_location(state, spanner_name)
                if spanner_location:
                    # Ensure spanner location is a known location in our graph
                    if spanner_location not in self.locations:
                         return float('inf') # Cannot calculate distances
                    usable_ground_spanners_with_loc.append((spanner_name, spanner_location))
                # else: # If usable but not carried and not at a location, ignore it.


        # 7. Calculate needed ground spanners
        N_pickups_needed = max(0, N_loose_goals - N_usable_carried)

        # 8. Identify required locations
        required_locations = set()
        # Add locations of loose goal nuts
        required_locations.update(loc for nut, loc in loose_goal_nuts_with_loc)

        # Add locations of needed ground spanners (the closest ones)
        needed_spanner_locations = set()
        if N_pickups_needed > 0:
            # Sort usable ground spanners by distance from man's current location
            # Use get() with float('inf') default for robustness if man_location wasn't in graph (though checked earlier)
            sorted_ground_spanners = sorted(usable_ground_spanners_with_loc,
                                            key=lambda item: self.distances.get(man_location, {}).get(item[1], float('inf')))

            # Take the locations of the N_pickups_needed closest ones
            needed_spanner_locations.update(loc for s, loc in sorted_ground_spanners[:N_pickups_needed])

        required_locations.update(needed_spanner_locations)

        # 9. Calculate estimated walk cost
        walk_cost = 0
        if required_locations:
            # Sum of distances from man's current location to each required location
            # This is an overestimate but simple and non-zero if movement is needed.
            # Ensure man_location is in the distances map (checked earlier)
            # Ensure required locations are in the distances map (checked when adding)
            for loc in required_locations:
                dist = self.distances[man_location][loc] # dist will be finite or inf (handled earlier)
                if dist == float('inf'):
                    # Should have been caught when identifying required locations, but double check
                    return float('inf')
                walk_cost += dist


        # 10. Total heuristic value
        total_cost = N_loose_goals  # Cost for tighten actions (minimum 1 per nut)
        total_cost += N_pickups_needed # Cost for pickup actions (minimum 1 per needed spanner)
        total_cost += walk_cost       # Estimated walk cost

        return total_cost
