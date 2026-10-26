from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to compute distances
def bfs_distances(start_node, graph):
    """Computes shortest path distances from start_node to all reachable nodes in the graph."""
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Ensure the node exists in the graph keys
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Define the heuristic class
# class spannerHeuristic(Heuristic): # Uncomment this line if inheriting from Heuristic base class
class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It sums the number of loose nuts (representing the tighten actions), the estimated
    cost for Bob to reach the nearest location with a loose nut, and the estimated
    cost for Bob to acquire a usable spanner if he isn't already carrying one.

    # Assumptions
    - The location graph defined by 'link' predicates is connected (or relevant parts are).
    - Usable spanners exist somewhere on the ground if Bob isn't carrying one and the problem is solvable.
    - Bob can carry at most one spanner at a time.
    - The 'usable' property of a spanner is static (determined in the initial state/static facts).
    - Objects named "bob" is the man, "nutX" are nuts, "spannerX" are spanners.

    # Heuristic Initialization
    - Identify all locations and build the location graph based on 'link' facts.
    - Compute shortest path distances between all pairs of locations using BFS.
    - Identify all usable spanners from static facts and initial state.
    - Store goal nuts (those that need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:
    1. Count the number of nuts that are currently 'loose' and are part of the goal. This count is the base heuristic value, representing the minimum number of 'tighten' actions required.
    2. If there are no loose goal nuts, the heuristic is 0 (goal state).
    3. Find Bob's current location.
    4. Find the locations of all loose goal nuts.
    5. Calculate the shortest path distance from Bob's current location to the nearest location containing a loose goal nut. Add this distance to the heuristic. This estimates the movement cost for Bob to get to where the work is needed. If no loose nuts are reachable, return infinity.
    6. Determine if Bob is currently carrying a usable spanner.
    7. If Bob is NOT carrying a usable spanner:
        a. Find the locations of all usable spanners that are currently on the ground.
        b. If there are no usable spanners on the ground, the state is likely unsolvable, return infinity.
        c. Calculate the shortest path distance from Bob's current location to the nearest usable spanner on the ground.
        d. Add this distance plus 1 (for the 'pickup' action) to the heuristic. This estimates the cost for Bob to acquire a spanner. If no usable spanners are reachable, return infinity.
    8. If Bob IS carrying a usable spanner, add 0 for spanner acquisition cost.
    9. The total heuristic is the sum of the base count, Bob's movement cost to the nearest nut, and the spanner acquisition cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and computing distances."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Identify all locations and build the location graph
        locations = set()
        graph = {}

        # Collect locations from initial state and static facts
        all_facts = set(static_facts) | set(initial_state)
        for fact in all_facts:
            parts = get_parts(fact)
            if parts[0] == "at":
                # (at obj loc)
                locations.add(parts[2])
            elif parts[0] == "link":
                # (link loc1 loc2)
                locations.add(parts[1])
                locations.add(parts[2])

        # Initialize graph adjacency list
        for loc in locations:
            graph[loc] = []

        # Build graph from link facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                if loc1 in graph and loc2 in graph: # Ensure locations are valid
                    graph[loc1].append(loc2)
                    graph[loc2].append(loc1) # Links are bidirectional

        self.locations = locations
        self.graph = graph

        # 2. Compute shortest path distances between all pairs of locations
        self._distance_cache = {}
        for start_loc in self.locations:
            distances_from_start = bfs_distances(start_loc, self.graph)
            for end_loc, dist in distances_from_start.items():
                 self._distance_cache[(start_loc, end_loc)] = dist

        # Helper to get distance, returns infinity if no path
        def get_distance(loc1, loc2):
             return self._distance_cache.get((loc1, loc2), float('inf'))
        self.get_distance = get_distance


        # 3. Identify all usable spanners (assuming usable is static or in initial state)
        self.usable_spanners = set()
        for fact in static_facts | initial_state:
            parts = get_parts(fact)
            if parts[0] == "usable":
                self.usable_spanners.add(parts[1])

        # 4. Store goal nuts (those that need to be tightened)
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened":
                self.goal_nuts.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Count loose goal nuts and find their locations
        loose_nuts_locations = {} # nut_name: location
        nut_locations_in_state = {} # nut_name: location mapping from state
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at" and parts[1] in self.goal_nuts:
                 nut_locations_in_state[parts[1]] = parts[2]

        num_loose_nuts = 0
        for nut_name in self.goal_nuts:
             if f"(loose {nut_name})" in state:
                  num_loose_nuts += 1
                  if nut_name in nut_locations_in_state:
                       loose_nuts_locations[nut_name] = nut_locations_in_state[nut_name]
                  # else: loose nut exists but its location is unknown? Treat as unsolvable.

        # 2. If no loose nuts, goal is reached
        if num_loose_nuts == 0:
            return 0

        # Base heuristic: number of tighten actions needed
        total_cost = num_loose_nuts

        # 3. Find Bob's current location
        bob_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] == "bob":
                bob_location = parts[2]
                break
        if bob_location is None:
             # Bob must always be somewhere in a solvable state
             return float('inf')

        # 4. Calculate distance for Bob to reach the nearest loose nut location
        min_dist_to_nut = float('inf')
        loose_nut_locations_set = set(loose_nuts_locations.values()) # Use set to handle multiple nuts at same location
        if not loose_nut_locations_set:
             # Should not happen if num_loose_nuts > 0 and nuts have locations
             # This means loose nuts exist but their location is not in the state, which is weird.
             # Treat as unsolvable.
             return float('inf')

        for nut_loc in loose_nut_locations_set:
             dist = self.get_distance(bob_location, nut_loc)
             min_dist_to_nut = min(min_dist_to_nut, dist)

        if min_dist_to_nut == float('inf'):
             # Cannot reach any loose nut location
             return float('inf')

        total_cost += min_dist_to_nut

        # 5. Check if Bob is carrying a usable spanner
        bob_carrying_usable = False
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "carrying" and parts[1] == "bob" and parts[2] in self.usable_spanners:
                 bob_carrying_usable = True
                 break # Assume Bob carries only one spanner

        # 6. Calculate spanner acquisition cost if needed
        spanner_acquisition_cost = 0
        if not bob_carrying_usable:
            # Bob is not carrying a usable spanner. Need to get one.
            # Find usable spanners on the ground and their locations
            usable_spanners_on_ground_locs = set()
            for fact in state:
                 parts = get_parts(fact)
                 if parts[0] == "at" and parts[1] in self.usable_spanners:
                     # Ensure it's not the one Bob might be carrying (redundant check if bob_carrying_usable is False)
                     is_carried = False
                     for carry_fact in state:
                         carry_parts = get_parts(carry_fact)
                         if carry_parts[0] == "carrying" and carry_parts[1] == "bob" and carry_parts[2] == parts[1]:
                             is_carried = True
                             break
                     if not is_carried:
                         usable_spanners_on_ground_locs.add(parts[2])

            if not usable_spanners_on_ground_locs:
                 # No usable spanners on the ground and Bob isn't carrying one. Unsolvable.
                 return float('inf')

            # Find distance for Bob to reach the nearest usable spanner on the ground
            min_dist_to_spanner = float('inf')
            for spanner_loc in usable_spanners_on_ground_locs:
                 dist = self.get_distance(bob_location, spanner_loc)
                 min_dist_to_spanner = min(min_dist_to_spanner, dist)

            if min_dist_to_spanner == float('inf'):
                 # Cannot reach any usable spanner on the ground
                 return float('inf')

            spanner_acquisition_cost = min_dist_to_spanner + 1 # move to spanner + pickup action

        total_cost += spanner_acquisition_cost

        # 7. Return the total estimated cost
        return total_cost
