import math
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all required nuts.
    It considers the cost of tightening each loose goal nut, the cost of picking up
    the necessary spanners, and the estimated travel cost for the man to move
    between his current location, spanner locations, and nut locations.

    # Assumptions:
    - The man can carry only one spanner at a time.
    - A spanner becomes unusable after tightening one nut.
    - Links between locations are bidirectional.
    - All loose nuts that are part of the goal must be tightened.
    - There are enough usable spanners available throughout the problem to tighten all goal nuts, or the problem is unsolvable.

    # Heuristic Initialization
    - Extracts the set of goal nuts from the task definition.
    - Identifies all locations and builds a graph based on `link` facts.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Identifies object types (man, spanner, nut, location) from initial state and static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify which spanner the man is carrying, if any, and if it is usable.
    3. Identify the locations of all spanners and nuts.
    4. Determine the set of nuts that are currently loose and are part of the goal (`N_loose_goal`).
    5. If `N_loose_goal` is empty, the heuristic is 0 (goal state).
    6. If there are loose goal nuts but no usable spanners available (neither carried nor on the ground), the problem is likely unsolvable, return infinity.
    7. Calculate the base cost:
       - Add 1 for each loose goal nut (representing the `tighten_nut` action).
       - Add 1 for each spanner that needs to be picked up. This is the number of loose goal nuts minus 1 if the man is already carrying a usable spanner (since that spanner can be used for the first nut).
    8. Estimate the travel cost:
       - The man starts at his current location (`loc_m`).
       - He needs to perform `num_nuts_to_tighten` tasks, each involving getting a spanner and going to a nut location.
       - If the man starts carrying a usable spanner:
         - The first trip is from `loc_m` to the nearest loose nut location.
         - Subsequent `num_nuts_to_tighten - 1` trips involve going from a nut location to the nearest usable spanner location, then to the next nearest loose nut location. Estimate this cost as `(num_nuts_to_tighten - 1) * (min_dist_nut_to_spanner + min_dist_spanner_to_nut)`.
       - If the man does not start carrying a usable spanner:
         - The first trip is from `loc_m` to the nearest usable spanner location, then to the nearest loose nut location. Estimate this cost as `min_dist_m_to_spanner + min_dist_spanner_to_nut`.
         - Subsequent `num_nuts_to_tighten - 1` trips are estimated as `(num_nuts_to_tighten - 1) * (min_dist_nut_to_spanner + min_dist_spanner_to_nut)`.
       - Minimum distances between sets of locations (man, nuts, spanners) are precomputed using BFS.
    9. The total heuristic value is the sum of the base cost and the estimated travel cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        self.static = task.static

        # Extract object types and names
        self.men = set()
        self.spanners = set()
        self.nuts = set()
        self.locations = set()

        # Helper to populate object sets
        def add_objects_from_fact(fact):
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                self.locations.add(parts[2])
            elif predicate == 'carrying':
                self.men.add(parts[1])
                self.spanners.add(parts[2])
            elif predicate == 'usable':
                self.spanners.add(parts[1])
            elif predicate == 'link':
                self.locations.add(parts[1])
                self.locations.add(parts[2])
            elif predicate in ['tightened', 'loose']:
                self.nuts.add(parts[1])

        for fact in task.initial_state:
            add_objects_from_fact(fact)
        for fact in task.static:
             add_objects_from_fact(fact)
        for goal in self.goals:
             add_objects_from_fact(goal)


        # Extract goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

        # Build location graph from static facts
        self.graph = {loc: [] for loc in self.locations}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Assuming links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all other locations."""
        distances = {loc: math.inf for loc in self.locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.graph: # Ensure the location exists in the graph keys
                for neighbor in self.graph[current_loc]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract dynamic information from the current state
        loc_m = None
        man_carrying_spanner = None
        spanner_locations = {}
        nut_locations = {}
        usable_spanners_available = [] # List of (spanner_name, location)

        # Assuming there is only one man based on domain structure
        man_name = list(self.men)[0] if self.men else None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj == man_name:
                    loc_m = loc
                elif obj in self.spanners:
                    spanner_locations[obj] = loc
                elif obj in self.nuts:
                    nut_locations[obj] = loc
            elif parts[0] == 'carrying' and parts[1] == man_name:
                 man_carrying_spanner = parts[2]

        # Identify usable spanners currently available (on ground or carried)
        for spanner in self.spanners:
             if f'(usable {spanner})' in state:
                 if spanner == man_carrying_spanner:
                     # Man is carrying this usable spanner
                     usable_spanners_available.append((spanner, loc_m))
                 elif spanner in spanner_locations:
                     # Usable spanner is on the ground
                     usable_spanners_available.append((spanner, spanner_locations[spanner]))


        man_carrying_usable_spanner = (man_carrying_spanner is not None) and (f'(usable {man_carrying_spanner})' in state)

        # Identify loose nuts that are goals
        N_loose_goal = {n for n in self.goal_nuts if f'(loose {n})' in state}

        # If all goal nuts are tightened, heuristic is 0
        if not N_loose_goal:
            return 0

        # Check for unsolvable case: loose goal nuts but no usable spanners anywhere
        if not usable_spanners_available and not man_carrying_usable_spanner:
             return float('inf') # Cannot tighten any nut without a usable spanner

        h = 0
        num_nuts_to_tighten = len(N_loose_goal)

        # Cost for tighten actions (1 per nut)
        h += num_nuts_to_tighten

        # Cost for pickup actions (1 per spanner needed)
        # Need one spanner per nut. If man carries a usable one, he needs one less pickup.
        num_pickups_needed = num_nuts_to_tighten - (1 if man_carrying_usable_spanner else 0)
        num_pickups_needed = max(0, num_pickups_needed) # Cannot need negative pickups
        h += num_pickups_needed

        # Travel cost estimation
        travel_cost = 0

        locations_of_loose_nuts = {nut_locations[n] for n in N_loose_goal if n in nut_locations}
        locations_of_usable_spanners_on_ground = {loc for s, loc in usable_spanners_available if s != man_carrying_spanner}


        # Calculate minimum distances between relevant location sets
        # Handle cases where sets might be empty (e.g., no usable spanners on ground)
        min_dist_m_to_nut = min((self.dist[loc_m][loc_n] for loc_n in locations_of_loose_nuts), default=math.inf)
        min_dist_m_to_spanner_on_ground = min((self.dist[loc_m][loc_s] for loc_s in locations_of_usable_spanners_on_ground), default=math.inf)

        min_dist_nut_to_spanner_on_ground = math.inf
        if locations_of_loose_nuts and locations_of_usable_spanners_on_ground:
            min_dist_nut_to_spanner_on_ground = min(
                self.dist[loc_n][loc_s]
                for loc_n in locations_of_loose_nuts
                for loc_s in locations_of_usable_spanners_on_ground
            )

        min_dist_spanner_on_ground_to_nut = math.inf
        if locations_of_usable_spanners_on_ground and locations_of_loose_nuts:
             min_dist_spanner_on_ground_to_nut = min(
                self.dist[loc_s][loc_n]
                for loc_s in locations_of_usable_spanners_on_ground
                for loc_n in locations_of_loose_nuts
            )


        if man_carrying_usable_spanner:
            # Travel to the first nut location
            travel_cost += min_dist_m_to_nut
            # For remaining nuts, need to go from a nut location to a spanner location on the ground, then to a nut location.
            # This sequence happens `num_nuts_to_tighten - 1` times.
            if num_nuts_to_tighten > 1:
                 # Need usable spanners on the ground for subsequent pickups
                 if not locations_of_usable_spanners_on_ground:
                      return float('inf') # Cannot get more spanners
                 travel_cost += (num_nuts_to_tighten - 1) * (min_dist_nut_to_spanner_on_ground + min_dist_spanner_on_ground_to_nut)
        else:
            # Travel to the first spanner location on the ground, then to a nut location.
            # This sequence happens once.
            if not locations_of_usable_spanners_on_ground:
                 return float('inf') # Cannot get the first spanner
            travel_cost += min_dist_m_to_spanner_on_ground + min_dist_spanner_on_ground_to_nut

            # For remaining nuts, need to go from a nut location to a spanner location on the ground, then to a nut location.
            # This sequence happens `num_nuts_to_tighten - 1` times.
            if num_nuts_to_tighten > 1:
                 # Need usable spanners on the ground for subsequent pickups
                 if not locations_of_usable_spanners_on_ground:
                      return float('inf') # Cannot get more spanners
                 travel_cost += (num_nuts_to_tighten - 1) * (min_dist_nut_to_spanner_on_ground + min_dist_spanner_on_ground_to_nut)

        h += travel_cost

        return h

