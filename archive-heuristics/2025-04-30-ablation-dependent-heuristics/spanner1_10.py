import collections

def parse_fact(fact_string):
    """Parses a PDDL fact string into (predicate, [objects])."""
    # Remove surrounding brackets and split by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class spannerHeuristic:
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    Estimates the cost to reach the goal state (all required nuts tightened)
    by summing the number of tighten actions needed, the number of spanner
    pickup actions needed, and an estimate of the travel cost.
    The travel cost is estimated using a greedy approach: repeatedly move
    to the closest required location (either a loose goal nut location or
    a location with an available usable spanner if pickups are needed).

    Assumptions:
    - There is exactly one man object.
    - The graph of locations connected by 'link' predicates is connected,
      or at least all relevant locations (initial man location, nut locations,
      spanner locations) are in the same connected component for solvable states.
    - Enough usable spanners exist in the problem instance to tighten all
      goal nuts.
    - The PDDL facts are represented as strings like '(predicate obj1 obj2)'.
    - The first argument of 'carrying' is always the man. If no 'carrying'
      fact exists, the man is the unique object appearing as the first argument
      of an 'at' fact that does not appear as an argument of 'usable', 'loose',
      or 'tightened'.

    Heuristic Initialization:
    - Stores the task object and extracts goal nuts.
    - Parses the static facts to build a graph of locations and their links.
    - Computes all-pairs shortest paths between locations using Breadth-First Search (BFS).
      This distance information is stored in `self.location_distances` for efficient
      lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the goal state is already reached using `self.task.goal_reached(state)`. If yes, return 0.
    2.  Parse the current state to identify:
        - All usable spanners currently existing (`(usable s)`).
        - The man object (using the assumption about domain structure).
        - The man's current location.
        - The set of usable spanners the man is currently carrying.
        - The locations of usable spanners that are not being carried, along with the count at each location.
        - The set of nuts that are currently loose.
        - The locations of all nuts (loose or tightened).
    3.  Identify the set of loose goal nuts by checking which loose nuts from step 2 are also in the goal set (identified during initialization).
    4.  Calculate the number of 'tighten_nut' actions needed: This is equal to the number of loose goal nuts.
    5.  Calculate the minimum number of 'pickup_spanner' actions needed: This is the maximum of 0 and (number of loose goal nuts - number of usable spanners currently carried). Each usable spanner carried can be used for one nut.
    6.  Estimate the travel cost using a greedy approach:
        - Initialize `current_location` to the man's location and `travel_cost` to 0.
        - Create a set of locations of loose goal nuts that still need to be visited.
        - Create a dictionary tracking the count of available usable spanners at each location.
        - Initialize `pickups_remaining` to the number of pickup actions needed from step 5.
        - While there are still nut locations to visit OR spanner pickups needed:
            - Find the minimum distance (`min_dist`) from `current_location` to any location in the set of remaining required nut locations OR any location with available usable spanners (if `pickups_remaining > 0`).
            - Identify the `next_location` that achieves this minimum distance and its `target_type` ('nut' or 'spanner').
            - If no reachable target is found (`min_dist` is infinity), return `float('inf')` as the state is likely unsolvable.
            - Add `min_dist` to `travel_cost`.
            - Update `current_location` to `next_location`.
            - If `next_location` was a nut location, remove it from the set of required nut locations.
            - If `next_location` was a spanner location and `pickups_remaining > 0`, decrement the count of available spanners at that location and decrement `pickups_remaining`.
    7.  The total heuristic value is the sum of the number of tighten actions, the number of pickup actions, and the estimated travel cost.
    """

    def __init__(self, task):
        self.task = task
        self.goal_nuts = set()
        # Extract goal nuts from goal facts
        for fact in task.goals:
            pred, objs = parse_fact(fact)
            if pred == 'tightened' and len(objs) > 0:
                self.goal_nuts.add(objs[0])

        # Build location graph from static link facts
        self.location_graph = collections.defaultdict(set)
        self.all_locations = set()
        for fact in task.static:
            pred, objs = parse_fact(fact)
            if pred == 'link' and len(objs) > 1:
                loc1, loc2 = objs
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Compute all-pairs shortest paths
        self.location_distances = {}
        for start_loc in self.all_locations:
            self.location_distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Computes shortest distances from start_node to all reachable nodes."""
        distances = {node: float('inf') for node in self.all_locations}
        if start_node not in self.all_locations:
             # Start node is not a known location, cannot compute distances
             return distances # Will remain all inf

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.location_graph: # Check if node exists in graph keys
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, state):
        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # --- Parse State ---
        state_facts = set(state)

        # Identify Usable Spanners, Loose Nuts, Tightened Nuts in State
        usable_spanners_in_state = set()
        loose_nuts_in_state = set()
        tightened_nuts_in_state = set()
        for fact in state_facts:
            pred, objs = parse_fact(fact)
            if len(objs) > 0:
                if pred == 'usable':
                    usable_spanners_in_state.add(objs[0])
                elif pred == 'loose':
                    loose_nuts_in_state.add(objs[0])
                elif pred == 'tightened':
                    tightened_nuts_in_state.add(objs[0])


        # Identify Man Object
        man_obj = None
        potential_men_at = set()
        non_man_locatables = usable_spanners_in_state.union(loose_nuts_in_state).union(tightened_nuts_in_state)

        for fact in state_facts:
            pred, objs = parse_fact(fact)
            if len(objs) > 0:
                if pred == 'carrying':
                    man_obj = objs[0]
                    break # Found the man!
                elif pred == 'at':
                    potential_men_at.add(objs[0])

        if man_obj is None: # Man is not carrying anything
            man_candidates = potential_men_at - non_man_locatables
            if len(man_candidates) == 1:
                man_obj = list(man_candidates)[0]
            else:
                # Cannot uniquely identify the man or no man found
                # This state is likely invalid or unsolvable
                return float('inf')

        if man_obj is None: # Should not happen if potential_men_at was not empty and unique candidate found.
             return float('inf')


        # Collect other state information using man_obj and usable_spanners_in_state
        man_location = None
        carried_usable_spanners = set()
        usable_spanners_at_loc = collections.defaultdict(list) # {loc: [spanner1, spanner2], ...}
        nut_locations = {} # {nut: loc}

        for fact in state_facts:
            pred, objs = parse_fact(fact)
            if len(objs) > 1:
                if pred == 'at':
                    obj, loc = objs
                    if obj == man_obj:
                        man_location = loc
                    elif obj in usable_spanners_in_state:
                         usable_spanners_at_loc[loc].append(obj)
                    elif obj in loose_nuts_in_state or obj in tightened_nuts_in_state:
                         nut_locations[obj] = loc
                elif pred == 'carrying':
                     carrier, spanner = objs
                     if carrier == man_obj and spanner in usable_spanners_in_state:
                         carried_usable_spanners.add(spanner)

        # Handle case where man_location is not found (invalid state?)
        if man_location is None:
             return float('inf')

        # --- Heuristic Calculation ---

        # Identify loose goal nuts and their locations
        loose_goal_nuts = loose_nuts_in_state.intersection(self.goal_nuts)
        required_nut_locations = {nut_locations.get(nut) for nut in loose_goal_nuts if nut in nut_locations}
        required_nut_locations.discard(None) # Remove None if any nut location wasn't found

        # 2. Tighten cost
        num_loose_goal_nuts = len(loose_goal_nuts)
        tighten_cost = num_loose_goal_nuts

        # 3. Pickup cost
        num_carried_usable = len(carried_usable_spanners)
        pickups_needed = max(0, num_loose_goal_nuts - num_carried_usable)
        pickup_cost = pickups_needed

        # 4. Travel cost (Greedy approach)
        current_location = man_location
        spanner_locations_with_available_count = {loc: len(spanners) for loc, spanners in usable_spanners_at_loc.items()}

        travel_cost = 0
        pickups_remaining = pickups_needed

        # Create mutable sets/dicts for the greedy loop
        current_required_nut_locs = set(required_nut_locations)
        current_spanner_locs_with_count = dict(spanner_locations_with_available_count)


        while current_required_nut_locs or pickups_remaining > 0:
            min_dist = float('inf')
            next_location = None
            target_type = None # 'nut' or 'spanner'

            # Consider nut locations as targets
            nut_targets = list(current_required_nut_locs)
            for loc in nut_targets:
                d = self.location_distances.get(current_location, {}).get(loc, float('inf'))
                if d < min_dist:
                    min_dist = d
                    next_location = loc
                    target_type = 'nut'

            # Consider spanner locations as targets if pickups are needed
            spanner_candidates = []
            if pickups_remaining > 0:
                spanner_candidates = [loc for loc, count in current_spanner_locs_with_count.items() if count > 0]

            for loc in spanner_candidates:
                 d = self.location_distances.get(current_location, {}).get(loc, float('inf'))
                 if d < min_dist:
                    min_dist = d
                    next_location = loc
                    target_type = 'spanner'

            # If no target found (disconnected graph or no available spanners/nuts)
            if next_location is None or min_dist == float('inf'):
                 # This state is likely unsolvable from here
                 return float('inf')

            travel_cost += min_dist
            current_location = next_location

            if target_type == 'nut':
                current_required_nut_locs.remove(current_location)
            elif target_type == 'spanner':
                current_spanner_locs_with_count[current_location] -= 1
                pickups_remaining -= 1

        # Total heuristic value
        total_heuristic = tighten_cost + pickup_cost + travel_cost

        return total_heuristic
