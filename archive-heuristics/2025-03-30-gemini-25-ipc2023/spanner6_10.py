import sys
from fnmatch import fnmatch
from collections import deque

# Try to import the base class Heuristic.
# This assumes the heuristic code is placed in a directory structure
# where 'heuristics.heuristic_base' can be resolved.
# If the import fails, a dummy class is defined for basic structure validation,
# but the code will not function correctly in a planner without the actual base class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using dummy Heuristic class.", file=sys.stderr)
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        def __init__(self, task):
            """Dummy init."""
            pass
        def __call__(self, node):
            """Dummy call."""
            raise NotImplementedError("Heuristic base class not found.")

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string.
    Example: "(at bob shed)" -> ['at', 'bob', 'shed']
    Returns an empty list if the fact is malformed or empty.
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return []


class SpannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    This heuristic estimates the remaining cost to tighten all loose nuts specified in the goal.
    The cost is calculated as the sum of actions required for each remaining goal nut,
    considering the minimum necessary steps (walking, picking up a spanner, tightening).
    It assumes each loose goal nut requires one 'tighten' action and potentially one 'pickup' action,
    plus the walking distance involved.

    # Assumptions
    - There is exactly one 'man' agent in the problem instance.
    - 'link' predicates define a symmetric graph where each edge has a cost of 1 (one 'walk' action).
    - The heuristic estimates cost by summing the minimum cost to achieve each remaining `(tightened nut)`
      goal independently from the current state. This simplification ignores resource contention
      (e.g., a spanner becoming unusable affects subsequent goals) beyond an initial check for
      sufficient usable spanners. This may overestimate the true cost but aims to be informative
      for greedy best-first search.
    - Object names might follow conventions (e.g., 'spannerX', 'nutY', 'bob'/'man') which are used
      as a fallback if explicit type information isn't readily available from the task object.

    # Heuristic Initialization
    - Parses static facts (`link`) to build a graph representation of locations.
    - Precomputes all-pairs shortest paths (APSP) between locations using Breadth-First Search (BFS),
      storing the minimum number of 'walk' actions required.
    - Identifies all relevant objects (nuts, spanners, the man) by parsing the initial state predicates
      (e.g., `at`, `carrying`, `usable`, `loose`).
    - Stores the set of nuts that must be in the `(tightened ?n)` state according to the goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Remaining Goals:** Determine the set of nuts `N_loose_goal` that are specified as goals
        `(tightened n)` but are currently in the `(loose n)` state.
    2.  **Check Goal Achievement:** If `N_loose_goal` is empty, the goal state is reached, and the
        heuristic value is 0.
    3.  **Identify Resources:** Find the set `S_usable` of spanners that are currently `(usable s)`.
    4.  **Check for Dead Ends:** If the number of loose goal nuts `len(N_loose_goal)` is greater than
        the number of usable spanners `len(S_usable)`, the goal is unreachable from this state. Return infinity.
    5.  **Parse Current State:** Determine the man's current location (`man_loc`), the spanner the man
        is carrying (`carried_spanner`, if any), and the locations of all relevant objects (`obj_locs`).
        Check if the `carried_spanner` is currently usable (`man_carrying_usable`).
    6.  **Estimate Cost per Remaining Nut:** For each nut `n` in `N_loose_goal`:
        a.  Retrieve the nut's location `nut_loc`. If the location is unknown, return infinity (invalid state).
        b.  Calculate `cost1`: The estimated cost if the man uses the currently carried *usable* spanner.
            This cost only involves walking: `dist(man_loc, nut_loc)`. If the man isn't carrying a
            usable spanner, `cost1` is effectively infinity.
        c.  Calculate `cost2`: The estimated cost if the man needs to pick up a *different* usable spanner `s`.
            This involves finding a spanner `s` in `S_usable` (excluding `carried_spanner`) that minimizes
            the travel distance: `dist(man_loc, spanner_loc) + dist(spanner_loc, nut_loc)`.
            The total cost for this path is `1 (for pickup action) + min_travel_distance`. If no such
            reachable spanner `s` exists, `cost2` is infinity.
        d.  The minimum estimated cost for walking and potentially picking up a spanner for nut `n` is
            `min_walk_pickup_cost_n = min(cost1, cost2)`.
        e.  If `min_walk_pickup_cost_n` is infinity for any nut `n`, it implies that nut is unreachable,
            making the overall goal unreachable. Return infinity.
    7.  **Sum Costs:** The total heuristic value is calculated as the sum of:
        - The number of remaining loose goal nuts `len(N_loose_goal)` (representing the cost of `tighten_nut` actions).
        - The sum of the minimum estimated walk/pickup costs (`min_walk_pickup_cost_n`) calculated in step 6
          for each remaining nut `n` in `N_loose_goal`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the task definition, precomputing distances,
        and identifying relevant objects and goals.
        """
        super().__init__(task) # Ensure base class initialization is called if needed
        self.goals = task.goals
        static_facts = task.static

        # Extract locations and links from static facts
        self.locations = set()
        links = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Ensure fact is a valid link predicate
            if parts and parts[0] == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                links.add(fact)

        # Compute all-pairs shortest paths using BFS
        self.distances = self._compute_distances(self.locations, links)

        # Find goal nuts from goal predicates
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure fact is a valid tightened predicate
            if parts and parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Find all spanners, the man, and all nuts from the initial state predicates
        self.all_spanners = set()
        self.man = None
        self.nuts = set() # Track all known nuts

        initial_state = task.initial_state
        for fact in initial_state:
             parts = get_parts(fact)
             if not parts: continue # Skip empty or malformed facts
             pred = parts[0]
             args = parts[1:]

             if pred == 'usable' and len(args) == 1:
                 self.all_spanners.add(args[0])
             elif pred == 'loose' and len(args) == 1:
                 self.nuts.add(args[0])
             elif pred == 'tightened' and len(args) == 1:
                 # Track nuts even if they start tightened
                 self.nuts.add(args[0])
             elif pred == 'carrying' and len(args) == 2:
                 # Assume the first object in a 'carrying' predicate is the man
                 if self.man is None: self.man = args[0]
                 # The second object is assumed to be a spanner
                 self.all_spanners.add(args[1])
             elif pred == 'at' and len(args) == 2:
                 obj = args[0]
                 # Use naming conventions as a fallback to identify object types
                 # This is less robust than explicit type information
                 if 'spanner' in obj and obj not in self.all_spanners:
                      self.all_spanners.add(obj)
                 elif 'nut' in obj and obj not in self.nuts:
                      self.nuts.add(obj)
                 # Try to identify the man object if not already found via 'carrying'
                 elif self.man is None and ('man' in obj or obj == 'bob'): # 'bob' is common in examples
                      self.man = obj

        # Fallback check for man if not found via carrying predicate
        if self.man is None:
             for fact in initial_state:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'at' and len(parts) == 3:
                     obj = parts[1]
                     # Refine this check based on typical object names if needed
                     if 'man' in obj or obj == 'bob':
                          self.man = obj
                          break

        # Critical check: If the man object couldn't be identified, the heuristic cannot function.
        if self.man is None:
            print("CRITICAL ERROR: SpannerHeuristic could not identify the man object.", file=sys.stderr)
            # Consider raising an error or ensuring __call__ returns infinity reliably.

        # Ensure all goal nuts are included in the set of known nuts
        self.nuts.update(self.goal_nuts)


    def _compute_distances(self, locations, links):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {loc: {other: float('inf') for other in locations} for loc in locations}
        adj = {loc: [] for loc in locations}

        # Build adjacency list from link facts
        for link_fact in links:
            parts = get_parts(link_fact)
            if len(parts) == 3 and parts[0] == 'link':
                 l1, l2 = parts[1], parts[2]
                 # Ensure locations are valid before adding edges
                 if l1 in locations and l2 in locations:
                     adj[l1].append(l2)
                     adj[l2].append(l1) # Assume links are symmetric

        # Run BFS from each location to find shortest paths
        for start_node in locations:
            # Check if start_node is valid (should be in locations set)
            if start_node not in distances: continue
            distances[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node}

            while queue:
                current_node = queue.popleft()
                # Check if current_node exists in adjacency list (might be isolated)
                if current_node not in adj: continue

                current_dist = distances[start_node][current_node]
                # If current node is somehow unreachable, skip its neighbors
                if current_dist == float('inf'): continue

                for neighbor in adj[current_node]:
                    # Process neighbor only if it's a valid location and not visited yet from start_node
                    if neighbor in locations and neighbor not in visited:
                        visited.add(neighbor)
                        # Path cost increases by 1 for each step (walk action)
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def _get_obj_locations(self, state):
        """Parses the state (set of facts) to find the current location of all objects."""
        obj_locs = {}
        for fact in state:
            parts = get_parts(fact)
            # Check for valid 'at' predicate: (at object location)
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                obj_locs[obj] = loc
        return obj_locs

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns an estimate of the remaining actions needed to reach the goal state.
        """
        state = node.state

        # 1. Identify remaining goals (loose nuts that are part of the goal)
        loose_goal_nuts = set()
        for nut in self.goal_nuts:
            # Format the fact string to check for existence in the state set
            if f'(loose {nut})' in state:
                loose_goal_nuts.add(nut)

        # 2. Check goal achievement
        if not loose_goal_nuts:
            # All goal nuts are tightened
            return 0

        num_loose_goals = len(loose_goal_nuts)

        # 3. Identify currently usable spanners
        usable_spanners = set()
        for spanner in self.all_spanners:
             if f'(usable {spanner})' in state:
                 usable_spanners.add(spanner)

        # 4. Check for dead end (insufficient usable spanners)
        if len(usable_spanners) < num_loose_goals:
            # Not enough spanners available to tighten all remaining nuts
            return float('inf')

        # 5. Parse current state details
        obj_locs = self._get_obj_locations(state)

        # Check if the man object was successfully identified during initialization
        if self.man is None:
             print("Error: Man object unknown in heuristic calculation.", file=sys.stderr)
             return float('inf') # Cannot proceed without the man object

        # Find the man's current location
        man_loc = obj_locs.get(self.man)
        if man_loc is None:
             # If the man has no 'at' predicate, the state might be invalid or unexpected
             print(f"Error: Location of man '{self.man}' not found in state.", file=sys.stderr)
             return float('inf')

        # Find the spanner carried by the man, if any
        carried_spanner = None
        for fact in state:
            parts = get_parts(fact)
            # Check for valid 'carrying' predicate involving the man
            if parts and parts[0] == 'carrying' and len(parts) == 3 and parts[1] == self.man:
                carried_spanner = parts[2]
                break

        # Check if the carried spanner is currently usable
        man_carrying_usable = False
        if carried_spanner is not None and carried_spanner in usable_spanners:
            man_carrying_usable = True

        # 6. Estimate cost per remaining nut
        total_estimated_walk_pickup_cost = 0
        for nut in loose_goal_nuts:
            # Get the location of the current nut
            nut_loc = obj_locs.get(nut)
            if nut_loc is None:
                 # If a required nut has no location, the state is likely invalid
                 print(f"Error: Location of nut '{nut}' not found in state.", file=sys.stderr)
                 return float('inf')

            # --- Calculate cost option 1: Use the currently carried usable spanner ---
            cost1 = float('inf')
            if man_carrying_usable:
                # Check if locations are valid keys in the precomputed distances dictionary
                if man_loc in self.distances and nut_loc in self.distances[man_loc]:
                     dist = self.distances[man_loc].get(nut_loc, float('inf'))
                     cost1 = dist
                # else: cost1 remains inf if locations are invalid or unreachable

            # --- Calculate cost option 2: Pick up a different usable spanner ---
            cost2 = float('inf')
            min_pickup_walk_cost = float('inf')
            # Identify usable spanners available for pickup (not the one being carried)
            spanners_to_consider = usable_spanners - ({carried_spanner} if carried_spanner else set())

            if spanners_to_consider:
                found_path_for_pickup = False
                for spanner in spanners_to_consider:
                    spanner_loc = obj_locs.get(spanner)
                    # Ensure the spanner has a known location
                    if spanner_loc is None: continue

                    # Check reachability using precomputed distances
                    # Ensure all locations involved are valid keys in self.distances
                    if (man_loc in self.distances and
                        spanner_loc in self.distances.get(man_loc, {}) and # Check man->spanner path
                        spanner_loc in self.distances and # Check spanner exists as source
                        nut_loc in self.distances.get(spanner_loc, {})): # Check spanner->nut path

                        dist_man_spanner = self.distances[man_loc].get(spanner_loc, float('inf'))
                        dist_spanner_nut = self.distances[spanner_loc].get(nut_loc, float('inf'))

                        # If both path segments are reachable
                        if dist_man_spanner != float('inf') and dist_spanner_nut != float('inf'):
                            current_pickup_walk_cost = dist_man_spanner + dist_spanner_nut
                            min_pickup_walk_cost = min(min_pickup_walk_cost, current_pickup_walk_cost)
                            found_path_for_pickup = True
                    # else: locations are invalid or unreachable, path cost remains inf

                # If a reachable path involving pickup was found
                if found_path_for_pickup:
                    # Cost includes 1 action for pickup + minimum walk distance
                    cost2 = 1 + min_pickup_walk_cost

            # --- Determine minimum cost for this nut ---
            # This is the minimum cost for walking and potentially picking up a spanner
            min_cost_for_nut = min(cost1, cost2)

            # If cost is infinite, this nut (and thus the goal) is unreachable from current state
            if min_cost_for_nut == float('inf'):
                return float('inf')

            # Accumulate the estimated walk/pickup cost for this nut
            total_estimated_walk_pickup_cost += min_cost_for_nut

        # 7. Sum costs: (Num Tighten Actions) + (Sum of Walk/Pickup Costs)
        # num_loose_goals represents the cost of the 'tighten_nut' actions.
        heuristic_value = num_loose_goals + total_estimated_walk_pickup_cost

        # Return the final heuristic value. It should be an integer unless infinity.
        return int(heuristic_value) if heuristic_value != float('inf') else float('inf')

