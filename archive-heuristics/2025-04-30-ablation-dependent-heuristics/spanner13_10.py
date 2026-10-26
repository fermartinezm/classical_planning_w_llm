import collections
# import re # Not strictly needed for this fact parsing

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Parses a fact string like '(predicate arg1 arg2)' into (predicate, [arg1, arg2])."""
    # Remove surrounding brackets and split by space
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing up
    the estimated costs for each loose nut that needs to be tightened.
    For each such nut, the estimated cost includes:
    1. The cost of the 'tighten_nut' action (1).
    2. The cost for the man to travel from his current location to the nut's location.
    3. The cost to acquire a usable spanner if the man doesn't have enough usable spanners
       to tighten all remaining loose goal nuts. This is estimated as the number of
       additional spanners needed, assuming 1 pickup action per spanner.

    The heuristic is calculated as:
    h = (Number of pending goal nuts)
      + (Shortest distance from man's current location to the closest pending goal nut location)
      + (Number of additional usable spanners the man needs to pick up)

    Assumptions:
    - There is exactly one man object.
    - Nut locations are static (appear in initial state or static facts with 'at').
    - Link facts define a graph for locations.
    - The problem is solvable (enough usable spanners exist initially and locations are connected).

    Heuristic Initialization:
    The constructor pre-processes the task information:
    - Identifies the man object, goal nuts, nut locations, and all locations by parsing initial, goal, and static facts.
    - Builds a graph of locations based on 'link' facts.
    - Computes all-pairs shortest paths between locations using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state using `self.task.goal_reached(state)`. If yes, return 0.
    2. Find the man's current location from the state facts by looking for the `(at man_name location)` fact.
    3. Identify all nuts that are part of the goal conditions (`self.goal_nuts`) but are not yet tightened in the current state (i.e., `(tightened nut)` is not in `state`). These are the pending goal nuts.
    4. Count the total number of such pending goal nuts (`NumPendingGoalNuts`). If 0, return 0 (this case is covered by step 1, but included for clarity).
    5. Count the number of usable spanners the man is currently carrying (`NumUsableCarried`) by checking for facts `(carrying man_name spanner)` and `(usable spanner)` in the current state.
    6. Determine the number of additional usable spanners the man needs to pick up to tighten all pending nuts:
       `AdditionalSpannersNeeded = max(0, NumPendingGoalNuts - NumUsableCarried)`.
       This is a lower bound on the number of 'pickup_spanner' actions required.
    7. Find the locations of all pending goal nuts using the pre-computed `self.nut_locations`.
    8. Calculate the shortest distance from the man's current location to each of these pending nut locations using the pre-computed `self.distances`.
    9. Find the minimum of these distances (`min_dist_to_nut`). If the man's location or any nut location is not in the pre-computed graph, or if no pending nuts are reachable, return a large value indicating a potential issue or unsolvable state.
    10. The heuristic value is calculated as the sum of the estimated costs:
        `h = NumPendingGoalNuts` (estimated tighten actions)
          `+ min_dist_to_nut` (estimated travel to the first nut)
          `+ AdditionalSpannersNeeded` (estimated pickup actions for spanners).
    11. Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        self.task = task
        self.goal_nuts = set()
        self.nut_locations = {} # nut_name -> location_name (static)
        self.locations = set()
        self.man_name = None
        self.adj = collections.defaultdict(set)
        self.distances = {} # (loc1, loc2) -> distance

        # --- Object and Type Identification ---
        nuts_set = set()
        spanners_set = set()
        locations_set = set()
        man_set = set()

        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)

        # Pass 1: Identify types based on specific predicates
        for fact_str in all_facts:
            pred, args = parse_fact(fact_str)
            if pred in ('loose', 'tightened'):
                nuts_set.add(args[0])
            elif pred == 'usable':
                 spanners_set.add(args[0])
            elif pred == 'carrying':
                 man_set.add(args[0])
                 spanners_set.add(args[1])
            elif pred == 'link':
                 locations_set.add(args[0])
                 locations_set.add(args[1])

        # Pass 2: Use 'at' predicate to find locations and potentially the man
        locatables_set = nuts_set | spanners_set | man_set
        for fact_str in all_facts:
             pred, args = parse_fact(fact_str)
             if pred == 'at':
                 obj, loc = args
                 locations_set.add(loc)
                 # If an object is at a location but not yet classified as nut/spanner/man,
                 # and it's not a location itself, it must be a locatable.
                 # In this domain, if it's not a known nut or spanner, assume it's the man.
                 if obj not in nuts_set and obj not in spanners_set and obj not in locations_set:
                      man_set.add(obj)
                 locatables_set.add(obj)

        self.locations = locations_set

        # Assuming one man, pick one from the set. If set is empty, man_name remains None.
        if man_set:
             self.man_name = list(man_set)[0]
        # else: self.man_name remains None, will cause issues later if not handled.
        # Assuming valid problems have a man.

        # --- Parse static facts for graph and nut locations ---
        # Static facts can include 'at' for locatables whose location is fixed.
        # Nuts have static locations.
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred == 'link':
                l1, l2 = args
                self.adj[l1].add(l2)
                self.adj[l2].add(l1) # Links are bidirectional
            elif pred == 'at' and args[0] in nuts_set:
                 self.nut_locations[args[0]] = args[1]

        # Also get nut locations from initial state if not in static (should be static in this domain)
        for fact_str in task.initial_state:
             pred, args = parse_fact(fact_str)
             if pred == 'at' and args[0] in nuts_set:
                 self.nut_locations[args[0]] = args[1]


        # Ensure all locations referenced in adj are in self.locations
        for loc in self.adj:
             self.locations.add(loc)
             for neighbor in self.adj[loc]:
                 self.locations.add(neighbor)

        # --- Compute all-pairs shortest paths ---
        for start_node in self.locations:
            self._bfs(start_node)

        # --- Identify goal nuts ---
        for goal_fact_str in task.goals:
            pred, args = parse_fact(goal_fact_str)
            if pred == 'tightened':
                self.goal_nuts.add(args[0])


    def _bfs(self, start_node):
        """Performs BFS from start_node to compute distances to all reachable nodes."""
        q = collections.deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            # Check if current_node is in adj before iterating
            if current_node in self.adj:
                for neighbor in self.adj[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: A frozenset of strings representing the current state facts.
        @return: An integer heuristic value.
        """
        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Ensure man_name was identified
        if self.man_name is None:
             # Should not happen in valid problems, but handle defensively
             return 1000000

        # 2. Find man's current location
        current_man_loc = None
        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'at' and args[0] == self.man_name:
                current_man_loc = args[1]
                break

        if current_man_loc is None:
             # Man's location must be known if he's not carrying a spanner and not at a location, which is impossible.
             # Or if he is carrying a spanner, (at man loc) should still be true.
             # This indicates a state representation issue or unsolvable state.
             return 1000000 # A large value indicating potential issue or far from goal

        # Ensure man's current location is in our graph of locations
        if current_man_loc not in self.locations:
             # This could happen if the initial state places the man at a location
             # not mentioned in any link or static 'at' fact for a nut.
             # Our init logic should add all locations from initial state/goals/static.
             # If it's still not found, it's an unknown location relative to the graph.
             return 1000000 # Indicate issue or unreachable state

        # 3. Identify pending goal nuts (goal nuts not yet tightened)
        pending_goal_nuts = {n for n in self.goal_nuts if f'(tightened {n})' not in state}

        NumPendingGoalNuts = len(pending_goal_nuts)

        # If no pending nuts, goal is reached (already checked, but double check logic)
        if NumPendingGoalNuts == 0:
             return 0

        # 5. Count usable spanners carried by the man
        NumUsableCarried = 0
        carried_spanners = set()
        usable_spanners_in_state = set()

        for fact_str in state:
             pred, args = parse_fact(fact_str)
             if pred == 'carrying' and args[0] == self.man_name:
                 carried_spanners.add(args[1])
             elif pred == 'usable':
                 usable_spanners_in_state.add(args[0])

        NumUsableCarried = len(carried_spanners.intersection(usable_spanners_in_state))


        # 6. Additional spanners needed
        AdditionalSpannersNeeded = max(0, NumPendingGoalNuts - NumUsableCarried)

        # 7. Find locations of pending goal nuts
        pending_nut_locations = [self.nut_locations.get(nut) for nut in pending_goal_nuts if nut in self.nut_locations]

        # 8. Calculate minimum distance from man's current location to a pending nut location
        min_dist_to_nut = float('inf')

        for nut_loc in pending_nut_locations:
            # Ensure nut location is one we know about from graph construction
            if nut_loc in self.locations:
                # Distance might not exist if graph is disconnected
                # Check if the pair (current_man_loc, nut_loc) exists in distances
                if (current_man_loc, nut_loc) in self.distances:
                    dist = self.distances[(current_man_loc, nut_loc)]
                    min_dist_to_nut = min(min_dist_to_nut, dist)
                # else: distance is effectively infinity, min_dist_to_nut remains float('inf')
            # else: nut_loc not in self.locations, effectively unreachable, min_dist_to_nut remains float('inf')


        # If no reachable pending nut location, return high value
        # This happens if min_dist_to_nut is still float('inf')
        # This could mean pending_nut_locations was empty (shouldn't happen if NumPendingGoalNuts > 0)
        # or all pending nut locations are unreachable from current_man_loc.
        if min_dist_to_nut == float('inf'):
             return 1000000 # Indicate unsolvable or very far

        # 10. Calculate heuristic
        # h = NumPendingGoalNuts (estimated tighten actions)
        #   + min_dist_to_nut (estimated travel to the first nut)
        #   + AdditionalSpannersNeeded (estimated pickup actions for spanners).
        h = NumPendingGoalNuts + min_dist_to_nut + AdditionalSpannersNeeded

        return h
