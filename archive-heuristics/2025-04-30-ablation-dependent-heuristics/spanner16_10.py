import collections
import math

def parse_fact(fact_str):
    """Parses a fact string like '(predicate arg1 arg2)' into a tuple."""
    # Remove surrounding brackets and split by space
    parts = fact_str[1:-1].split()
    # The first part is the predicate, the rest are arguments
    return (parts[0],) + tuple(parts[1:])

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the cost to reach the goal (tighten all specified nuts)
    by summing the estimated number of required actions:
    1.  The number of 'tighten_nut' actions needed (equal to the number of loose goal nuts).
    2.  The number of 'pickup_spanner' actions needed (equal to the number of additional
        usable spanners required beyond those currently carried).
    3.  The estimated travel cost for the man to reach a location where the *first*
        necessary action (either picking up a spanner or tightening a nut) can occur.

    Assumptions:
    -   The problem instance is solvable with the initially available usable spanners.
    -   Spanners, once used for tightening, become permanently unusable.
    -   Spanners cannot be dropped or transferred between locatables.
    -   The man is the only agent.
    -   Nut locations are static.
    -   Link facts define an undirected graph.
    -   The man object name can be inferred from initial state facts (e.g., by looking
        for 'carrying' facts or the unique locatable object that isn't a nut or spanner).
    -   All spanners are initially identifiable via 'usable' or 'carrying' facts.
    -   All goal nuts have an initial 'at' fact specifying their location.

    Heuristic Initialization:
    -   Parses static 'link' facts to build a graph of locations.
    -   Computes all-pairs shortest paths between locations using Breadth-First Search (BFS).
        These distances are stored in `self.distances`.
    -   Identifies all nut objects that are part of the goal and stores their initial
        locations in `self.nut_locations`.
    -   Identifies all spanner objects.
    -   Infers the man object name.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to find the man's current location, the set of spanners
        being carried, the set of usable spanners (carried or at locations), and the
        set of loose goal nuts.
    2.  Identify the set of goal nuts that are currently loose in the state. Let this count be N_loose_goals.
    3.  If N_loose_goals is 0, the goal is reached, return 0.
    4.  Identify the set of usable spanners currently carried by the man. Let this count be N_carried_usable.
    5.  Calculate the number of additional usable spanners needed: N_needed_spanners = max(0, N_loose_goals - N_carried_usable).
    6.  Check if the total number of usable spanners available anywhere in the state
        (`usable_spanners_available`) is less than N_loose_goals. If so, the state is
        likely unsolvable (not enough spanners exist), return infinity.
    7.  Calculate the base cost: This is the sum of the minimum required 'tighten_nut'
        actions (N_loose_goals) and 'pickup_spanner' actions (N_needed_spanners).
        Base cost = N_loose_goals + N_needed_spanners.
    8.  Calculate the estimated travel cost:
        -   If N_needed_spanners > 0, the man needs to acquire a spanner. Find the minimum
            shortest path distance from the man's current location to any location
            containing a usable spanner that is not currently carried. This is the
            estimated travel cost. If no such reachable location exists, return infinity.
        -   If N_needed_spanners == 0 (man has enough usable spanners carried), the man
            needs to tighten nuts. Find the minimum shortest path distance from the man's
            current location to any location containing a loose goal nut. This is the
            estimated travel cost. If no such reachable location exists, return infinity.
    9.  The total heuristic value is the sum of the base cost and the estimated travel cost.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        self.locations = set()
        self.location_graph = collections.defaultdict(set)
        self._process_static_facts()

        self.distances = self._compute_all_pairs_shortest_paths()

        # Identify all nuts that are part of the goal and their initial locations
        self.all_goal_nuts = {parse_fact(f)[1] for f in self.goals if parse_fact(f)[0] == 'tightened'}
        self.nut_locations = {}
        self._process_initial_state_for_nut_locations(task.initial_state)

        # Identify all spanners
        self.all_spanners = set()
        self._find_all_spanners(task.initial_state)

        # Identify the man object name
        self.man_obj = None
        self._find_man_object(task)


    def _process_static_facts(self):
        """Builds the location graph from static link facts."""
        for fact_str in self.static:
            pred, *args = parse_fact(fact_str)
            if pred == 'link':
                loc1, loc2 = args
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Links are bidirectional


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        for start_loc in self.locations:
            distances[start_loc] = self._bfs(start_loc)
        return distances

    def _bfs(self, start_loc):
        """Performs BFS starting from start_loc to find distances to all other locations."""
        q = collections.deque([(start_loc, 0)])
        visited = {start_loc}
        dist = {start_loc: 0}

        while q:
            current_loc, current_dist = q.popleft()

            for neighbor in self.location_graph.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist[neighbor] = current_dist + 1
                    q.append((neighbor, current_dist + 1))
        return dist

    def _process_initial_state_for_nut_locations(self, initial_state):
        """Stores initial locations for all goal nuts."""
        for fact_str in initial_state:
            pred, *args = parse_fact(fact_str)
            if pred == 'at' and len(args) == 2:
                 obj, loc = args
                 if obj in self.all_goal_nuts:
                      self.nut_locations[obj] = loc # Assuming nuts don't move

        # Check if locations for all goal nuts were found
        if len(self.nut_locations) != len(self.all_goal_nuts):
             # This indicates an issue with the problem instance or assumptions.
             # For a valid solvable instance, all goal nuts should have an initial location.
             # Missing nut locations will result in infinite heuristic later.
             pass


    def _find_all_spanners(self, initial_state):
         """Identifies all spanner object names from the initial state facts."""
         # Spanners can be identified by being 'usable' or being the second argument of 'carrying'.
         # We collect all objects mentioned in these roles.
         for fact_str in initial_state:
              pred, *args = parse_fact(fact_str)
              if pred == 'usable' and len(args) >= 1:
                   self.all_spanners.add(args[0])
              elif pred == 'carrying' and len(args) >= 2:
                   self.all_spanners.add(args[1]) # The second argument is the spanner


    def _find_man_object(self, task):
         """Infers the man object name from the initial state facts."""
         # 1. Look for an object in a 'carrying' fact.
         for fact_str in task.initial_state:
              pred, *args = parse_fact(fact_str)
              if pred == 'carrying' and len(args) >= 1:
                   self.man_obj = args[0]
                   return

         # 2. If no 'carrying' fact, collect all objects in 'at' facts.
         locatables_in_init = {parse_fact(f)[1] for f in task.initial_state if parse_fact(f)[0] == 'at'}

         # 3. Identify nuts and spanners from initial state facts.
         # Use all nuts found, not just goal nuts, for better exclusion.
         all_nuts_in_init = {parse_fact(f)[1] for f in task.initial_state if parse_fact(f)[0] in ('loose', 'tightened', 'at') and len(parse_fact(f)) > 1} # Include nuts mentioned in 'at'
         all_spanners_in_init = self.all_spanners # Already found in _find_all_spanners

         # 4. The man is a locatable that is not a nut or spanner.
         potential_men = locatables_in_init - all_nuts_in_init - all_spanners_in_init

         # 5. If there is exactly one such object, assume it's the man.
         if len(potential_men) == 1:
              self.man_obj = list(potential_men)[0]
              return

         # Fallback: If inference fails, assume 'bob' (based on example).
         # This is a weak assumption but necessary without type information.
         # In a real system, this should ideally be provided or inferred more robustly.
         # print("Warning: Could not uniquely identify man object from initial state facts. Assuming 'bob'.")
         self.man_obj = 'bob' # Fallback assumption


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost to reach the goal.
        """
        # 1. Parse state
        man_loc = None
        carried_spanners = set()
        usable_spanners_at_loc = collections.defaultdict(set)
        usable_spanners_available = set()
        loose_nuts_in_state = set()

        for fact_str in state:
            pred, *args = parse_fact(fact_str)
            if pred == 'at':
                obj, loc = args
                if obj == self.man_obj:
                    man_loc = loc
                # Spanner locations are handled below after identifying usable spanners
            elif pred == 'carrying':
                carrier, obj = args
                if carrier == self.man_obj and obj in self.all_spanners:
                    carried_spanners.add(obj)
            elif pred == 'usable':
                spanner = args[0]
                if spanner in self.all_spanners:
                    usable_spanners_available.add(spanner)
            elif pred == 'loose':
                nut = args[0]
                # We only care about nuts that are goal nuts
                if nut in self.all_goal_nuts:
                    loose_nuts_in_state.add(nut)
            # 'tightened' facts are implicitly handled by checking if goal nuts are NOT loose

        # Populate usable_spanners_at_loc based on 'at' facts and usable_spanners_available
        for fact_str in state:
             pred, *args = parse_fact(fact_str)
             if pred == 'at' and len(args) == 2:
                  obj, loc = args
                  if obj in self.all_spanners and obj in usable_spanners_available:
                       usable_spanners_at_loc[loc].add(obj)


        # 2. Identify loose goal nuts
        # These are goal nuts that are currently in the 'loose' state.
        loose_goal_nuts = loose_nuts_in_state # Because we filtered loose_nuts_in_state to only include goal nuts

        # 3. Check if goal reached
        if not loose_goal_nuts:
            return 0

        # 4. N_loose_goals
        N_loose_goals = len(loose_goal_nuts)

        # 5. N_carried_usable
        carried_usable_spanners = carried_spanners.intersection(usable_spanners_available)
        N_carried_usable = len(carried_usable_spanners)

        # 6. N_needed_spanners
        N_needed_spanners = max(0, N_loose_goals - N_carried_usable)

        # 7. Check solvability based on total usable spanners
        # If the total number of usable spanners available *anywhere* is less than the number of nuts to tighten, it's unsolvable.
        if len(usable_spanners_available) < N_loose_goals:
             return float('inf')

        # 8. Base cost
        base_cost = N_loose_goals + N_needed_spanners

        # 9. Travel cost
        travel_cost = 0
        if man_loc is None:
             # Man's location must be known in a valid state
             return float('inf')

        if N_needed_spanners > 0:
            # Need to pick up spanners. Go to the nearest location with a usable spanner not carried.
            locations_with_usable_spanners_not_carried = {
                loc for loc, spanners in usable_spanners_at_loc.items() if spanners
            }

            if not locations_with_usable_spanners_not_carried:
                 # This case implies all usable spanners are carried.
                 # If N_needed_spanners > 0, this contradicts len(usable_spanners_available) >= N_loose_goals.
                 # It should be caught by the earlier solvability check.
                 # As a safeguard, return inf if we need spanners but none are at locations.
                 return float('inf')

            min_dist_to_spanner_loc = float('inf')
            for loc in locations_with_usable_spanners_not_carried:
                 # Ensure man_loc and target loc are in the computed distances (i.e., reachable)
                 if man_loc in self.distances and loc in self.distances[man_loc]:
                      min_dist_to_spanner_loc = min(min_dist_to_spanner_loc, self.distances[man_loc][loc])

            if min_dist_to_spanner_loc == float('inf'):
                 # No reachable location with a usable spanner
                 return float('inf')
            travel_cost = min_dist_to_spanner_loc

        else: # N_needed_spanners == 0, man has enough spanners
            # Need to tighten nuts. Go to the nearest location with a loose goal nut.
            loose_nut_locations = set()
            for nut in loose_goal_nuts:
                 if nut in self.nut_locations:
                      loose_nut_locations.add(self.nut_locations[nut])
                 else:
                      # Location of a loose goal nut is unknown (should not happen in valid problems)
                      return float('inf')


            if not loose_nut_locations:
                 # Should not happen if N_loose_goals > 0
                 return 0 # Goal reached case handled earlier

            min_dist_to_nut_loc = float('inf')
            for loc in loose_nut_locations:
                 # Ensure man_loc and target loc are in the computed distances (i.e., reachable)
                 if man_loc in self.distances and loc in self.distances[man_loc]:
                      min_dist_to_nut_loc = min(min_dist_to_nut_loc, self.distances[man_loc][loc])

            if min_dist_to_nut_loc == float('inf'):
                 # No reachable location with a loose goal nut
                 return float('inf')
            travel_cost = min_dist_to_nut_loc

        # 10. Total heuristic
        return base_cost + travel_cost
