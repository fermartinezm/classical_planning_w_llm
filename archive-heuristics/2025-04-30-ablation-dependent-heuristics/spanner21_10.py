import math
import collections

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the cost to reach the goal state (all goal nuts
    tightened) by summing up three main cost components:
    1. The number of remaining loose goal nuts (representing the minimum number
       of tighten actions required).
    2. The estimated travel cost for the man to reach the locations of all
       remaining loose goal nuts (sum of shortest path distances from the man's
       current location to each loose goal nut's location).
    3. The estimated cost to acquire enough usable spanners to tighten all
       remaining loose goal nuts (sum of costs to reach and pick up the
       required number of nearest usable spanners).

    Assumptions:
    - There is exactly one man object in the domain.
    - Nuts, spanners, and locations are identifiable by their usage in predicates
      in the task facts.
    - Links between locations are bidirectional.
    - All locations relevant to objects (man, spanners, nuts) are part of the
      linked graph.
    - Loose nuts that are goal nuts are always located somewhere (i.e., an
      '(at nut loc)' fact exists for them in the state).
    - Usable spanners are either carried or at a location (i.e., an
      '(at spanner loc)' fact exists for them if not carried).
    - The state representation includes all necessary facts (e.g., '(loose nut)'
      if a nut is loose, '(at obj loc)' for object locations).

    Heuristic Initialization:
    The constructor performs the following steps once:
    1. Identifies the man object, spanner objects, nut objects, and location
       objects by analyzing the predicate structure and arguments across all
       possible ground facts in the task.
    2. Identifies the set of goal nuts from the task's goal conditions.
    3. Builds an adjacency list representation of the location graph based on
       the static '(link l1 l2)' facts.
    4. Computes all-pairs shortest path distances between all identified
       locations using Breadth-First Search (BFS) starting from each location.
       These distances are stored for efficient lookup during heuristic
       computation.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Parse the state facts to determine:
       - The man's current location.
       - Which nuts are currently loose.
       - Which spanners are currently usable.
       - Which spanners are currently carried by the man.
       - The locations of all nuts and spanners currently at locations.
    2. Identify the set of loose goal nuts (nuts that are goals and are currently loose).
    3. If there are no loose goal nuts, the goal is reached, and the heuristic is 0.
    4. Count the total number of usable spanners available anywhere (carried or at locations).
    5. If the total number of usable spanners is less than the number of loose
       goal nuts, the problem is unsolvable from this state, return infinity.
    6. Initialize the heuristic value `h` with the number of loose goal nuts
       (representing the cost of the tighten actions).
    7. Calculate the sum of shortest path distances from the man's current
       location to the location of each loose goal nut. Add this sum to `h`.
       If any nut location is unreachable, return infinity.
    8. Determine the number of additional usable spanners (`k`) the man needs
       to pick up. This is the number of loose goal nuts minus the number of
       usable spanners the man is currently carrying, clamped at a minimum of 0.
    9. If `k > 0`:
       - For each usable spanner currently at a location, calculate the cost
         to reach that spanner's location from the man's current location and
         pick it up (distance + 1 for the pickup action).
       - Collect these pickup costs. If any spanner location is unreachable,
         return infinity.
       - Sort the collected pickup costs and sum the smallest `k` costs.
       - Add this sum to `h`. If there are fewer than `k` usable spanners at
         locations, return infinity (this case should ideally be covered by
         the total spanner check earlier, but this adds robustness).
    10. Return the final calculated value of `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.man_obj = None
        self.spanner_objs = set()
        self.nut_objs = set()
        self.locations = set()
        self.goal_nuts = set()
        self.adj = {}
        self.distances = {}

        # --- Object Identification ---
        # Infer types based on predicate usage patterns across all ground facts
        obj_in_carrying_arg1 = set()
        obj_in_carrying_arg2 = set()
        obj_in_usable_arg1 = set()
        obj_in_loose_arg1 = set()
        obj_in_tightened_arg1 = set()
        obj_in_link_arg1 = set()
        obj_in_link_arg2 = set()
        obj_in_at_arg1 = set()
        obj_in_at_arg2 = set()

        for fact_str in task.facts:
            pred, *args = self.parse_fact(fact_str)
            if pred == 'carrying':
                if len(args) == 2: obj_in_carrying_arg1.add(args[0]); obj_in_carrying_arg2.add(args[1])
            elif pred == 'usable':
                if len(args) == 1: obj_in_usable_arg1.add(args[0])
            elif pred == 'loose':
                if len(args) == 1: obj_in_loose_arg1.add(args[0])
            elif pred == 'tightened':
                if len(args) == 1: obj_in_tightened_arg1.add(args[0])
            elif pred == 'link':
                if len(args) == 2: obj_in_link_arg1.add(args[0]); obj_in_link_arg2.add(args[1])
            elif pred == 'at':
                if len(args) == 2: obj_in_at_arg1.add(args[0]); obj_in_at_arg2.add(args[1])

        # Man: Appears as arg1 of 'carrying'. Is a locatable (appears as arg1 of 'at').
        man_candidates = obj_in_carrying_arg1.intersection(obj_in_at_arg1)
        if len(man_candidates) == 1:
            self.man_obj = list(man_candidates)[0]
        else:
            # Fallback: If no 'carrying' facts, man is the only locatable not a spanner/nut.
            spanner_candidates_from_predicates = obj_in_carrying_arg2.union(obj_in_usable_arg1)
            nut_candidates_from_predicates = obj_in_loose_arg1.union(obj_in_tightened_arg1)
            locatable_candidates = obj_in_at_arg1
            other_locatables = locatable_candidates - spanner_candidates_from_predicates - nut_candidates_from_predicates
            if len(other_locatables) == 1:
                self.man_obj = list(other_locatables)[0]
            # Else: Man object couldn't be uniquely identified. Heuristic might fail.
            # print("Warning: Could not uniquely identify man object.") # For debugging

        # Spanners: Appear as arg2 of 'carrying' or arg1 of 'usable'. Are locatable (appear as arg1 of 'at').
        self.spanner_objs = (obj_in_carrying_arg2.union(obj_in_usable_arg1)).intersection(obj_in_at_arg1)

        # Nuts: Appear as arg1 of 'loose' or 'tightened'. Are locatable (appear as arg1 of 'at').
        self.nut_objs = (obj_in_loose_arg1.union(obj_in_tightened_arg1)).intersection(obj_in_at_arg1)

        # Locations: Appear as args of 'link' or arg2 of 'at'.
        self.locations = obj_in_link_arg1.union(obj_in_link_arg2).union(obj_in_at_arg2)

        # --- Identify Goal Nuts ---
        for goal_fact_str in task.goals:
            pred, *args = self.parse_fact(goal_fact_str)
            if pred == 'tightened' and len(args) == 1:
                self.goal_nuts.add(args[0])

        # --- Build Location Graph ---
        self.adj = {loc: set() for loc in self.locations}
        for fact_str in task.static:
            pred, *args = self.parse_fact(fact_str)
            if pred == 'link' and len(args) == 2:
                l1, l2 = args
                if l1 in self.adj and l2 in self.adj: # Ensure locations are in our set
                     self.adj[l1].add(l2)
                     self.adj[l2].add(l1) # Links are bidirectional

        # --- Compute All-Pairs Shortest Paths ---
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = {}
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            while q:
                curr_node, dist = q.popleft()
                self.distances[start_node][curr_node] = dist
                for neighbor in self.adj.get(curr_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for a given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost to reach the goal, or float('inf') if
                 unreachable.
        """
        return self.spannerHeuristic(state)

    def spannerHeuristic(self, state):
        """
        Computes the domain-dependent heuristic value.

        See class docstring for detailed explanation.
        """
        # --- Parse State ---
        man_loc = None
        loose_nuts = set() # nut objects that are loose
        usable_spanners = set() # spanner objects that are usable
        carried_spanners = set() # spanner objects carried
        nut_locations = {} # nut -> loc
        spanner_locations = {} # spanner -> loc

        for fact in state:
            pred, *args = self.parse_fact(fact)
            if pred == 'at':
                if len(args) == 2:
                    obj, loc = args
                    if obj == self.man_obj:
                        man_loc = loc
                    elif obj in self.spanner_objs:
                        spanner_locations[obj] = loc
                    elif obj in self.nut_objs:
                        nut_locations[obj] = loc
            elif pred == 'carrying':
                if len(args) == 2:
                    m, s = args
                    if m == self.man_obj:
                        carried_spanners.add(s)
            elif pred == 'usable':
                if len(args) == 1:
                    usable_spanners.add(args[0])
            elif pred == 'loose':
                 if len(args) == 1 and args[0] in self.nut_objs:
                      loose_nuts.add(args[0])

        # Identify loose goal nuts
        loose_goal_nuts = loose_nuts.intersection(self.goal_nuts)
        N_loose_goals = len(loose_goal_nuts)

        # --- Check Goal Reached ---
        if N_loose_goals == 0:
            return 0

        # --- Check Solvability (Spanners) ---
        # Total usable spanners available anywhere
        N_total_usable_spanners = len(usable_spanners)
        if N_total_usable_spanners < N_loose_goals:
             return float('inf')

        # Identify usable spanners that are carried or at locations
        carried_usable_spanners = carried_spanners.intersection(usable_spanners)
        usable_spanners_at_loc = {s: loc for s, loc in spanner_locations.items() if s in usable_spanners}

        N_carried_usable = len(carried_usable_spanners)
        k = max(0, N_loose_goals - N_carried_usable) # Number of spanners to pick up

        # --- Compute Heuristic Components ---
        h = N_loose_goals # Cost for tighten actions

        # Cost to reach nuts (sum of distances from man to each loose goal nut)
        sum_dist_to_nuts = 0
        for nut in loose_goal_nuts:
            loc_n = nut_locations.get(nut)
            if loc_n is None:
                 # Loose goal nut has no location fact - problem likely unsolvable or state is malformed
                 # Based on assumptions, this shouldn't happen in valid reachable states
                 return float('inf') # Or handle as error
            dist = self.get_distance(man_loc, loc_n)
            if dist == float('inf'):
                 # Nut location is unreachable from man's location
                 return float('inf')
            sum_dist_to_nuts += dist
        h += sum_dist_to_nuts

        # Cost to pickup k spanners
        if k > 0:
            pickup_costs = []
            for s, loc_s in usable_spanners_at_loc.items():
                dist = self.get_distance(man_loc, loc_s)
                if dist == float('inf'):
                     # Usable spanner location is unreachable from man's location
                     return float('inf')
                pickup_costs.append(dist + 1) # +1 for pickup action

            if len(pickup_costs) < k:
                # Not enough usable spanners at locations to pick up the required amount
                # This should be caught by the total usable spanner check earlier, but good safety
                return float('inf')

            pickup_costs.sort()
            h += sum(pickup_costs[:k])

        return h

    def get_distance(self, loc1, loc2):
        """
        Gets the precomputed shortest distance between two locations.

        @param loc1: Start location.
        @param loc2: End location.
        @return: Shortest distance, or float('inf') if no path exists or locations are unknown.
        """
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
            # One or both locations are not in our graph, or no path exists.
            # This implies unsolvability or an issue with the location set/graph.
            return float('inf')
        return self.distances[loc1][loc2]

    def parse_fact(self, fact_string):
        """
        Parses a PDDL fact string into a list of strings [predicate, arg1, arg2, ...].

        @param fact_string: The fact string (e.g., '(at bob shed)').
        @return: A list of strings.
        """
        # Removes surrounding brackets and splits by space
        return fact_string.strip("()").split()

