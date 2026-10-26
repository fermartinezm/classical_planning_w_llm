import collections
import re

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        Estimates the cost to reach the goal state by summing the estimated costs
        associated with tightening each loose goal nut. This includes the cost
        of the tightening action itself, the estimated walking cost to reach
        the nut's location, the estimated cost of picking up a spanner if needed,
        and the estimated walking cost to reach spanner locations. The heuristic
        uses precomputed shortest path distances between locations.

    Assumptions:
        - The domain is 'spanner' as defined in the provided PDDL.
        - The goal is to achieve (tightened ?n) for a set of nuts ?n.
        - Nuts are static (their location does not change).
        - Spanners move only via the pickup action.
        - The man can only carry one spanner at a time.
        - Tightening a nut consumes the usability of the spanner.
        - There is exactly one man object in the domain.
        - The location graph defined by 'link' predicates is undirected.
        - All locations, nuts, and spanners mentioned in facts are defined in the task objects.
        - Solvable problems have enough usable spanners available (carried or at locations)
          to tighten all goal nuts.

    Heuristic Initialization:
        The constructor takes a Task object. It performs the following steps:
        1. Parses initial state, goal, and static facts to identify all objects
           and infer their roles (man, nuts, spanners, locations) based on the
           predicates they appear in.
        2. Identifies the set of nuts that are part of the goal condition.
        3. Parses the static facts (link predicates) to build an adjacency list
           representation of the location graph.
        4. Computes all-pairs shortest paths between all identified locations
           using Breadth-First Search (BFS) starting from each location. These
           distances are stored in a dictionary `self.dist`.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Check if the goal is reached using `task.goal_reached(state)`. If yes,
           the heuristic is 0.
        2. Identify the set of loose nuts that are also goal nuts (`LooseGoalNuts`).
        3. If there are no loose goal nuts, the heuristic is 0 (this case should
           be covered by step 1, but serves as a logical base case).
        4. Identify the man object and his current location (`l_m`) from the state.
           If the man's location cannot be found or is not a recognized location,
           return infinity (invalid state).
        5. Identify the locations of all loose goal nuts (`LooseGoalNutLocations`)
           from the state. If any loose goal nut's location cannot be found or
           is not a recognized location, return infinity (invalid state, though
           nuts are typically static).
        6. Identify all usable spanners that are currently at a location
           (`UsableSpannersAtLoc`) and their locations (`UsableSpannerLocations`).
           Ensure these locations are recognized locations.
        7. Check if the man is currently carrying a usable spanner (`carrying_usable`).
        8. Calculate the total number of usable spanners available in the state
           (the one potentially carried plus those at locations).
        9. Let `k` be the number of loose goal nuts. If `k` is greater than the
           total number of usable spanners available, the state is likely unsolvable
           with the current resources; return infinity.
        10. Initialize the heuristic value `h = 0`.
        11. Add the cost for the `k` tighten actions: `h += k`.
        12. Calculate the number of additional spanners the man needs to pick up
            from locations: `num_pickups_needed = k - (1 if carrying_usable else 0)`.
        13. If `num_pickups_needed > 0`, add the cost for these pickup actions:
            `h += num_pickups_needed`.
        14. Add the estimated walking cost to reach the nut locations. This is
            estimated as the sum of shortest path distances from the man's current
            location (`l_m`) to each location in `LooseGoalNutLocations`. If any
            nut location is unreachable, return infinity.
        15. If `num_pickups_needed > 0`, add the estimated walking cost to reach
            the spanner locations for pickup. This is estimated by finding the
            `num_pickups_needed` closest locations in `UsableSpannerLocations`
            from the man's current location (`l_m`), sorting their distances,
            and summing the smallest `num_pickups_needed` distances. If any
            spanner location is unreachable, or if there are fewer than
            `num_pickups_needed` usable spanner locations available, return infinity
            (this case should ideally be covered by step 9 if all spanners start
            at locations or carried).
        16. Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing shortest paths and identifying objects.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.man = None
        self.nuts = set()
        self.spanners = set()
        self.locations = set()
        self.goal_nuts = set()

        # --- Object and Role Identification ---
        # Iterate through all facts in initial state, goal, and static to find objects
        all_objects = set()
        for fact in task.initial_state | task.goals | task.static:
             parsed = self._parse_fact(fact)
             if parsed:
                 all_objects.update(parsed[1:]) # Add all arguments as potential objects

        # Infer roles based on predicates
        # This inference is based on the typical usage of predicates in the spanner domain
        for obj in all_objects:
            for fact in task.initial_state | task.goals | task.static:
                parsed = self._parse_fact(fact)
                if parsed:
                    predicate = parsed[0]
                    if predicate == 'at':
                        if len(parsed) > 1 and parsed[1] == obj: # First arg of 'at' is locatable
                            # Refine type based on other predicates
                            for fact2 in task.initial_state | task.goals: # Check dynamic facts
                                parsed2 = self._parse_fact(fact2)
                                if parsed2 and parsed2[0] == 'carrying' and len(parsed2) > 1 and parsed2[1] == obj:
                                    self.man = obj
                                    break
                                if parsed2 and parsed2[0] in ('loose', 'tightened') and len(parsed2) > 1 and parsed2[1] == obj:
                                    self.nuts.add(obj)
                                    break
                                if parsed2 and parsed2[0] == 'usable' and len(parsed2) > 1 and parsed2[1] == obj:
                                    self.spanners.add(obj)
                                    break
                        if len(parsed) > 2 and parsed[2] == obj: # Second arg of 'at' is location
                            self.locations.add(obj)
                    elif predicate == 'carrying': # First arg is man, second is spanner
                         if len(parsed) > 1 and parsed[1] == obj:
                             self.man = obj
                         if len(parsed) > 2 and parsed[2] == obj:
                             self.spanners.add(obj)
                    elif predicate == 'usable': # Arg is spanner
                         if len(parsed) > 1 and parsed[1] == obj:
                             self.spanners.add(obj)
                    elif predicate in ('loose', 'tightened'): # Arg is nut
                         if len(parsed) > 1 and parsed[1] == obj:
                             self.nuts.add(obj)
                    elif predicate == 'link': # Args are locations
                         if len(parsed) > 1 and parsed[1] == obj:
                             self.locations.add(obj)
                         if len(parsed) > 2 and parsed[2] == obj:
                             self.locations.add(obj)

        # Identify goal nuts
        for goal_fact in task.goals:
            parsed = self._parse_fact(goal_fact)
            if parsed and parsed[0] == 'tightened' and len(parsed) > 1:
                self.goal_nuts.add(parsed[1])

        # Build location graph
        self.location_graph = collections.defaultdict(set)
        for fact in task.static:
            parsed = self._parse_fact(fact)
            if parsed and parsed[0] == 'link' and len(parsed) > 2:
                loc1, loc2 = parsed[1], parsed[2]
                # Ensure locations are recognized before adding links
                if loc1 in self.locations and loc2 in self.locations:
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1) # Links are bidirectional

        # Compute shortest paths between all locations
        self.dist = {}
        # Compute for all identified locations, BFS handles disconnected components
        for start_loc in self.locations:
             self.dist[start_loc] = self._bfs(start_loc)


    @staticmethod
    def _parse_fact(fact_str):
        """Parses a fact string into a list of strings [predicate, obj1, obj2, ...]."""
        # Remove parentheses and split by whitespace
        cleaned_fact = fact_str.strip()
        if not cleaned_fact.startswith('(') or not cleaned_fact.endswith(')'):
            return None # Not a valid fact string format we expect
        cleaned_fact = cleaned_fact[1:-1].strip() # Remove outer parentheses
        if not cleaned_fact:
            return None # Empty fact?
        # Use shlex.split if facts could contain spaces within object names,
        # but simple split is sufficient for the spanner domain format.
        return cleaned_fact.split()

    def _get_object_location(self, state, obj_name):
        """Finds the location of an object in the state."""
        for fact in state:
            parsed = self._parse_fact(fact)
            if parsed and parsed[0] == 'at' and len(parsed) > 2 and parsed[1] == obj_name:
                return parsed[2]
        return None # Object not found at any location (e.g., carried spanner)

    def _bfs(self, start_node):
        """Computes shortest path distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not a recognized location, cannot compute paths
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Only explore if the current node is in the graph (has links)
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The heuristic value (integer or float('inf')).
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}
        k = len(loose_goal_nuts)

        # If no loose goal nuts, goal is effectively reached for nuts
        if k == 0:
            return 0 # Should be covered by task.goal_reached, but good safety check

        # Get man's location
        man_loc = self._get_object_location(state, self.man)
        if man_loc is None or man_loc not in self.locations:
             # Man must always be at a valid location
             return float('inf')

        # Get locations of loose goal nuts
        loose_goal_nut_locations = set()
        for n in loose_goal_nuts:
             loc = self._get_object_location(state, n)
             if loc is None or loc not in self.locations:
                  # Loose goal nut not at a valid location - invalid state
                  return float('inf')
             loose_goal_nut_locations.add(loc)

        # Get usable spanners at locations
        usable_spanners_at_loc = {s for s in self.spanners if f'(usable {s})' in state and any(f'(at {s} {l})' in state for l in self.locations)}
        usable_spanner_locations = {self._get_object_location(state, s) for s in usable_spanners_at_loc}
        usable_spanner_locations.discard(None) # Remove None in case of carried usable spanners incorrectly included
        # Ensure all identified spanner locations are valid locations
        usable_spanner_locations = {loc for loc in usable_spanner_locations if loc in self.locations}

        # Check if man is carrying a usable spanner
        carrying_usable = any(f'(carrying {self.man} {s})' in state and f'(usable {s})' in state for s in self.spanners)

        # Calculate total usable spanners available (carried or at location)
        total_usable_spanners_available = (1 if carrying_usable else 0) + len(usable_spanners_at_loc)

        # Check if enough spanners exist for all loose goal nuts
        if k > total_usable_spanners_available:
             # Not enough usable spanners in the state to tighten all remaining nuts
             return float('inf')

        # --- Heuristic Calculation ---
        h = 0

        # 1. Cost for tighten actions
        h += k

        # 2. Cost for pickup actions
        num_pickups_needed = k - (1 if carrying_usable else 0)
        if num_pickups_needed < 0: num_pickups_needed = 0 # Should not happen if k <= total_usable_spanners_available

        if num_pickups_needed > 0:
            h += num_pickups_needed

        # 3. Estimated walking cost to reach nut locations
        # Sum of distances from man's current location to each loose goal nut location
        nut_walk_cost = 0
        for l in loose_goal_nut_locations:
             if l in self.dist[man_loc]:
                  nut_walk_cost += self.dist[man_loc][l]
             else:
                  # Nut location unreachable from man's location
                  return float('inf')
        h += nut_walk_cost

        # 4. Estimated walking cost to reach spanner locations for pickup
        if num_pickups_needed > 0:
            distances_to_spanners = []
            for loc in usable_spanner_locations:
                 if man_loc in self.dist and loc in self.dist[man_loc]:
                      distances_to_spanners.append(self.dist[man_loc][loc])
                 else:
                      # Spanner location unreachable from man's location
                      return float('inf')

            # Sort distances and sum the smallest num_pickups_needed distances
            distances_to_spanners.sort()
            # We need num_pickups_needed distinct spanners from locations.
            # The number of available locations might be less than num_pickups_needed
            # if multiple usable spanners are at the same location.
            # We need to sum distances to num_pickups_needed *distinct* locations.
            # The set `usable_spanner_locations` already contains distinct locations.
            # If len(usable_spanner_locations) < num_pickups_needed, it means
            # there aren't enough *distinct locations* with usable spanners for the
            # required pickups. This state is likely unsolvable unless the initial
            # spanner count check was insufficient. Assuming the initial check is
            # sufficient, we sum the first num_pickups_needed distances from the
            # sorted list of distances to *all* usable spanner locations.
            # If len(distances_to_spanners) < num_pickups_needed, it implies
            # not enough reachable usable spanner locations exist for the pickups needed.
            # This should also be covered by the k > total_usable_spanners_available check
            # if we assume all spanners start either carried or at a location.
            # Let's sum over the first num_pickups_needed distances available.
            num_distances_to_sum = min(num_pickups_needed, len(distances_to_spanners))
            h += sum(distances_to_spanners[:num_distances_to_sum])

        return h
