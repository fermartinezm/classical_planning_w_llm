import collections
import math

class spannerHeuristic:
    """
    Domain-dependent heuristic for the spanner domain.

    Estimates the required number of actions to reach a goal state.
    The heuristic is calculated as the sum of:
    1. The number of loose nuts that are specified in the goal.
    2. The number of usable spanners the man needs to pick up to tighten
       all loose goal nuts.
    3. The shortest travel distance for the man to reach the location
       of the nearest loose goal nut.

    Returns infinity if the state is detected as unsolvable (e.g., not enough
    usable spanners in the entire problem, or unreachable goal nuts).
    """

    def __init__(self, task):
        """
        Initializes the spanner heuristic.

        Builds the location graph and computes shortest paths.
        Identifies the man, all objects, and goal nuts.
        """
        self.task = task
        self.location_graph = collections.defaultdict(set)
        self.locations = set()
        self.man = None
        self.goal_nuts = set()
        self.all_objects = set() # Store all objects mentioned in init/goal

        # Extract static information
        for fact_str in task.static:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'link':
                l1, l2 = objs
                self.location_graph[l1].add(l2)
                self.location_graph[l2].add(l1) # Assume links are bidirectional for travel
                self.locations.add(l1)
                self.locations.add(l2)

        # Compute shortest paths between all locations
        self.distances = self._compute_all_pairs_shortest_paths()

        # Identify all objects and goal nuts from initial state and goals
        nuts_and_spanners_candidates = set()
        for fact_str in task.initial_state | task.goals:
            pred, objs = self._parse_fact(fact_str)
            # Collect all object names (exclude predicates and locations)
            if pred not in ('link',): # Predicates to ignore as objects
                 for obj in objs:
                      # Heuristically exclude things that look like locations
                      # A better way needs type info. Assume locations are in self.locations
                      if obj not in self.locations:
                           self.all_objects.add(obj)

            if pred in ('loose', 'tightened') and len(objs) == 1:
                self.goal_nuts.add(objs[0])
                nuts_and_spanners_candidates.add(objs[0])
            if pred in ('usable',) and len(objs) == 1:
                nuts_and_spanners_candidates.add(objs[0])
            if pred == 'carrying' and len(objs) == 2:
                 # The second arg is the spanner
                 nuts_and_spanners_candidates.add(objs[1])


        # Find the man: object at a location in initial state that is not a known nut or spanner candidate
        # This inference is fragile without type information.
        for fact_str in task.initial_state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'at' and len(objs) == 2:
                obj_name, loc_name = objs
                if obj_name not in nuts_and_spanners_candidates:
                    self.man = obj_name
                    break # Assuming only one man

        if self.man is None:
             # Fallback if inference failed (e.g., man not at a location initially, or complex scenario)
             # Assume 'bob' as a last resort based on examples.
             # In a real scenario, this would need a proper parser or task representation.
             # print("Warning: Could not infer man object name. Assuming 'bob'.") # Avoid printing during search
             self.man = 'bob' # Fragile fallback
             self.all_objects.add(self.man) # Ensure man is in all_objects if inferred this way


    def _parse_fact(self, fact_string):
        """Helper to parse a fact string into predicate and objects."""
        # Remove parentheses and split by space
        parts = fact_string[1:-1].split()
        if not parts: # Handle empty string case just in case
            return None, []
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest paths between all pairs of locations using BFS."""
        distances = {}
        for start_node in self.locations:
            distances[start_node] = {}
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                current_node, dist = queue.popleft()
                distances[start_node][current_node] = dist
                for neighbor in self.location_graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return distances

    def get_distance(self, loc1, loc2):
        """Gets the shortest distance between two locations."""
        # Return infinity if locations are not in the computed distances (e.g., disconnected graph)
        return self.distances.get(loc1, {}).get(loc2, math.inf)


    def __call__(self, state):
        """
        Computes the domain-dependent heuristic for the spanner domain.

        Summary:
        Estimates the required number of actions to reach a goal state.
        The heuristic is calculated as the sum of:
        1. The number of loose nuts that are specified in the goal.
        2. The number of usable spanners the man needs to pick up to tighten
           all loose goal nuts.
        3. The shortest travel distance for the man to reach the location
           of the nearest loose goal nut.

        Returns infinity if the state is detected as unsolvable (e.g., not enough
        usable spanners in the entire problem, or unreachable goal nuts).

        Assumptions:
        - There is exactly one man object in the problem.
        - Nuts and spanners are locatable objects.
        - Links between locations defined by the 'link' predicate are
          bidirectional for travel purposes.
        - The location graph formed by 'link' predicates is connected,
          at least for all locations relevant to goals and initial state.
        - The man object name can be inferred from the initial state facts
          or defaults to 'bob' if inference fails.
        - The heuristic value is 0 if and only if the state is a goal state.
        - Objects that are not the man and not goal nuts are assumed to be spanners
          for the purpose of counting available spanners.

        Heuristic Initialization:
        - Parses static facts to build an undirected graph representing
          locations and links.
        - Computes all-pairs shortest paths between all known locations
          using Breadth-First Search (BFS).
        - Identifies all object names mentioned in the initial state or goals,
          excluding locations.
        - Identifies the set of nuts that are specified as 'tightened'
          in the task's goal conditions.
        - Identifies the unique man object name by examining initial state
          facts and excluding known nuts and spanners.

        Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current `state` satisfies the task's goal conditions
           using `self.task.goal_reached(state)`. If true, the heuristic is 0.
        2. Find the current location of the man (`man_location`) by searching
           for an `(at <man_name> ?l)` fact within the `state`. If the man's
           location cannot be found, the state is likely invalid or unreachable,
           and infinity is returned.
        3. Identify the set of nuts that are present in the task's goal
           conditions (`self.goal_nuts`) and are currently loose (`(loose <nut_name>)`
           is a fact in the `state`). Store these in `loose_goal_nuts`.
        4. If `loose_goal_nuts` is empty, it implies all goal nuts are already
           tightened. Given the domain, this should mean the goal is reached,
           and step 1 should have returned 0. If not, it indicates an issue
           or unhandled goal types; return 0 defensively.
        5. The number of `tighten_nut` actions required is equal to the number
           of loose goal nuts: `tighten_cost = len(loose_goal_nuts)`. Add this
           to the total heuristic value.
        6. Count the number of usable spanners the man is currently carrying.
           Find facts `(carrying <man_name> ?s)` and check if `(usable ?s)`
           is also in the `state`. Let this be `carried_usable_spanners`.
        7. Count the number of usable spanners available in the world (not carried
           by the man). Iterate through all identified objects that are not the man
           and not goal nuts (assumed to be spanners). Check if they are at a
           location (`(at <obj> ?l)` in state) and are usable (`(usable <obj>)`
           in state). Let this be `available_usable_spanners_world`.
        8. Check if the total number of usable spanners available in the problem
           (`carried_usable_spanners + available_usable_spanners_world`) is less
           than the number of loose goal nuts. If so, the problem is unsolvable
           from this state, return infinity.
        9. Calculate the number of additional usable spanners the man needs
           to pick up: `spanners_to_pickup = max(0, len(loose_goal_nuts) -
           carried_usable_spanners)`. Add this `pickup_cost = spanners_to_pickup`
           to the total heuristic value. This estimates the number of
           `pickup_spanner` actions.
        10. Find the location (`L_N`) for each nut `N` in `loose_goal_nuts`
            by searching for an `(at N L_N)` fact in the `state`.
        11. Calculate the minimum travel distance for the man from his current
            location (`man_location`) to any of the locations of the
            `loose_goal_nuts`. This is `min(self.get_distance(man_location, L_N)
            for N in loose_goal_nuts)`. Let this be `min_travel_to_nut`.
            If no loose goal nut is reachable, return infinity.
        12. Add `travel_cost = min_travel_to_nut` to the total heuristic value.
        13. Return the accumulated heuristic value.
        """
        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # 2. Identify man's current location
        man_location = None
        for fact_str in state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'at' and len(objs) == 2 and objs[0] == self.man:
                man_location = objs[1]
                break
        if man_location is None:
             # Man's location must be known in a valid state
             return math.inf

        # 3. Identify loose goal nuts and their locations
        loose_goal_nuts = set()
        nut_locations = {}
        for nut in self.goal_nuts:
            loose_fact = f'(loose {nut})'
            if loose_fact in state:
                loose_goal_nuts.add(nut)
                # Find the location of this nut
                nut_location = None
                for fact_str in state:
                    pred, objs = self._parse_fact(fact_str)
                    if pred == 'at' and len(objs) == 2 and objs[0] == nut:
                        nut_location = objs[1]
                        break
                if nut_location:
                    nut_locations[nut] = nut_location
                else:
                    # Nut location must be known in a valid state
                    return math.inf

        # 4. Handle case where no loose goal nuts remain (should be caught by step 1)
        if not loose_goal_nuts:
             return 0

        # 5. Number of tighten actions
        tighten_cost = len(loose_goal_nuts)

        # 6. Count carried usable spanners
        carried_usable_spanners = 0
        carried_spanners = set()
        for fact_str in state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'carrying' and len(objs) == 2 and objs[0] == self.man:
                carried_spanners.add(objs[1])

        for spanner in carried_spanners:
            usable_fact = f'(usable {spanner})'
            if usable_fact in state:
                carried_usable_spanners += 1

        # 7. Count usable spanners available in the world (not carried)
        available_usable_spanners_world = 0
        for obj in self.all_objects:
             # Assume objects that are not the man and not goal nuts are spanners
             if obj != self.man and obj not in self.goal_nuts:
                  # Check if it's at a location and usable
                  is_at_location = False
                  for fact_str in state:
                       pred, objs_at = self._parse_fact(fact_str)
                       if pred == 'at' and len(objs_at) == 2 and objs_at[0] == obj:
                            is_at_location = True
                            break
                  if is_at_location:
                       usable_fact = f'(usable {obj})'
                       if usable_fact in state:
                            available_usable_spanners_world += 1

        # 8. Check if enough spanners exist in total (carried + world)
        total_usable_spanners = carried_usable_spanners + available_usable_spanners_world
        if len(loose_goal_nuts) > total_usable_spanners:
             # Not enough usable spanners in the entire problem to tighten all nuts
             return math.inf # Indicate unsolvable state

        # 9. Calculate the number of additional usable spanners the man needs
        #    to pick up (only if enough exist in total)
        spanners_to_pickup = max(0, len(loose_goal_nuts) - carried_usable_spanners)
        pickup_cost = spanners_to_pickup

        # 10 & 11. Minimum travel distance to a loose goal nut location
        min_travel_to_nut = math.inf
        for nut in loose_goal_nuts:
            nut_loc = nut_locations.get(nut) # nut_loc is guaranteed to exist by step 3 check
            dist = self.get_distance(man_location, nut_loc)
            min_travel_to_nut = min(min_travel_to_nut, dist)

        # If min_travel_to_nut is still inf, it means no loose goal nut is reachable.
        if min_travel_to_nut == math.inf:
             return math.inf

        travel_cost = min_travel_to_nut

        # 12 & 13. Total heuristic
        heuristic_value = tighten_cost + pickup_cost + travel_cost

        return heuristic_value
