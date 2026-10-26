import collections
import math

def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple (predicate, [arg1, arg2, ...])."""
    # Remove parentheses and split by spaces
    parts = fact_str.strip('()').split()
    if not parts:
        return None, []
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the cost to reach the goal (tighten all specified nuts)
    by summing the cost of required actions and estimated travel.
    It counts the number of loose goal nuts (representing the minimum tighten actions).
    If the man is not carrying a usable spanner and needs one, it adds the cost
    to walk to the nearest usable spanner on the ground and pick it up.
    Finally, it adds the estimated travel cost for the man (starting from his
    current location, or the spanner pickup location if a spanner was needed)
    to reach the location of the *nearest* loose goal nut.

    Assumptions:
    - The PDDL task is well-formed according to the spanner domain.
    - There is exactly one man object.
    - Nut locations are static (defined in static facts).
    - The location graph defined by 'link' predicates is undirected.
    - The problem is solvable if and only if there are enough usable spanners
      initially and all required locations are reachable.

    Heuristic Initialization:
    1.  Build the location graph from 'link' predicates in static facts.
    2.  Compute all-pairs shortest paths between locations using BFS. Store distances.
    3.  Identify the man object, all spanner objects, and all nut objects by examining predicate usage in initial state, static facts, and operators.
    4.  Store the static locations of nuts from static facts.
    5.  Identify the set of goal nuts from the task goals.
    6.  Check if the initial state has enough usable spanners for the goal nuts. If not, mark as unsolvable.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to find:
        -   The man's current location.
        -   Whether the man is carrying a spanner, and if so, which one.
        -   Which spanners are usable.
        -   Which nuts are loose.
        -   The locations of spanners on the ground.
    2.  Identify the set of goal nuts that are currently loose (`loose_goal_nuts`).
    3.  If `loose_goal_nuts` is empty, the goal is reached, return 0.
    4.  Check if the current state has enough usable spanners for the remaining loose goal nuts. If not, return infinity.
    5.  Initialize the heuristic value `h` with the number of loose goal nuts (cost for tighten actions).
    6.  Determine if the man is currently carrying a usable spanner.
    7.  Initialize `spanner_acquisition_cost` to 0 and `travel_start_loc` to the man's current location.
    8.  If the man is not carrying a usable spanner:
        -   Find the nearest usable spanner on the ground from the man's current location using precomputed distances.
        -   If no reachable usable spanner is found on the ground, return infinity.
        -   Set `spanner_acquisition_cost` to the distance to the nearest usable spanner plus 1 (for the pickup action).
        -   Set `travel_start_loc` to the location of the nearest usable spanner.
    9.  Find the location of the nearest loose goal nut from `travel_start_loc` using precomputed distances.
    10. If no reachable loose goal nut location is found, return infinity.
    11. Add `spanner_acquisition_cost` and the distance to the nearest loose goal nut location to `h`.
    12. Return `h`.
    """
    def __init__(self, task):
        self.task = task
        self.location_graph = collections.defaultdict(list)
        self.locations = set()
        self.nut_location = {}
        self.goal_nuts = set()
        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()

        # Identify object types and build location graph
        all_facts_init_static = set(task.initial_state) | set(task.static)
        all_facts_ops = set()
        for op in task.operators:
             all_facts_ops |= op.preconditions | op.add_effects | op.del_effects
        all_relevant_facts = all_facts_init_static | all_facts_ops

        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()

        for fact_str in all_relevant_facts:
             pred, args = parse_fact(fact_str)
             if pred == 'link' and len(args) == 2:
                 loc1, loc2 = args
                 self.location_graph[loc1].append(loc2)
                 self.location_graph[loc2].append(loc1) # Links are undirected
                 self.locations.add(loc1)
                 self.locations.add(loc2)
             elif pred == 'carrying' and len(args) == 2:
                 potential_men.add(args[0])
                 potential_spanners.add(args[1])
             elif pred in ['tightened', 'loose'] and len(args) == 1:
                 potential_nuts.add(args[0])
             elif pred == 'usable' and len(args) == 1:
                 potential_spanners.add(args[0])

        self.spanner_names = potential_spanners
        self.nut_names = potential_nuts

        # Identify man: Assume the object in initial state that is involved in 'at' or 'carrying'
        # and is not a known spanner or nut is the man.
        initial_locatables = set()
        for fact_str in task.initial_state:
             pred, args = parse_fact(fact_str)
             if pred == 'at' and len(args) >= 1: # at can have 1 or 2 args depending on PDDL version, but here it's 2
                  initial_locatables.add(args[0])
             elif pred == 'carrying' and len(args) >= 1: # carrying has 2 args
                  initial_locatables.add(args[0])

        inferred_men = initial_locatables - self.spanner_names - self.nut_names - self.locations # Locations are not locatable
        if len(inferred_men) == 1:
             self.man_name = list(inferred_men)[0]
        elif len(potential_men) == 1: # Fallback
             self.man_name = list(potential_men)[0]
        else:
             # Fallback: Assume the first object in initial state not in other categories
             all_initial_objects = set()
             for fact_str in task.initial_state:
                 all_initial_objects.update(parse_fact(fact_str)[1])
             inferred_men_from_initial = list(all_initial_objects - self.spanner_names - self.nut_names - self.locations)
             if inferred_men_from_initial:
                  self.man_name = inferred_men_from_initial[0]
             else:
                  # Should not happen in valid problems
                  # print("Warning: Could not definitively identify the man object.")
                  self.man_name = "unknown_man" # Placeholder


        # Compute all-pairs shortest paths
        self.dist = {}
        all_nodes = list(self.locations)
        for start_loc in all_nodes:
            self.dist.update(self._bfs(start_loc))

        # Store static nut locations
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred == 'at' and len(args) == 2:
                 obj, loc = args
                 if obj in self.nut_names:
                      self.nut_location[obj] = loc

        # Identify goal nuts
        for goal_fact_str in task.goals:
            pred, args = parse_fact(goal_fact_str)
            if pred == 'tightened' and len(args) == 1 and args[0] in self.nut_names:
                self.goal_nuts.add(args[0])

        # Check if enough usable spanners exist initially for the goal nuts
        initial_usable_spanners_count = 0
        initial_usable_spanners = set()
        initial_carried_spanner = None
        initial_spanners_on_ground = {}

        for fact_str in task.initial_state:
             pred, args = parse_fact(fact_str)
             if pred == 'usable' and len(args) == 1 and args[0] in self.spanner_names:
                  initial_usable_spanners.add(args[0])
             elif pred == 'carrying' and len(args) == 2 and args[0] == self.man_name and args[1] in self.spanner_names:
                  initial_carried_spanner = args[1]
             elif pred == 'at' and len(args) == 2 and args[0] in self.spanner_names:
                  initial_spanners_on_ground[args[0]] = args[1]

        if initial_carried_spanner in initial_usable_spanners:
             initial_usable_spanners_count += 1
        for spanner in initial_spanners_on_ground:
             if spanner in initial_usable_spanners:
                  initial_usable_spanners_count += 1

        if len(self.goal_nuts) > initial_usable_spanners_count:
             self.unsolvable = True
        else:
             self.unsolvable = False


    def _bfs(self, start_node):
        """Performs BFS from start_node to find distances to all reachable nodes."""
        distances = {node: math.inf for node in self.locations}
        if start_node not in self.locations:
             # Start node is not a known location, cannot compute distances
             return {} # Return empty map

        distances[start_node] = 0
        queue = collections.deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            for neighbor in self.location_graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        # Store distances as (start, end) -> dist
        dist_map = {}
        for end_node, d in distances.items():
             dist_map[(start_node, end_node)] = d
        return dist_map


    def __call__(self, state):
        if self.unsolvable:
             return math.inf

        # 1. Parse current state
        man_loc = None
        carried_spanner = None
        usable_spanners_in_state = set()
        loose_nuts_in_state = set()
        spanner_ground_loc = {} # Spanners currently on the ground

        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'at' and len(args) == 2:
                obj, loc = args
                if obj == self.man_name:
                    man_loc = loc
                elif obj in self.spanner_names: # Check if object is a known spanner
                     spanner_ground_loc[obj] = loc
            elif pred == 'carrying' and len(args) == 2 and args[0] == self.man_name:
                carried_spanner = args[1]
            elif pred == 'usable' and len(args) == 1 and args[0] in self.spanner_names: # Check if usable spanner is known
                usable_spanners_in_state.add(args[0])
            elif pred == 'loose' and len(args) == 1 and args[0] in self.nut_names: # Check if loose nut is known
                loose_nuts_in_state.add(args[0])

        # Ensure man_loc is found
        if man_loc is None:
             # Man's location is not in the state facts, problem state is malformed?
             # Return infinity as a safe fallback for unexpected state.
             # print(f"Error: Man {self.man_name} location not found in state.")
             return math.inf


        # 2. Identify loose goal nuts
        loose_goal_nuts = self.goal_nuts.intersection(loose_nuts_in_state)

        # 3. If loose_goal_nuts is empty, goal is reached
        if not loose_goal_nuts:
            return 0

        # 4. Count usable spanners available in current state
        num_usable_available = 0
        man_has_usable_spanner = False
        if carried_spanner in usable_spanners_in_state:
             num_usable_available += 1
             man_has_usable_spanner = True

        usable_spanners_on_ground = [s for s in usable_spanners_in_state if s in spanner_ground_loc]
        num_usable_available += len(usable_spanners_on_ground)

        # 5. Check if enough usable spanners exist
        if len(loose_goal_nuts) > num_usable_available:
            return math.inf

        # 6. Initialize heuristic
        h = len(loose_goal_nuts) # Cost for tighten actions

        # 7. Determine spanner acquisition cost and travel start location
        spanner_acquisition_cost = 0
        travel_start_loc = man_loc
        nearest_usable_spanner_loc = None

        if not man_has_usable_spanner:
            # Find nearest usable spanner on the ground
            min_dist_to_spanner = math.inf

            for spanner in usable_spanners_on_ground:
                loc_s = spanner_ground_loc[spanner]
                d = self.dist.get((man_loc, loc_s), math.inf)
                if d < min_dist_to_spanner:
                    min_dist_to_spanner = d
                    nearest_usable_spanner_loc = loc_s

            if nearest_usable_spanner_loc is None or min_dist_to_spanner == math.inf:
                 # This implies we need a spanner but cannot reach any usable one on the ground.
                 # This should be covered by the num_usable_available check if initial state was solvable.
                 # It means we are in a state where remaining nuts > remaining usable spanners,
                 # or the remaining usable spanners are unreachable.
                 return math.inf

            spanner_acquisition_cost = min_dist_to_spanner + 1
            travel_start_loc = nearest_usable_spanner_loc

        # 9. Find nearest loose goal nut location from travel_start_loc
        min_dist_to_nut = math.inf
        nearest_nut_loc = None

        for nut in loose_goal_nuts:
            nut_loc = self.nut_location.get(nut)
            if nut_loc is None:
                 # Should not happen if nut_location is populated correctly from static facts
                 # print(f"Warning: Location for nut {nut} not found in static facts.")
                 return math.inf # Cannot reach nut

            d = self.dist.get((travel_start_loc, nut_loc), math.inf)
            if d < min_dist_to_nut:
                 min_dist_to_nut = d
                 nearest_nut_loc = nut_loc


        if nearest_nut_loc is None or min_dist_to_nut == math.inf:
             # Cannot reach any loose goal nut from travel_start_loc
             return math.inf

        travel_cost = min_dist_to_nut

        # 11. Add costs to heuristic
        h += spanner_acquisition_cost + travel_cost

        return h
