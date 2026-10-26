import math
from collections import deque

# Assume the following classes/structures are available from the planner environment:
# - Task: Represents the planning task, includes initial_state, goals, static facts.
# - state: A frozenset of strings representing facts.

def parse_fact(fact_string):
    """Parses a PDDL fact string into a list [predicate, arg1, arg2, ...]."""
    # Remove parentheses and split by spaces
    return fact_string.strip('()').split()

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the number of remaining 'tighten_nut' actions, the number of
        'pickup_spanner' actions required, and the minimum walking distance
        to a relevant location (either a loose goal nut location or an
        available usable spanner location if spanners are needed).

    Assumptions:
        - The problem instance is solvable.
        - The location graph defined by 'link' facts is undirected.
        - Object names in facts do not include the type suffix (e.g., 'bob' instead of 'bob - man').
        - There is exactly one man object in the domain, identifiable as the first argument of a 'carrying' fact in the initial state, or failing that, an object named 'bob' or containing 'man' in an initial 'at' fact.
        - Unreachable locations have infinite distance.

    Heuristic Initialization:
        The constructor processes the static information and initial state
        from the planning task.
        1. It identifies the man object, goal nuts, and initial locations
           of all nuts and spanners by parsing the initial state and goals.
        2. It builds a graph of locations based on the 'link' static facts.
           The graph is represented as an adjacency list.
        3. It computes the shortest path distances between all pairs of
           locations using Breadth-First Search (BFS). These distances are
           stored in a dictionary for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Identify the man's current location from the state facts. If the man's
           location cannot be determined (e.g., man object not found or not at
           a location), return infinity.
        2. Identify the set of nuts that are part of the goal and are currently 'loose'
           in the state ('LooseGoalNuts'). If this set is empty, the goal is reached,
           and the heuristic value is 0.
        3. Count the number of usable spanners the man is currently carrying
           ('CarriedUsableSpanners') by checking 'carrying' and 'usable' facts in the state.
        4. Calculate the number of additional spanners the man needs to pick up
           to tighten all 'LooseGoalNuts'. This is `max(0, len(LooseGoalNuts) - len(CarriedUsableSpanners))`.
        5. The base cost includes one action for each 'tighten_nut' and one action
           for each spanner that needs to be picked up.
           `base_cost = len(LooseGoalNuts) + max(0, len(LooseGoalNuts) - len(CarriedUsableSpanners))`.
        6. Determine the set of 'RelevantLocations' the man needs to visit.
           This set always includes the locations of all 'LooseGoalNuts' (looked up
           from the precomputed initial nut locations).
           If the man needs to pick up spanners (i.e., `len(CarriedUsableSpanners) < len(LooseGoalNuts)`),
           this set also includes the locations of all available usable spanners
           found at locations in the current state facts.
           If there are loose goal nuts but no relevant locations could be identified,
           return infinity (indicates an likely unsolvable state from here).
        7. Calculate the minimum shortest path distance from the man's current
           location to any location in the 'RelevantLocations' set using the
           precomputed distances. This estimates the walking cost to reach the
           first location where progress can be made. If a location is unreachable,
           its distance is considered infinite. If the minimum distance is infinity,
           return infinity.
        8. The final heuristic value is the sum of the `base_cost` and the
           minimum walking distance calculated in the previous step.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing task information.

        @param task: The planning task object.
        """
        self.task = task
        self.man = None
        self.goal_nuts = set()
        self.nut_locations = {}  # nut_name -> location_name (initial location)
        self.spanner_locations = {}  # spanner_name -> location_name (initial location)
        self.locations = set()
        self.graph = {}  # location_name -> set of neighbor_location_names
        self.distances = {}  # (loc1, loc2) -> distance

        # --- Parse Task Information ---

        all_facts = task.initial_state | task.goals | task.static

        # Identify potential objects and their initial locations
        potential_men = set()
        potential_nuts = set()
        potential_spanners = set()
        potential_locations = set()
        initial_at_facts = {} # obj -> loc from initial state

        for fact_str in all_facts:
            parts = parse_fact(fact_str)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at':
                obj, loc = args
                potential_locations.add(loc)
                if fact_str in task.initial_state:
                    initial_at_facts[obj] = loc
            elif predicate == 'carrying':
                carrier, carried = args
                potential_men.add(carrier)
                potential_spanners.add(carried)
            elif predicate == 'loose' or predicate == 'tightened':
                nut = args[0]
                potential_nuts.add(nut)
                if predicate == 'tightened' and fact_str in task.goals:
                    self.goal_nuts.add(nut)
            elif predicate == 'usable':
                spanner = args[0]
                potential_spanners.add(spanner)
            elif predicate == 'link':
                 l1, l2 = args
                 potential_locations.add(l1)
                 potential_locations.add(l2)
                 self.graph.setdefault(l1, set()).add(l2)
                 self.graph.setdefault(l2, set()).add(l1) # Links are typically bidirectional

        # Identify the man object
        if potential_men:
             # Assume the man is the object that can carry things
             self.man = list(potential_men)[0]
        else:
             # If no 'carrying' facts, try to find an object named 'bob' or 'man' in initial state 'at' facts
             for obj, loc in initial_at_facts.items():
                  if 'bob' in obj.lower() or 'man' in obj.lower():
                       self.man = obj
                       break
             # If self.man is still None, the heuristic might not work correctly

        # Store initial locations for nuts and spanners
        for obj, loc in initial_at_facts.items():
             if obj in potential_nuts:
                  self.nut_locations[obj] = loc
             elif obj in potential_spanners:
                  self.spanner_locations[obj] = loc

        # Ensure all locations from links and initial state are in self.locations
        self.locations.update(potential_locations)
        for loc in self.graph:
             self.locations.add(loc)
        for neighbors in self.graph.values():
             self.locations.update(neighbors)

        # Compute all-pairs shortest paths
        self._compute_distances()

    def _compute_distances(self):
        """Computes shortest path distances between all pairs of locations."""
        # Handle case with no locations or disconnected graph
        if not self.locations:
            return

        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node: 0}
            self.distances[(start_node, start_node)] = 0

            # Ensure start_node is in graph keys, even if it has no neighbors
            if start_node not in self.graph:
                 self.graph[start_node] = set()

            while q:
                curr_node, dist = q.popleft()

                # Ensure curr_node is in graph keys, even if it has no neighbors
                if curr_node not in self.graph:
                    self.graph[curr_node] = set() # Should already be added if in self.locations

                for neighbor in self.graph[curr_node]:
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # Distances for unreachable pairs remain absent in the dictionary,
        # which is handled by .get(..., math.inf) lookup.


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost to reach the goal.
        """
        # 1. Find man's current location
        man_loc = None
        if self.man: # Ensure man object was identified during init
            for fact_str in state:
                parts = parse_fact(fact_str)
                if parts and parts[0] == 'at' and parts[1] == self.man:
                    man_loc = parts[2]
                    break

        if man_loc is None:
             # Man's location not found in state (e.g., man object not identified, or state is malformed)
             # This state is likely unreachable or invalid.
             return math.inf

        # 2. Identify loose goal nuts in the current state
        loose_goal_nuts = {n for n in self.goal_nuts if '(loose %s)' % n in state}

        # 3. If no loose goal nuts, goal is reached
        if not loose_goal_nuts:
            return 0

        # 4. Count usable spanners the man is carrying
        carried_usable_spanners = set()
        for fact_str in state:
             parts = parse_fact(fact_str)
             if parts and parts[0] == 'carrying' and parts[1] == self.man:
                  spanner = parts[2]
                  # Check if the carried spanner is usable in this state
                  if '(usable %s)' % spanner in state:
                       carried_usable_spanners.add(spanner)

        # 5. Count available usable spanners at locations
        available_usable_spanners_at_loc = set() # Stores (spanner, location)
        # Iterate through all facts to find usable objects at locations
        for fact_str in state:
             parts = parse_fact(fact_str)
             if parts and parts[0] == 'at' and parts[1] != self.man: # Object at a location, not the man
                  obj, loc = parts[1], parts[2]
                  # Check if this object is usable
                  if '(usable %s)' % obj in state:
                       # Assume any usable object at a location is a usable spanner
                       available_usable_spanners_at_loc.add((obj, loc))


        # 6. Calculate number of spanners to pick up
        num_spanners_to_pick_up = max(0, len(loose_goal_nuts) - len(carried_usable_spanners))

        # 7. Base cost (tighten actions + pickup actions)
        base_cost = len(loose_goal_nuts) + num_spanners_to_pick_up

        # 8. Identify RelevantLocations
        nut_locations = {self.nut_locations[n] for n in loose_goal_nuts if n in self.nut_locations}
        spanner_locations = {loc for (s, loc) in available_usable_spanners_at_loc}

        relevant_locations = set()
        relevant_locations.update(nut_locations)

        # If we need to pick up spanners, the locations of available spanners are also relevant
        if num_spanners_to_pick_up > 0:
             relevant_locations.update(spanner_locations)

        # If there are loose goal nuts but no relevant locations could be identified,
        # it means the locations of the goal nuts were not found during init,
        # or there are loose goal nuts but no spanners available/carried when needed
        # and no spanner locations were found. This state is likely unsolvable.
        if not relevant_locations and loose_goal_nuts:
             return math.inf


        # 9. Calculate minimum distance to a relevant location
        min_dist = math.inf
        for loc in relevant_locations:
            dist = self.distances.get((man_loc, loc), math.inf)
            min_dist = min(min_dist, dist)

        # If min_dist is still infinity, it means all relevant locations are unreachable from man_loc
        if min_dist == math.inf:
             return math.inf

        # 10. Final heuristic value
        heuristic_value = base_cost + min_dist

        return heuristic_value
