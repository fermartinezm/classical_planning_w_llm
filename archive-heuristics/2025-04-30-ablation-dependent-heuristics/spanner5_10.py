import collections
import math

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the cost to reach a goal state by summing up:
    1. The number of loose nuts that need tightening (minimum tighten actions).
    2. The number of spanners that need to be picked up (based on the number of loose nuts and usable spanners already carried).
    3. An estimate of the walking cost, calculated as the sum of the shortest distance from the man's current location to the nearest loose nut location and the shortest distance from the man's current location to the nearest usable spanner location (if spanners are needed).

    Assumptions:
    - There is exactly one man object in the domain.
    - Nut locations are static (defined in the static part of the PDDL problem).
    - Spanners can be picked up and carried.
    - Tightening a nut consumes the usability of the spanner used.
    - The graph of locations defined by 'link' predicates is undirected.
    - The problem instance is solvable (unless the heuristic returns infinity).
    - Object types (man, spanner, nut, location) can be inferred from predicate usage in the problem description provided in the Task object's facts, initial_state, static, and goals.

    Heuristic Initialization:
    The constructor identifies objects by type based on predicate usage in the provided Task information. It extracts static nut locations and the graph of locations defined by 'link' predicates. It then precomputes the shortest path distances between all pairs of locations using Breadth-First Search (BFS).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Parse the state to find the man's current location, carried spanners, usable spanners, loose nuts, and spanner locations on the ground.
    2. Identify the set of loose nuts that are also goal nuts (`loose_goal_nuts`). If this set is empty, the goal is reached, and the heuristic is 0.
    3. Calculate the base heuristic value as the number of `loose_goal_nuts`. This represents the minimum number of `tighten_nut` actions required.
    4. Count the number of usable spanners the man is currently carrying.
    5. Calculate the number of additional spanners needed: `max(0, len(loose_goal_nuts) - num_usable_spanners_carried)`. Add this number to the heuristic, as each needed spanner requires a `pickup_spanner` action.
    6. Calculate the walking cost component:
        a. Find the set of locations where loose goal nuts are located (`loose_nut_locations`).
        b. Find the shortest distance from the man's current location to the nearest location in `loose_nut_locations`. Add this distance to the heuristic. If no loose nut locations are reachable, the state is likely unsolvable, return infinity.
        c. If additional spanners are needed (step 5 > 0):
            i. Find the set of locations where usable spanners are available on the ground (`usable_available_spanner_locations`).
            ii. Find the shortest distance from the man's current location to the nearest location in `usable_available_spanner_locations`. Add this distance to the heuristic. If no usable spanner locations are reachable when spanners are needed, the state is likely unsolvable, return infinity.
    7. Return the total calculated heuristic value.
    """
    def __init__(self, task):
        self.task = task
        # Extract goal nuts
        self.goal_nuts = {self._parse_fact(g)[1] for g in task.goals if self._parse_fact(g)[0] == 'tightened'}

        # Extract static information and identify objects by type
        self.nut_locations = {}
        self.locations = set()
        self.links = collections.defaultdict(list)
        self.man = None
        self.objects_by_type = collections.defaultdict(set)

        # Collect all objects and facts from initial state, static, and goals
        all_facts_strings = list(task.initial_state) + list(task.static) + list(task.goals)

        # First pass: Collect all objects appearing as arguments in facts
        all_args = set()
        for fact_string in all_facts_strings:
             pred, *args = self._parse_fact(fact_string)
             all_args.update(args)

        # Second pass: Classify objects based on predicate usage
        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()
        potential_locations = set()

        for fact_string in all_facts_strings:
             pred, *args = self._parse_fact(fact_string)
             if pred == 'at':
                  if len(args) == 2:
                       obj, loc = args
                       potential_locations.add(loc)
                  # else: print(f"Warning: Unexpected 'at' fact format: {fact_string}") # Suppress warnings for brevity
             elif pred == 'link':
                  if len(args) == 2:
                       loc1, loc2 = args
                       potential_locations.add(loc1)
                       potential_locations.add(loc2)
                  # else: print(f"Warning: Unexpected 'link' fact format: {fact_string}")
             elif pred == 'carrying':
                  if len(args) == 2:
                       m, s = args
                       potential_men.add(m)
                       potential_spanners.add(s)
                  # else: print(f"Warning: Unexpected 'carrying' fact format: {fact_string}")
             elif pred == 'usable':
                  if len(args) == 1:
                       s = args[0]
                       potential_spanners.add(s)
                  # else: print(f"Warning: Unexpected 'usable' fact format: {fact_string}")
             elif pred == 'loose' or pred == 'tightened':
                  if len(args) == 1:
                       n = args[0]
                       potential_nuts.add(n)
                  # else: print(f"Warning: Unexpected nut fact format: {fact_string}")

        # Refine classification (simple approach: assume disjoint types based on common predicate usage)
        # Objects appearing as first arg of 'carrying' are likely men
        self.objects_by_type['man'] = potential_men
        # Objects appearing as second arg of 'carrying' or arg of 'usable' are likely spanners
        self.objects_by_type['spanner'] = potential_spanners - self.objects_by_type['man']
        # Objects appearing as arg of 'loose' or 'tightened' are likely nuts
        self.objects_by_type['nut'] = potential_nuts - self.objects_by_type['man'] - self.objects_by_type['spanner']
        # Objects appearing as arg of 'link' or second arg of 'at' are likely locations
        self.objects_by_type['location'] = potential_locations - self.objects_by_type['man'] - self.objects_by_type['spanner'] - self.objects_by_type['nut']

        # Assume there's only one man
        if len(self.objects_by_type['man']) == 1:
             self.man = list(self.objects_by_type['man'])[0]
        elif len(self.objects_by_type['man']) > 1:
             # print(f"Warning: Found multiple potential men: {self.objects_by_type['man']}. Picking one arbitrarily.")
             self.man = list(self.objects_by_type['man'])[0]
        else:
             # print("Warning: Could not identify the man object.")
             self.man = None # Heuristic will likely return inf if man is not found/located

        # Extract static nut locations using the identified nut objects
        for fact_string in task.static:
             pred, *args = self._parse_fact(fact_string)
             if pred == 'at' and len(args) == 2:
                  obj, loc = args
                  if obj in self.objects_by_type['nut']:
                       self.nut_locations[obj] = loc

        # Populate locations set for BFS from identified locations
        self.locations = set(self.objects_by_type['location'])

        # Populate links for BFS
        for fact_string in task.static:
             pred, *args = self._parse_fact(fact_string)
             if pred == 'link' and len(args) == 2:
                  loc1, loc2 = args
                  if loc1 in self.locations and loc2 in self.locations:
                       self.links[loc1].append(loc2)
                       self.links[loc2].append(loc1)
                  # else: print(f"Warning: 'link' fact involves non-location object: {fact_string}")


        # Compute shortest path distances between all locations
        self.dist = self._compute_all_pairs_shortest_paths()

    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a tuple."""
        # Remove parentheses and split by space
        parts = fact_string[1:-1].split()
        return tuple(parts)

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all locations using BFS."""
        dist_map = {}
        for start_loc in self.locations:
            dist_map[start_loc] = self._bfs(start_loc)
        return dist_map

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all reachable locations."""
        q = collections.deque([(start_loc, 0)])
        distances = {start_loc: 0}
        visited = {start_loc}

        while q:
            current_loc, current_dist = q.popleft()

            # Ensure current_loc is in links keys, handle locations with no links
            for neighbor in self.links.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    q.append((neighbor, current_dist + 1))
        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # If man object wasn't identified during init, cannot compute heuristic
        if self.man is None:
             # print("Error: Man object not identified during heuristic initialization.")
             return math.inf # Cannot solve without a man

        # Parse state to find dynamic information
        man_location = None
        carried_spanners = set()
        usable_spanners_in_state = set()
        loose_nuts_in_state = set()
        spanner_locations_in_state = {} # {spanner_name: location}

        for fact_string in state:
            pred, *args = self._parse_fact(fact_string)
            if pred == 'at':
                if len(args) == 2:
                    obj, loc = args
                    if obj == self.man:
                        man_location = loc
                    elif obj in self.objects_by_type['spanner']:
                        spanner_locations_in_state[obj] = loc
            elif pred == 'carrying':
                if len(args) == 2:
                    m, s = args
                    if m == self.man and s in self.objects_by_type['spanner']:
                        carried_spanners.add(s)
            elif pred == 'usable':
                if len(args) == 1:
                    s = args[0]
                    if s in self.objects_by_type['spanner']:
                        usable_spanners_in_state.add(s)
            elif pred == 'loose':
                if len(args) == 1:
                    n = args[0]
                    if n in self.objects_by_type['nut']:
                        loose_nuts_in_state.add(n)

        # If man location is unknown, cannot compute heuristic
        if man_location is None:
             # print("Error: Man location not found in state.")
             return math.inf # Cannot solve if man's location is unknown

        # Identify loose goal nuts in the current state
        loose_goal_nuts = self.goal_nuts.intersection(loose_nuts_in_state)

        # If no loose goal nuts, the goal is reached
        if not loose_goal_nuts:
            return 0

        # Calculate heuristic components

        # 1. Cost for tighten actions
        h = len(loose_goal_nuts)

        # 2. Cost for pickup actions needed
        # Count usable spanners currently carried by the man
        usable_spanners_carried = carried_spanners.intersection(usable_spanners_in_state)
        num_usable_spanners_carried = len(usable_spanners_carried)

        # Number of additional spanners needed
        spanners_needed = max(0, len(loose_goal_nuts) - num_usable_spanners_carried)
        h += spanners_needed # Each needed spanner requires a pickup action

        # 3. Walking cost
        # Find locations of loose goal nuts
        loose_nut_locations = {self.nut_locations[n] for n in loose_goal_nuts if n in self.nut_locations}
        if not loose_nut_locations:
             # This should not happen if loose_goal_nuts is not empty and nut_locations are static and known
             # print("Error: Loose goal nuts found, but their static locations are unknown.")
             return math.inf # Should be unreachable state

        # Find locations of usable spanners available on the ground
        usable_available_spanners = usable_spanners_in_state - carried_spanners
        usable_available_spanner_locations = {
            spanner_locations_in_state[s]
            for s in usable_available_spanners
            if s in spanner_locations_in_state # Spanner must be on the ground to have a location in state
        }

        # Calculate distance from man's location
        if man_location not in self.dist:
             # Man is in a location not connected to the link graph
             # print(f"Error: Man location '{man_location}' not found in link graph.")
             return math.inf # Unreachable locations

        # Distance to nearest loose nut location
        min_dist_to_nut = math.inf
        min_dist_to_nut = min(self.dist[man_location].get(loc, math.inf) for loc in loose_nut_locations)

        # If nearest nut location is unreachable, return infinity
        if min_dist_to_nut == math.inf:
             # print(f"Error: Nearest loose nut location unreachable from {man_location}.")
             return math.inf

        h += min_dist_to_nut

        # Distance to nearest usable spanner location (if spanners are needed)
        if spanners_needed > 0:
             min_dist_to_spanner = math.inf
             if usable_available_spanner_locations:
                 min_dist_to_spanner = min(self.dist[man_location].get(loc, math.inf) for loc in usable_available_spanner_locations)
             else:
                 # Spanners are needed, but no usable spanners are available on the ground.
                 # This state is likely unsolvable regarding spanners.
                 # print("Error: Spanners needed, but no usable spanners available on the ground.")
                 return math.inf

             # If nearest spanner location is unreachable, return infinity
             if min_dist_to_spanner == math.inf:
                  # print(f"Error: Nearest usable spanner location unreachable from {man_location}.")
                  return math.inf

             h += min_dist_to_spanner

        return h
