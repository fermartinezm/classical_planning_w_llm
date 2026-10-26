import math
from collections import deque
# import heapq # Not strictly needed for O(V^2) Prim's

# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """
    Parses a PDDL fact string into a predicate and arguments.
    e.g., '(at obj loc)' -> ('at', ['obj', 'loc'])
    """
    # Removes parentheses and splits by space
    parts = fact_str[1:-1].split()
    if not parts: # Handle empty fact string, though unlikely in valid PDDL state
        return None, []
    return parts[0], parts[1:]

# Helper function to find object location from state facts
def get_location_from_state(state, obj_name):
    """
    Finds the location of an object from the current state facts.
    Returns the location string or None if the object is not 'at' a location.
    """
    for fact_str in state:
        pred, args = parse_fact(fact_str)
        if pred == 'at' and len(args) == 2 and args[0] == obj_name:
            return args[1]
    return None # Object not found at any location (e.g., carried or not in state)

# BFS for shortest path in an unweighted graph
def bfs(graph, start_node, all_nodes):
    """
    Computes shortest path distances from start_node to all other nodes in the graph.
    Returns a dictionary mapping node to distance.
    """
    distances = {node: math.inf for node in all_nodes}
    if start_node in all_nodes:
        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_loc = queue.popleft()

            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if neighbor in all_nodes and neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
    return distances

# Compute all-pairs shortest paths
def compute_all_pairs_shortest_paths(graph, locations):
    """
    Computes shortest path distances between all pairs of locations.
    Returns a dictionary of dictionaries dist_matrix[l1][l2].
    """
    dist_matrix = {}
    location_list = list(locations) # Ensure consistent order
    for start_loc in location_list:
        dist_matrix[start_loc] = bfs(graph, start_loc, locations)
    return dist_matrix

# Compute MST cost using Prim's algorithm (O(V^2) implementation)
def compute_mst_cost(locations_subset, dist_matrix):
    """
    Computes the Minimum Spanning Tree cost for a subset of locations
    using precomputed pairwise distances. Returns math.inf if the subset
    spans disconnected components.
    """
    if not locations_subset:
        return 0

    # Convert set to list for indexing
    locations_list = list(locations_subset)
    num_locations = len(locations_list)

    # Map location name to index
    loc_to_idx = {loc: i for i, loc in enumerate(locations_list)}

    min_cost = [math.inf] * num_locations
    in_mst = [False] * num_locations

    # Start with the first location in the subset
    start_idx = 0
    min_cost[start_idx] = 0

    total_mst_cost = 0
    visited_count = 0

    for _ in range(num_locations):
        # Find vertex with minimum key value from the set of vertices not yet included in MST
        min_val = math.inf
        min_idx = -1

        for v in range(num_locations):
            if not in_mst[v] and min_cost[v] < min_val:
                min_val = min_cost[v]
                min_idx = v

        # If min_idx is -1, it means there are unreachable nodes within the subset
        # or the subset spans disconnected components.
        if min_idx == -1:
             return math.inf # Indicate unsolvable

        u_idx = min_idx
        in_mst[u_idx] = True
        
        # Add the cost of the edge connecting this vertex to the MST
        # The first vertex has min_cost 0, so it doesn't add to the total cost.
        total_mst_cost += min_cost[u_idx]
        visited_count += 1

        # Update key values of the adjacent vertices of the picked vertex.
        # Consider only those vertices not yet included in MST
        u_loc = locations_list[u_idx]
        for v_idx in range(num_locations):
            if not in_mst[v_idx]:
                v_loc = locations_list[v_idx]
                # Get distance from the precomputed matrix
                # Handle cases where u_loc or v_loc might not be in dist_matrix keys
                # (e.g., locations only in subset, not in original graph nodes - should not happen if all_locations is built correctly)
                distance = dist_matrix.get(u_loc, {}).get(v_loc, math.inf)

                if distance < min_cost[v_idx]:
                    min_cost[v_idx] = distance

    # Check if all locations in the subset were visited
    if visited_count != num_locations:
        return math.inf # Subset spans disconnected components

    return total_mst_cost


class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the cost to reach the goal state (all required nuts tightened)
    by summing the estimated costs of the necessary actions: tighten_nut, pickup_spanner,
    and walk. It calculates the number of tighten actions needed (number of loose goal nuts)
    and the number of pickup actions needed (number of additional usable spanners required).
    The travel cost is estimated as the distance from the man's current location to the
    first required location (either a spanner to pick up or a nut to tighten), plus
    the cost of a Minimum Spanning Tree (MST) connecting all required locations
    (locations of loose goal nuts and locations of spanners to be picked up).
    The distances between locations are precomputed using BFS on the location graph.

    Assumptions:
    - There is exactly one man object. The heuristic attempts to identify the man
      by looking for an object in an 'at' fact in the initial state that is not
      identified as a spanner or nut based on predicate patterns.
    - Goal is a conjunction of (tightened nut_name) facts.
    - Spanners become unusable after one tighten_nut action.
    - The location graph defined by 'link' facts is undirected (link l1 l2 implies link l2 l1).
    - All locations mentioned in initial state, goal, and static links are part of the graph.
    - Object names do not contain spaces or parentheses.
    - Nut locations are static and provided in the initial state.

    Heuristic Initialization:
    The constructor precomputes static information:
    - Identifies the man's name.
    - Identifies the set of goal nuts.
    - Identifies all locations present in the problem by scanning initial state and static facts.
    - Builds the location graph from 'link' facts.
    - Computes all-pairs shortest paths between all locations using BFS.
    - Stores the static locations of all nuts found in the initial state.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to identify:
        - The man's current location (`loc_m`).
        - The set of usable spanners (`S_usable_S`).
        - The set of spanners carried by the man (`S_carried_S`).
        - The current location of all objects that are 'at' a location (`loc_map`).
    2.  Identify the set of nuts that are loose and are part of the goal (`N_loose_goal_S`) by checking the state and the precomputed `goal_nuts`.
    3.  If `N_loose_goal_S` is empty, the goal is reached, return 0.
    4.  Calculate the set of usable spanners carried by the man (`S_usable_carried_S`).
    5.  Calculate the set of usable spanners not carried by the man (`S_usable_not_carried_S`).
    6.  Calculate the number of spanners needed: `k_needed_spanners = len(N_loose_goal_S)`.
    7.  Calculate the number of usable spanners currently held: `k_carried_usable = len(S_usable_carried_S)`.
    8.  Calculate the number of additional spanners to pick up: `k_pickup = max(0, k_needed_spanners - k_carried_usable)`.
    9.  Check for unsolvability: If `k_needed_spanners > len(S_usable_S)` (total usable spanners in state), return `math.inf`.
    10. Initialize heuristic `h = k_needed_spanners` (cost for tighten actions).
    11. Add cost for pickup actions: `h += k_pickup`.
    12. Identify the set of locations the man *must* visit for tightening (`L_nuts_to_visit`) using the precomputed static nut locations.
    13. Identify the set of usable spanners not carried that are currently at a location (`S_usable_not_carried_located`).
    14. Select `k_pickup` spanners from `S_usable_not_carried_located` whose locations are closest to the man's current location (`loc_m`). Get their locations `L_spanners_to_pickup`. Handle cases where not enough reachable usable spanners exist.
    15. Define the set of required locations for travel: `L_required_locations = L_nuts_to_visit.union(L_spanners_to_pickup)`.
    16. Estimate travel cost:
        - If `L_required_locations` is empty (only happens if `N_loose_goal_S` was empty, handled in step 3), travel cost is 0.
        - Otherwise, calculate the minimum distance from `loc_m` to any location in `L_required_locations`. Handle unreachable required locations.
        - Calculate the MST cost connecting all locations in `L_required_locations` using the precomputed distance matrix. Handle disconnected required locations.
        - Travel cost = minimum distance to first required location + MST cost.
    17. Add travel cost to `h`.
    18. Return `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.man_name = None
        self.goal_nuts = set()
        self.all_locations = set()
        self.location_graph = {}
        self.dist_matrix = {}
        self.all_spanners = set()
        self.all_nuts = set()
        self.nut_locations = {} # Static nut locations

        # Identify all potential objects based on roles in initial state and goal
        potential_spanners = set()
        potential_nuts = set()
        potential_men = set()

        for fact_str in task.initial_state:
            pred, args = parse_fact(fact_str)
            if pred == 'at' and len(args) == 2:
                 obj, loc = args
                 self.all_locations.add(loc)
            elif pred == 'usable' and len(args) == 1:
                 potential_spanners.add(args[0])
            elif pred == 'carrying' and len(args) == 2:
                 potential_men.add(args[0])
                 potential_spanners.add(args[1])
            elif pred == 'loose' and len(args) == 1:
                 potential_nuts.add(args[0])
            elif pred == 'tightened' and len(args) == 1:
                 potential_nuts.add(args[0])

        # Identify goal nuts
        if isinstance(task.goals, str):
             goal_facts = {task.goals}
        else: # Assuming it's a set/frozenset
             goal_facts = task.goals

        for fact_str in goal_facts:
             pred, args = parse_fact(fact_str)
             if pred == 'tightened' and len(args) == 1:
                 self.goal_nuts.add(args[0])
                 potential_nuts.add(args[0]) # Goal nuts are nuts

        # Refine object types - assuming objects in potential_nuts are nuts,
        # objects in potential_spanners are spanners. The remaining object
        # in initial state 'at' facts is the man.
        known_nuts_spanners = potential_nuts.union(potential_spanners)
        man_candidate = None
        for fact_str in task.initial_state:
             pred, args = parse_fact(fact_str)
             if pred == 'at' and len(args) == 2:
                 obj, loc = args
                 if obj not in known_nuts_spanners:
                     man_candidate = obj
                     break # Assuming only one man

        if man_candidate:
             self.man_name = man_candidate
             # potential_men.add(self.man_name) # Not strictly needed

        self.all_nuts = potential_nuts
        self.all_spanners = potential_spanners

        # Store static nut locations and add initial man location to all_locations
        for fact_str in task.initial_state:
             pred, args = parse_fact(fact_str)
             if pred == 'at' and len(args) == 2:
                 obj, loc = args
                 if obj in self.all_nuts:
                     self.nut_locations[obj] = loc
                 if obj == self.man_name:
                     self.all_locations.add(loc)


        # Build location graph from static links
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred == 'link':
                l1, l2 = args
                self.all_locations.add(l1)
                self.all_locations.add(l2)
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1) # Assuming links are bidirectional

        # Compute all-pairs shortest paths
        self.dist_matrix = compute_all_pairs_shortest_paths(self.location_graph, list(self.all_locations))


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost to reach the goal, or math.inf if unsolvable.
        """
        # 1. Parse state
        loc_m = None
        S_usable_S = set()
        S_carried_S = set()
        loc_map = {} # Map object to its current location if 'at'

        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'at':
                obj, loc = args
                loc_map[obj] = loc
                if obj == self.man_name:
                    loc_m = loc
            elif pred == 'carrying' and len(args) == 2 and args[0] == self.man_name:
                S_carried_S.add(args[1])
            elif pred == 'usable' and len(args) == 1:
                S_usable_S.add(args[0])

        # Ensure man's location is found
        if loc_m is None:
             # Man must always be at a location in a valid state
             return math.inf

        # 2. Identify loose goal nuts in current state
        N_loose_goal_S = set()
        for nut in self.goal_nuts:
             # Check if the fact '(loose nut)' is in the state
             if f'(loose {nut})' in state:
                 N_loose_goal_S.add(nut)

        # 3. Check if goal reached
        if not N_loose_goal_S:
            return 0

        # 4. Calculate usable spanners carried
        S_usable_carried_S = S_usable_S.intersection(S_carried_S)

        # 5. Calculate usable spanners not carried
        S_usable_not_carried_S = S_usable_S - S_carried_S

        # 6. Calculate spanner needs
        k_needed_spanners = len(N_loose_goal_S)
        k_carried_usable = len(S_usable_carried_S)
        k_pickup = max(0, k_needed_spanners - k_carried_usable)

        # 7. Check for unsolvability based on total usable spanners
        if k_needed_spanners > len(S_usable_S):
             return math.inf

        # 8. Base cost (actions)
        h = k_needed_spanners # tighten_nut actions
        h += k_pickup         # pickup_spanner actions

        # 9. Identify locations to visit for nuts
        L_nuts_to_visit = set()
        for nut in N_loose_goal_S:
             # Nut locations are static, use precomputed map
             if nut in self.nut_locations:
                 L_nuts_to_visit.add(self.nut_locations[nut])
             else:
                 # Should not happen if nut locations were correctly identified in init
                 return math.inf # Invalid problem state/definition

        # 10. Identify usable spanners not carried that are at a location
        S_usable_not_carried_located = {s for s in S_usable_not_carried_S if s in loc_map}

        # 11. Select spanners to pickup locations
        L_spanners_to_pickup = set()
        if k_pickup > 0:
            # Find k_pickup closest usable spanners not carried and located
            available_spanners_with_loc = [(s, loc_map[s]) for s in S_usable_not_carried_located]

            # Filter out spanners at locations unreachable from the man's current location
            reachable_available_spanners = []
            if loc_m in self.dist_matrix: # Ensure man's location is in the distance matrix
                for s, loc in available_spanners_with_loc:
                     if loc in self.dist_matrix[loc_m]: # Ensure spanner location is reachable/in matrix
                         if self.dist_matrix[loc_m][loc] != math.inf:
                             reachable_available_spanners.append((s, loc))

            if len(reachable_available_spanners) < k_pickup:
                 # Not enough reachable usable spanners to satisfy the need
                 return math.inf

            # Sort by distance from man's current location and take the top k_pickup
            reachable_available_spanners.sort(key=lambda item: self.dist_matrix[loc_m][item[1]])

            L_spanners_to_pickup = {loc for s, loc in reachable_available_spanners[:k_pickup]}

        # 12. Define required locations for travel
        L_required_locations = L_nuts_to_visit.union(L_spanners_to_pickup)

        # 13. Estimate travel cost
        travel_cost = 0
        if L_required_locations:
            # Find distance from man's current location to the closest required location
            min_dist_to_first = math.inf
            
            # Ensure man's location is a valid key in the distance matrix
            if loc_m not in self.dist_matrix:
                 # This implies man is at a location not in the precomputed graph. Invalid.
                 return math.inf

            for req_loc in L_required_locations:
                 # Ensure required location is a valid key in the distance matrix
                 if req_loc in self.dist_matrix[loc_m]:
                     min_dist_to_first = min(min_dist_to_first, self.dist_matrix[loc_m][req_loc])
                 # else: # req_loc not in dist_matrix[loc_m] means it's unreachable or not in graph
                     # This case is handled by the check below (min_dist_to_first == math.inf)


            if min_dist_to_first == math.inf:
                 # No required location is reachable from man's current location
                 return math.inf

            travel_cost += min_dist_to_first

            # Calculate MST cost on required locations
            mst_cost = compute_mst_cost(L_required_locations, self.dist_matrix)

            if mst_cost == math.inf:
                 # Required locations are in disconnected components
                 return math.inf

            travel_cost += mst_cost

        # 14. Add travel cost to h
        h += travel_cost

        # 15. Return h
        return h
