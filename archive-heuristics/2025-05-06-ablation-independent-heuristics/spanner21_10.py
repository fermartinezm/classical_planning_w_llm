# Helper function to parse a fact string
def parse_fact(fact_string):
    """Parses a PDDL fact string into a predicate and list of objects."""
    # Remove surrounding brackets and split by space
    parts = fact_string.strip('()').split()
    if not parts:
        return None, [] # Handle empty string case
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

# BFS to find shortest paths from a start location
def bfs(start_loc, graph, locations):
    """Performs BFS from start_loc to find distances to all other locations."""
    distances = {loc: float('inf') for loc in locations}
    distances[start_loc] = 0
    queue = deque([start_loc])
    visited = {start_loc}

    while queue:
        current_loc = queue.popleft()

        # Check if current_loc exists in the graph (it should if it's in locations)
        # and iterate over its neighbors.
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor in locations and neighbor not in visited: # Ensure neighbor is a known location
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)
    return distances

# All-pairs shortest path
def compute_all_pairs_shortest_paths(graph, locations):
    """Computes shortest paths between all pairs of locations using BFS."""
    dist = {}
    for start_loc in locations:
        dist[start_loc] = bfs(start_loc, graph, locations)
    return dist

# Extract objects by type from initial state facts
def get_objects_by_type(initial_state_facts):
    """Infers object types (man, spanner, nut, location) from initial state facts."""
    man_obj = None
    spanner_objs = set()
    nut_objs = set()
    location_objs = set()

    for fact_string in initial_state_facts:
        predicate, obj_list = parse_fact(fact_string)
        if not predicate: continue # Skip empty facts

        if predicate == 'at':
            # (at ?locatable ?location)
            if len(obj_list) == 2:
                item, loc = obj_list
                location_objs.add(loc)
                # item could be man, spanner, nut - identify from other predicates
        elif predicate == 'carrying':
            # (carrying ?man ?spanner)
            if len(obj_list) == 2:
                man_obj = obj_list[0]
                spanner_objs.add(obj_list[1])
        elif predicate == 'usable':
            # (usable ?spanner)
            if len(obj_list) == 1:
                 spanner_objs.add(obj_list[0])
        elif predicate == 'loose' or predicate == 'tightened':
            # (loose ?nut) or (tightened ?nut)
            if len(obj_list) == 1:
                 nut_objs.add(obj_list[0])
        elif predicate == 'link':
             # (link ?location ?location)
             if len(obj_list) == 2:
                 location_objs.add(obj_list[0])
                 location_objs.add(obj_list[1])

    # Collect all unique object names found
    all_objects = {man_obj} | spanner_objs | nut_objs | location_objs
    all_objects.discard(None) # Remove None if man_obj wasn't found (shouldn't happen in valid problems)

    # Store objects by type
    objects_by_type = {
        'man': [man_obj] if man_obj else [],
        'spanner': list(spanner_objs),
        'nut': list(nut_objs),
        'location': list(location_objs)
    }
    return objects_by_type, list(all_objects)


from heuristics.heuristic_base import Heuristic
from collections import deque
import math

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal (all nuts tightened) by summing:
    1. The number of loose nuts (representing tighten_nut actions).
    2. The number of additional usable spanners needed (representing pickup_spanner actions).
    3. The shortest path distance from the man's current location to the farthest
       relevant location (either a loose nut location or a location of a needed
       usable spanner), estimating the required travel cost.

    Assumptions:
    - The location graph defined by 'link' predicates is connected for all relevant locations.
    - The PDDL facts follow the expected structure for predicates like 'at', 'link',
      'carrying', 'usable', 'loose', 'tightened'.
    - Object types (man, spanner, nut, location) can be inferred from initial state facts
      or are otherwise available. This implementation infers types from predicate usage
      in the initial state.

    Heuristic Initialization:
    - Parses 'link' facts from the static information to build the location graph.
    - Computes all-pairs shortest paths between all locations using BFS.
    - Identifies all objects (man, spanners, nuts, locations) present in the initial state facts.
    - Stores the initial locations of nuts and spanners (those not carried).
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals

        # --- Initialization: Build location graph and compute distances ---
        self.location_graph = {}
        all_locations_set = set()

        for fact_string in task.static:
            predicate, obj_list = parse_fact(fact_string)
            if predicate == 'link':
                if len(obj_list) == 2:
                    loc1, loc2 = obj_list
                    self.location_graph.setdefault(loc1, []).append(loc2)
                    self.location_graph.setdefault(loc2, []).append(loc1)
                    all_locations_set.add(loc1)
                    all_locations_set.add(loc2)

        # Ensure all locations mentioned in initial state 'at' facts are included
        # even if they have no links (e.g., isolated locations)
        for fact_string in task.initial_state:
             predicate, obj_list = parse_fact(fact_string)
             if predicate == 'at' and len(obj_list) == 2:
                  _item, loc = obj_list
                  all_locations_set.add(loc)
        self.locations = list(all_locations_set)

        self.dist = compute_all_pairs_shortest_paths(self.location_graph, self.locations)

        # --- Initialization: Identify objects and initial fixed locations ---
        self.objects_by_type, self.all_objects = get_objects_by_type(task.initial_state)
        self.man = self.objects_by_type.get('man', [None])[0] # Assuming one man
        self.spanners = set(self.objects_by_type.get('spanner', []))
        self.nuts = set(self.objects_by_type.get('nut', []))

        self.nut_location = {}
        # Store initial location for spanners not carried. This is used as the
        # static location of the spanner if it's not currently carried.
        self.spanner_static_location = {}

        for fact_string in task.initial_state:
            predicate, obj_list = parse_fact(fact_string)
            if predicate == 'at':
                if len(obj_list) == 2:
                    item, loc = obj_list
                    if item in self.nuts:
                        self.nut_location[item] = loc
                    elif item in self.spanners:
                        # This is the initial location if not carried
                        self.spanner_static_location[item] = loc
            # Note: 'carrying' facts in initial state mean the spanner is not at a location

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.

        Keyword arguments:
        node -- the current state node

        Step-By-Step Thinking for Computing Heuristic:
        1.  Parse the current state facts to identify:
            -   The man's current location (`l_m`).
            -   The set of spanners currently carried by the man (`S_carried`).
            -   The set of spanners that are currently usable (`S_usable`).
            -   The set of nuts that are currently loose (`N_loose_set`).
            -   The current locations of spanners that are at locations (`spanner_current_location`).
        2.  Determine the locations of the loose nuts (`L_nuts_loose`) using the pre-calculated `self.nut_location`.
        3.  Identify the set of usable spanners that are currently at locations (not carried). Map these to their locations (`S_usable_at_locs_set`).
        4.  Count the total number of loose nuts (`N_loose`).
        5.  Count the number of usable spanners currently carried by the man (`S_usable_carried`).
        6.  Count the number of usable spanners currently at locations (`S_usable_at_locs`).
        7.  Calculate the total number of usable spanners available (`S_usable_total = S_usable_carried + S_usable_at_locs`).
        8.  If `N_loose` is 0, the goal is reached, return 0.
        9.  If `N_loose` is greater than `S_usable_total`, the goal is unreachable with the available usable spanners, return `float('inf')`.
        10. Calculate the number of additional usable spanners that need to be picked up (`Needed_spanners_count = max(0, N_loose - S_usable_carried)`).
        11. Identify the set of locations (`LS_pickup_needed_locations`) from which these additional spanners should ideally be picked up. This is done by finding the `Needed_spanners_count` usable spanners at locations that are closest to the man's current location.
            -   Create a list of `(distance_from_man, location, spanner)` for each usable spanner `s` that is at a location (`s` in `S_usable_at_locs_set`) and is usable (`s` in `S_usable`). The distance is `self.dist[l_m][location_of_s]`. Handle cases where `l_m` or `location_of_s` might not be in the distance map (should return inf).
            -   Sort this list by distance.
            -   Take the first `Needed_spanners_count` entries.
            -   Collect the unique locations from these entries into `LS_pickup_needed_locations`.
        12. Define the set of `Relevant_locations` the man needs to visit. This includes all locations with loose nuts (`L_nuts_loose`) and, if spanners are needed, the locations identified in `LS_pickup_needed_locations`.
        13. Calculate the `farthest_dist`: the maximum shortest path distance from the man's current location (`l_m`) to any location in `Relevant_locations`. Handle potential `inf` distances if a location is unreachable.
        14. The heuristic value is the sum of:
            -   `N_loose` (estimated tighten_nut actions)
            -   `Needed_spanners_count` (estimated pickup_spanner actions)
            -   `farthest_dist` (estimated walk actions related to reaching the furthest necessary point).
        """
        state = node.state

        # 1. Parse current state facts
        l_m = None
        S_carried = set()
        S_usable = set()
        N_loose_set = set()
        S_at_locs_set = set() # All spanners at locations
        spanner_current_location = {} # Current location for spanners at locations

        for fact_string in state:
            predicate, obj_list = parse_fact(fact_string)
            if not predicate: continue # Skip empty facts

            if predicate == 'at':
                if len(obj_list) == 2:
                    item, loc = obj_list
                    if item == self.man:
                        l_m = loc
                    elif item in self.spanners:
                        S_at_locs_set.add(item)
                        spanner_current_location[item] = loc
                    # Nut locations are static, use self.nut_location
            elif predicate == 'carrying':
                # (carrying ?man ?spanner)
                if len(obj_list) == 2:
                    S_carried.add(obj_list[1])
            elif predicate == 'usable':
                # (usable ?spanner)
                if len(obj_list) == 1:
                    S_usable.add(obj_list[0])
            elif predicate == 'loose':
                # (loose ?nut)
                if len(obj_list) == 1:
                    N_loose_set.add(obj_list[0])
            # Ignore 'tightened' facts as N_loose_set captures the remaining goal

        # Ensure man's location is found (should always be the case in valid states)
        if l_m is None:
             # This indicates an unexpected state structure, treat as unreachable
             return float('inf')

        # 2. Locations of loose nuts
        L_nuts_loose = {self.nut_location[n] for n in N_loose_set if n in self.nut_location}

        # 3. Usable spanners at locations
        S_usable_at_locs_set = {s for s in S_at_locs_set if s in S_usable}

        # 4. Counts
        N_loose = len(N_loose_set)
        S_usable_carried = len(S_carried.intersection(S_usable))
        S_usable_at_locs = len(S_usable_at_locs_set)

        # 5. Total usable spanners
        S_usable_total = S_usable_carried + S_usable_at_locs

        # 6. Goal check
        if N_loose == 0:
            return 0

        # 7. Unreachable check (based on usable spanners)
        if N_loose > S_usable_total:
            return float('inf')

        # 8. Needed spanners count
        Needed_spanners_count = max(0, N_loose - S_usable_carried)

        # 9. Identify LS_pickup_needed_locations
        LS_pickup_needed_locations = set()
        if Needed_spanners_count > 0:
            usable_spanners_at_locs_list = []
            for s in S_usable_at_locs_set:
                 loc_s = spanner_current_location.get(s) # Use current location from state
                 # Ensure both man's location and spanner's location are in the distance map
                 if l_m in self.dist and loc_s in self.dist[l_m]:
                    dist_to_spanner_loc = self.dist[l_m][loc_s]
                    # If distance is inf, this spanner is unreachable, skip it
                    if dist_to_spanner_loc != float('inf'):
                        usable_spanners_at_locs_list.append((dist_to_spanner_loc, loc_s, s))
                 else:
                     # This location is unreachable from the man's current location
                     pass # Skip this spanner

            # If after filtering, we don't have enough reachable usable spanners at locations
            if len(usable_spanners_at_locs_list) < Needed_spanners_count:
                 # This means the goal is unreachable even if S_usable_total >= N_loose
                 # because the needed spanners are not reachable.
                 return float('inf')

            # Sort by distance
            usable_spanners_at_locs_list.sort()

            # Take the locations of the closest Needed_spanners_count spanners
            for i in range(Needed_spanners_count): # Take exactly Needed_spanners_count
                 _dist, loc, _spanner = usable_spanners_at_locs_list[i]
                 LS_pickup_needed_locations.add(loc)


        # 10. Define Relevant_locations
        Relevant_locations = set(L_nuts_loose)
        if Needed_spanners_count > 0:
            Relevant_locations.update(LS_pickup_needed_locations)

        # 11. Calculate farthest_dist
        farthest_dist = 0
        if Relevant_locations:
            try:
                # Calculate distances from man's current location to all relevant locations
                distances_to_relevant = []
                for loc in Relevant_locations:
                    # Ensure the location is in the distance map from l_m
                    if l_m in self.dist and loc in self.dist[l_m]:
                        distances_to_relevant.append(self.dist[l_m][loc])
                    else:
                        # This location is unreachable from the man's current location
                        return float('inf')

                # If any distance is infinity, it means a relevant location is unreachable
                if any(d == float('inf') for d in distances_to_relevant):
                    return float('inf')
                farthest_dist = max(distances_to_relevant)
            except KeyError:
                 # This might happen if l_m or a location in Relevant_locations is not in the dist map
                 # (e.g., due to parsing error or disconnected graph)
                 return float('inf')


        # 12. Heuristic value
        h_value = N_loose + Needed_spanners_count + farthest_dist

        return h_value
