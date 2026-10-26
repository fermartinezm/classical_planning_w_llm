from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Helper to parse a PDDL fact string into a list of parts."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we have the same number of parts and args for matching
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_location_graph(static_facts):
    """Builds an adjacency list graph from link facts."""
    graph = {}
    for fact in static_facts:
        if match(fact, "link", "*", "*"):
            _, loc1, loc2 = get_parts(fact)
            graph.setdefault(loc1, []).append(loc2)
            graph.setdefault(loc2, []).append(loc1) # Links are bidirectional
    return graph

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of locations using BFS."""
    distances = {}
    # Ensure all nodes in the graph have an entry, even if isolated
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    for start_node in all_nodes:
        distances[start_node] = {}
        queue = deque([(start_node, 0)])
        visited = {start_node}
        while queue:
            current_node, dist = queue.popleft()
            distances[start_node][current_node] = dist
            # Use graph.get(current_node, []) to handle nodes with no neighbors
            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return distances

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal by summing:
    1. The number of loose goal nuts (minimum tighten actions).
    2. The number of pickup actions needed to acquire enough usable spanners
       from locations.
    3. The minimum walking distance from the man's current location to any
       location of interest (either a loose goal nut or a location with a
       needed usable spanner).

    Assumptions:
    - The problem is solvable (there are enough usable spanners in total).
    - Links between locations are bidirectional.
    - Man can carry multiple spanners.
    - Nuts are static (do not move).
    - Spanners are static unless carried.
    - Usable predicate for spanners is dynamic (becomes false after use).
    - The object involved in a 'carrying' fact is the man. If no 'carrying'
      fact initially, the man is inferred from 'at' facts by exclusion of
      spanners/nuts.
    - Objects involved in 'usable', 'loose', or 'tightened' predicates are spanners or nuts.
    - Objects involved as the second argument in 'at' or arguments in 'link' are locations.

    Heuristic Initialization:
    - Parses initial state and static facts to identify the man, all spanners,
      all nuts, and all locations based on predicate usage.
    - Stores initial locations of nuts (as nuts are static).
    - Stores the set of goal nuts.
    - Builds the location graph from 'link' predicates and computes all-pairs
      shortest paths.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the goal is already reached (all goal nuts are tightened). If so, heuristic is 0.
    2. Identify the man's current location from the state by finding the 'at' fact for the man object.
       If the man's location cannot be found, return infinity (problematic state).
    3. Count the number of loose goal nuts (N_loose_goals) by checking which goal nuts
       are currently in a 'loose' state. Only considers goal nuts whose initial
       locations were successfully identified.
    4. Count the number of usable spanners the man is currently carrying (N_carried_usable)
       by checking 'carrying' and 'usable' facts in the state.
    5. Count the number of usable spanners currently at locations (N_at_loc_usable)
       by finding spanners that are 'at' a location and are 'usable', excluding those carried.
    6. Check for unsolvability: If the total number of usable spanners available
       (carried + at locations) is less than N_loose_goals, return infinity.
    7. Calculate the number of additional usable spanners the man needs to pick up
       from locations (N_needed_from_loc = max(0, N_loose_goals - N_carried_usable)).
    8. Identify the locations of all loose goal nuts using the pre-calculated initial locations
       (nuts are assumed static).
    9. Identify the current locations of all usable spanners that are currently at locations.
    10. If N_needed_from_loc > 0, find the N_needed_from_loc usable spanners at locations
        that are closest to the man's current location. Get their locations. This involves
        calculating distances from the man's current location to all usable spanner locations,
        sorting them, and selecting the locations of the top N_needed_from_loc spanners.
        If any of these needed spanner locations are unreachable, return infinity.
    11. Determine the set of "target" locations: These are the locations of loose
        goal nuts (if any) and the locations of the needed usable spanners (if any).
    12. Calculate the minimum distance from the man's current location to any
        location in the target set using the pre-computed shortest paths. If any
        target location is unreachable, return infinity.
    13. The heuristic value is the sum of:
        - N_loose_goals (estimated cost for the tighten actions)
        - N_needed_from_loc (estimated cost for the pickup actions)
        - The minimum distance calculated in step 12 (estimated cost for the first walk towards a target).
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.man = None
        self.all_spanners = set()
        self.all_nuts = set()
        self.all_locations = set()
        self.nut_locations = {} # {nut: initial_location} - Nuts are static

        # Collect all objects mentioned in initial state and static facts
        all_facts = set(initial_state) | set(static_facts)

        # Try to identify object types based on predicates
        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()
        potential_locations = set()

        for fact in all_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                potential_locations.add(loc)
                potential_men.add(obj)
                potential_spanners.add(obj)
                potential_nuts.add(obj)
            elif pred == 'carrying':
                man_obj, spanner_obj = parts[1], parts[2]
                potential_men = {man_obj} # This object is definitely the man
                potential_spanners.add(spanner_obj) # This object is definitely a spanner
            elif pred == 'usable':
                spanner_obj = parts[1]
                potential_spanners.add(spanner_obj) # This object is definitely a spanner
            elif pred == 'loose' or pred == 'tightened':
                nut_obj = parts[1]
                potential_nuts.add(nut_obj) # This object is definitely a nut
            elif pred == 'link':
                loc1, loc2 = parts[1], parts[2]
                potential_locations.add(loc1)
                potential_locations.add(loc2)

        # Refine object sets based on identified types
        # Man is the one identified as potential_men (should be a single element set)
        if not potential_men:
             # Fallback: if no 'carrying' fact initially, try to find the object
             # at an initial location that isn't a spanner or nut.
             non_spanner_nut_potential_men = potential_men - potential_spanners - potential_nuts
             if non_spanner_nut_potential_men:
                  self.man = list(non_spanner_nut_potential_men)[0]
             else:
                  # Cannot identify man robustly, use a placeholder.
                  # This might indicate an unusual problem structure.
                  self.man = "unknown_man"
                  # print("Warning: Could not robustly identify the man object.")
        else:
             self.man = list(potential_men)[0]


        # Spanners are potential spanners that are not the man and not nuts
        self.all_spanners = potential_spanners - {self.man} - potential_nuts
        # Nuts are potential nuts that are not the man and not spanners
        self.all_nuts = potential_nuts - {self.man} - potential_spanners
        # Locations are potential locations that are not man, spanners, or nuts
        self.all_locations = potential_locations - {self.man} - self.all_spanners - self.all_nuts


        # Store initial locations of nuts
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj in self.all_nuts:
                    self.nut_locations[obj] = loc

        # --- Build Location Graph and Compute Distances ---
        self.location_graph = build_location_graph(static_facts)
        # Ensure all locations found are in the graph, even if isolated
        for loc in self.all_locations:
             self.location_graph.setdefault(loc, [])
        self.distances = compute_all_pairs_shortest_paths(self.location_graph)

        # --- Identify Goal Nuts ---
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Basic check: Ensure all goal nuts were identified and have initial locations
        if not self.goal_nuts.issubset(self.all_nuts):
             # print("Warning: Goal nuts not found in initial state objects.")
             pass # Continue assuming they exist but might not be in initial 'at' facts
        if not all(n in self.nut_locations for n in self.goal_nuts):
             # print("Warning: Initial location not found for all goal nuts.")
             # This might happen if a nut is initially carried, which is not standard
             # but the heuristic relies on nut_locations. If a goal nut isn't here,
             # the heuristic might be inaccurate or fail.
             pass # Continue, hoping nut_locations is sufficient


    def get_man_location(self, state):
        """Finds the man's current location in the state."""
        for fact in state:
            if match(fact, "at", self.man, "*"):
                return get_parts(fact)[2]
        return None # Should not happen in a valid state where man is at a location

    def get_spanners_at_locations(self, state):
        """Finds the current locations of spanners that are not carried."""
        spanners_at_loc = {}
        # Find spanners currently carried by the man
        carried_spanners = {s for fact in state if match(fact, "carrying", self.man, s := "*")}

        # Iterate through all known spanners
        for spanner in self.all_spanners:
            # If the spanner is not carried, check if it's at a location
            if spanner not in carried_spanners:
                 for fact in state:
                     if match(fact, "at", spanner, "*"):
                         spanners_at_loc[spanner] = get_parts(fact)[2]
                         break # Assuming a spanner is at most at one location if not carried
        return spanners_at_loc


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        # Goal is reached if all goal nuts are tightened.
        goal_reached = all(f'(tightened {n})' in state for n in self.goal_nuts)
        if goal_reached:
            return 0

        # 2. Identify man's location
        l_m = self.get_man_location(state)
        if l_m is None:
             # Man is not at any location? Problematic state.
             return float('inf')

        # 3. Count loose goal nuts
        # Only consider goal nuts whose locations we know
        G_loose = {n for n in self.goal_nuts if f'(loose {n})' in state and n in self.nut_locations}
        N_loose_goals = len(G_loose)

        # If N_loose_goals is 0, goal is reached (already checked, but defensive)
        if N_loose_goals == 0:
             return 0

        # 4. Count carried usable spanners
        S_carried = {s for fact in state if match(fact, "carrying", self.man, s := "*")}
        S_usable_carried = {s for s in S_carried if f'(usable {s})' in state}
        N_carried_usable = len(S_usable_carried)

        # 5. Count usable spanners at locations
        Spanners_at_loc = self.get_spanners_at_locations(state)
        S_at_loc_usable = {s for s in Spanners_at_loc if f'(usable {s})' in state}
        N_at_loc_usable = len(S_at_loc_usable)

        # 6. Check for unsolvability
        if N_carried_usable + N_at_loc_usable < N_loose_goals:
             # Not enough usable spanners in the world to tighten all goal nuts
             return float('inf')

        # 7. Calculate needed spanners from locations
        N_needed_from_loc = max(0, N_loose_goals - N_carried_usable)

        # 8. Identify locations of loose goal nuts
        L_nut_loose_goal = {self.nut_locations[n] for n in G_loose}

        # 9. Identify locations of usable spanners at locations
        L_spanner_at_loc_usable = {Spanners_at_loc[s] for s in S_at_loc_usable if s in Spanners_at_loc} # Defensive check

        # 10. Find locations of needed spanners closest to man
        L_spanner_needed = set()
        if N_needed_from_loc > 0:
            # Get distances from man's location to all usable spanners at locations
            usable_spanner_locs_with_dist = []
            for loc in L_spanner_at_loc_usable:
                 # Ensure man's location and spanner location are in the distance map
                 if l_m in self.distances and loc in self.distances[l_m]:
                     usable_spanner_locs_with_dist.append((self.distances[l_m][loc], loc))
                 else:
                     # This spanner location is unreachable from the man's current location.
                     # If the problem is solvable, all relevant locations should be reachable.
                     # If not reachable, return inf.
                     return float('inf')


            # Sort by distance and take the locations of the N_needed_from_loc closest ones
            usable_spanner_locs_with_dist.sort()
            # Take up to N_needed_from_loc locations, handling cases where fewer are available
            L_spanner_needed = {loc for dist, loc in usable_spanner_locs_with_dist[:N_needed_from_loc]}

            # If we needed spanners but couldn't find enough reachable usable ones at locations,
            # the check in step 6 should have caught the total count issue.
            # If step 6 passed, but we can't find N_needed_from_loc *reachable* spanners,
            # it implies a reachability issue, which is handled by returning inf above.


        # 11. Determine target locations
        T = set()
        if N_loose_goals > 0:
             T.update(L_nut_loose_goal)
        if N_needed_from_loc > 0:
             T.update(L_spanner_needed)

        # 12. Calculate minimum distance to a target location
        min_dist_to_target_location = float('inf')
        if T:
            for target_loc in T:
                 # Ensure man's location and target location are in the distance map
                 if l_m in self.distances and target_loc in self.distances[l_m]:
                     min_dist_to_target_location = min(min_dist_to_target_location, self.distances[l_m][target_loc])
                 else:
                     # Target location is unreachable from man's current location.
                     return float('inf')


        # If no targets (should only happen if N_loose_goals=0, handled at start), dist is 0
        if min_dist_to_target_location == float('inf'):
             # This case should only be reached if T was empty, which implies N_loose_goals was 0.
             # Or if T was not empty but all targets were unreachable, which we handled by returning inf above.
             # So, if we reach here and min_dist is inf, it means T was empty.
             min_dist_to_target_location = 0


        # 13. Compute heuristic value
        heuristic_value = N_loose_goals + N_needed_from_loc + min_dist_to_target_location

        return heuristic_value
