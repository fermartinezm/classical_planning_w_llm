from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class is available

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_path(start_node, end_node, graph):
    """
    Find the shortest path distance between start_node and end_node in a graph.
    Graph is represented as an adjacency dictionary {location: [linked_locations]}.
    Returns distance or float('inf') if no path exists.
    """
    if start_node == end_node:
        return 0
    visited = {start_node}
    queue = deque([(start_node, 0)]) # (node, distance)

    while queue:
        current_node, distance = queue.popleft()

        if current_node in graph: # Ensure the node exists in the graph keys
            for neighbor in graph[current_node]:
                if neighbor == end_node:
                    return distance + 1
                if neighbor not in visited:
                    visited[neighbor] = distance + 1
                    queue.append((neighbor, distance + 1))

    return float('inf') # No path found

def precompute_distances(locations, links):
    """
    Precompute shortest path distances between all pairs of locations.
    Returns a dictionary {(loc1, loc2): distance}.
    """
    graph = {}
    all_locations_set = set(locations) # Use set for quick lookup
    for l1, l2 in links:
        # Ensure locations from links are in the set of all locations
        if l1 in all_locations_set and l2 in all_locations_set:
            graph.setdefault(l1, []).append(l2)
            graph.setdefault(l2, []).append(l1) # Links are bidirectional

    distances = {}
    for start_loc in locations:
        # Perform BFS from each start location
        queue = deque([(start_loc, 0)])
        visited = {start_loc: 0} # Store distance in visited set

        while queue:
            current_loc, dist = queue.popleft()
            distances[(start_loc, current_loc)] = dist

            if current_loc in graph: # Ensure the node exists in the graph keys
                for neighbor in graph[current_loc]:
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    # Fill in distances for pairs where start != end and no path exists
    for l1 in locations:
        for l2 in locations:
            if (l1, l2) not in distances:
                 distances[(l1, l2)] = float('inf') # No path

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost based on the number of loose goal nuts,
    the actions needed (pickup, tighten), and estimated travel costs.
    Travel cost is estimated as the cost for the first task (get spanner if needed, go to nut)
    plus an estimated cost for subsequent tasks (go from nut to spanner, then to nut).

    # Heuristic Initialization
    - Extracts locations and links to build a graph and precompute all-pairs shortest paths.
    - Identifies all nut and spanner objects and their initial locations.
    - Precomputes minimum distances between any initial nut location and any initial spanner location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of loose goal nuts (K). If K=0, return 0.
    2. Check if there are enough usable spanners (including the one carried) for all K nuts. If not, return infinity.
    3. Calculate a base cost: K for tighten actions + (K-1 if carrying usable spanner initially, else K) for pickup actions.
    4. Estimate the walk cost for the first task:
       - If carrying a usable spanner: Cost is the shortest path from the man's current location to the closest loose goal nut location.
       - If not carrying a usable spanner: Cost is the shortest path from the man's current location to the closest usable spanner location, plus the shortest path from that spanner location to the closest loose goal nut location.
       - If this first leg is impossible (infinite distance), return infinity.
    5. Estimate the walk cost for the remaining K-1 tasks:
       - Each remaining task conceptually involves going from a nut location to a spanner location, then to a nut location.
       - Estimate the cost of one such sequence as the minimum distance between *any* initial nut location and *any* initial spanner location, plus the minimum distance between *any* initial spanner location and *any* initial nut location.
       - If this sequence cost is infinite and K > 1, return infinity.
       - Add (K-1) times this estimated sequence cost to the total cost.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Extract locations and links to build the graph
        locations = set()
        links = []
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                links.append((l1, l2))
                locations.add(l1)
                locations.add(l2)

        # 2. Identify all objects by type and their initial locations
        initial_at_facts_dict = {get_parts(fact)[1]: get_parts(fact)[2] for fact in initial_state if match(fact, "at", "*", "*")}

        # Collect all objects mentioned in initial state and goal facts
        all_objects_mentioned = set()
        for fact in initial_state | self.goals:
            all_objects_mentioned.update(get_parts(fact)[1:])

        nut_names = set()
        spanner_names = set()
        self.man_name = None

        # Categorize objects based on predicates
        for obj in all_objects_mentioned:
            is_nut = any(match(fact, "loose", obj) or match(fact, "tightened", obj) for fact in initial_state | self.goals)
            is_spanner = any(match(fact, "usable", obj) for fact in initial_state | self.goals) # Check for (usable obj)

            # Find the man object name. Assume it's the object involved in (carrying man spanner)
            # or the single object in (at obj loc) not identified as nut/spanner.
            if self.man_name is None:
                 for fact in initial_state:
                      if match(fact, "carrying", obj, "*"):
                           self.man_name = obj # Found the man via carrying
                           break

            if is_nut:
                nut_names.add(obj)
            elif is_spanner:
                spanner_names.add(obj)

        # Fallback for man_name if not found via carrying
        if self.man_name is None:
             located_objects = {get_parts(fact)[1] for fact in initial_at_facts_dict.keys()}
             potential_men = located_objects - nut_names - spanner_names
             if len(potential_men) == 1:
                  self.man_name = list(potential_men)[0]
             elif len(potential_men) > 1:
                  # Handle case with multiple potential men - pick one?
                  # Assume the first object in the initial (at obj loc) list that isn't a nut or spanner is the man.
                  for obj in initial_at_facts_dict.keys():
                       if obj not in nut_names and obj not in spanner_names:
                            self.man_name = obj
                            break
                  if self.man_name is None: # Still not found? Problematic.
                       # As a last resort, pick the first located object.
                       if located_objects:
                            self.man_name = list(located_objects)[0]
                       else:
                            self.man_name = "unknown_man" # Should not happen in valid problems


        # Collect all locations mentioned in initial state (at facts)
        for loc in initial_at_facts_dict.values():
             locations.add(loc)

        self.locations = list(locations) # Store all unique locations
        self.all_pairs_distances = precompute_distances(self.locations, links)

        # Collect initial locations for all nuts and spanners
        self.all_nut_locations = {initial_at_facts_dict[nut] for nut in nut_names if nut in initial_at_facts_dict}
        self.all_spanner_locations = {initial_at_facts_dict[spanner] for spanner in spanner_names if spanner in initial_at_facts_dict}

        # 3. Precompute min distances between any nut location and any spanner location
        self.min_dist_any_nut_to_any_spanner = float('inf')
        if self.all_nut_locations and self.all_spanner_locations: # Only compute if both sets are non-empty
            for nut_loc in self.all_nut_locations:
                for spanner_loc in self.all_spanner_locations:
                     self.min_dist_any_nut_to_any_spanner = min(self.min_dist_any_nut_to_any_spanner, self.all_pairs_distances.get((nut_loc, spanner_loc), float('inf')))

        self.min_dist_any_spanner_to_any_nut = float('inf')
        if self.all_nut_locations and self.all_spanner_locations: # Only compute if both sets are non-empty
            for spanner_loc in self.all_spanner_locations:
                for nut_loc in self.all_nut_locations:
                     self.min_dist_any_spanner_to_any_nut = min(self.min_dist_any_spanner_to_any_nut, self.all_pairs_distances.get((spanner_loc, nut_loc), float('inf')))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Get man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man's location not found? Problem state error.
             return float('inf') # Should not happen in valid states

        # 2. Check if man is carrying a usable spanner
        carrying_usable = False
        carried_spanner_name = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner_name = get_parts(fact)[2]
                # Check if this specific spanner is usable in the current state
                if f"(usable {carried_spanner_name})" in state:
                     carrying_usable = True
                break # Assume man carries at most one spanner

        # 3. Identify loose goal nuts and their current locations
        loose_goal_nuts_info = {} # {nut_name: location}
        # First, find all goal nuts
        goal_nut_names = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Then, check their status and location in the current state
        nut_locations_in_state = {} # {nut_name: location} for nuts currently at a location
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at" and parts[1] in goal_nut_names:
                  nut_locations_in_state[parts[1]] = parts[2]

        for nut_name in goal_nut_names:
             # A goal nut is loose if it's in the state as (loose nut_name)
             # and it is currently at a location.
             if f"(loose {nut_name})" in state and nut_name in nut_locations_in_state:
                  loose_goal_nuts_info[nut_name] = nut_locations_in_state[nut_name]


        # 4. Identify usable spanners not carried and their current locations
        usable_spanners_at_loc_info = {} # {spanner_name: location}
        # Find all usable spanners in the state
        usable_spanner_names_in_state = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        # Filter out the one being carried (if any)
        if carried_spanner_name in usable_spanner_names_in_state:
             usable_spanner_names_in_state.remove(carried_spanner_name)

        # Find locations for the remaining usable spanners
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at" and parts[1] in usable_spanner_names_in_state:
                  usable_spanners_at_loc_info[parts[1]] = parts[2]

        # --- Heuristic Calculation ---

        K = len(loose_goal_nuts_info)

        # If all goal nuts are tightened, the heuristic is 0.
        if K == 0:
            return 0

        # Check for unsolvability: Not enough usable spanners available (including the one carried).
        total_usable_spanners = len(usable_spanners_at_loc_info) + (1 if carrying_usable else 0)
        if total_usable_spanners < K:
             return float('inf') # Not enough spanners to tighten all nuts

        # Check for unsolvability: No nut locations or no spanner locations identified initially.
        # This would make min_dist_any_* calculations potentially fail or be misleading.
        # If K > 0 but no nuts/spanners were identified initially, it's likely unsolvable.
        if K > 0 and (not self.all_nut_locations or not self.all_spanner_locations):
             # This might happen if all nuts/spanners are carried initially, or problem is malformed.
             # If there are loose nuts but no initial nut locations or spanner locations recorded,
             # it suggests something is fundamentally wrong or unreachable.
             # However, the BFS handles unreachable locations. Let's rely on distance checks.
             pass


        # Base cost: K tighten actions + (K - (1 if carrying usable spanner else 0)) pickup actions
        cost = K # tighten actions
        cost += K - (1 if carrying_usable else 0) # pickup actions

        # Walk cost for the first task (get spanner if needed, go to first nut)
        current_loc = man_location
        first_leg_cost = float('inf')

        if carrying_usable:
            # Man has the spanner, go to the closest loose nut location
            for nut, loc in loose_goal_nuts_info.items():
                dist = self.all_pairs_distances.get((current_loc, loc), float('inf'))
                first_leg_cost = min(first_leg_cost, dist)
        else: # Not carrying usable spanner
            # Go to closest spanner, then to closest nut from there.
            min_dist_man_to_spanner = float('inf')
            closest_spanner_loc = None
            # Need to iterate over usable spanners *at locations* in the current state
            for spanner, loc in usable_spanners_at_loc_info.items():
                dist = self.all_pairs_distances.get((current_loc, loc), float('inf'))
                if dist < min_dist_man_to_spanner:
                    min_dist_man_to_spanner = dist
                    closest_spanner_loc = loc

            if closest_spanner_loc is not None and min_dist_man_to_spanner != float('inf'):
                min_dist_spanner_to_nut = float('inf')
                # Need to iterate over loose goal nut locations in the current state
                for nut, loc in loose_goal_nuts_info.items():
                    dist = self.all_pairs_distances.get((closest_spanner_loc, loc), float('inf'))
                    min_dist_spanner_to_nut = min(min_dist_spanner_to_nut, dist)

                if min_dist_spanner_to_nut != float('inf'):
                     first_leg_cost = min_dist_man_to_spanner + min_dist_spanner_to_nut

        if first_leg_cost == float('inf'):
            return float('inf') # Unsolvable: Cannot reach first spanner/nut combination

        cost += first_leg_cost

        # Walk cost for remaining K-1 tasks
        if K > 1:
            # Each remaining task requires going from a nut location to a spanner location, then to a nut location.
            # Estimate using the minimum distances between *any* nut/spanner locations found initially.
            cost_per_sequence = self.min_dist_any_nut_to_any_spanner + self.min_dist_any_spanner_to_any_nut

            if cost_per_sequence == float('inf'):
                 # If even the minimum path between *any* nut and *any* spanner is inf,
                 # and we need K-1 > 0 more sequences, it's unsolvable.
                 return float('inf')

            cost += (K - 1) * cost_per_sequence

        return cost
