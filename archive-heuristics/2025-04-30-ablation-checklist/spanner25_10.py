# Assuming heuristic_base.py is available in the same directory or PYTHONPATH
# from heuristics.heuristic_base import Heuristic
# If the above import fails, you might need to provide a dummy Heuristic class:
class Heuristic:
    """
    Dummy base class for heuristic functions.
    Replace with the actual base class provided by the planner framework.
    """
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

from collections import deque
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def build_location_graph(static_facts):
    """Builds an adjacency list graph from link facts."""
    graph = {}
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'link' and len(parts) == 3:
            loc1, loc2 = parts[1], parts[2]
            graph.setdefault(loc1, []).append(loc2)
            graph.setdefault(loc2, []).append(loc1)
            locations.add(loc1)
            locations.add(loc2)
    return graph, list(locations)

def bfs_distances(graph, start_node):
    """Computes shortest distances from start_node to all other nodes using BFS."""
    distances = {node: math.inf for node in graph}
    # Ensure start_node is in the graph keys, even if isolated
    if start_node not in graph:
         graph[start_node] = [] # Add isolated node to graph structure
         distances[start_node] = math.inf # Initialize distance

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        # current_node is guaranteed to be in graph keys now
        for neighbor in graph[current_node]:
            if distances[neighbor] == math.inf:
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

def precompute_distances(static_facts, initial_state):
    """Precomputes distances between all pairs of locations."""
    # Collect all locations mentioned in static facts (links) and initial state (at)
    locations = set()
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'link' and len(parts) == 3:
            loc1, loc2 = parts[1], parts[2]
            graph.setdefault(loc1, []).append(loc2)
            graph.setdefault(loc2, []).append(loc1)
            locations.add(loc1)
            locations.add(loc2)
    for fact in initial_state:
         parts = get_parts(fact)
         if parts and parts[0] == 'at' and len(parts) == 3:
             obj, loc = parts[1], parts[2]
             locations.add(loc)

    # Ensure all found locations are keys in the graph, even if isolated
    for loc in locations:
        graph.setdefault(loc, [])

    dist_map = {}
    for start_loc in locations:
        dist_map[start_loc] = bfs_distances(graph, start_loc)
    return dist_map, list(locations)


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost by summing the minimum actions required for each
    loose goal nut, assuming a greedy approach: go to the nearest loose
    goal nut, and if a usable spanner is not carried, detour to get the
    nearest usable spanner and return, then tighten the nut.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Identify nuts and spanners from initial state and goals
        nuts = set()
        spanners = set()
        locatables_in_init_at = set()
        self.nut_locations = {} # Store fixed location for each nut

        # First pass to identify nuts and spanners based on predicates
        for fact in task.initial_state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'loose' and len(parts) == 2:
                nuts.add(parts[1])
            elif parts[0] == 'usable' and len(parts) == 2:
                spanners.add(parts[1])
            elif parts[0] == 'carrying' and len(parts) == 2: # (carrying ?m ?s)
                 spanners.add(parts[2]) # The second arg is a spanner

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == 'tightened' and len(parts) == 2:
                nuts.add(parts[1])

        # Second pass to get locations and identify the man
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'at' and len(parts) == 3:
                  obj, loc = parts[1], parts[2]
                  locatables_in_init_at.add(obj)
                  # Store nut locations
                  if obj in nuts:
                       self.nut_locations[obj] = loc


        # Identify the man
        self.man = None
        for obj in locatables_in_init_at:
            if obj not in nuts and obj not in spanners:
                self.man = obj
                break # Found the man

        if self.man is None:
            # Error if man not found - indicates unexpected problem structure
            raise ValueError("Could not identify the man object from initial state.")

        # Identify goal nuts (subset of all nuts that are in the goal)
        self.goal_nuts = {nut for nut in nuts if f'(tightened {nut})' in self.goals}

        # Ensure locations for all goal nuts are found
        for nut in self.goal_nuts:
            if nut not in self.nut_locations:
                # This indicates a problem with the input instance.
                raise ValueError(f"Location for goal nut {nut} not found in initial state.")

        # Precompute distances
        self.dist_map, self.all_locations = precompute_distances(task.static, task.initial_state)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts in the current state
        loose_goal_nuts_in_state = {
            nut for nut in self.goal_nuts
            if f'(loose {nut})' in state
        }

        # If all goal nuts are tightened, cost is 0
        if not loose_goal_nuts_in_state:
            return 0

        # 2. Get man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] == self.man:
                man_location = parts[2]
                break

        if man_location is None:
             # Man's location not found? Should not happen in valid states.
             return math.inf # Indicate unsolvable

        # 3. Identify usable spanners currently carried by the man
        carried_spanners_in_state = {
            parts[2] for fact in state if (parts := get_parts(fact)) and parts[0] == 'carrying' and parts[1] == self.man
        }
        current_UsableS_carried = {
            s for s in carried_spanners_in_state if f'(usable {s})' in state
        }

        # 4. Identify usable spanners at locations in the current state
        usable_spanners_at_loc_in_state = set() # Stores (spanner_name, location)
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Check if this object is a usable spanner and not carried by the man
                if f'(usable {obj})' in state and obj not in carried_spanners_in_state:
                     usable_spanners_at_loc_in_state.add((obj, loc))


        # --- Heuristic Calculation ---
        h = 0
        current_L_M = man_location
        # Use copies of the sets/lists that change during the heuristic simulation
        sim_UsableS_carried = set(current_UsableS_carried)
        sim_usable_spanners_at_loc = set(usable_spanners_at_loc_in_state)

        nuts_remaining_with_loc = [(nut, self.nut_locations.get(nut)) for nut in loose_goal_nuts_in_state]
        # Filter out nuts whose location wasn't found (should have been caught in init)
        nuts_remaining_with_loc = [(nut, loc) for nut, loc in nuts_remaining_with_loc if loc is not None]


        while nuts_remaining_with_loc:
            # Find the nut N in nuts_to_tighten that minimizes dist(current_L_M, nut_locations[N]).
            nearest_nut_info = None
            min_dist_to_nut = math.inf

            for nut, nut_loc in nuts_remaining_with_loc:
                # Check if current_L_M and nut_loc are in the distance map
                if current_L_M not in self.dist_map or nut_loc not in self.dist_map.get(current_L_M, {}):
                     dist = math.inf
                else:
                     dist = self.dist_map[current_L_M][nut_loc]

                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
                    nearest_nut_info = (nut, nut_loc)

            if nearest_nut_info is None or min_dist_to_nut == math.inf:
                # Cannot reach any remaining nut
                return math.inf

            nut_to_tighten, nut_loc = nearest_nut_info

            # Cost to reach nut
            h += min_dist_to_nut
            current_L_M = nut_loc # Man is now at the nut location

            # Cost to get spanner at current_L_M (nut_loc)
            spanner_cost = 0
            if not sim_UsableS_carried:
                # Need to get a spanner from a location
                nearest_spanner_info = None # (spanner_name, location)
                min_dist_to_spanner = math.inf

                if not sim_usable_spanners_at_loc:
                     # No usable spanners available anywhere
                     return math.inf

                for spanner, spanner_loc in sim_usable_spanners_at_loc:
                     # Check if current_L_M and spanner_loc are in the distance map
                     if current_L_M not in self.dist_map or spanner_loc not in self.dist_map.get(current_L_M, {}):
                         dist = math.inf
                     else:
                         dist = self.dist_map[current_L_M][spanner_loc]

                     if dist < min_dist_to_spanner:
                         min_dist_to_spanner = dist
                         nearest_spanner_info = (spanner, spanner_loc)

                if nearest_spanner_info is None or min_dist_to_spanner == math.inf:
                     # Cannot reach any usable spanner
                     return math.inf

                spanner_to_pickup, spanner_loc = nearest_spanner_info

                # Cost to walk to spanner + pickup + walk back to nut location
                spanner_cost = min_dist_to_spanner + 1 # Walk from nut_loc to spanner_loc + pickup
                # Man is now at spanner_loc carrying the spanner.
                # He needs to walk back to the nut location (which is current_L_M).
                # Check if spanner_loc and current_L_M are in the distance map
                if spanner_loc not in self.dist_map or current_L_M not in self.dist_map.get(spanner_loc, {}):
                     # Should not happen if graph is connected and locations are valid
                     return math.inf # Indicate unsolvable

                spanner_cost += self.dist_map[spanner_loc][current_L_M] # Walk back from spanner_loc to nut_loc

                sim_usable_spanners_at_loc.remove((spanner_to_pickup, spanner_loc))
                # The picked up spanner is conceptually used immediately for this nut.
                # It does not get added to sim_UsableS_carried for future nuts in this heuristic trace.
                # This is because tighten_nut makes it unusable.

            else:
                # Man is carrying a usable spanner, use one.
                spanner_to_use = next(iter(sim_UsableS_carried)) # Pick any
                sim_UsableS_carried.remove(spanner_to_use) # It becomes unusable (conceptually)

            # Add spanner cost (0 if carried, detour cost if picked up)
            h += spanner_cost

            # Cost to tighten
            h += 1
            nuts_remaining_with_loc.remove((nut_to_tighten, nut_loc))

        return h
