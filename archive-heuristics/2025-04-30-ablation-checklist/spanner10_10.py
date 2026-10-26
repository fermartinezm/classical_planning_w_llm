from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't match if pattern is longer than fact parts
    if len(args) > len(parts):
         return False
    # Use zip which stops at the shortest sequence
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to calculate shortest path distances in the location graph
def bfs(start_loc, graph, all_locations):
    """
    Performs BFS to find shortest path distances from start_loc to all other locations.
    Returns a dictionary {location: distance}.
    """
    distances = {loc: float('inf') for loc in all_locations}
    if start_loc not in all_locations:
        # Start location is not in the known locations
        return distances

    distances[start_loc] = 0
    queue = deque([start_loc])

    while queue:
        curr = queue.popleft()

        # If distance is already infinity, it means we reached here via an unreachable path, skip.
        if distances[curr] == float('inf'):
             continue

        if curr in graph: # Handle locations with no links
            for neighbor in graph[curr]:
                # Only update if we found a shorter path (which is always the case in BFS for unweighted graph)
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
    return distances

# Inherit from Heuristic if available, otherwise define a simple class
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

class spannerHeuristic: # Replace with class spannerHeuristic(Heuristic): if base class is provided
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all goal nuts by summing:
    1. The number of tighten actions needed (one per loose goal nut).
    2. The number of pickup actions needed (one per usable spanner required but not carried).
    3. An estimate of the walk actions needed to visit all required spanner and nut locations.
       The walk cost is estimated using the Nearest Neighbor approach starting from the man's location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying key objects.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Build location graph and collect all locations and objects
        self.location_graph = {}
        all_locations_set = set()
        
        # Infer object types and collect locations from initial state and static facts
        self.man = None
        self.all_nuts = set()
        self.all_spanners = set()

        # Collect all objects mentioned in initial state and static facts
        # This helps in identifying all possible locations later
        all_objects_set = set()
        for fact in self.initial_state | self.static_facts:
             parts = get_parts(fact)
             if not parts: continue
             all_objects_set.update(parts[1:]) # Add all arguments as potential objects

        # Infer types based on predicates, assuming standard PDDL usage
        for fact in self.initial_state | self.static_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            args = parts[1:]

            if pred == 'link' and len(args) == 2:
                loc1, loc2 = args
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)
                all_locations_set.add(loc1)
                all_locations_set.add(loc2)
            elif pred == 'at' and len(args) == 2:
                obj, loc = args
                all_locations_set.add(loc)
            
            # Infer types based on predicate names and argument positions
            if pred == 'carrying' and len(args) == 2:
                 self.man = args[0] # Assume first arg of carrying is the man
                 self.all_spanners.add(args[1]) # Assume second arg is spanner
            elif pred == 'usable' and len(args) == 1:
                 self.all_spanners.add(args[0]) # Arg of usable is spanner
            elif pred == 'loose' and len(args) == 1:
                 self.all_nuts.add(args[0]) # Arg of loose is nut
            elif pred == 'tightened' and len(args) == 1:
                 self.all_nuts.add(args[0]) # Arg of tightened is nut

        self.all_locations = list(all_locations_set)

        # 2. Precompute all-pairs shortest paths
        self.all_pairs_dist = {}
        for loc in self.all_locations:
            self.all_pairs_dist[loc] = bfs(loc, self.location_graph, self.all_locations)

        # 3. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returning infinity if path doesn't exist."""
        if loc1 not in self.all_pairs_dist or loc2 not in self.all_pairs_dist.get(loc1, {}):
             # If either location is unknown or unreachable
             return float('inf')
        return self.all_pairs_dist[loc1][loc2]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man's location must always be known in a valid state
             return float('inf') # Should not happen in valid states

        # 2. Identify loose goal nuts in the current state and their locations
        loose_goal_nuts_in_state = set()
        nut_locations = {} # {nut: location}
        for nut in self.goal_nuts:
            if f"(loose {nut})" in state:
                loose_goal_nuts_in_state.add(nut)
                # Find the location of this nut
                for fact in state:
                    if match(fact, "at", nut, "*"):
                        nut_locations[nut] = get_parts(fact)[2]
                        break # Found location, move to next nut

        k = len(loose_goal_nuts_in_state)
        if k == 0:
            return 0 # Goal reached for these nuts

        # 3. Identify usable spanners currently carried by the man
        usable_carried_spanners = {
            get_parts(fact)[2] for fact in state
            if match(fact, "carrying", self.man, "*") and f"(usable {get_parts(fact)[2]})" in state
        }
        c = len(usable_carried_spanners)

        # 4. Identify usable spanners on the ground and their locations
        usable_ground_spanners = {} # {spanner: location}
        for spanner in self.all_spanners:
             # Check if usable and not carried
             if f"(usable {spanner})" in state and f"(carrying {self.man} {spanner})" not in state:
                  # Find location on the ground
                  for fact in state:
                       if match(fact, "at", spanner, "*"):
                            usable_ground_spanners[spanner] = get_parts(fact)[2]
                            break # Found location, move to next spanner

        g = len(usable_ground_spanners)

        # 5. Calculate base heuristic (tighten + pickup actions)
        # Each loose goal nut needs one tighten action.
        # The man needs k usable spanners in total. He has 'c' carried. He needs max(0, k - c) more.
        needed_pickups = max(0, k - c)

        # Check if enough usable spanners exist in total (carried + on ground)
        if c + g < k:
             # Goal is unreachable, not enough spanners exist
             return float('inf')

        # If we need to pick up more spanners than are available on the ground, something is wrong
        # (This check is technically redundant if c + g >= k, but good for clarity)
        if needed_pickups > g:
             return float('inf') # Should not happen if c + g >= k

        heuristic = k # Cost for tighten actions
        heuristic += needed_pickups # Cost for pickup actions

        # 6. Estimate walk actions
        # The man needs to visit the location of each loose goal nut.
        # The man needs to visit the location of each spanner he needs to pick up.

        L_nuts_loose = {nut_locations[nut] for nut in loose_goal_nuts_in_state}

        # Select the locations of the 'needed_pickups' closest usable ground spanners
        L_spanners_to_get = set()
        if needed_pickups > 0:
             # Sort usable ground spanners by distance from man_loc
             # Handle cases where man_loc might not be in all_pairs_dist (e.g., isolated location)
             # or spanner location might not be in all_pairs_dist[man_loc] (unreachable spanner)
             sorted_ground_spanners = sorted(
                 usable_ground_spanners.items(),
                 key=lambda item: self.get_distance(man_loc, item[1])
             )
             # Take the locations of the needed_pickups closest ones
             # Ensure we don't try to take more spanners than available on the ground
             for i in range(min(needed_pickups, len(sorted_ground_spanners))):
                  L_spanners_to_get.add(sorted_ground_spanners[i][1])

        # Combine target locations for walking
        target_locations = list(L_nuts_loose | L_spanners_to_get)

        # Calculate walk cost using Nearest Neighbor starting from man_loc
        walk_cost = 0
        current_loc = man_loc
        remaining_targets = set(target_locations)

        # If man_loc is one of the targets, the distance to it is 0 in the first step,
        # which is handled correctly by the loop.

        while remaining_targets:
            # Find the closest remaining target location
            next_loc = None
            min_dist = float('inf')

            for target in remaining_targets:
                dist = self.get_distance(current_loc, target)
                if dist < min_dist:
                    min_dist = dist
                    next_loc = target
                # If dist is inf, this target is unreachable from current_loc or any future loc in this path.
                # The Nearest Neighbor will pick reachable ones first. If all remaining are unreachable,
                # min_dist stays inf, next_loc stays None, and the loop breaks.

            if next_loc is not None:
                # If the closest target is unreachable, min_dist will be inf.
                # Adding inf to walk_cost makes the total inf.
                walk_cost += min_dist
                current_loc = next_loc
                remaining_targets.remove(next_loc)
            else:
                 # No reachable targets left from current_loc.
                 # If remaining_targets was not empty, it means the remaining targets are unreachable.
                 # We should return inf in this case.
                 if remaining_targets: # If there are still targets but none were reachable
                     return float('inf') # Goal unreachable
                 break # Exit loop

        # If walk_cost became infinity at any point, the heuristic is infinity
        if walk_cost == float('inf'):
             return float('inf')

        heuristic += walk_cost

        return heuristic
