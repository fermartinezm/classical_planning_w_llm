# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

# Helper functions
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
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation
def bfs(graph, start):
    """
    Performs Breadth-First Search to find shortest distances from a start node.
    Assumes graph is an adjacency dictionary {node: [neighbor1, neighbor2, ...]}
    Returns a dictionary {node: distance}.
    """
    # Initialize distances for all nodes in the graph
    distances = {node: float('inf') for node in graph}

    # If start node is not in the graph, we can't reach anything from it
    if start not in graph:
         # If start is a location mentioned elsewhere but isolated, distance to itself is 0
         if start in distances:
             distances[start] = 0
         return distances

    distances[start] = 0
    queue = deque([start])

    while queue:
        current = queue.popleft()
        # Ensure current node has neighbors in the graph dictionary
        if current in graph:
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions (tighten, pickup, walk)
    required to tighten all loose nuts that are part of the goal. It considers
    the cost of tightening each nut, picking up spanners as needed, and the
    estimated walk distance to visit necessary locations (spanners and nuts).

    # Assumptions:
    - Links between locations are bidirectional for walking.
    - The man can only carry one spanner at a time.
    - Although there is no explicit 'drop' action, the heuristic assumes
      the man can effectively acquire a *new* usable spanner when needed
      for a subsequent nut, potentially leaving the old one behind.
      Solvable instances are assumed to have enough usable spanners.
    - The cost of actions is 1 (walk, pickup_spanner, tighten_nut).
    - Object types (man, spanner, nut, location) are inferred from predicate
      structure and naming conventions (e.g., 'spanner' prefix for spanners).
      Assumes there is only one man object.

    # Heuristic Initialization
    - Extracts static 'link' facts to build a graph of locations.
    - Collects all potential locations mentioned in initial state, goals, and links.
    - Computes all-pairs shortest paths between all identified locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Determine if the man is currently carrying a usable spanner.
    3. Identify all loose nuts that are specified in the goal state.
    4. Identify the locations of all usable spanners currently on the ground.
    5. If there are no loose goal nuts, the heuristic is 0.
    6. Check if there are enough usable spanners (carried usable + on ground usable)
       to tighten all loose goal nuts. If not, the state is likely unsolvable,
       return infinity.
    7. Calculate the base cost: This is the sum of the minimum required
       tighten actions (one for each loose goal nut) and pickup actions
       (one for each loose goal nut, except possibly the first if the man
       starts with a usable spanner).
       `base_cost = num_loose_goal_nuts + (num_loose_goal_nuts - (1 if man_has_usable_spanner else 0))`
    8. Identify the set of 'required locations' the man must visit. This set
       includes the location of every loose goal nut (unique locations). If the
       man is not carrying a usable spanner, it also includes the location of the
       closest usable spanner on the ground (as he must pick one up first).
    9. Calculate the walk cost: This is estimated as the shortest distance
       from the man's current location to the closest location in the set
       of required locations, plus a simplified cost of 1 for each
       additional unique required location that needs to be visited.
       `walk_cost = dist(man_loc, closest_required_loc) + (num_unique_required_locations - 1)`
    10. The total heuristic value is the sum of the base cost and the walk cost.
        `h = base_cost + walk_cost`
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing
        shortest paths between locations.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Also need initial state to find all locations

        # Collect all potential locations from initial state, goals, and static links
        all_locations_in_task = set()
        self.graph = {}

        for fact in static_facts | initial_state | self.goals:
             parts = get_parts(fact)
             if parts[0] == 'link' and len(parts) == 3:
                  l1, l2 = parts[1:]
                  all_locations_in_task.add(l1)
                  all_locations_in_task.add(l2)
                  self.graph.setdefault(l1, []).append(l2)
                  self.graph.setdefault(l2, []).append(l1) # Assuming bidirectional links
             elif parts[0] == 'at' and len(parts) == 3:
                 # (at obj loc) - the third part is a location
                 all_locations_in_task.add(parts[2])

        # Ensure all identified locations are keys in the graph dictionary for BFS
        for loc in all_locations_in_task:
             self.graph.setdefault(loc, [])

        self.locations = all_locations_in_task # Store all known locations

        # Compute all-pairs shortest paths
        self.all_pairs_dist = {}
        for start_loc in self.locations:
            distances = bfs(self.graph, start_loc)
            for end_loc, dist in distances.items():
                 self.all_pairs_dist[(start_loc, end_loc)] = dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract relevant information from the state ---
        man_loc = None
        carried_spanner = None
        is_carried_usable = False
        loose_goal_nuts = set()
        nut_locations = {} # Map nut name to location
        usable_ground_spanners = set()
        usable_ground_spanner_locations = set()
        all_spanners = set() # Collect all spanner names seen
        all_nuts = set() # Collect all nut names seen

        # First pass to identify object types by name pattern and man's location/carried item
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1:]
                if obj.startswith("spanner"):
                    all_spanners.add(obj)
                elif obj.startswith("nut"):
                    all_nuts.add(obj)
                    nut_locations[obj] = loc # Store nut locations
                # Assume anything else at a location is the man (simplification)
                # A more robust way would be to get object types from the task description
                # or assume 'bob' is the man based on examples. Let's stick to 'not spanner/nut'.
                elif obj not in all_spanners and obj not in all_nuts:
                     man_loc = loc
            elif parts[0] == 'carrying' and len(parts) == 3:
                 m, s = parts[1:]
                 # Assume m is the man
                 carried_spanner = s
                 all_spanners.add(s)
            elif parts[0] == 'usable' and len(parts) == 2:
                 s = parts[1]
                 all_spanners.add(s)

        # Check if carried spanner is usable
        if carried_spanner and f"(usable {carried_spanner})" in state:
             is_carried_usable = True

        # Identify loose goal nuts
        goal_nut_names = set()
        for goal in self.goals:
             if match(goal, "tightened", "*"):
                  goal_nut_names.add(get_parts(goal)[1])

        for nut_name in goal_nut_names:
             if f"(loose {nut_name})" in state:
                  loose_goal_nuts.add(nut_name)

        # Identify usable spanners on the ground
        for spanner_name in all_spanners:
             if f"(usable {spanner_name})" in state:
                  # Check if it's on the ground (not carried)
                  is_carried = False
                  for fact in state:
                       if match(fact, "carrying", "*", spanner_name):
                            is_carried = True
                            break
                  if not is_carried:
                       usable_ground_spanners.add(spanner_name)
                       # Find its location
                       for fact in state:
                            if match(fact, "at", spanner_name, "*"):
                                 usable_ground_spanner_locations.add(get_parts(fact)[2])
                                 break # Assuming spanner is at only one location


        # --- Heuristic Calculation ---

        num_loose_goal_nuts = len(loose_goal_nuts)

        # 1. If goal is reached
        if num_loose_goal_nuts == 0:
            return 0

        # Check if man location is known
        if man_loc is None:
             # Man's location is unknown, cannot plan
             return float('inf')

        # 2. Check solvability based on spanners
        num_usable_spanners_available = (1 if is_carried_usable else 0) + len(usable_ground_spanners)
        if num_usable_spanners_available < num_loose_goal_nuts:
             # Not enough usable spanners to tighten all required nuts
             return float('inf') # Problem is unsolvable from this state

        # 3. Calculate base cost (tighten + pickup actions)
        # Each loose goal nut needs one tighten action (cost 1).
        # Each loose goal nut needs one pickup action (cost 1), except possibly the first
        # if the man starts with a usable spanner.
        num_pickups_needed = num_loose_goal_nuts - (1 if is_carried_usable else 0)
        base_cost = num_loose_goal_nuts + num_pickups_needed # Sum of tighten and pickup actions

        # 4. Identify set of unique required locations the man must visit
        required_locations = set()
        # Man must visit the location of every loose goal nut
        for nut_name in loose_goal_nuts:
             if nut_name in nut_locations:
                  required_locations.add(nut_locations[nut_name])
             # else: nut not found at a location? Should not happen in valid states.

        # If man needs to pick up a spanner first, he must visit a spanner location
        if not is_carried_usable:
             if not usable_ground_spanner_locations:
                  # Should have been caught by the spanner count check, but safeguard
                  return float('inf')
             # Find closest usable ground spanner location from man_loc
             closest_spanner_loc = None
             min_dist_spanner = float('inf')
             for s_loc in usable_ground_spanner_locations:
                  dist = self.all_pairs_dist.get((man_loc, s_loc), float('inf'))
                  if dist < min_dist_spanner:
                       min_dist_spanner = dist
                       closest_spanner_loc = s_loc
             if closest_spanner_loc is None or min_dist_spanner == float('inf'):
                  # Cannot reach any usable ground spanner
                  return float('inf')
             # Add the location of the closest usable ground spanner to required locations
             required_locations.add(closest_spanner_loc)


        # 5. Calculate walk cost
        walk_cost = 0
        if not required_locations:
             # No required locations to visit (e.g., loose nuts have no location?)
             # Should not happen if num_loose_goal_nuts > 0 and nuts have locations
             pass # walk_cost remains 0
        else:
             # Find the closest required location from the man's current location
             closest_req_loc = None
             min_dist_req = float('inf')
             for req_loc in required_locations:
                  dist = self.all_pairs_dist.get((man_loc, req_loc), float('inf'))
                  if dist < min_dist_req:
                       min_dist_req = dist
                       closest_req_loc = req_loc

             if closest_req_loc is None or min_dist_req == float('inf'):
                  # Cannot reach any required location
                  return float('inf')

             walk_cost += min_dist_req # Walk to the first required location

             # Simplified walk cost for visiting the remaining required locations
             # Assume 1 action per additional unique required location visit
             walk_cost += len(required_locations) - 1


        # 6. Total heuristic value
        total_cost = base_cost + walk_cost

        return total_cost
