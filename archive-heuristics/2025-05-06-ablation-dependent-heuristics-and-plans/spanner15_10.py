from collections import deque
from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided in the execution environment
# In a real scenario, this would be provided by the planner framework
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass

# Utility functions for parsing PDDL facts
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

# BFS implementation for shortest path on location graph
def bfs(start_node, graph):
    """
    Performs BFS starting from start_node on the given graph.
    Returns a dictionary of distances from start_node to all reachable nodes.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# All-pairs shortest path using BFS
def compute_all_pairs_shortest_paths(locations, graph):
    """
    Computes shortest path distances between all pairs of locations.
    Returns a dictionary distances[loc1][loc2] = dist.
    Unreachable locations will have distance infinity.
    """
    all_distances = {}
    for start_loc in locations:
        all_distances[start_loc] = bfs(start_loc, graph)

    # Fill in unreachable distances with infinity
    for loc1 in locations:
        for loc2 in locations:
            if loc2 not in all_distances[loc1]:
                 all_distances[loc1][loc2] = float('inf')

    return all_distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts
    that are part of the goal. It counts the necessary pickup and tighten actions
    for each nut, accounts for spanner drops between nuts, and estimates the travel
    cost based on minimum distances between the man, usable spanners, and loose nuts.

    # Assumptions:
    - The man can only carry one spanner at a time.
    - Using a spanner makes it unusable.
    - To tighten k nuts, k distinct usable spanners are needed throughout the plan.
    - The travel cost is estimated by summing minimum distances between relevant locations
      in a simplified sequence: Man -> Spanner -> Nut -> Spanner -> Nut ...
    - The man object can be identified by checking if its name starts with 'bob' or contains 'man'.

    # Heuristic Initialization
    - Extracts all location names from static facts, initial state, and goal conditions.
    - Builds the location graph based on `link` predicates.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies goal nuts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify the spanner the man is currently carrying, if any, and check if it's usable.
    3. Identify all loose nuts in the current state that are also goal nuts, and record their locations.
    4. Identify all usable spanners in the current state (on the ground or carried), and record their locations.
    5. If there are no loose nuts to tighten (`num_loose_nuts == 0`), the heuristic is 0.
    6. If the total number of usable spanners is less than the number of loose nuts, the problem is likely unsolvable; return infinity.
    7. Calculate the fixed action costs:
       - `num_loose_nuts` tighten actions (cost 1 each).
       - `num_loose_nuts` pickup actions (cost 1 each, assuming a new spanner is picked up for each nut).
       - `max(0, num_loose_nuts - 1)` drop actions (cost 1 each, needed to drop a used spanner before picking up the next).
       - `1` initial drop action (cost 1) if the man is currently carrying a non-usable spanner.
    8. Calculate minimum travel distances required for the sequence Man -> Spanner -> Nut -> Spanner -> Nut ...
       - `min_dist_m_s`: Minimum distance from the man's current location to any usable spanner location. (0 if man carries usable).
       - `min_dist_s_n`: Minimum distance from any usable spanner location to any loose nut location.
       - `min_dist_n_s`: Minimum distance from any loose nut location to any usable spanner location.
       If any required minimum distance is infinity (locations are disconnected or no required objects exist), the problem is likely unsolvable; return infinity.
    9. Estimate total travel cost:
        - Travel to the first spanner: `min_dist_m_s`.
        - Travel from spanner to nut: `num_loose_nuts * min_dist_s_n`.
        - Travel from nut to next spanner: `max(0, num_loose_nuts - 1) * min_dist_n_s`.
    10. Sum all costs: `(num_loose_nuts * 2) + max(0, num_loose_nuts - 1) + initial_drop_cost + min_dist_m_s + (num_loose_nuts * min_dist_s_n) + (max(0, num_loose_nuts - 1) * min_dist_n_s)`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and computing distances."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Identify all locations
        locations = set()
        location_graph = {}

        # Add locations from static links
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                location_graph.setdefault(loc1, set()).add(loc2)
                location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        # Add locations from initial state (objects at locations)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 locations.add(loc)
                 location_graph.setdefault(loc, set()) # Ensure all locations are keys even if no links

        # Add locations from goal state (objects at locations)
        # Goal can be conjunction, need to parse it
        goal_facts = set()
        if isinstance(self.goals, frozenset): # Standard goal representation
             goal_facts = self.goals
        elif isinstance(self.goals, tuple) and self.goals[0] == 'and': # Goal is (and ...)
             goal_facts = frozenset(self.goals[1:])
        else: # Single goal fact
             goal_facts = frozenset({self.goals})

        for fact in goal_facts:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 locations.add(loc)
                 location_graph.setdefault(loc, set()) # Ensure all locations are keys

        self.locations = list(locations) # Store as list for consistent indexing if needed, though dict is used
        self.location_graph = location_graph

        # Compute all-pairs shortest paths
        self.distances = compute_all_pairs_shortest_paths(self.locations, self.location_graph)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in goal_facts:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.goal_nuts.add(nut)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_location = None
        # Assuming the man object name contains 'bob' or 'man'
        man_name = None
        for fact in state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 if 'bob' in obj or 'man' in obj:
                      man_name = obj
                      man_location = loc
                      break # Found the man

        if man_name is None:
             # Should not happen in valid instances, but handle defensively
             return float('inf')


        # 2. Identify carried spanner and its usability
        carried_spanner = None
        is_carrying_usable = False
        is_carrying_non_usable = False

        for fact in state:
             if match(fact, "carrying", man_name, "*"):
                  _, m, s = get_parts(fact)
                  carried_spanner = s
                  break # Found carried spanner

        if carried_spanner:
             if f"(usable {carried_spanner})" in state:
                  is_carrying_usable = True
             else:
                  is_carrying_non_usable = True


        # 3. Identify loose nuts that are goal nuts and their locations
        loose_goal_nuts = set()
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                _, nut = get_parts(fact)
                if nut in self.goal_nuts:
                    loose_goal_nuts.add(nut)
            # Also collect nut locations from 'at' facts
            elif match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 if obj in self.goal_nuts: # Assuming goal nuts are the relevant nuts
                      nut_locations[obj] = loc


        num_loose_nuts = len(loose_goal_nuts)

        # 5. If no loose nuts to tighten, heuristic is 0
        if num_loose_nuts == 0:
            return 0

        # 4. Identify usable spanners and their locations
        usable_spanners_ground = set()
        spanner_locations_ground = {} # Location of spanners on the ground
        usable_spanners_carried = set() # Usable spanner if carried

        for fact in state:
             if match(fact, "usable", "*"):
                  _, spanner = get_parts(fact)
                  if spanner == carried_spanner:
                       usable_spanners_carried.add(spanner)
                  else:
                       usable_spanners_ground.add(spanner)
             # Also collect spanner locations from 'at' facts
             elif match(fact, "at", "*", "*"):
                  _, obj, loc = get_parts(fact)
                  # Assuming objects starting with 'spanner' are spanners
                  if obj.startswith('spanner'):
                       spanner_locations_ground[obj] = loc


        all_usable_spanners = usable_spanners_ground | usable_spanners_carried

        # 6. Check if enough usable spanners exist
        if len(all_usable_spanners) < num_loose_nuts:
             # Problem is likely unsolvable
             return float('inf')

        # 7. Calculate fixed action costs
        # Each nut needs pickup (1), tighten (1).
        # Each nut except the last needs a drop (1) afterwards to free hand for next pickup.
        fixed_actions = num_loose_nuts * 2 # pickup + tighten
        fixed_actions += max(0, num_loose_nuts - 1) # drops between nuts
        fixed_actions += 1 if is_carrying_non_usable else 0 # initial drop if needed

        # 8. Calculate minimum travel distances

        # Helper to get location of any spanner (ground or carried)
        def get_spanner_current_loc(spanner_name):
             if spanner_name == carried_spanner:
                  return man_location
             return spanner_locations_ground.get(spanner_name)


        # min_dist_m_s: ManLoc to any usable spanner location
        min_dist_m_s = float('inf')
        if is_carrying_usable:
             min_dist_m_s = 0 # Man is already at the spanner location (his own)
        else:
             for spanner in usable_spanners_ground:
                  spanner_loc = get_spanner_current_loc(spanner)
                  if spanner_loc and man_location in self.distances and spanner_loc in self.distances[man_location]:
                       min_dist_m_s = min(min_dist_m_s, self.distances[man_location][spanner_loc])

        # If num_loose_nuts > 0, we MUST be able to reach a usable spanner.
        # If min_dist_m_s is still inf here, it means no usable spanners are reachable from man_location.
        if min_dist_m_s == float('inf'):
             return float('inf')


        # min_dist_s_n: Any usable spanner location to any loose nut location
        min_dist_s_n = float('inf')
        for spanner in all_usable_spanners:
             spanner_loc = get_spanner_current_loc(spanner)
             if spanner_loc:
                  for nut in loose_goal_nuts:
                       nut_loc = nut_locations.get(nut)
                       if nut_loc and spanner_loc in self.distances and nut_loc in self.distances[spanner_loc]:
                            min_dist_s_n = min(min_dist_s_n, self.distances[spanner_loc][nut_loc])

        # If num_loose_nuts > 0, we MUST be able to reach a loose nut from a usable spanner location.
        if min_dist_s_n == float('inf'):
             return float('inf')


        # min_dist_n_s: Any loose nut location to any usable spanner location
        min_dist_n_s = float('inf')
        # This is only relevant if num_loose_nuts > 1
        if num_loose_nuts > 1:
            for nut in loose_goal_nuts:
                 nut_loc = nut_locations.get(nut)
                 if nut_loc:
                      for spanner in all_usable_spanners:
                           spanner_loc = get_spanner_current_loc(spanner)
                           if spanner_loc and nut_loc in self.distances and spanner_loc in self.distances[nut_loc]:
                                min_dist_n_s = min(min_dist_n_s, self.distances[nut_loc][spanner_loc])

            # If num_loose_nuts > 1, we MUST be able to travel from a nut location to the next spanner location.
            if min_dist_n_s == float('inf'):
                 return float('inf')
        else:
             # If num_loose_nuts is 0 or 1, this term is not used in the formula (multiplied by 0),
             # so its value doesn't matter.
             pass


        # 9. Estimate total travel cost
        travel_cost = min_dist_m_s \
                      + num_loose_nuts * min_dist_s_n \
                      + max(0, num_loose_nuts - 1) * min_dist_n_s

        # 10. Sum all costs
        total_cost = fixed_actions + travel_cost

        # Ensure heuristic is non-negative (distances are non-negative, counts are non-negative)
        return max(0, total_cost) # Should always be non-negative with this formula

