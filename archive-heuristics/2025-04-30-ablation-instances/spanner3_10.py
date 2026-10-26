from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

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
    # Ensure we have at least as many parts as non-wildcard args
    if len(parts) < len([arg for arg in args if arg != '*']):
         return False
    # Check if each part matches the corresponding arg pattern
    # Use min length to avoid index error if parts is shorter than args (shouldn't happen with correct patterns)
    for i in range(min(len(parts), len(args))):
        if not fnmatch(parts[i], args[i]):
            return False
    # If args is longer than parts, check if remaining args are all wildcards
    if len(args) > len(parts):
        return all(arg == '*' for arg in args[len(parts):])

    return True


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It sums the number of required tighten actions, the number of required spanner
    pickup actions, and an estimated walk cost for the man to reach necessary locations.

    # Assumptions
    - The goal is to tighten all nuts that are initially loose.
    - Nuts and spanners stay at their initial locations unless moved by actions.
    - The man can carry multiple spanners.
    - There are enough usable spanners available (carried or on the ground)
      to tighten all loose nuts in solvable instances.
    - Locations are connected by 'link' predicates forming a graph.
    - The man object is typically named 'bob'. Nut objects start with 'nut'. Spanner objects start with 'spanner'.

    # Heuristic Initialization
    - Extracts static 'link' facts to build a graph of locations.
    - Computes all-pairs shortest paths between locations using BFS.
    - Infers the set of all nuts in the problem from initial state and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location. Assume the man is the object named 'bob'.
    2. Identify all nuts that are currently loose and their locations.
    3. Identify all usable spanners the man is currently carrying.
    4. Identify all usable spanners currently on the ground and their locations.
    5. Calculate the number of loose nuts (`num_loose`). This is the minimum number of `tighten_nut` actions required.
    6. Calculate the number of usable spanners the man is carrying (`num_carried_usable`).
    7. Calculate the number of additional usable spanners the man needs to pick up from the ground (`num_pickups_needed`). This is `max(0, num_loose - num_carried_usable)`. This is the minimum number of `pickup_spanner` actions required.
    8. Calculate the base action cost as `num_loose + num_pickups_needed`.
    9. Estimate the walk cost for the man:
       - Identify the set of 'required' locations the man must visit. This includes all locations with loose nuts.
       - If `num_pickups_needed > 0`, also include locations with usable spanners on the ground as potential pickup spots.
       - If there are no required locations (all nuts tightened, no pickups needed), the walk cost is 0.
       - Otherwise, the walk cost is estimated as the distance from the man's current location to the nearest required location, plus a penalty for each additional distinct required location. The penalty is simply the count of required locations minus one. This encourages moving towards *any* necessary location and penalizes having many distinct locations to visit.
    10. The total heuristic value is the sum of the base action cost and the estimated walk cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Build the location graph from 'link' facts
        self.location_graph = {}
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.all_locations:
            self._bfs(start_loc)

        # Infer all relevant nuts from initial state and goal facts
        self.all_relevant_nuts = set()
        for fact in initial_state:
             if match(fact, "loose", "*") or match(fact, "tightened", "*"):
                 self.all_relevant_nuts.add(get_parts(fact)[1])
        for goal_fact in self.goals:
             if match(goal_fact, "tightened", "*"):
                 self.all_relevant_nuts.add(get_parts(goal_fact)[1])


    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all other locations."""
        q = deque([(start_loc, 0)])
        visited = {start_loc}
        self.distances[(start_loc, start_loc)] = 0

        while q:
            current_loc, dist = q.popleft()

            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_loc, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Return precomputed distance or infinity if unreachable or location not in graph
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify the man's current location. Assume man is 'bob'.
        man_name = 'bob' # Domain-specific assumption
        man_location = None
        for fact in state:
             if match(fact, "at", man_name, "*"):
                 man_location = get_parts(fact)[2]
                 break

        if man_location is None:
             # Man must be somewhere in a solvable state.
             # This indicates an unexpected state structure.
             # print(f"Warning: Man '{man_name}' location not found in state.")
             return float('inf') # Should not happen in solvable states

        # Check if goal is reached (all relevant nuts are tightened)
        num_tightened_nuts = sum(1 for nut in self.all_relevant_nuts if f'(tightened {nut})' in state)

        if len(self.all_relevant_nuts) > 0 and num_tightened_nuts == len(self.all_relevant_nuts):
             return 0 # Goal reached

        # 2. Identify all nuts that are currently loose and their locations.
        loose_nuts = {} # {nut_name: location}
        for nut_name in self.all_relevant_nuts:
             if f'(loose {nut_name})' in state:
                 # Find the location of this loose nut
                 for loc_fact in state:
                     if match(loc_fact, "at", nut_name, "*"):
                         loose_nuts[nut_name] = get_parts(loc_fact)[2]
                         break # Assuming each nut is at only one location

        # 3. Identify all usable spanners the man is currently carrying.
        usable_spanners_carried = set()
        # Assuming spanners start with 'spanner'
        for fact in state:
            if match(fact, "carrying", man_name, "*"):
                spanner_name = get_parts(fact)[2]
                if spanner_name.startswith('spanner') and f'(usable {spanner_name})' in state:
                    usable_spanners_carried.add(spanner_name)

        # 4. Identify all usable spanners currently on the ground and their locations.
        usable_spanners_on_ground = {} # {spanner_name: location}
        # Assuming spanners start with 'spanner'
        for fact in state:
            if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 obj_name = parts[1]
                 if obj_name.startswith('spanner'):
                     spanner_location = parts[2]
                     # Check if this spanner is usable and not carried
                     if f'(usable {obj_name})' in state and f'(carrying {man_name} {obj_name})' not in state:
                         usable_spanners_on_ground[obj_name] = spanner_location

        # 5. Calculate the number of loose nuts.
        num_loose = len(loose_nuts)

        # 6. Calculate the number of usable spanners the man is carrying.
        num_carried_usable = len(usable_spanners_carried)

        # 7. Calculate the number of additional usable spanners the man needs to pick up.
        num_pickups_needed = max(0, num_loose - num_carried_usable)

        # Check if enough spanners exist in total for a solvable problem
        num_usable_available_total = num_carried_usable + len(usable_spanners_on_ground)
        if num_loose > num_usable_available_total:
             # print(f"Warning: Not enough usable spanners ({num_usable_available_total}) for loose nuts ({num_loose}). State likely unsolvable.")
             return float('inf') # Indicate unsolvable state

        # 8. Calculate the base action cost.
        tighten_cost = num_loose
        pickup_cost = num_pickups_needed

        # 9. Estimate the walk cost.
        required_locations = set(loose_nuts.values()) # Locations of loose nuts

        # If pickups are needed, add locations with usable spanners on the ground
        if num_pickups_needed > 0:
             required_locations.update(usable_spanners_on_ground.values())

        walk_cost = 0
        if required_locations:
             # Find the minimum distance from the man's current location to any required location
             min_dist_to_target = float('inf')
             for loc in required_locations:
                 dist = self.get_distance(man_location, loc)
                 min_dist_to_target = min(min_dist_to_target, dist)

             # If any required location is unreachable from the man, state is unsolvable
             if min_dist_to_target == float('inf'):
                  # print(f"Warning: Required location unreachable from man at {man_location}")
                  return float('inf') # Indicate unsolvable state

             # Estimate walk cost: distance to first target + penalty for remaining targets
             # Penalty is number of additional distinct locations to visit
             walk_cost = min_dist_to_target + (len(required_locations) - 1)

        # 10. Total heuristic value.
        total_cost = tighten_cost + pickup_cost + walk_cost

        return total_cost
