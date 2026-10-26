# The base class Heuristic is assumed to be available in the environment.
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at bob shed)" -> ["at", "bob", "shed"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to precompute distances
def get_distance_map(locations, links):
    """Computes shortest path distances between all pairs of locations using BFS."""
    dist_map = {}
    adj = {}
    for l1, l2 in links:
        adj.setdefault(l1, []).append(l2)
        adj.setdefault(l2, []).append(l1) # Assuming links are bidirectional

    for start_loc in locations:
        dist_map[start_loc] = {}
        q = deque([(start_loc, 0)])
        visited = {start_loc}
        while q:
            current_loc, dist = q.popleft()
            dist_map[start_loc][current_loc] = dist
            if current_loc in adj:
                for neighbor in adj[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))
    return dist_map


# Define the domain-dependent heuristic class
# class spannerHeuristic(Heuristic): # Uncomment if inheriting from Heuristic base
class spannerHeuristic: # Use this if Heuristic base is not provided
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to reach the goal (all nuts tightened) by summing:
    1. The number of loose nuts (representing the required tighten actions).
    2. The number of usable spanners that need to be picked up from the ground.
    3. The minimum distance the man needs to travel to reach the first relevant
       location (either a loose nut location or a usable spanner location if needed).

    This heuristic is not admissible but aims to guide a greedy search efficiently.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing distances between locations.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Needed to identify the man

        # Precompute distances between locations
        locations = set()
        links = []
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                links.append((l1, l2))
                locations.add(l1)
                locations.add(l2)

        # Add locations mentioned in initial state 'at' predicates
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                  locations.add(parts[2])

        # Add locations mentioned in goal 'at' predicates (if any)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at' and len(parts) == 3:
                  locations.add(parts[2])

        # Handle cases where locations might be defined but not linked or used in init/goal
        # This is a potential limitation if the domain structure is complex.
        # For typical PDDL, locations are usually involved in links or initial/goal positions.

        self.dist_map = get_distance_map(list(locations), links)

        # Store goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Find the man's name (assuming there's only one man)
        self.man_name = None
        # Try finding the object involved in a 'carrying' predicate in the initial state
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'carrying' and len(parts) == 3:
                  self.man_name = parts[1]
                  break
        # If not found, try finding an object 'at' a location that is not a spanner or nut pattern
        if self.man_name is None:
             all_spanners_nuts_in_init = set()
             for fact in self.initial_state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and len(parts) == 3:
                       obj = parts[1]
                       # Fragile assumption based on naming convention 'spanner*' and 'nut*'
                       if obj.startswith('spanner') or obj.startswith('nut'):
                            all_spanners_nuts_in_init.add(obj)
             for fact in self.initial_state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and len(parts) == 3:
                       obj_name = parts[1]
                       if obj_name not in all_spanners_nuts_in_init:
                            self.man_name = obj_name
                            break

        # If man_name is still None, the problem setup is unexpected.
        # The heuristic will return inf later if man_loc isn't found.


    def get_distance(self, loc1, loc2):
        """Safely get distance, return infinity if no path."""
        if loc1 not in self.dist_map or loc2 not in self.dist_map[loc1]:
             # If locations are not in the precomputed map, they might be isolated.
             # If loc1 == loc2, distance is 0. Otherwise, infinity.
             return 0 if loc1 == loc2 else float('inf')
        return self.dist_map[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If man_name wasn't found in init, we can't proceed.
        if self.man_name is None:
             # print("Error: Man object not identified during heuristic initialization.") # Debugging
             return float('inf') # Cannot compute heuristic without knowing the man

        # 1. Identify man's current location
        man_loc = None
        for fact in state:
            if match(fact, 'at', self.man_name, '*'):
                man_loc = get_parts(fact)[2]
                break

        if man_loc is None:
             # Man is not at any location? This state is likely invalid or unsolvable.
             # print(f"Warning: Man location not found in state for man '{self.man_name}'. State: {state}") # Debugging
             return float('inf')


        # 2. Identify loose nuts and their locations
        loose_nuts = {} # {nut_name: location}
        for nut_name in self.goal_nuts:
             # Check if the nut is loose in the current state
             if f'(loose {nut_name})' in state:
                  # Find its location (should be static, but check state for robustness)
                  nut_loc = None
                  for fact in state:
                       if match(fact, 'at', nut_name, '*'):
                            nut_loc = get_parts(fact)[2]
                            break
                  if nut_loc is not None:
                       loose_nuts[nut_name] = nut_loc
                  else:
                       # Loose nut exists but its location is unknown? Unsolvable.
                       # print(f"Warning: Location of loose nut {nut_name} not found in state: {state}") # Debugging
                       return float('inf')


        # If no loose nuts, goal is reached. Heuristic is 0.
        if not loose_nuts:
            return 0

        # 3. Identify usable spanners and their locations (on ground)
        usable_spanners_ground = {} # {spanner_name: location}
        for fact in state:
             if match(fact, 'usable', '*'):
                  spanner_name = get_parts(fact)[1]
                  # Check if it's on the ground (at a location)
                  if match(fact, 'at', spanner_name, '*'):
                       spanner_loc = get_parts(fact)[2]
                       usable_spanners_ground[spanner_name] = spanner_loc

        # 4. Check if man is currently carrying a usable spanner
        man_carrying_usable_spanner = False
        # current_carried_spanner_name = None # Not strictly needed for the heuristic value calculation
        for fact in state:
             if match(fact, 'carrying', self.man_name, '*'):
                  current_carried_spanner_name = get_parts(fact)[2]
                  if f'(usable {current_carried_spanner_name})' in state:
                       man_carrying_usable_spanner = True
                  break # Man can only carry one spanner

        # Heuristic Calculation
        h = 0
        num_loose_nuts = len(loose_nuts)

        # Base cost: 1 action per loose nut (tighten)
        h += num_loose_nuts

        # Cost for spanner acquisition
        # The man needs a usable spanner for each nut.
        # If he starts with one usable, he needs num_loose_nuts - 1 more from the ground.
        # If he doesn't start with one usable, he needs num_loose_nuts from the ground.
        spanners_needed_from_ground = max(0, num_loose_nuts - (1 if man_carrying_usable_spanner else 0))

        # Check if enough usable spanners exist in total
        total_usable_spanners = len(usable_spanners_ground) + (1 if man_carrying_usable_spanner else 0)
        if num_loose_nuts > total_usable_spanners:
             # print(f"Warning: Not enough usable spanners ({total_usable_spanners}) for loose nuts ({num_loose_nuts}). Unsolvable.") # Debugging
             return float('inf') # Unsolvable

        # Cost for pickup actions
        h += spanners_needed_from_ground # Each pickup costs 1

        # Cost for movement
        # The man needs to travel from his current location to the first location he needs to interact with.
        # This first location is either a loose nut location (if he has a spanner) or a usable spanner location (if he needs one).

        locations_to_visit_first = set()
        if loose_nuts:
             locations_to_visit_first.update(loose_nuts.values())

        if spanners_needed_from_ground > 0:
             locations_to_visit_first.update(usable_spanners_ground.values())

        min_dist_to_first_target = float('inf')
        if locations_to_visit_first:
             for target_loc in locations_to_visit_first:
                  dist = self.get_distance(man_loc, target_loc)
                  if dist == float('inf'):
                       # Cannot reach a required location
                       # print(f"Warning: Cannot reach required location {target_loc} from man's location {man_loc}.") # Debugging
                       return float('inf')
                  if dist < min_dist_to_first_target:
                       min_dist_to_first_target = dist

             if min_dist_to_first_target != float('inf'):
                  h += min_dist_to_first_target
             # else: locations_to_visit_first is not empty but all targets are unreachable, handled by dist == inf check.
        # else: locations_to_visit_first is empty, meaning no loose nuts, which is handled at the start (h=0).

        return h
