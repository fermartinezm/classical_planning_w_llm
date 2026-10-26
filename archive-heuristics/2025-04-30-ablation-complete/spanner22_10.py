from fnmatch import fnmatch
from collections import deque
import math # Use math.inf for unreachable locations

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If running standalone without the planning framework, uncomment the following
# dummy base class definition:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass

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


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It considers the number of nuts to tighten, the number of
    spanners to pick up, and the travel cost to reach the first required
    location (either a nut or a spanner).

    # Assumptions
    - There is exactly one man object.
    - Locations form a connected graph (or relevant parts are connected).
    - Sufficient usable spanners exist on the ground or are carried to tighten
      all goal nuts. If not, the heuristic returns a large value.
    - Nuts stay at their initial locations.
    - Spanners become unusable after one use.

    # Heuristic Initialization
    - Identify all locations mentioned in static facts, initial state, or goals.
    - Build the location graph from 'link' facts.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify the man object by looking for the first argument of an 'at'
      predicate in the initial state that is not identified as a spanner or nut.
    - Identify the goal nuts from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify the set of goal nuts that are currently 'loose'.
    3. If there are no loose goal nuts, the heuristic is 0 (goal reached).
    4. Identify usable spanners (carried or on the ground).
    5. Check if the total number of available usable spanners is less than the
       number of loose goal nuts. If so, the problem is likely unsolvable
       from this state in this domain, return a large value (e.g., 1000000).
    6. Initialize heuristic value `h` with the number of loose goal nuts
       (representing the 'tighten' actions, each costs 1).
    7. Determine how many spanners need to be picked up from the ground:
       This is the number of loose goal nuts minus 1 if the man is already
       carrying a usable spanner, clamped at a minimum of 0. Add this number
       to `h` (representing 'pickup' actions, each costs 1).
    8. Calculate the minimum travel cost to reach the first required location.
       The man needs to reach at least one loose nut location.
       If spanners need to be picked up, the man also needs to reach at least
       one usable spanner location on the ground.
       The first action will be walking to the nearest location among the
       loose nut locations and (if pickups are needed) usable spanner locations
       on the ground. Calculate the minimum distance from the man's current
       location to any of these required initial target locations. Add this
       minimum distance to `h`.
    9. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing shortest paths.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state
        # self.all_facts = task.facts # Contains all possible facts, including types (less reliable than parsing state/static/goals)

        # 1. Identify all locations
        self.locations = set()
        # Locations from links
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                if len(parts) == 3: # Ensure correct structure
                    self.locations.add(parts[1])
                    self.locations.add(parts[2])
        # Locations from initial state and goals 'at' predicates
        for fact in self.initial_state | self.goals:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 # The second argument of 'at' is a location
                 self.locations.add(parts[2])

        # 2. Build location graph
        self.graph = {loc: [] for loc in self.locations}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                if l1 in self.graph and l2 in self.graph: # Ensure locations were identified
                    self.graph[l1].append(l2)
                    self.graph[l2].append(l1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        self.dist_matrix = {}
        for start_loc in self.locations:
            self.dist_matrix[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.dist_matrix[start_loc][start_loc] = 0

            while q:
                curr_loc, d = q.popleft()
                for neighbor in self.graph.get(curr_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1))
                        self.dist_matrix[start_loc][neighbor] = d + 1

        # 4. Identify the man object
        self.man_obj = None
        # A simple way: find the object that is the first argument of 'at' in initial state
        # and is not a spanner or nut based on predicates like 'usable', 'loose', 'tightened'.
        potential_men = set()
        spanners_or_nuts = set()

        # Scan initial state for potential man (first arg of 'at' or 'carrying')
        # and spanners/nuts (arg of 'usable', 'loose', 'tightened', second arg of 'carrying')
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                 potential_men.add(parts[1])
            if parts[0] == 'carrying' and len(parts) == 3:
                 potential_men.add(parts[1]) # First arg of carrying is man
                 spanners_or_nuts.add(parts[2]) # Second arg of carrying is spanner
            if parts[0] in ['usable', 'loose', 'tightened'] and len(parts) == 2:
                 spanners_or_nuts.add(parts[1])

        # The man is a potential man who is not a spanner or nut
        for obj in potential_men:
            if obj not in spanners_or_nuts:
                self.man_obj = obj
                break

        # Fallback: If not found this way, assume the first object in the first 'at' fact is the man
        if self.man_obj is None:
             for fact in self.initial_state:
                 parts = get_parts(fact)
                 if parts[0] == 'at' and len(parts) == 3:
                      self.man_obj = parts[1]
                      break # Assume the first 'at' fact's first arg is the man

        # 5. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_obj, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man's location must be known in a valid state
             return math.inf # Should not happen in valid states

        # 2. Identify loose goal nuts
        loose_goal_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}

        # 3. If no loose goal nuts, goal reached
        if not loose_goal_nuts:
            return 0

        # 4. Identify usable spanners and their locations/status
        usable_spanners = {s.split()[1] for s in state if match(s, "usable", "*")}
        spanner_locations = {} # Map spanner to its location or man_obj if carried
        for s in usable_spanners:
             # Check if carried
             if f"(carrying {self.man_obj} {s})" in state:
                  spanner_locations[s] = self.man_obj
             else:
                  # Check if on ground
                  for fact in state:
                       if match(fact, "at", s, "*"):
                            spanner_locations[s] = get_parts(fact)[2]
                            break

        usable_spanners_on_ground_locs = {loc for s, loc in spanner_locations.items() if loc != self.man_obj}
        man_carrying_usable = any(loc == self.man_obj for loc in spanner_locations.values())

        # 5. Check if sufficient usable spanners exist
        s_available = len(usable_spanners_on_ground_locs) + (1 if man_carrying_usable else 0)
        if s_available < len(loose_goal_nuts):
             # Not enough spanners to tighten all nuts
             return 1000000 # Large value representing unsolvable

        # 6. Initialize heuristic value
        h = len(loose_goal_nuts) # Base cost: one 'tighten' action per loose goal nut

        # 7. Calculate number of spanners to pick up
        num_pickups_needed = max(0, len(loose_goal_nuts) - (1 if man_carrying_usable else 0))
        h += num_pickups_needed # Add cost for 'pickup' actions

        # 8. Calculate travel cost to the first required location
        # The man needs to reach a loose nut location AND (if pickups are needed) a usable spanner location.
        # The first move will be to the nearest of these locations.

        min_dist_to_nut = math.inf
        nut_locations = {}
        for nut in loose_goal_nuts:
             # Find location of the loose nut
             loc_n = None
             for fact in state:
                  if match(fact, "at", nut, "*"):
                       loc_n = get_parts(fact)[2]
                       nut_locations[nut] = loc_n
                       break
             if loc_n and man_loc in self.dist_matrix and loc_n in self.dist_matrix[man_loc]:
                  min_dist_to_nut = min(min_dist_to_nut, self.dist_matrix[man_loc][loc_n])
             else:
                 # Nut location not found or unreachable - problem likely unsolvable
                 return math.inf

        min_dist_to_spanner = math.inf
        if num_pickups_needed > 0:
             if usable_spanners_on_ground_locs:
                  for loc_s in usable_spanners_on_ground_locs:
                       if man_loc in self.dist_matrix and loc_s in self.dist_matrix[man_loc]:
                            min_dist_to_spanner = min(min_dist_to_spanner, self.dist_matrix[man_loc][loc_s])
                       else:
                           # Spanner location not found or unreachable - problem likely unsolvable
                           return math.inf
             # else: num_pickups_needed > 0 but no usable spanners on ground. This case
             # is already covered by the s_available check returning 1000000.

        # Add the minimum distance to the first target location
        if num_pickups_needed > 0:
             # First target is either nearest nut or nearest spanner
             h += min(min_dist_to_nut, min_dist_to_spanner)
        else:
             # No pickups needed, first target is nearest nut
             h += min_dist_to_nut

        return h
