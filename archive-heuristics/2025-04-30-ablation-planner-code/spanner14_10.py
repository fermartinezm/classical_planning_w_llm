from fnmatch import fnmatch
import math # Use math.inf for clarity

# Assuming Heuristic base class is available in the environment like:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

# If the base class is not provided, a dummy one might be needed for execution context,
# but the final output should only be the class below.
# For this response, we assume the Heuristic base class exists.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return [] # Return empty list for malformed facts
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the total number of actions required to tighten all
    goal nuts. It sums the estimated minimum cost for each individual loose goal nut,
    considering the man's current location and the locations of usable spanners.

    # Assumptions
    - Each goal nut must be tightened.
    - Each tightening action requires one usable spanner, which becomes unusable afterwards.
    - The man is the only agent capable of moving, picking up spanners, and tightening nuts.
    - The cost of each action (walk, pickup, tighten) is 1.
    - The heuristic calculates the minimum cost for each nut independently, potentially
      overestimating if resources (man, spanners) or paths are shared efficiently
      in the optimal plan. This is a relaxed, non-admissible heuristic suitable for
      greedy best-first search.
    - If there are more loose goal nuts than available usable spanners, the problem
      is considered unsolvable, and the heuristic returns infinity.
    - The state representation includes 'at' facts for all relevant locatable objects
      (man, nuts, spanners) and 'usable' facts for usable spanners.

    # Heuristic Initialization
    - Extracts all location names and the links between them from static facts.
    - Computes all-pairs shortest paths between locations using the Floyd-Warshall algorithm.
    - Identifies the set of nuts that are required to be tightened in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all nuts that are in the goal state but are currently loose (not tightened).
       These are the 'loose goal nuts'. A nut is loose if it's a goal nut and the fact
       '(tightened <nut-name>)' is not present in the state.
    2. If there are no loose goal nuts, the state is a goal state, and the heuristic is 0.
    3. Identify the man's current location by finding the fact '(at <man-name> <location>)'.
       The man's name is inferred from 'carrying' facts or by finding the unique locatable
       object that is not a spanner or nut based on facts in the state.
    4. Identify all usable spanners by finding facts '(usable <spanner-name>)'.
    5. For each usable spanner, find its current location by finding the fact
       '(at <spanner-name> <location>)' or determine if it is carried by finding
       '(carrying <man-name> <spanner-name>)'.
    6. If the number of loose goal nuts exceeds the number of available usable spanners,
       return infinity (problem is considered unsolvable in this domain).
    7. Initialize the total heuristic cost to 0.
    8. For each loose goal nut `n` at its current location `l_n` (found via '(at <n> <l_n>)'):
       a. Calculate the minimum cost to tighten this specific nut, considering the man's
          current location (`man_loc`) and all available usable spanners.
       b. The cost to tighten nut `n` involves:
          - Getting the man to `l_n` with a usable spanner.
          - Performing the `tighten_nut` action (cost 1).
       c. The cost to get the man to `l_n` with a usable spanner is the minimum of:
          - If the man is currently carrying a usable spanner (and it is usable):
            The shortest distance from `man_loc` to `l_n`.
          - If the man is not carrying a usable spanner, or to find a potentially cheaper path
            using a usable spanner at a location:
            For each usable spanner `s` located at `l_s`: The shortest distance from `man_loc`
            to `l_s`, plus 1 (for pickup), plus the shortest distance from `l_s` to `l_n`.
       d. The minimum cost for nut `n` is `min(cost_from_carried_spanner, min_cost_from_spanners_at_locations) + 1` (for the tighten action).
       e. Add this minimum cost for nut `n` to the total heuristic cost.
    9. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing
        shortest paths.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Extract location graph and all locations
        self.locations = set()
        links = []
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                links.append((loc1, loc2))
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Add any locations mentioned in goals.
        # This helps include locations that might not be part of a link but are goal points.
        for goal in self.goals:
             # Goals are typically (tightened ?n), but a goal could theoretically be (at ?obj ?loc)
             # Check for 'at' predicates in goals to include goal locations.
             if match(goal, "at", "*", "*"):
                  _, obj, loc = get_parts(goal)
                  self.locations.add(loc)

        self.locations = sorted(list(self.locations)) # Consistent ordering for matrix
        self.location_to_idx = {loc: i for i, loc in enumerate(self.locations)}
        num_locations = len(self.locations)

        # Initialize distance matrix
        self.dist = [[math.inf] * num_locations for _ in range(num_locations)]
        for i in range(num_locations):
            self.dist[i][i] = 0

        # Add direct links (cost 1)
        for loc1, loc2 in links:
            if loc1 in self.location_to_idx and loc2 in self.location_to_idx:
                i, j = self.location_to_idx[loc1], self.location_to_idx[loc2]
                self.dist[i][j] = 1
                self.dist[j][i] = 1 # Links are bidirectional
            # else: Link involves a location not inferred? Ignore.

        # Floyd-Warshall to compute all-pairs shortest paths
        for k in range(num_locations):
            for i in range(num_locations):
                for j in range(num_locations):
                    if self.dist[i][k] != math.inf and self.dist[k][j] != math.inf:
                        self.dist[i][j] = min(self.dist[i][j], self.dist[i][k] + self.dist[k][j])

        # Store shortest paths lookup
        self.shortest_paths = {}
        for i in range(num_locations):
            for j in range(num_locations):
                 self.shortest_paths[(self.locations[i], self.locations[j])] = self.dist[i][j]

        # 2. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut_name = get_parts(goal)
                self.goal_nuts.add(nut_name)

    def get_distance(self, loc1, loc2):
        """Lookup shortest distance between two locations."""
        if loc1 not in self.location_to_idx or loc2 not in self.location_to_idx:
             # One or both locations are not in our graph of known locations
             return math.inf
        return self.shortest_paths.get((loc1, loc2), math.inf)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify objects and their current status/location from state facts
        object_locations = {} # Map object name to its location
        usable_spanners_set = set() # Set of spanner names that are usable
        tightened_nuts_set = set() # Set of nut names that are tightened
        man_name = None
        carried_spanner_name = None

        # First pass to collect basic facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                object_locations[obj] = loc
            elif parts[0] == 'carrying' and len(parts) == 3:
                 m_name, s_name = parts[1], parts[2]
                 man_name = m_name # Found man name
                 carried_spanner_name = s_name # Found carried spanner name
            elif parts[0] == 'usable' and len(parts) == 2:
                 s_name = parts[1]
                 usable_spanners_set.add(s_name)
            elif parts[0] == 'tightened' and len(parts) == 2:
                 n_name = parts[1]
                 tightened_nuts_set.add(n_name)

        # If man name wasn't found via 'carrying', try finding the unique object at a location
        # that isn't a spanner or nut based on the objects we've seen.
        # This is an inference step necessary without full type information.
        if man_name is None:
             # Collect all objects seen at locations or being carried
             all_objects_in_state = set(object_locations.keys())
             if carried_spanner_name:
                 all_objects_in_state.add(carried_spanner_name)
             # Approximate sets of spanners and nuts seen
             spanners_seen = usable_spanners_set.copy() # All usable ones
             if carried_spanner_name: spanners_seen.add(carried_spanner_name) # Carried one
             nuts_seen = tightened_nuts_set.copy() # All tightened ones
             nuts_seen.update(n for n in self.goal_nuts if n in object_locations) # Goal nuts at locations

             potential_men = all_objects_in_state - spanners_seen - nuts_seen
             if len(potential_men) == 1:
                  man_name = list(potential_men)[0]
             # else: Cannot reliably determine man name. State might be unusual.

        # Get man's location
        man_loc = object_locations.get(man_name)
        if man_name is None or man_loc is None:
             # Man or his location is essential. If not found, state is likely malformed or unreachable.
             return math.inf


        # 2. Identify loose goal nuts in current state and their locations
        loose_goal_nuts_in_state = {} # Map nut name to location
        for nut in self.goal_nuts:
            if nut not in tightened_nuts_set: # Check if nut is NOT tightened
                 # Find the location of the loose nut
                 nut_current_loc = object_locations.get(nut)
                 if nut_current_loc is None:
                      # Loose goal nut location not found in state.
                      # This nut is unreachable or state is malformed.
                      return math.inf
                 loose_goal_nuts_in_state[nut] = nut_current_loc

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts_in_state:
            return 0

        # 3. Identify usable spanners available (at locations or carried)
        usable_spanners_available = {} # Map spanner name to location or 'carrying'
        for s_name in usable_spanners_set:
             if s_name == carried_spanner_name:
                  usable_spanners_available[s_name] = 'carrying'
             elif s_name in object_locations:
                  usable_spanners_available[s_name] = object_locations[s_name]
             # else: usable spanner exists but is not at a location and not carried? Ignore.


        # 4. Check solvability based on spanners
        if len(loose_goal_nuts_in_state) > len(usable_spanners_available):
             return math.inf # Not enough usable spanners for all loose goal nuts

        # 5. Calculate total heuristic cost
        total_h = 0

        for nut, l_n in loose_goal_nuts_in_state.items():
            min_nut_cost = math.inf

            # Option 1: Use the spanner the man is currently carrying (if usable)
            is_carried_spanner_usable = (carried_spanner_name is not None and
                                         carried_spanner_name in usable_spanners_set)
            if is_carried_spanner_usable:
                 # Cost = travel from man_loc to nut_loc + tighten action
                 travel_cost = self.get_distance(man_loc, l_n)
                 if travel_cost != math.inf:
                      min_nut_cost = min(min_nut_cost, travel_cost + 1) # +1 for tighten

            # Option 2: Use a usable spanner available at a location
            for s_name, s_loc in usable_spanners_available.items():
                 if s_loc != 'carrying': # Consider spanners at physical locations
                      # Cost = travel from man_loc to spanner_loc + pickup + travel from spanner_loc to nut_loc + tighten
                      travel_to_spanner_cost = self.get_distance(man_loc, s_loc)
                      travel_spanner_to_nut_cost = self.get_distance(s_loc, l_n)
                      if travel_to_spanner_cost != math.inf and travel_spanner_to_nut_cost != math.inf:
                           cost = travel_to_spanner_cost + 1 + travel_spanner_to_nut_cost + 1 # +1 pickup, +1 tighten
                           min_nut_cost = min(min_nut_cost, cost)

            # If min_nut_cost is still infinity, this nut is unreachable
            if min_nut_cost == math.inf:
                 # If even one loose goal nut is unreachable, the whole goal is unreachable
                 return math.inf

            total_h += min_nut_cost

        return total_h
