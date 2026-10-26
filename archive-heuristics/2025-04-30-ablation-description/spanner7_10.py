from fnmatch import fnmatch
import collections
import math

# Assume Heuristic base class is NOT provided and define the class directly.
# If a base class 'Heuristic' from 'heuristics.heuristic_base' is expected,
# the class definition should be 'class spannerHeuristic(Heuristic):'

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle unexpected fact format, return empty list
         return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all required nuts. It sums the
    estimated cost for each loose goal nut independently. The estimated cost
    for a single nut includes the fixed costs of pickup and tighten actions,
    plus the estimated movement cost for the man to reach a usable spanner,
    pick it up, and carry it to the nut's location.

    # Assumptions
    - Each loose goal nut requires a distinct usable spanner.
    - The man is the only agent capable of movement and action execution.
    - The shortest path distance between locations is a reasonable estimate
      of movement cost.
    - Any currently usable spanner can be used for any remaining loose goal nut.
      (This is a simplification that makes the heuristic non-admissible but
       potentially effective for greedy search).
    - There is exactly one object of type 'man' in the problem.
    - Objects involved in 'usable', 'loose', 'tightened', 'carrying' are spanners or nuts.
    - The man's location can be determined from 'at' or 'carrying' facts.

    # Heuristic Initialization
    - Extract the goal conditions to identify which nuts need tightening.
    - Build a graph of locations based on `link` facts from static information.
    - Compute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all nuts that are currently loose but are required to be tightened in the goal state. These are the 'loose goal nuts'.
    2. If there are no loose goal nuts, the heuristic value is 0 (goal state or sub-goal state where all required nuts are tightened).
    3. Identify the man object and his current location. This is done by examining state facts: first check 'carrying' facts, then look for the unique object in an 'at' fact that is not identified as a spanner or nut.
    4. Identify all usable spanners and their current locations (either on the ground or being carried by the man).
    5. If there are loose goal nuts but no usable spanners, the problem is likely unsolvable in this domain (as spanners are consumed and cannot be made usable again), so return a very large value (infinity).
    6. For each loose goal nut `n` at location `l_n`:
       a. Initialize the estimated cost for this nut to 0.
       b. Add the cost of the `tighten_nut` action (1).
       c. Add the cost of the `pickup_spanner` action (1).
       d. Calculate the minimum movement cost required to get the man from his current location (`man_location`) to a usable spanner's location (`l_s`) and then carry that spanner from `l_s` to the nut's location (`l_n`). This minimum is calculated over all currently available usable spanners `s` at location `l_s`: `dist(man_location, l_s) + dist(l_s, l_n)`.
       e. Add this minimum movement cost to the nut's estimated cost.
    7. The total heuristic value is the sum of the estimated costs for all loose goal nuts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and computing
        shortest path distances.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'tightened':
                if len(parts) > 1:
                    self.goal_nuts.add(parts[1])

        # Build location graph and compute distances
        self.locations = set()
        self.graph = collections.defaultdict(set)

        # Find all locations and build graph from link facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link':
                if len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    self.locations.add(l1)
                    self.locations.add(l2)
                    self.graph[l1].add(l2)
                    self.graph[l2].add(l1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in list(self.locations): # Use list to avoid modifying set during iteration
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while queue:
                current_loc, dist = queue.popleft()

                for neighbor in self.graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_loc, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Distances are stored bidirectionally, so order doesn't matter
        # Check both (loc1, loc2) and (loc2, loc1) keys
        if (loc1, loc2) in self.distances:
             return self.distances[(loc1, loc2)]
        elif (loc2, loc1) in self.distances:
             return self.distances[(loc2, loc1)]
        else:
             return math.inf # Return infinity if no path

    def __call__(self, node):
        """
        Estimate the minimum cost to tighten all remaining loose goal nuts.
        """
        state = node.state

        # 1. Identify loose goal nuts and their locations
        loose_goal_nuts_locs = {}
        nut_names_in_state = set()

        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'loose' and len(parts) > 1:
                 nut = parts[1]
                 nut_names_in_state.add(nut)
                 if nut in self.goal_nuts:
                     # Find the location of this loose goal nut
                     nut_loc = None
                     for fact2 in state:
                          parts2 = get_parts(fact2)
                          if parts2 and parts2[0] == 'at' and len(parts2) > 1 and parts2[1] == nut:
                              nut_loc = parts2[2]
                              break
                     if nut_loc:
                          loose_goal_nuts_locs[nut] = nut_loc
                     # If nut_loc is None, the nut is not 'at' any location, which is weird state.
             elif parts[0] == 'tightened' and len(parts) > 1:
                  nut_names_in_state.add(parts[1])


        # Remove nuts that are already tightened from loose_goal_nuts_locs
        tightened_nuts_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'tightened' and len(parts) > 1:
                 tightened_nuts_in_state.add(parts[1])

        loose_goal_nuts_locs = {
            nut: loc for nut, loc in loose_goal_nuts_locs.items()
            if nut not in tightened_nuts_in_state
        }


        # 2. If no loose goal nuts, return 0
        if not loose_goal_nuts_locs:
            return 0

        # 3. Identify the man object and his current location
        man_name = None
        man_location = None
        spanner_names_in_state = set()

        # First, try finding the carrier in 'carrying' facts
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'carrying' and len(parts) > 2:
                 man_name = parts[1] # Assume carrier is the man
                 spanner_names_in_state.add(parts[2])
                 # Don't break, collect all spanners being carried

        # Collect spanner names from 'usable' facts as well
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'usable' and len(parts) > 1:
                 spanner_names_in_state.add(parts[1])


        # If man_name wasn't found via 'carrying', try finding the unique 'at' object that isn't a spanner or nut
        if man_name is None:
             potential_man_locs = []
             for fact in state:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'at' and len(parts) > 2:
                     obj, loc = parts[1], parts[2]
                     if obj not in spanner_names_in_state and obj not in nut_names_in_state:
                         potential_man_locs.append((obj, loc))

             if len(potential_man_locs) == 1:
                 man_name, man_location = potential_man_locs[0]
             elif len(potential_man_locs) > 1:
                 # Ambiguous state, multiple potential men? Or heuristic logic is wrong.
                 return math.inf
             else:
                 # Man's location not found in state? Problematic.
                 return math.inf
        else: # man_name was found via 'carrying', now find his location
             man_location = None
             for fact in state:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'at' and len(parts) > 2 and parts[1] == man_name:
                     man_location = parts[2]
                     break
             if man_location is None:
                  # Man is carrying but not at a location? Invalid state.
                  return math.inf


        # Check if we successfully found man_location
        if man_location is None:
             return math.inf # Should be caught by the logic above, but double check


        # 4. Identify all usable spanners and their current locations
        usable_spanners_locs = [] # List of (spanner_name, location)
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'usable' and len(parts) > 1:
                 spanner = parts[1]
                 # Find location of this usable spanner
                 spanner_loc = None
                 for fact2 in state:
                     parts2 = get_parts(fact2)
                     if parts2 and parts2[0] == 'at' and len(parts2) > 2 and parts2[1] == spanner:
                         spanner_loc = parts2[2]
                         break
                     elif parts2 and parts2[0] == 'carrying' and len(parts2) > 2 and parts2[2] == spanner and parts2[1] == man_name:
                         # Spanner is carried by the man, its location is man's location
                         spanner_loc = man_location
                         break
                 if spanner_loc:
                     usable_spanners_locs.append((spanner, spanner_loc))

        # 5. If no usable spanners but loose goal nuts exist, return infinity
        if not usable_spanners_locs:
             return math.inf

        # 6. Calculate heuristic sum
        total_cost = 0
        for nut, nut_loc in loose_goal_nuts_locs.items():
            # Cost for this nut = 1 (tighten) + 1 (pickup) + min_spanner_travel_cost
            tighten_cost = 1
            pickup_cost = 1

            # Calculate min_spanner_travel_cost(man_location, nut_loc)
            min_travel_cost = math.inf
            for spanner, spanner_loc in usable_spanners_locs:
                dist_man_to_spanner = self.get_distance(man_location, spanner_loc)
                dist_spanner_to_nut = self.get_distance(spanner_loc, nut_loc)

                if dist_man_to_spanner == math.inf or dist_spanner_to_nut == math.inf:
                     # Cannot reach spanner or nut location, this spanner is not useful
                     continue

                travel_cost = dist_man_to_spanner + dist_spanner_to_nut
                min_travel_cost = min(min_travel_cost, travel_cost)

            if min_travel_cost == math.inf:
                 # Cannot reach any usable spanner or bring it to the nut
                 return math.inf # Problem unsolvable from this state

            total_cost += tighten_cost + pickup_cost + min_travel_cost

        return total_cost
