import collections
from fnmatch import fnmatch
# Assuming heuristic_base.py is available in a 'heuristics' directory
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided externally for standalone testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic must be callable")


# Helper functions to parse PDDL facts represented as strings
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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
        return False
    # Check if there are extra parts in the fact not covered by args (unless the last arg is '*')
    if len(parts) > len(args) and args and args[-1] != '*':
         return False
    return True


# BFS function to compute shortest paths on the location graph
def bfs(graph, start_node):
    """
    Performs BFS starting from start_node to find shortest distances to all reachable nodes.
    Returns a dictionary mapping node -> distance.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
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

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost based on:
    1. The number of loose goal nuts (representing tighten actions).
    2. The travel cost for the man to reach the closest loose goal nut.
    3. The cost to acquire necessary usable spanners if the man isn't carrying enough.
       This includes travel to the closest usable spanner on the ground + pickup
       for the first needed spanner, plus pickup costs for subsequent needed spanners.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts, locations,
        building the location graph, and precomputing distances.
        Also identifies man, spanners, and nuts, and checks initial solvability.
        """
        super().__init__(task) # Call base class constructor if needed

        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # Identify goal nuts from the goal conditions
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Identify all locations and build the graph
        self.locations = set()
        self.graph = collections.defaultdict(list)

        # Locations from initial state facts (objects at locations)
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                self.locations.add(loc)

        # Locations and links from static facts
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                # Links are bidirectional
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Precompute all-pairs shortest paths using BFS from each location
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(self.graph, start_loc)

        # Identify man, all spanners, and all nuts from initial state/goals
        # This is a simple inference based on common naming patterns and facts.
        self.man = None
        self.all_spanners = set()
        self.all_nuts = set()

        # Collect objects and infer types based on predicates they appear in
        for fact in self.initial_state | self.goals:
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate == 'at' and len(parts) == 3:
                  obj_name = parts[1]
                  loc_name = parts[2]
                  # Cannot infer type from 'at' alone without type info in fact
             elif predicate == 'carrying' and len(parts) == 3:
                  man_name = parts[1]
                  spanner_name = parts[2]
                  self.man = man_name # Found the man
                  self.all_spanners.add(spanner_name)
             elif predicate in ['loose', 'tightened'] and len(parts) == 2:
                  nut_name = parts[1]
                  self.all_nuts.add(nut_name)
             elif predicate == 'usable' and len(parts) == 2:
                  spanner_name = parts[1]
                  self.all_spanners.add(spanner_name)

        # Fallback for man identification if not found via 'carrying'
        if self.man is None:
             for fact in self.initial_state:
                  if match(fact, "at", "*", "*"):
                       obj_name = get_parts(fact)[1]
                       # Assume the man is the only locatable that isn't a spanner or nut
                       if obj_name not in self.all_spanners and obj_name not in self.all_nuts and obj_name not in self.locations:
                            self.man = obj_name
                            break


        # Check if problem is solvable in principle (enough usable spanners total initially)
        initial_usable_spanners = {s for s in self.all_spanners if f"(usable {s})" in self.initial_state}
        initially_loose_goal_nuts = {n for n in self.goal_nuts if f"(loose {n})" in self.initial_state}

        if len(initial_usable_spanners) < len(initially_loose_goal_nuts):
             # Problem is impossible if there are fewer total usable spanners initially than goal nuts
             self.is_solvable = False
        else:
             self.is_solvable = True


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a state where all goal nuts are tightened.
        Returns float('inf') if the state is estimated to be a dead end.
        """
        state = node.state

        # If the problem is known to be unsolvable from the start, return infinity
        if not self.is_solvable:
             return float('inf')

        # Find loose goal nuts in the current state that are part of the goal
        loose_goal_nuts_in_state = {n for n in self.goal_nuts if f"(loose {n})" in state}

        # If all goal nuts are tightened, the goal is reached
        if not loose_goal_nuts_in_state:
            return 0

        # Base cost: one tighten action for each loose goal nut
        h = len(loose_goal_nuts_in_state)

        # Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man's location should always be known in a valid state
             return float('inf') # Indicates an unexpected state representation

        # Add cost to reach the closest loose goal nut
        min_dist_to_nut = float('inf')
        nut_locations = {}
        for nut in loose_goal_nuts_in_state:
             # Find the location of the nut in the current state
             nut_loc = None
             for fact in state:
                  if match(fact, "at", nut, "*"):
                       nut_loc = get_parts(fact)[2]
                       break
             if nut_loc:
                  nut_locations[nut] = nut_loc
                  if man_location in self.distances and nut_loc in self.distances[man_location]:
                       min_dist_to_nut = min(min_dist_to_nut, self.distances[man_location][nut_loc])
                  else:
                       # A nut is at a location unreachable from the man's current location
                       return float('inf') # Dead end
             else:
                  # A loose goal nut is not at any location? Invalid state?
                  return float('inf')

        if min_dist_to_nut != float('inf'):
             h += min_dist_to_nut
        else:
             # This implies a nut is in a location not found in precomputed distances
             # (e.g., not connected to the main graph component)
             return float('inf') # Dead end

        # Find usable spanners carried by the man and on the ground
        usable_carried = set()
        usable_on_ground = {} # Map location -> list of spanners
        all_usable_spanners_in_state = set()

        for spanner in self.all_spanners:
             is_usable = f"(usable {spanner})" in state
             if is_usable:
                  all_usable_spanners_in_state.add(spanner)
                  if f"(carrying {self.man} {spanner})" in state:
                       usable_carried.add(spanner)
                  else:
                       # Find spanner location if on ground
                       spanner_loc = None
                       for fact in state:
                            if match(fact, "at", spanner, "*"):
                                 spanner_loc = get_parts(fact)[2]
                                 break
                       if spanner_loc:
                            if spanner_loc not in usable_on_ground:
                                 usable_on_ground[spanner_loc] = []
                            usable_on_ground[spanner_loc].append(spanner)
                       # else: spanner is usable but not carried and not at a location? Invalid state?

        num_carried_usable = len(usable_carried)
        num_loose_goal_nuts = len(loose_goal_nuts_in_state)

        # Check if enough usable spanners exist in total in the current state for remaining nuts
        if len(all_usable_spanners_in_state) < num_loose_goal_nuts:
             return float('inf') # Not enough usable spanners anywhere for all remaining nuts

        # Add cost for acquiring additional spanners if needed
        num_needed_additional_spanners = max(0, num_loose_goal_nuts - num_carried_usable)

        if num_needed_additional_spanners > 0:
            # Check if there are enough usable spanners on the ground to pick up
            num_usable_on_ground_total = sum(len(spanners) for spanners in usable_on_ground.values())
            if num_usable_on_ground_total < num_needed_additional_spanners:
                 return float('inf') # Not enough usable spanners on the ground to meet the need

            # Cost to get the first additional spanner: travel to closest + pickup
            min_dist_to_spanner_pickup = float('inf')
            for s_loc in usable_on_ground.keys(): # Iterate locations with usable spanners
                 if man_location in self.distances and s_loc in self.distances[man_location]:
                      min_dist_to_spanner_pickup = min(min_dist_to_spanner_pickup, self.distances[man_location][s_loc] + 1) # +1 for pickup action
                 else:
                      # A spanner is at a location unreachable from the man's current location
                      return float('inf') # Dead end

            if min_dist_to_spanner_pickup == float('inf'):
                 # Needed spanners but none are reachable on the ground
                 return float('inf') # Dead end

            h += min_dist_to_spanner_pickup

            # Cost for subsequent needed spanners (simplified to just pickup actions)
            h += max(0, num_needed_additional_spanners - 1)


        return h

