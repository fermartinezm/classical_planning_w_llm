from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available from the environment if needed,
# but the problem description asks only for the class definition.
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
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

# Helper function for BFS
def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    in an unweighted graph.
    """
    distances = {node: math.inf for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., an object location not linked)
         # We can't reach anywhere from here.
         return distances # All distances remain infinity

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current = queue.popleft()

        # current is guaranteed to be in graph keys because it came from the queue,
        # and the queue only gets nodes from graph[current] or the initial start_node.
        # The check `if current in graph:` is technically redundant if graph keys are comprehensive.
        for neighbor in graph[current]:
            if distances[neighbor] == math.inf:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


# Class definition
# class spannerHeuristic(Heuristic): # Inherit if base class is used
class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the minimum number of actions required to tighten
    all goal nuts that are currently loose. It sums the estimated cost for
    each loose goal nut independently. The cost for a single nut is estimated
    based on the travel needed for the man to reach the nut's location and
    acquire a usable spanner, plus the final tighten action.

    # Assumptions
    - The graph of locations connected by 'link' predicates is static and
      can be used to compute shortest paths (number of walk actions).
    - Nuts do not move from their initial locations.
    - There is exactly one man object.
    - There are usable spanners available initially to solve the problem.
      If no usable spanners are found in a state where one is needed,
      the heuristic returns infinity.
    - The cost of any action (walk, pickup, tighten) is 1.

    # Heuristic Initialization
    - Extracts the goal conditions to identify goal nuts.
    - Extracts static 'link' facts to build the location graph.
    - Identifies all potential locations from links, initial state, and goals.
    - Computes all-pairs shortest paths between all identified locations using BFS.
    - Extracts initial locations of all goal nuts (assuming they don't move).
    - Identifies the man object.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man object and his current location (`l_m`).
    2. Determine if the man is currently carrying a usable spanner.
    3. Identify all usable spanners and their current locations (`l_s`).
    4. Initialize the total heuristic cost to 0.
    5. Iterate through each goal nut `?n`.
    6. For each goal nut `?n`:
       - Check if `(tightened ?n)` is already true in the current state. If yes, the cost for this nut is 0.
       - If `(tightened ?n)` is not true (i.e., `(loose ?n)` is effectively true):
         - Find the location of the nut `l_n` (precomputed during initialization).
         - If the man is currently carrying a usable spanner:
           - The estimated cost for this nut is the distance from `l_m` to `l_n` plus 1 (for `tighten_nut`). `cost = dist(l_m, l_n) + 1`.
         - If the man is NOT currently carrying a usable spanner:
           - The man needs to acquire a usable spanner first. This involves going to a spanner location `l_s`, picking it up, and then going to the nut location `l_n`.
           - The estimated cost for this sequence is `dist(l_m, l_s) + 1 (pickup) + dist(l_s, l_n) + 1 (tighten)`.
           - This cost must be minimized over all currently available usable spanners at their respective locations `l_s`.
           - If there are no usable spanners available in the current state, the problem is unsolvable from here; return infinity.
           - The estimated cost for this nut is the minimum of `dist(l_m, l_s) + 1 + dist(l_s, l_n) + 1` over all usable spanners.
         - Add the calculated cost for this nut to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Identify all potential locations.
        potential_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                potential_locations.add(loc1)
                potential_locations.add(loc2)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)[1:]
                 potential_locations.add(loc)
        for goal in self.goals:
             # Goals might specify object locations, though spanner goal is tightened nuts
             # Include locations mentioned in goal 'at' facts if any
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 potential_locations.add(loc)

        self.all_locations = list(potential_locations)

        # 2. Build the location graph from static 'link' facts.
        self.location_graph = {loc: [] for loc in self.all_locations} # Initialize with all potential locations
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations from links are in our identified locations set
                if loc1 in self.location_graph and loc2 in self.location_graph:
                    self.location_graph[loc1].append(loc2)
                    self.location_graph[loc2].append(loc1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths.
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = bfs(self.location_graph, start_loc)

        # 4. Store goal nuts and their initial locations (nuts don't move).
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.goal_nuts.add(nut)

        self.nut_locations = {}
        # Find initial locations of all goal nuts
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.goal_nuts:
                     self.nut_locations[obj] = loc

        # 5. Identify the man object.
        # Assume the object involved in any (carrying ?m ?s) fact in the initial state is the man.
        # If not found, assume the first object at a location in the initial state that is not a goal nut is the man.
        self.man_obj = None
        for fact in initial_state:
             if match(fact, "carrying", "*", "*"):
                 self.man_obj = get_parts(fact)[1]
                 break
        if self.man_obj is None:
             for fact in initial_state:
                 if match(fact, "at", "*", "*"):
                     obj, loc = get_parts(fact)[1:]
                     # Check if this object is a known nut. We don't know spanners yet from types.
                     if obj not in self.goal_nuts and loc in self.all_locations:
                          self.man_obj = obj
                          break
        # If man_obj is still None, the problem instance might be malformed or
        # the identification logic needs refinement based on how objects are listed/typed.
        # For valid spanner problems, the above should identify the man.


    def __call__(self, node):
        """Estimate the minimum cost to tighten all remaining loose goal nuts."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # 1. Find man's current location
        man_loc = None
        if self.man_obj:
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        if man_loc is None or man_loc not in self.all_locations:
             # Man location unknown or not a recognized location. Unsolvable.
             return math.inf

        # 2. Check if man is carrying a usable spanner
        is_carrying_usable_spanner = False
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_obj, "*"):
                carried_spanner = get_parts(fact)[2]
                break
        if carried_spanner is not None:
            if f"(usable {carried_spanner})" in state:
                 is_carrying_usable_spanner = True

        # 3. Find locations of all usable spanners in the current state
        usable_spanners_locs = {} # {spanner_obj: location}
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Assume any object that is 'usable' is a spanner.
                 if f"(usable {obj})" in state:
                      usable_spanners_locs[obj] = loc

        total_cost = 0

        # 4. Iterate through each goal nut that is loose
        loose_goal_nuts = [nut for nut in self.goal_nuts if f"(tightened {nut})" not in state]

        # 5. Sum costs for each loose goal nut
        for nut in loose_goal_nuts:
            nut_loc = self.nut_locations.get(nut)
            if nut_loc is None or nut_loc not in self.all_locations:
                 # Location of a goal nut not found or invalid. Unsolvable.
                 return math.inf

            # Cost for this nut
            cost_for_nut = 0

            if is_carrying_usable_spanner:
                # Man has a spanner, just needs to go to the nut and tighten
                if man_loc in self.dist and nut_loc in self.dist[man_loc] and self.dist[man_loc][nut_loc] != math.inf:
                    cost_for_nut = self.dist[man_loc][nut_loc] + 1 # Walk + Tighten
                else:
                    # Nut location unreachable from man's location. Unsolvable.
                    return math.inf
            else:
                # Man needs to get a spanner first, then go to the nut
                min_path_via_spanner_cost = math.inf
                if not usable_spanners_locs:
                     # No usable spanners available. Unsolvable.
                     return math.inf

                for spanner_obj, spanner_loc in usable_spanners_locs.items():
                     # Check if paths exist: man_loc -> spanner_loc and spanner_loc -> nut_loc
                     if (spanner_loc in self.all_locations and # Ensure spanner location is recognized
                         man_loc in self.dist and spanner_loc in self.dist[man_loc] != math.inf and # Path to spanner exists
                         spanner_loc in self.dist and nut_loc in self.dist[spanner_loc] != math.inf): # Path from spanner to nut exists

                          # Cost: Walk to spanner + Pickup + Walk to nut + Tighten
                          path_cost = self.dist[man_loc][spanner_loc] + 1 + self.dist[spanner_loc][nut_loc] + 1
                          min_path_via_spanner_cost = min(min_path_via_spanner_cost, path_cost)
                     # else: This specific spanner path is not viable, continue to check others.

                if min_path_via_spanner_cost == math.inf:
                     # No usable spanner found from which man can reach the nut. Unsolvable.
                     return math.inf

                cost_for_nut = min_path_via_spanner_cost

            total_cost += cost_for_nut

        return total_cost
