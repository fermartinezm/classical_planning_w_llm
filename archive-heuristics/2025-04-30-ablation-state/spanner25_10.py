from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
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

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all reachable nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {node: math.inf for node in graph}
    if start_node not in graph: # Handle cases where start_node is not in the graph
        return distances # All distances remain infinity

    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        if current_node in graph: # Handle nodes with no outgoing links
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing the estimated
    minimum costs for tightening each loose nut independently. The cost for a single
    nut includes travel to the nut's location, acquiring a usable spanner if needed,
    and the tighten action itself.

    # Assumptions
    - The man can carry multiple spanners simultaneously.
    - A spanner becomes unusable after tightening one nut.
    - The problem is unsolvable from a state if the number of currently usable spanners
      (carried or on ground) is less than the number of loose nuts that need tightening.
    - The cost of actions is 1 (walk, pickup_spanner, tighten_nut).

    # Heuristic Initialization
    - Extracts static information from the initial state: the man object name, all nut names,
      all spanner names, all location names, and nut locations (nuts are static).
    - Builds a graph of locations based on `link` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location. If the man's location is unknown or unreachable, return infinity.
    2. Identify all loose nuts in the current state. If none, the heuristic is 0.
    3. Identify all usable spanners (carried or on the ground).
    4. Check if the number of currently usable spanners is less than the number of loose nuts. If so, the problem is unsolvable from this state, return infinity.
    5. For each loose nut `N` that needs tightening (i.e., is loose) at location `L_N`:
        a. Initialize the cost for this nut to 1 (for the `tighten_nut` action).
        b. Calculate the minimum cost to get the man to `L_N` while carrying a usable spanner.
           - If the man is currently carrying *any* usable spanner: The cost is the distance from the man's current location to `L_N`. If `L_N` is unreachable, this cost is infinity.
           - If the man is not currently carrying *any* usable spanner:
             - Calculate the minimum cost to acquire a usable spanner and reach `L_N`. This involves considering two sub-options:
               - Sub-option 2a: Travel to `L_N` and pick up a usable spanner *if one is available at `L_N`*. The cost is the distance from the man to `L_N` plus 1 (for pickup). This is only possible if `L_N` is reachable and a usable spanner is on the ground at `L_N`.
               - Sub-option 2b: Travel to the closest usable spanner on the ground, pick it up, and then travel to `L_N`. The cost is the distance from the man to the spanner's location, plus 1 (for pickup), plus the distance from the spanner's location to `L_N`. This is only possible if there are usable spanners on the ground and all necessary locations are reachable.
             - The minimum cost to get a spanner and reach `L_N` is the minimum of the applicable sub-options (2a and 2b). If no usable spanners are available anywhere or locations are unreachable, this cost is infinity.
        c. Add the minimum cost calculated in step 5b to the cost for this nut.
        d. If the cost calculated in step 5b was infinity, the problem is unsolvable from this state, return infinity immediately.
    6. The total heuristic value is the sum of the costs calculated for each loose nut.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and computing distances."""
        self.goals = task.goals
        # Static facts are not explicitly used, derived from initial state or task structure

        self.man_obj = None
        self.all_nuts = set()
        self.all_spanners = set()
        self.all_locations = set()
        self.nut_locations = {} # nut -> location
        links = []

        # Parse initial state to find objects, locations, links
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                # Infer object types based on naming convention or typical roles
                if obj.startswith("bob"): self.man_obj = obj
                elif obj.startswith("spanner"): self.all_spanners.add(obj)
                elif obj.startswith("nut"):
                     self.all_nuts.add(obj)
                     self.nut_locations[obj] = loc
                self.all_locations.add(loc)
            elif parts[0] == "loose":
                self.all_nuts.add(parts[1])
            elif parts[0] == "tightened":
                 self.all_nuts.add(parts[1])
            elif parts[0] == "usable":
                self.all_spanners.add(parts[1])
            elif parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                links.append((l1, l2))
                self.all_locations.add(l1)
                self.all_locations.add(l2)
            elif parts[0] == "carrying":
                 self.all_spanners.add(parts[2]) # Add carried spanner to list of all spanners

        # Build location graph
        self.graph = {loc: [] for loc in self.all_locations}
        for l1, l2 in links:
            if l1 in self.graph and l2 in self.graph: # Ensure locations are in the graph
                self.graph[l1].append(l2)
                self.graph[l2].append(l1) # Assuming links are bidirectional

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = bfs(self.graph, start_loc)

    def get_location(self, obj, state):
        """Finds the location of an object in the current state."""
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]
            if self.man_obj and match(fact, "carrying", self.man_obj, obj):
                 # If carried, its effective location for pickup/drop is the man's location
                 return self.get_location(self.man_obj, state)
        return None # Object not found at a location or carried

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify man's location
        man_loc = self.get_location(self.man_obj, state)
        if man_loc is None or man_loc not in self.all_locations: # Man must be at a known location
             return math.inf

        # Identify loose nuts
        loose_nuts = {n for n in self.all_nuts if f"(loose {n})" in state}
        num_loose_nuts = len(loose_nuts)

        # If no loose nuts, goal is reached
        if num_loose_nuts == 0:
            return 0

        # Identify usable spanners and their status (carried or on ground)
        usable_spanners = {s for s in self.all_spanners if f"(usable {s})" in state}
        carried_spanners = {s for s in self.all_spanners if f"(carrying {self.man_obj} {s})" in state}
        usable_spanners_on_ground = {s for s in usable_spanners if s not in carried_spanners}
        carried_usable_spanners = {s for s in usable_spanners if s in carried_spanners}
        man_carrying_usable = len(carried_usable_spanners) > 0

        # Check if problem is solvable from this state based on available usable spanners
        # If the number of currently usable spanners is less than the number of loose nuts,
        # it's impossible to tighten all remaining nuts.
        if len(usable_spanners) < num_loose_nuts:
             return math.inf

        total_heuristic_cost = 0

        # Calculate cost for each loose nut independently
        for nut in loose_nuts:
            nut_loc = self.nut_locations.get(nut)
            if nut_loc is None or nut_loc not in self.all_locations: # Nut location must be known
                 return math.inf

            # Cost for the tighten_nut action
            cost_for_this_nut = 1

            # Cost to get man to nut_loc with a usable spanner
            cost_get_spanner_and_reach_nut = math.inf

            # Option 1: Man is already carrying a usable spanner
            if man_carrying_usable:
                if man_loc in self.dist and nut_loc in self.dist[man_loc]:
                     cost_get_spanner_and_reach_nut = min(cost_get_spanner_and_reach_nut, self.dist[man_loc][nut_loc])
                # else: nut_loc is unreachable from man_loc, cost remains inf

            # Option 2: Man is not carrying a usable spanner, needs to acquire one and reach nut_loc
            if not man_carrying_usable:
                min_spanner_acquisition_cost_to_nut = math.inf

                # Sub-option 2a: Pick up spanner at nut_loc if available and usable
                usable_spanner_at_nut = any(self.get_location(s, state) == nut_loc for s in usable_spanners_on_ground)
                if usable_spanner_at_nut:
                     # Cost to reach nut and pick up there
                     if man_loc in self.dist and nut_loc in self.dist[man_loc]:
                         cost_option_2a = self.dist[man_loc][nut_loc] + 1
                         min_spanner_acquisition_cost_to_nut = min(min_spanner_acquisition_cost_to_nut, cost_option_2a)
                     # else: nut_loc is unreachable from man_loc, this option is infinite

                # Sub-option 2b: Go to closest usable spanner on ground, pick up, go to nut_loc
                if usable_spanners_on_ground:
                    min_cost_option_2b = math.inf
                    for spanner in usable_spanners_on_ground:
                        spanner_loc = self.get_location(spanner, state)
                        # Ensure spanner_loc is a valid location (not carried)
                        if spanner_loc in self.all_locations:
                            # Ensure all locations are reachable
                            if man_loc in self.dist and spanner_loc in self.dist[man_loc] and nut_loc in self.dist[spanner_loc]:
                                 cost_option_2b_spanner = self.dist[man_loc][spanner_loc] + 1 + self.dist[spanner_loc][nut_loc] # Travel to spanner, pickup, travel to nut
                                 min_cost_option_2b = min(min_cost_option_2b, cost_option_2b_spanner)

                    if min_cost_option_2b != math.inf:
                         min_spanner_acquisition_cost_to_nut = min(min_spanner_acquisition_cost_to_nut, min_cost_option_2b)

                # The cost to get a spanner and reach the nut location is the minimum of these options
                cost_get_spanner_and_reach_nut = min_spanner_acquisition_cost_to_nut

            # If after considering all options, the cost is still infinity, this nut cannot be tightened
            if cost_get_spanner_and_reach_nut == math.inf:
                return math.inf # Problem unsolvable from this state

            cost_for_this_nut += cost_get_spanner_and_reach_nut
            total_heuristic_cost += cost_for_this_nut

        return total_heuristic_cost
