from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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

# Helper function for Breadth-First Search
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Handle nodes with no outgoing links
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all required nuts by summing the estimated
    cost of the first nut-tightening cycle and the estimated cost of
    subsequent cycles. A cycle involves getting a usable spanner, traveling
    to a nut, and tightening it.

    Assumes the man can carry only one spanner at a time.
    Assumes links between locations are traversable in both directions for distance calculation.
    Assumes solvable problems, meaning necessary spanners and nuts are reachable.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by building the location graph and
        precomputing all-pairs shortest paths, and identifying goal nuts.
        """
        # Store task information
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Build location graph from static facts (treating links as undirected)
        self.locations = set()
        self.graph = {} # Adjacency list {loc: [neighbor1, neighbor2, ...]}

        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Add reverse link for undirected graph

        # Ensure all locations found are keys in the graph dictionary
        for loc in self.locations:
             if loc not in self.graph:
                 self.graph[loc] = []

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = bfs(self.graph, start_node)

        # Identify goal nuts and their initial locations (nuts are static)
        self.goal_nuts_locations = {}
        goal_nut_names = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut_name = get_parts(goal)
                goal_nut_names.add(nut_name)

        # Find initial location of these nuts from the initial state
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                _, obj_name, loc_name = get_parts(fact)
                if obj_name in goal_nut_names:
                    self.goal_nuts_locations[obj_name] = loc_name

        # Find the man's name (assuming exactly one man object)
        self.man_name = None
        # Identify objects that are initially at a location
        initial_locatables_at_loc = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "at", "*", "*")}
        # Identify objects that are initially usable (likely spanners)
        initial_usables = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "usable", "*")}
        # Identify objects that are initially loose (likely nuts)
        initial_looses = {get_parts(fact)[1] for fact in self.initial_state if match(fact, "loose", "*")}

        # The man is the locatable object that is not initially usable or loose
        for obj_name in initial_locatables_at_loc:
             if obj_name not in initial_usables and obj_name not in initial_looses:
                  self.man_name = obj_name
                  break

        if not self.man_name:
             # Fallback or error if man not found based on this logic
             # Given the domain structure and examples, this should find the man.
             raise ValueError("Could not identify the man in the initial state.")


    def __call__(self, node):
        """
        Computes the heuristic estimate for the given state.
        """
        state = node.state

        # 1. Identify man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man must always be somewhere if state is valid
             return float('inf') # Should not happen in solvable problems

        # 2. Check if man is carrying a usable spanner
        carrying_usable_spanner = False
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner = get_parts(fact)[2]
                break
        if carried_spanner:
             if f"(usable {carried_spanner})" in state:
                 carrying_usable_spanner = True

        # 3. Identify loose nuts needing tightening and their locations
        nuts_to_tighten = []
        nut_locations_needed = {} # Map nut name to location
        nut_locs_set = set() # Set of unique locations of nuts needing tightening

        for nut_name, loc_name in self.goal_nuts_locations.items():
            # A nut needs tightening if it's a goal AND it's currently loose (i.e., not tightened)
            if f"(tightened {nut_name})" not in state:
                 nuts_to_tighten.append(nut_name)
                 nut_locations_needed[nut_name] = loc_name
                 nut_locs_set.add(loc_name)

        K = len(nuts_to_tighten)

        # If all goal nuts are tightened, heuristic is 0
        if K == 0:
            return 0

        # 4. Identify usable spanners on the ground and their locations
        spanner_locs_on_ground_set = set() # Set of unique locations of usable spanners on ground
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj_name = parts[1]
                loc_name = parts[2]
                # Check if this object is usable. Assume usable objects at locations are spanners.
                if f"(usable {obj_name})" in state:
                     spanner_locs_on_ground_set.add(loc_name)

        # 5. Calculate required minimum distances between relevant location sets
        # Use precomputed distances. Handle cases where a location might not be in distances
        # if the graph construction was incomplete, though self.locations should cover all linked nodes.
        # Handle cases where source/target sets for min distance are empty.

        # Minimum distance from man to any nut location needing tightening
        min_dist_man_to_nut = float('inf')
        if man_location in self.distances:
             for loc in nut_locs_set:
                  if loc in self.distances[man_location]:
                       min_dist_man_to_nut = min(min_dist_man_to_nut, self.distances[man_location][loc])

        # Minimum distance from man to any usable spanner location on the ground
        min_dist_man_to_spanner = float('inf')
        if man_location in self.distances and spanner_locs_on_ground_set:
             for loc in spanner_locs_on_ground_set:
                  if loc in self.distances[man_location]:
                       min_dist_man_to_spanner = min(min_dist_man_to_spanner, self.distances[man_location][loc])

        # Minimum distance from any nut location needing tightening to any usable spanner location on the ground
        min_dist_nut_to_spanner = float('inf')
        if nut_locs_set and spanner_locs_on_ground_set:
             for l_n in nut_locs_set:
                  if l_n in self.distances:
                       for l_s in spanner_locs_on_ground_set:
                            if l_s in self.distances[l_n]:
                                 min_dist_nut_to_spanner = min(min_dist_nut_to_spanner, self.distances[l_n][l_s])

        # Minimum distance from any usable spanner location on the ground to any nut location needing tightening
        min_dist_spanner_to_nut = float('inf')
        if spanner_locs_on_ground_set and nut_locs_set:
             for l_s in spanner_locs_on_ground_set:
                  if l_s in self.distances:
                       for l_n in nut_locs_set:
                            if l_n in self.distances[l_s]:
                                 min_dist_spanner_to_nut = min(min_dist_spanner_to_nut, self.distances[l_s][l_n])

        # 6. Calculate heuristic based on estimated costs for cycles

        # Cost for the first nut-tightening cycle
        # This involves getting the first spanner (if needed), traveling to the first nut, and tightening it.
        cost_first_nut_cycle = float('inf')
        if carrying_usable_spanner:
            # Man has a spanner, first step is travel to a nut and tighten
            # Minimum cost: Travel from man's current location to the nearest nut location + tighten action (1)
            if min_dist_man_to_nut != float('inf'):
                 cost_first_nut_cycle = min_dist_man_to_nut + 1
        else:
            # Man needs to get a spanner first, then go to a nut and tighten
            # Minimum cost: Travel man->nearest spanner + pickup (1) + travel spanner->nearest nut + tighten (1)
            # This is only possible if there are usable spanners on the ground
            if min_dist_man_to_spanner != float('inf') and min_dist_spanner_to_nut != float('inf'):
                 cost_first_nut_cycle = min_dist_man_to_spanner + 1 + min_dist_spanner_to_nut + 1

        # If the first cycle is impossible (e.g., no reachable spanners when needed),
        # the state is likely a dead end for K > 0. Return infinity to prune this branch.
        if cost_first_nut_cycle == float('inf'):
             return float('inf')


        # Cost for remaining K-1 nut-tightening cycles
        # Each remaining cycle requires travel from previous nut location to a spanner,
        # pickup, travel from spanner to next nut, tighten.
        # We use the minimum distances between *any* nut loc and *any* spanner loc on ground
        # as an estimate for the travel cost in subsequent cycles.
        cost_remaining_nuts_cycles = 0
        if K > 1:
            # This is only possible if there are usable spanners on the ground to pick up
            if min_dist_nut_to_spanner != float('inf') and min_dist_spanner_to_nut != float('inf'):
                 cost_per_remaining_nut_cycle = min_dist_nut_to_spanner + 1 + min_dist_spanner_to_nut + 1
                 cost_remaining_nuts_cycles = (K - 1) * cost_per_remaining_nut_cycle
            else:
                 # If no path exists between *any* nut loc and *any* spanner loc on ground,
                 # subsequent nuts cannot be tightened after the first (unless man started with K spanners,
                 # which is unlikely with K>1 and single spanner assumption).
                 # This state is likely a dead end if K > 1 and no ground spanners.
                 return float('inf')


        # Total heuristic is sum of cost for the first cycle and the remaining cycles
        total_cost = cost_first_nut_cycle + cost_remaining_nuts_cycles

        return total_cost
