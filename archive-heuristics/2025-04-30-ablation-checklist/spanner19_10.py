from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

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

def bfs(graph, start):
    """
    Performs Breadth-First Search to find shortest distances from a start node.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbor1, ...]}
        start: The starting node.

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from start.
    """
    distances = {start: 0}
    queue = deque([start])
    visited = {start}

    while queue:
        current = queue.popleft()
        distance = distances[current]

        if current in graph: # Handle nodes with no outgoing links
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the number of actions needed to tighten all loose nuts.
    It calculates the cost for the first nut based on the man's current
    location and available spanners, and adds an estimated average cost
    for each subsequent nut.

    Heuristic Components:
    - Cost to tighten each loose nut (1 action per nut).
    - Cost to pick up a spanner for each loose nut (1 action per nut).
    - Walk cost:
        - For the first nut: Walk from man's current location to a spanner,
          then from spanner to the nut location.
        - For subsequent nuts: Estimated average walk from a nut location
          to a spanner location, then from spanner to the next nut location.

    Precomputation:
    - Computes all-pairs shortest paths between locations using BFS.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying
        static locations and objects.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Identify all locations
        locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                locations.add(parts[1])
                locations.add(parts[2]) # Assuming links are bidirectional for walking

        # 2. Build location graph (adjacency list)
        self.location_graph = {loc: [] for loc in locations}
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.location_graph[l1].append(l2)
                self.location_graph[l2].append(l1) # Assume links are bidirectional

        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in locations:
            self.distances[start_loc] = bfs(self.location_graph, start_loc)

        # 4. Identify all nuts, spanners, and the man from initial state and goals
        self.all_nuts = set()
        self.all_spanners = set()

        # Infer nuts from loose/tightened predicates in initial state and goals
        for fact in self.initial_state | self.goals:
             parts = get_parts(fact)
             if parts[0] in ['loose', 'tightened'] and len(parts) > 1:
                 self.all_nuts.add(parts[1])

        # Infer spanners from usable/carrying predicates in initial state and goals
        for fact in self.initial_state | self.goals:
             parts = get_parts(fact)
             if parts[0] in ['usable', 'carrying'] and len(parts) > 1:
                 self.all_spanners.add(parts[1])

        # Infer the man from 'at' predicate in initial state, assuming he's the only locatable not a nut or spanner
        self.the_man = None
        for fact in self.initial_state:
            if match(fact, 'at', '*', '*'):
                obj, loc = get_parts(fact)[1:]
                if obj not in self.all_nuts and obj not in self.all_spanners:
                     self.the_man = obj
                     break # Assuming only one man

        # If man not found, heuristic cannot function. This is an error case.
        if not self.the_man:
             # In a real system, this might raise an error or log a critical warning.
             # For this exercise, we'll proceed, but man_location will be None,
             # likely leading to inf heuristic values or errors later.
             pass


    def get_distance(self, loc1, loc2):
        """Safely get distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # If locations are unknown or unreachable, return infinity.
             # This indicates an impossible path.
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Extract state information
        man_location = None
        man_carrying_usable_spanner = False
        usable_spanners_on_ground = {} # {spanner_obj: location}
        loose_nuts = {} # {nut_obj: location}

        carried_spanner_obj = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1:]
                if obj == self.the_man:
                     man_location = loc
                elif obj in self.all_spanners:
                    if '(usable ' + obj + ')' in state:
                         usable_spanners_on_ground[obj] = loc
                elif obj in self.all_nuts:
                    if '(loose ' + obj + ')' in state:
                        loose_nuts[obj] = loc

            elif parts[0] == 'carrying':
                 carrier, carried_obj = parts[1:]
                 if carrier == self.the_man and carried_obj in self.all_spanners:
                     carried_spanner_obj = carried_obj
                     if '(usable ' + carried_obj + ')' in state:
                         man_carrying_usable_spanner = True

        # If man_location could not be determined, the heuristic cannot proceed.
        if man_location is None:
             # This state is likely unreachable or malformed if the man exists but isn't 'at' a location.
             # Return infinity as a safe fallback.
             return float('inf')


        num_loose_nuts = len(loose_nuts)

        # Locations of all currently usable spanners (carried or on ground)
        usable_spanner_locations = set(usable_spanners_on_ground.values())
        if man_carrying_usable_spanner:
             usable_spanner_locations.add(man_location)

        usable_spanners_available_count = len(usable_spanners_on_ground) + (1 if man_carrying_usable_spanner else 0)


        # 2. Base cases
        if num_loose_nuts == 0:
            return 0 # Goal reached

        if usable_spanners_available_count < num_loose_nuts:
            # Not enough usable spanners to tighten all nuts
            return float('inf')

        # Locations of all loose nuts
        loose_nut_locations = set(loose_nuts.values())

        # Calculate min distance between any usable spanner location and any loose nut location
        # This is used for the estimated average cost of subsequent nuts
        min_dist_spanner_to_nut_pair = float('inf')
        if usable_spanner_locations and loose_nut_locations:
            for l_s in usable_spanner_locations:
                for l_n in loose_nut_locations:
                    dist = self.get_distance(l_s, l_n)
                    if dist != float('inf'): # Only consider reachable pairs
                         min_dist_spanner_to_nut_pair = min(min_dist_spanner_to_nut_pair, dist)

        # If min_dist_spanner_to_nut_pair is still inf here, it means no usable spanner can reach any loose nut.
        # This implies the problem is unsolvable from this state.
        if min_dist_spanner_to_nut_pair == float('inf') and num_loose_nuts > 0:
             return float('inf')


        # Calculate cost for the first nut
        cost_first_nut = float('inf')

        if man_carrying_usable_spanner:
            # Man is carrying a usable spanner, just needs to walk to the closest nut and tighten
            for l_n in loose_nut_locations:
                 dist = self.get_distance(man_location, l_n)
                 if dist != float('inf'): # Only consider reachable nuts
                     cost_first_nut = min(cost_first_nut, dist + 1) # +1 for tighten

        else:
            # Man needs to get a spanner first, then go to the nut
            if usable_spanner_locations: # Ensure there's a spanner to pick up
                for l_n in loose_nut_locations:
                    for l_s in usable_spanner_locations:
                        # Cost = walk M->S + pickup + walk S->N + tighten
                        dist_ms = self.get_distance(man_location, l_s)
                        dist_sn = self.get_distance(l_s, l_n)
                        if dist_ms != float('inf') and dist_sn != float('inf'): # Only consider reachable paths
                            cost = dist_ms + 1 + dist_sn + 1
                            cost_first_nut = min(cost_first_nut, cost)
            # If cost_first_nut is still inf here, it means man cannot reach any usable spanner
            # or no usable spanner can reach any loose nut. This is handled correctly.


        # If cost_first_nut is inf, the problem is unsolvable from this state.
        if cost_first_nut == float('inf'):
             return float('inf')


        # Calculate estimated average cost for remaining nuts
        cost_remaining_nut_avg = 0
        if num_loose_nuts > 1:
             # Each remaining nut requires getting a spanner (walk from nut loc to spanner loc + pickup)
             # and going to the next nut loc (walk spanner loc to nut loc) + tighten.
             # We approximate the walk costs using the minimum distance between any spanner/nut pair.
             # Walk N->S + Pickup + Walk S->N + Tighten
             # Note: min_dist_spanner_to_nut_pair is symmetric (min_dist_nut_to_spanner_pair)
             cost_remaining_nut_avg = min_dist_spanner_to_nut_pair + 1 + min_dist_spanner_to_nut_pair + 1


        # 4. Combine components
        total_heuristic = cost_first_nut
        if num_loose_nuts > 1:
            # If cost_remaining_nut_avg is inf, the total will be inf.
            total_heuristic += (num_loose_nuts - 1) * cost_remaining_nut_avg

        return total_heuristic
