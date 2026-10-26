from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

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
    # Ensure the number of parts matches the number of arguments in the pattern
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all loose goal nuts.
    Uses an H_add-like approach: sums the minimum cost to achieve each
    '(tightened nut)' goal fact independently.
    The cost for a single nut is the minimum of:
    1. Man is carrying a usable spanner: Walk to nut location + tighten.
    2. Man is not carrying a usable spanner: Walk to nearest usable spanner,
       pickup, walk to nut location + tighten.

    Shortest paths between locations are precomputed using BFS.
    Assumes action costs are 1.
    Assumes there is exactly one man object in the domain.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and precomputing distances."""
        self.goals = task.goals
        self.static_facts = task.static

        # Identify goal nuts: any nut that needs to be tightened in the goal state.
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Build location graph from static link facts. The graph is directed.
        self.location_graph = {}
        self.all_locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Precompute shortest path distances from all locations.
        # This is needed for distance(SpannerLocation, NutLocation) in Option B.
        # Running BFS from every location might be slow for very large graphs,
        # but is acceptable for typical planning benchmarks and this domain structure.
        self.all_pairs_distances = {}
        # Convert set to list to iterate over a fixed sequence
        for start_loc in list(self.all_locations):
             self.all_pairs_distances[start_loc] = self._bfs(start_loc)

        # Identify the man's name. We assume there is exactly one man object.
        # A robust way would parse object types, but that info isn't in the Task object.
        # We'll infer it from initial state facts based on domain structure.
        self.man_name = None
        # The man is the object that can carry spanners. Look for 'carrying' fact.
        for fact in task.initial_state:
             if match(fact, "carrying", "*", "*"):
                  self.man_name = get_parts(fact)[1]
                  break
        # If not carrying anything initially, look for the object in an 'at' fact
        # that is likely the man (e.g., not a spanner or nut based on naming, or just 'bob').
        # Based on example instances, 'bob' is the man.
        if self.man_name is None:
             # This is a fallback based on example instance names.
             # A more general approach might look for the single object of type 'man'
             # if type information were available in the Task object.
             self.man_name = 'bob'


    def _bfs(self, start_loc):
        """Compute shortest path distances from start_loc to all reachable locations using BFS."""
        distances = {loc: float('inf') for loc in self.all_locations}

        # If the start location isn't in the graph (e.g., an isolated location
        # mentioned in an 'at' fact but not in any 'link' fact), return all infinities.
        if start_loc not in self.all_locations:
             return distances

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            u = queue.popleft()
            # Check if u has outgoing links in the graph
            if u in self.location_graph:
                for v in self.location_graph[u]:
                    if distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts).

        # 1. Extract relevant state information
        man_location = None
        carried_usable_spanners = set()
        usable_spanners_at_locs = {} # {spanner: location}
        loose_goal_nuts_at_locs = {} # {nut: location}

        # Collect all 'at' facts first for quick lookup of object locations
        at_facts = {}
        for fact in state:
             if match(fact, "at", "*", "*"):
                  obj, loc = get_parts(fact)[1:]
                  at_facts[obj] = loc

        # Now process other facts using collected locations
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                 # Man's location is already in at_facts, just retrieve it
                 man_location = at_facts.get(self.man_name)
            elif match(fact, "carrying", self.man_name, "*"):
                 spanner = get_parts(fact)[2]
                 # Check if the carried spanner is usable
                 if f"(usable {spanner})" in state:
                      carried_usable_spanners.add(spanner)
            elif match(fact, "usable", "*"):
                 spanner = get_parts(fact)[1]
                 # Check if this usable spanner is at a location (i.e., not carried)
                 # If it's usable and not carried, it must be at a location according to the domain
                 if spanner in at_facts:
                      usable_spanners_at_locs[spanner] = at_facts[spanner]
            elif match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                # Check if this loose nut is one of the goal nuts
                if nut in self.goal_nuts:
                    # Find location of this loose goal nut using collected at_facts
                    if nut in at_facts:
                         loose_goal_nuts_at_locs[nut] = at_facts[nut]
                    # else: A goal nut is loose but has no location? (Should not happen in valid instances)


        # If man_location is not found, the state is likely invalid or terminal in an unexpected way.
        # Return infinity as the goal is unreachable from such a state.
        if man_location is None:
             return float('inf')

        # 2. Check if goal is reached: all goal nuts are no longer loose.
        if not loose_goal_nuts_at_locs:
            return 0 # All goal nuts are tightened

        # 3. Check spanner availability: need at least one usable spanner per loose goal nut.
        total_usable_spanners = len(carried_usable_spanners) + len(usable_spanners_at_locs)
        if total_usable_spanners < len(loose_goal_nuts_at_locs):
             # Not enough usable spanners exist in the state to tighten all remaining nuts.
             return float('inf') # Unreachable

        # 4. Get precomputed distances from man's current location
        # This might be None if man_location is an isolated node not in the graph
        dist_M = self.all_pairs_distances.get(man_location, {})

        total_heuristic = 0

        # 5. Calculate cost for each loose goal nut independently (H_add style)
        # Summing these costs provides an overestimate, suitable for greedy search.
        for nut, nut_location in loose_goal_nuts_at_locs.items():
            min_cost_for_nut = float('inf')

            # Option A: Use a carried spanner
            # This option is possible if the man is currently carrying *any* usable spanner.
            if carried_usable_spanners:
                # Cost is walk from man's current location to nut location + tighten action (cost 1).
                # Check if the nut location is reachable from the man's location.
                if nut_location in dist_M and dist_M[nut_location] != float('inf'):
                     min_cost_for_nut = min(min_cost_for_nut, dist_M[nut_location] + 1)

            # Option B: Pick up a spanner from a location
            # This option is possible if there are usable spanners available at locations.
            if usable_spanners_at_locs:
                min_pickup_and_travel = float('inf')
                # Iterate through all usable spanners currently at locations
                for spanner, spanner_location in usable_spanners_at_locs.items():
                    # Cost: walk from man's location to spanner location + pickup action (cost 1)
                    # + walk from spanner location to nut location + tighten action (cost 1).
                    # Check if spanner location is reachable from man's location.
                    if spanner_location in dist_M and dist_M[spanner_location] != float('inf'):
                        # Need distance from spanner_location to nut_location.
                        # Use precomputed distances from spanner_location.
                        dist_S_to_N = self.all_pairs_distances.get(spanner_location, {}).get(nut_location, float('inf'))
                        # Check if nut location is reachable from spanner location.
                        if dist_S_to_N != float('inf'):
                            cost = dist_M[spanner_location] + 1 + dist_S_to_N + 1
                            min_pickup_and_travel = min(min_pickup_and_travel, cost)

                # Update min_cost_for_nut with the best cost from Option B
                min_cost_for_nut = min(min_cost_for_nut, min_pickup_and_travel)


            # If min_cost_for_nut is still infinity, it means this specific nut
            # is unreachable from the current state with any available spanner/path.
            # The overall goal is therefore unreachable.
            if min_cost_for_nut == float('inf'):
                return float('inf')

            # Add the minimum cost required for this nut to the total heuristic
            total_heuristic += min_cost_for_nut

        return total_heuristic
