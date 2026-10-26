from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for representing infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all loose nuts. It sums the estimated cost for each loose nut independently. The estimated cost for a single loose nut includes the cost to get the man to the nut's location carrying a usable spanner, plus the cost of the tighten action.

    # Assumptions
    - There is only one man.
    - Spanners are single-use (become unusable after one tighten action).
    - The graph of locations defined by `link` predicates is connected (or relevant parts are connected).
    - Problems are solvable (enough usable spanners exist or can be acquired).
    - The man object can be identified (e.g., by being involved in a 'carrying' fact or named 'bob').

    # Heuristic Initialization
    - Build a graph of locations based on `link` facts.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify the man object from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts and their current locations from the state. If no nuts are loose, the heuristic is 0.
    2. Identify the man's current location from the state.
    3. Determine if the man is currently carrying *any* usable spanner based on the state.
    4. Identify all usable spanners currently on the ground and their locations from the state.
    5. For each loose nut `N` at location `L_N`:
       a. Calculate the minimum cost to get the man from his current location (`L_M`) to `L_N` while ensuring he has a usable spanner *when he arrives at `L_N`*.
          - If the man is currently carrying *a* usable spanner: The cost is the shortest distance from `L_M` to `L_N`. (This is a relaxation; it assumes the carried spanner is available for this nut, even if it's the only one and needed elsewhere).
          - If the man is NOT currently carrying *any* usable spanner: The man must first go pick up a usable spanner from the ground. Find the minimum cost path from `L_M` to any location `L_S` with a usable spanner on the ground, perform the pickup action (cost 1), and then travel from `L_S` to `L_N`. The minimum cost for this sequence is `min(dist(L_M, L_S) + 1 + dist(L_S, L_N))` over all locations `L_S` with a usable spanner on the ground. If no usable spanners are on the ground, the problem is likely unsolvable, and this cost is infinite.
       b. Add 1 to the cost calculated in step 5a for the `tighten_nut` action itself.
    6. The total heuristic value is the sum of the costs calculated for each loose nut in step 5. This sum-of-independent-costs approach overestimates the true cost by summing travel and pickup costs for each nut independently, but provides a non-negative and goal-aware estimate suitable for greedy search. If any required location is unreachable or a spanner is needed but none are available, the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and computing distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Build location graph from link facts
        self.location_graph = {}
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = self._bfs(start_loc)

        # Identify the man object (assuming it's the one involved in 'carrying' or named 'bob')
        self.man_object = None
        # Check initial state for carrying or at bob
        for fact in initial_state:
             if match(fact, "carrying", "*", "*"):
                 self.man_object = get_parts(fact)[1]
                 break
             if match(fact, "at", "bob", "*"):
                  self.man_object = 'bob'
                  break

        # Fallback: If not found, assume 'bob' as a common convention
        if self.man_object is None:
             self.man_object = 'bob'


    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: math.inf for node in self.all_locations}
        # If start_node is not in the graph, no paths are possible from it
        if start_node not in self.all_locations:
             return distances

        distances[start_node] = 0
        queue = [start_node]

        while queue:
            current_node = queue.pop(0)

            # If current_node is not in graph (shouldn't happen if start_node was in all_locations), skip
            if current_node not in self.location_graph:
                 continue

            for neighbor in self.location_graph[current_node]:
                if distances[neighbor] == math.inf: # Not visited
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Data structures to hold relevant state information
        loose_nuts_locs = {}
        locatable_locations = {} # Map object to its location
        usable_spanners = set() # Set of usable spanner objects
        carried_spanners = set() # Set of spanners carried by the man

        # Parse the current state to populate data structures
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1:]
                locatable_locations[obj] = loc
            elif predicate == "loose":
                nut = parts[1]
                # We need the location of this nut, will get it from locatable_locations later
                loose_nuts_locs[nut] = None # Placeholder
            elif predicate == "usable":
                 spanner = parts[1]
                 usable_spanners.add(spanner)
            elif predicate == "carrying":
                 carrier, spanner = parts[1:]
                 if carrier == self.man_object:
                      carried_spanners.add(spanner)


        # Populate loose_nuts_locs with actual locations
        nuts_to_remove = []
        for nut in loose_nuts_locs:
             if nut in locatable_locations:
                  loose_nuts_locs[nut] = locatable_locations[nut]
             else:
                  # Loose nut exists but has no location? Cannot plan for this nut.
                  nuts_to_remove.append(nut)

        for nut in nuts_to_remove:
             del loose_nuts_locs[nut]


        # If no loose nuts, goal is reached
        if not loose_nuts_locs:
            return 0

        # Get man's location
        man_loc = locatable_locations.get(self.man_object)
        if man_loc is None or man_loc not in self.dist:
             # Man is not at a known location? Cannot compute distances.
             return math.inf

        # Determine if man is carrying *any* usable spanner
        man_carrying_usable = any(s in usable_spanners for s in carried_spanners)

        # Identify usable spanners on ground and their locations
        usable_spanners_ground_locs = {} # Map spanner object to its location
        for spanner in usable_spanners:
            # Check if this usable spanner is NOT carried by the man
            if spanner not in carried_spanners:
                # Find its location on the ground
                if spanner in locatable_locations:
                     spanner_loc = locatable_locations[spanner]
                     # Check if spanner_loc is a known location in the graph
                     if spanner_loc in self.dist:
                          usable_spanners_ground_locs[spanner] = spanner_loc
                     # else: usable spanner on ground at unknown location? Ignore it for planning.


        # Calculate heuristic sum for each loose nut
        total_cost = 0
        for nut, nut_loc in loose_nuts_locs.items():
            # Check if nut_loc is a known location in the graph
            if nut_loc not in self.dist:
                 # Loose nut at unknown location? Cannot compute distances.
                 return math.inf

            # Cost for this nut = Cost to get man to nut_loc carrying usable spanner + 1 (tighten)

            cost_to_reach_nut_with_spanner = math.inf

            if man_carrying_usable:
                # Man is already carrying a usable spanner, just need to walk to the nut
                # Check if nut_loc is reachable from man_loc
                if nut_loc in self.dist[man_loc] and self.dist[man_loc][nut_loc] != math.inf:
                     cost_to_reach_nut_with_spanner = self.dist[man_loc][nut_loc]
                # else: nut_loc is unreachable from man_loc

            else:
                # Man is not carrying a usable spanner, must pick one up first
                min_path_cost_via_spanner = math.inf
                if not usable_spanners_ground_locs:
                    # No usable spanners on the ground and man not carrying one -> unsolvable
                    return math.inf

                # Find the minimum cost path via any usable spanner on the ground
                for spanner_loc in usable_spanners_ground_locs.values():
                    # Check if spanner_loc is reachable from man_loc AND nut_loc is reachable from spanner_loc
                    if (spanner_loc in self.dist[man_loc] and self.dist[man_loc][spanner_loc] != math.inf and
                        nut_loc in self.dist[spanner_loc] and self.dist[spanner_loc][nut_loc] != math.inf):

                         # Path: man_loc -> spanner_loc (walk) + 1 (pickup) + spanner_loc -> nut_loc (walk)
                         path_cost = self.dist[man_loc][spanner_loc] + 1 + self.dist[spanner_loc][nut_loc]
                         min_path_cost_via_spanner = min(min_path_cost_via_spanner, path_cost)

                if min_path_cost_via_spanner == math.inf:
                     # Cannot find a path via any usable spanner on the ground to this loose nut
                     return math.inf

                cost_to_reach_nut_with_spanner = min_path_cost_via_spanner

            if cost_to_reach_nut_with_spanner == math.inf:
                 # If cost is still infinity, it means this nut is unreachable with a spanner
                 return math.inf

            # Add cost for this nut: travel/pickup + tighten action
            total_cost += cost_to_reach_nut_with_spanner + 1

        return total_cost
