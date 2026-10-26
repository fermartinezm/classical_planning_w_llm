from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Ensure the number of parts is at least the number of args for matching
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to compute shortest paths in an unweighted graph
def bfs_shortest_paths(graph, start_node):
    """
    Computes shortest path distances from a start_node to all other nodes
    in an unweighted graph using BFS.

    Args:
        graph: Adjacency list representation of the graph (dict: node -> list of neighbors).
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from start_node.
        Returns float('inf') for unreachable nodes.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # Start node is not in the graph, no paths possible
        return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node is a key in the graph before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

# All-pairs shortest paths
def compute_all_pairs_shortest_paths(graph):
    """
    Computes shortest path distances between all pairs of nodes in an unweighted graph.

    Args:
        graph: Adjacency list representation of the graph.

    Returns:
        A dictionary of dictionaries, where distances[u][v] is the shortest distance from u to v.
        Returns float('inf') for unreachable pairs.
    """
    all_distances = {}
    # Iterate through all nodes that are keys in the graph
    for start_node in graph:
        all_distances[start_node] = bfs_shortest_paths(graph, start_node)

    return all_distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts
    that are specified as goals. It sums the cost for each individual nut,
    considering the actions needed to get Bob to the nut's location with a spanner,
    plus the tightening action itself.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Bob can carry one or more spanners simultaneously. The heuristic only checks if Bob is carrying *at least one* usable spanner.
    - Spanners are usable if the 'usable' predicate holds for them (assumed static).
    - The 'link' predicate defines a bidirectional graph of locations.
    - The cost of each action (move, pick-up, tighten) is 1.
    - All locations and objects mentioned in the state/goal/static facts are valid.
    - If the goal is reachable, there is at least one usable spanner available somewhere.
    - Nuts remain at their initial locations throughout the plan.

    # Heuristic Initialization
    - Build a graph of locations based on 'link' facts and locations mentioned in initial state 'at' predicates.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify the names of all usable spanners based on static 'usable' facts.
    - Store the names of all nuts that need to be tightened according to the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify Bob's current location. If unknown or invalid, return infinity.
    2. Determine if Bob is currently carrying *any* spanner.
    3. Identify the current location of all usable spanners (either at a location or carried by Bob).
    4. Identify all nuts that are currently 'loose' and are specified as 'tightened' in the goal. These are the nuts that still need work. Also find their current locations. If a nut needing tightening has no known location, return infinity.
    5. If there are no nuts needing tightening, the heuristic is 0 (goal state).
    6. Initialize the total heuristic cost to 0.
    7. For each nut that needs tightening:
       a. Get the current location of this nut.
       b. Calculate the estimated cost to tighten *this specific nut*:
          - This cost includes getting Bob to the nut's location with a spanner, plus the tighten action (cost 1).
          - If Bob is already carrying *any* spanner:
            - The cost to get a spanner is 0. Bob has one at his current location.
            - The cost to reach the nut's location is the shortest distance from Bob's current location to the nut's location. If unreachable, return infinity.
            - Total cost for this nut = Distance(BobLoc, NutLoc) + 1 (Tighten).
          - If Bob is NOT carrying *any* spanner:
            - He needs to pick one up first. Find the usable spanner whose *current* location is closest to Bob's current location. Let this location be SpannerLoc.
            - If no usable spanners are currently available, return infinity.
            - The cost to get a spanner is Distance(BobLoc, SpannerLoc) + 1 (Pick-up). Bob is now effectively at SpannerLoc with a spanner. If SpannerLoc is unreachable, return infinity.
            - The cost to reach the nut's location is the shortest distance from SpannerLoc to the NutLoc. If unreachable, return infinity.
            - Total cost for this nut = Distance(BobLoc, SpannerLoc) + 1 (Pick-up) + Distance(SpannerLoc, NutLoc) + 1 (Tighten).
       c. Add the calculated cost for this nut to the total heuristic cost.
    8. Return the total heuristic cost.

    Note: This heuristic sums the costs independently for each nut, which might overestimate the true cost as actions (like moving or picking up a spanner) can contribute to satisfying the requirements for multiple nuts. However, it provides a reasonable estimate for greedy search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Location graph and distances.
        - Usable spanner names.
        - Goal nuts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Need initial state to find initial nut locations if goal doesn't specify

        # Build location graph from 'link' facts and locations mentioned in initial state
        self.location_graph = {}
        all_locations_set = set() # Collect all locations mentioned

        # Locations from links
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Links are bidirectional
                all_locations_set.add(loc1)
                all_locations_set.add(loc2)

        # Locations from initial state 'at' predicates
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                  _, obj, loc = get_parts(fact)
                  # Assume any object 'at' a location implies the location exists
                  # Add the location if it's not already in the graph keys
                  if loc not in self.location_graph:
                       self.location_graph[loc] = []
                  all_locations_set.add(loc)

        # Ensure all collected locations are keys in the graph, even if they have no links
        for loc in all_locations_set:
             self.location_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        self.distances = compute_all_pairs_shortest_paths(self.location_graph)

        # Identify usable spanner names (assuming 'usable' is static)
        self.usable_spanners = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "usable", "*")
        }

        # Store goal nuts (those that need to be tightened)
        self.goal_nuts = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "tightened", "*")
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Bob's current location.
        bob_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_location = get_parts(fact)[2]
                break
        # If Bob's location is unknown or not a valid location in our graph, it's an impossible state
        if bob_location is None or bob_location not in self.distances:
             return float('inf')

        # 2. Determine if Bob is currently carrying any spanner.
        bob_carrying_any_spanner = any(match(fact, "carrying", "bob", "*") for fact in state)

        # 3. Identify the current location of all usable spanners.
        current_usable_spanner_locations = {}
        for spanner in self.usable_spanners:
             # Check if spanner is at a location
             found_loc = False
             for fact in state:
                 if match(fact, "at", spanner, "*"):
                     current_usable_spanner_locations[spanner] = get_parts(fact)[2]
                     found_loc = True
                     break
             # If not at a location, check if Bob is carrying it
             if not found_loc:
                 # Check specifically if Bob is carrying this spanner
                 if match(f"(carrying bob {spanner})", "carrying", "bob", spanner):
                      current_usable_spanner_locations[spanner] = bob_location # Spanner is with Bob

        # 4. Identify nuts needing tightening and their current locations
        nuts_to_tighten = set()
        current_nut_locations = {}
        for nut in self.goal_nuts:
            is_loose = False
            nut_loc = None
            for fact in state:
                if match(fact, "loose", nut):
                    is_loose = True
                if match(fact, "at", nut, "*"):
                    nut_loc = get_parts(fact)[2]

            # If the nut is loose and its location is known, it needs tightening
            if is_loose and nut_loc is not None:
                 nuts_to_tighten.add(nut)
                 current_nut_locations[nut] = nut_loc
            elif nut in self.goal_nuts and nut_loc is None:
                 # A goal nut exists but its location is unknown - impossible state
                 return float('inf')


        # 5. If no nuts need tightening, it's a goal state (or sub-goal towards goal)
        if not nuts_to_tighten:
            return 0

        # 6. Initialize total cost
        total_cost = 0

        # 7. Calculate cost for each nut needing tightening
        for nut in nuts_to_tighten:
            nut_loc = current_nut_locations[nut] # Location is guaranteed to be known if added to nuts_to_tighten

            # Ensure nut_loc is in our distance graph
            if nut_loc not in self.distances:
                 return float('inf') # Cannot reach nut location

            # Cost includes the tighten action itself
            cost_for_this_nut = 1 # for the Tighten action

            # Cost to get Bob to NutLoc with a spanner
            if bob_carrying_any_spanner:
                # Bob has a spanner, just need to move to the nut's location
                move_cost = self.distances[bob_location].get(nut_loc, float('inf'))
                if move_cost == float('inf'): return float('inf') # Cannot reach nut location
                cost_for_this_nut += move_cost
            else:
                # Bob needs to get a spanner first
                min_spanner_acquisition_path_cost = float('inf')

                # Find the nearest usable spanner location
                if not current_usable_spanner_locations:
                     # No usable spanners available - impossible to tighten
                     return float('inf')

                for spanner, spanner_loc in current_usable_spanner_locations.items():
                    # Ensure spanner_loc is in our distance graph
                    if spanner_loc not in self.distances:
                         continue # Cannot use this spanner location

                    # Cost to reach spanner: Distance(BobLoc, SpannerLoc) + 1 (Pick-up)
                    dist_to_spanner = self.distances[bob_location].get(spanner_loc, float('inf'))
                    if dist_to_spanner == float('inf'): continue # Cannot reach this spanner

                    # Cost to move from SpannerLoc to NutLoc
                    dist_spanner_to_nut = self.distances[spanner_loc].get(nut_loc, float('inf'))
                    if dist_spanner_to_nut == float('inf'): continue # Cannot reach nut from spanner location

                    # Total path cost: move to spanner + pick up + move to nut
                    path_cost = dist_to_spanner + 1 + dist_spanner_to_nut

                    if path_cost < min_spanner_acquisition_path_cost:
                        min_spanner_acquisition_path_cost = path_cost

                if min_spanner_acquisition_path_cost == float('inf'):
                     # Cannot reach any usable spanner and then the nut
                     return float('inf')

                cost_for_this_nut += min_spanner_acquisition_path_cost

            total_cost += cost_for_this_nut

        # 8. Return total heuristic cost
        return total_cost
