# Assume Heuristic base class exists with __init__(self, task) and __call__(self, node)
# from heuristics.heuristic_base import Heuristic # Commented out as base class is not provided

from fnmatch import fnmatch
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
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the number of loose nuts, the travel cost for the man to reach
    a nut location, and the cost to acquire a usable spanner if needed. It accounts
    for the constraint that the man can only carry one spanner and needs a usable one
    for each nut.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A spanner becomes unusable after tightening one nut.
    - The man cannot pick up a spanner if he is already carrying one.
    - The 'link' predicate represents bidirectional connections between locations.
    - Solvable instances guarantee that if loose nuts exist, the man is either
      carrying a usable spanner or is carrying no spanner and usable spanners
      are available on the ground and reachable. States where the man is carrying
      a non-usable spanner and loose nuts exist are considered unsolvable from that point.
    - The man object is identifiable (assumed to be named 'bob' if not found carrying a spanner).

    # Heuristic Initialization
    - Extracts all location objects mentioned in 'link' facts.
    - Builds a graph of locations based on 'link' predicates.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners that are currently on the ground and their locations.
    4. Check if the man is currently carrying *any* spanner.
    5. Check if the man is currently carrying a *usable* spanner.
    6. Count the total number of loose nuts (`N_loose`).
    7. If `N_loose` is 0, the state is a goal state, and the heuristic is 0.
    8. If `N_loose > 0`:
       a. Check for unsolvable states based on spanner constraints:
          - If the man is carrying *any* spanner, but it is *not* usable, and there are loose nuts, return infinity (unsolvable from here).
       b. The base heuristic value is `N_loose` (representing the minimum number of tighten actions).
       c. Add the shortest distance from the man's current location to the location of the *nearest* loose nut. This estimates the travel cost to start working on the first nut.
       d. Determine the cost to acquire a usable spanner if the man doesn't currently possess one:
          - If the man is carrying a usable spanner, this cost is 0.
          - If the man is carrying *no* spanner, find the minimum distance from the man's current location to any location containing a usable spanner on the ground. Add this minimum distance plus 1 (for the pickup action) to the heuristic. If no usable spanners are on the ground or reachable, return infinity (unsolvable).
       e. Sum the base cost, the minimum travel cost to a nut, and the spanner acquisition cost.

    This heuristic is non-admissible. It provides an estimate based on the number of remaining tasks and the immediate costs to start the next task (reaching a nut and getting the first required spanner).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and computing
        shortest paths between locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract all location objects from link facts
        self.locations = set()
        for fact in static_facts:
             if match(fact, "link", "*", "*"):
                 _, loc1, loc2 = get_parts(fact)
                 self.locations.add(loc1)
                 self.locations.add(loc2)

        # 2. Build the location graph
        self.location_graph = {loc: set() for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Assuming links are bidirectional

        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS to find shortest paths from start_node to all other nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node in self.locations: # Ensure start_node is part of the graph
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                if current_node in self.location_graph: # Ensure node exists in graph keys
                    for neighbor in self.location_graph[current_node]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # One or both locations are not in the graph built from links.
             # If they are the same location, distance is 0.
             if loc1 == loc2:
                 return 0
             # Otherwise, they are unreachable from each other within the linked graph.
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # Find the man object name and carried spanner in the current state
        man_obj = None
        carried_spanner = None
        man_is_carrying = False
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                man_obj = get_parts(fact)[1]
                carried_spanner = get_parts(fact)[2]
                man_is_carrying = True
                break
        # If man is not carrying anything, assume the name is 'bob' based on examples
        if man_obj is None:
             man_obj = 'bob' # Assuming 'bob' is the man object name

        # 1. Identify man's location
        man_location = None
        for fact in state:
            if match(fact, "at", man_obj, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None:
             # Man's location must be known in a valid state.
             # print(f"Warning: Man location not found for {man_obj} in state: {state}")
             return float('inf') # Should not happen in solvable states

        # 2. Identify loose nuts and their locations
        loose_nuts = {} # {nut_name: location}
        for fact in state:
            if match(fact, "loose", "*"):
                nut_name = get_parts(fact)[1]
                # Find the location of this nut
                for loc_fact in state:
                    if match(loc_fact, "at", nut_name, "*"):
                        loose_nuts[nut_name] = get_parts(loc_fact)[2]
                        break

        # 3. Identify usable spanners on the ground
        usable_spanners_on_ground = {} # {spanner_name: location}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner_name = get_parts(fact)[1]
                # Check if this spanner is on the ground (not carried)
                # Check if the spanner is the one being carried by the man
                if carried_spanner != spanner_name:
                    # Find the location of this spanner
                    for loc_fact in state:
                        if match(loc_fact, "at", spanner_name, "*"):
                            usable_spanners_on_ground[spanner_name] = get_parts(loc_fact)[2]
                            break

        # 4. Check if man is carrying a usable spanner
        man_carrying_usable_spanner = False
        if carried_spanner:
             for fact in state:
                 if match(fact, "usable", carried_spanner):
                     man_carrying_usable_spanner = True
                     break

        # 5. Count loose nuts
        num_loose_nuts = len(loose_nuts)

        # 6. Goal state check
        if num_loose_nuts == 0:
            return 0

        # 7. Check for unsolvable states based on spanner constraints
        # If man is carrying a spanner, but it's not usable, and there are loose nuts,
        # he cannot pick up a new spanner and is stuck.
        if man_is_carrying and not man_carrying_usable_spanner:
             # print(f"Warning: Man {man_obj} carrying non-usable spanner {carried_spanner} with {num_loose_nuts} nuts loose. State considered unsolvable.")
             return float('inf') # Unsolvable from this state

        # 8. Compute heuristic components
        h = num_loose_nuts # Base cost for tighten actions

        # a. Cost to reach the nearest loose nut location
        min_dist_to_nut = float('inf')
        nut_locations = list(loose_nuts.values()) # Get list of locations
        if not nut_locations: # Should not happen if num_loose_nuts > 0, but safety check
             min_dist_to_nut = 0 # This branch is unreachable if num_loose_nuts > 0
        else:
            for nut_loc in nut_locations:
                 dist = self.get_distance(man_location, nut_loc)
                 min_dist_to_nut = min(min_dist_to_nut, dist)

        if min_dist_to_nut == float('inf'):
             # Nearest nut location is unreachable. Problem unsolvable.
             # print(f"Warning: Nearest nut location unreachable from {man_location}. State considered unsolvable.")
             return float('inf')

        h += min_dist_to_nut

        # b. Cost to acquire a usable spanner if needed for the first nut
        cost_get_spanner = 0
        if not man_carrying_usable_spanner:
            # Man is not carrying a usable spanner. He needs to pick one up.
            # This is only possible if he is carrying *no* spanner (handled by unsolvable check above).
            min_pickup_cost = float('inf')
            if not usable_spanners_on_ground:
                 # No usable spanners on the ground and man isn't carrying one.
                 # Problem unsolvable if num_loose_nuts > 0.
                 # print(f"Warning: No usable spanners available on ground for {num_loose_nuts} loose nuts. State considered unsolvable.")
                 return float('inf') # Should not happen in solvable instances

            for spanner_loc in usable_spanners_on_ground.values():
                 dist = self.get_distance(man_location, spanner_loc)
                 if dist != float('inf'):
                     min_pickup_cost = min(min_pickup_cost, dist + 1) # +1 for pickup action

            if min_pickup_cost == float('inf'):
                 # All usable spanners on ground are unreachable. Problem unsolvable.
                 # print(f"Warning: Usable spanners on ground unreachable from {man_location}. State considered unsolvable.")
                 return float('inf')

            cost_get_spanner = min_pickup_cost

        h += cost_get_spanner

        return h
