from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to calculate distances in the location graph
def bfs_distances(start_node, graph, all_nodes):
    """
    Performs BFS from a start_node to find distances to all reachable nodes.

    Args:
        start_node: The node to start BFS from.
        graph: Adjacency list representation of the graph {node: [neighbor1, neighbor2, ...]}
        all_nodes: A set of all nodes in the graph.

    Returns:
        A dictionary {node: distance} from start_node to node.
        Distance is float('inf') if unreachable.
    """
    distances = {node: float('inf') for node in all_nodes}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node exists in the graph keys (it should if from all_nodes)
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It sums the estimated costs for:
    1. Tightening each loose goal nut (1 action per nut).
    2. Picking up necessary usable spanners (1 action per spanner).
    3. Walking to the locations where spanners need to be picked up and where nuts need to be tightened.

    # Assumptions
    - There is only one man object in the problem instance.
    - The man object can be identified (inferred from 'carrying' predicate or assumed name 'bob').
    - Spanner objects can be identified (inferred from 'at' or 'carrying' facts, assumed name prefix 'spanner').
    - Links between locations are bidirectional.
    - The graph of locations is connected (or relevant parts are connected).
    - Enough usable spanners exist in the problem instance to tighten all goal nuts.
    - The man can carry multiple spanners simultaneously.
    - Spanners become unusable after tightening one nut but remain carried.

    # Heuristic Initialization
    - Identify all goal nuts from the task definition.
    - Identify the man object name and all spanner object names from the initial state facts.
    - Build the graph of locations based on `link` facts. Collect all locations mentioned in `link` and `at` facts in the initial state.
    - Compute all-pairs shortest path distances between all locations using BFS. Store these distances.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location. If not found, return infinity.
    2. Identify all goal nuts that are currently loose in the state. Let this count be `k`.
    3. If `k` is 0, the heuristic is 0 (goal state).
    4. Count the number of usable spanners the man is currently carrying. Let this be `carried_usable`.
    5. Identify all usable spanners currently on the ground (at a location and not carried) and their locations.
    6. Calculate the number of additional usable spanners needed from the ground: `needed_from_ground = max(0, k - carried_usable)`.
    7. If `needed_from_ground` is greater than the number of usable spanners available on the ground, the state is likely unsolvable, return infinity.
    8. Initialize heuristic value `h = 0`.
    9. Add the cost of tightening each loose goal nut: `h += k`.
    10. Add the cost of picking up the needed spanners: `h += needed_from_ground`.
    11. Identify the locations of the `needed_from_ground` usable spanners on the ground that are closest to the man's current location (using precomputed distances).
    12. Identify the locations of all loose goal nuts in the current state.
    13. Combine these into a set of unique "required locations" that the man must visit.
    14. Calculate the walk cost as the sum of the shortest path distances from the man's current location to each location in the set of required locations. If any required location is unreachable, the walk cost is infinity.
    15. Add the walk cost to `h`.
    16. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts, finding objects,
        building the location graph, and computing all-pairs shortest paths.
        """
        super().__init__(task)

        # 1. Identify all goal nuts
        self.goal_nuts = {
            args[0] for goal in task.goals if match(goal, "tightened", "*")
        }

        # Identify the man object name and all spanner object names
        self.man_name = None
        self.all_spanners = set()

        # Collect all objects mentioned in 'at' or 'carrying' facts in the initial state
        # and infer types based on predicate structure and naming conventions from examples.
        all_objects_in_init = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 obj = parts[1]
                 loc = parts[2]
                 all_objects_in_init.add(obj)
                 # Assume objects starting with 'spanner' are spanners
                 if obj.startswith("spanner"):
                     self.all_spanners.add(obj)
             elif parts[0] == "carrying" and len(parts) == 3:
                  man_obj = parts[1]
                  spanner_obj = parts[2]
                  all_objects_in_init.add(man_obj)
                  all_objects_in_init.add(spanner_obj)
                  # The first argument of 'carrying' is the man
                  self.man_name = man_obj
                  # The second argument of 'carrying' is a spanner
                  self.all_spanners.add(spanner_obj)

        # Fallback for man name if not found in 'carrying' facts initially
        # This is a limitation based on the input format lacking explicit types in state facts.
        if self.man_name is None:
             # Assume the single object that is not a spanner or nut (if identifiable) is the man.
             # Or, fall back to a common name from examples.
             # We don't have nut names easily here. Fallback to 'bob'.
             self.man_name = 'bob' # Fallback based on examples

        # 2. Collect all locations
        self.locations = set()
        # Locations from link facts
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
        # Locations from initial 'at' facts
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 self.locations.add(loc) # Add all locations mentioned in 'at' facts

        # 3. Build the location graph (adjacency list)
        self.graph = {loc: [] for loc in self.locations}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Assuming links are bidirectional
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # 4. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs_distances(start_loc, self.graph, self.locations)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                 man_loc = get_parts(fact)[2]
                 break
        if man_loc is None:
             # Man must always be at a location in a valid state
             return float('inf')

        # 2. Identify loose goal nuts in the current state
        loose_goal_nuts = set()
        # Need to find the current status of goal nuts
        current_tightened_nuts = {
             args[0] for fact in state if match(fact, "tightened", "*")
        }
        # A goal nut is loose if it's a goal nut AND it's not currently tightened
        loose_goal_nuts = self.goal_nuts - current_tightened_nuts

        k = len(loose_goal_nuts)

        # 3. If k is 0, goal is reached
        if k == 0:
            return 0

        # 4. Count usable spanners carried by the man
        carried_spanners = set()
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                spanner = get_parts(fact)[2]
                carried_spanners.add(spanner)

        carried_usable = 0
        for spanner in carried_spanners:
             if f"(usable {spanner})" in state:
                 carried_usable += 1

        # 5. Identify usable spanners on the ground and their locations
        available_usable_on_ground = [] # List of (spanner, location)

        for spanner in self.all_spanners: # Use the set collected in __init__
            # Check if it's usable in the current state
            if f"(usable {spanner})" in state:
                # Check if it's on the ground (at a location)
                spanner_loc = None
                is_on_ground = False
                for fact in state:
                    if match(fact, "at", spanner, "*"):
                        spanner_loc = get_parts(fact)[2]
                        is_on_ground = True
                        break

                # Check if it's carried by the man (if carried, it's not on the ground)
                is_carried = False
                for fact in state:
                    if match(fact, "carrying", self.man_name, spanner):
                        is_carried = True
                        break

                if is_on_ground and not is_carried:
                     available_usable_on_ground.append((spanner, spanner_loc))


        # 6. Calculate needed spanners from ground
        needed_from_ground = max(0, k - carried_usable)

        # 7. Check for unsolvability (not enough usable spanners in total)
        if needed_from_ground > len(available_usable_on_ground):
             # This state is likely unsolvable if we need more spanners than exist usable on the ground
             # and we don't have enough carried.
             return float('inf')

        # 8. Initialize heuristic value
        h = 0

        # 9. Add cost of tighten actions (1 action per loose goal nut)
        h += k

        # 10. Add cost of pickup actions (1 action per spanner picked up from ground)
        h += needed_from_ground

        # 11. Identify spanner pickup locations
        # Sort available usable spanners on ground by distance from man_loc
        # Use .get() with infinity fallback in case a location is not in self.distances (shouldn't happen if locations collected correctly)
        available_usable_on_ground.sort(key=lambda item: self.distances.get(man_loc, {}).get(item[1], float('inf')))

        spanner_pickup_locations = [loc for spanner, loc in available_usable_on_ground[:needed_from_ground]]

        # 12. Identify nut locations
        nut_locations = []
        # Need to find the location of each loose goal nut in the current state
        for nut in loose_goal_nuts:
            nut_loc = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_loc = get_parts(fact)[2]
                    break
            if nut_loc: # Should always find the location if the nut exists and is 'at' a location
                 nut_locations.append(nut_loc)
            else:
                 # Nut exists but is not 'at' a location? Invalid state.
                 return float('inf')


        # 13. Combine into required locations set (unique locations)
        required_locations = set(spanner_pickup_locations + nut_locations)

        # 14. Calculate walk cost
        walk_cost = 0
        for loc in required_locations:
            # Get distance from man_loc to the required location
            dist = self.distances.get(man_loc, {}).get(loc, float('inf'))
            if dist == float('inf'):
                 # A required location is unreachable from the man's current location
                 return float('inf')
            walk_cost += dist # Summing distances from start to each required location

        # 15. Add walk cost to h
        h += walk_cost

        # 16. Return h
        return h
