# Need to import Heuristic base class and deque for BFS
from fnmatch import fnmatch
from collections import deque
# Assuming the Heuristic base class is available at this path
# If not, replace with the actual import path or include the base class definition
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the base class is not found
    print("Warning: heuristics.heuristic_base not found. Using dummy Heuristic class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found.")


# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at obj loc)" -> ["at", "obj", "loc"]
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, considering wildcards
    # Use zip to handle prefix matching with wildcards
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    target nuts. It considers the cost of tightening each loose target nut,
    the cost of acquiring a spanner if Bob isn't carrying one, and the
    estimated travel cost to visit all necessary locations (spanner location
    if needed, and all loose target nut locations).

    # Assumptions
    - Bob can carry at most one spanner at a time.
    - Links between locations are bidirectional.
    - Any spanner Bob is carrying can be used for tightening.
    - Usable spanners exist and are reachable if needed in solvable problems.
    - The heuristic calculates travel cost using a greedy approach to visiting locations.

    # Heuristic Initialization
    - Extracts the set of nuts that need to be tightened from the goal conditions.
    - Builds a graph of locations based on static `link` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that are required to be tightened in the goal.
    2. Check the current state to find which of these goal nuts are still `loose`.
    3. If no goal nuts are loose, the state is a goal state, and the heuristic is 0.
    4. Initialize the heuristic value `h` with the number of loose goal nuts (each requires a `tighten` action).
    5. Find Bob's current location. If Bob's location is unknown, the state is unsolvable (return infinity).
    6. Determine if Bob is currently carrying a spanner.
    7. Identify the locations of all loose goal nuts. If any loose goal nut has no location, the state is unsolvable (return infinity).
    8. Create a set of locations Bob needs to visit, initially containing the locations of all loose goal nuts.
    9. If Bob is *not* carrying a spanner:
       - Find all usable spanners and their current locations.
       - Find the usable spanner location that is closest to Bob's current location using precomputed distances.
       - If a reachable usable spanner is found:
         - Add 1 to `h` for the `pickup` action.
         - Add the location of the closest usable spanner to the set of locations Bob needs to visit.
       - If no reachable usable spanner is found, the problem is likely unsolvable from this state; return infinity.
    10. Calculate the estimated move cost:
        - Start from Bob's current location (or the spanner location if he needed to pick one up first).
        - Use a greedy approach: repeatedly move to the closest unvisited location in the set of locations to visit until all are visited. Sum the distances of these moves.
        - If any required location is unreachable during the greedy traversal, return infinity.
    11. Add the estimated move cost to `h`.
    12. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts, building the location
        graph, and computing shortest paths.
        """
        super().__init__(task) # Call base class constructor if needed

        self.goals = task.goals
        # Identify nuts that must be tightened in the goal
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Collect all unique locations mentioned in initial state and static facts
        locations = set()
        # Combine initial state and static facts for location discovery
        all_relevant_facts = set(task.initial_state) | set(task.static)

        for fact in all_relevant_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                locations.add(parts[2]) # (at obj loc)
            elif parts and parts[0] == 'link' and len(parts) == 3:
                locations.add(parts[1]) # (link loc1 loc2)
                locations.add(parts[2]) # (link loc1 loc2)

        # Build adjacency list graph from link facts
        self.graph = {loc: [] for loc in locations}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Assuming links are bidirectional
                if loc1 in self.graph and loc2 in self.graph:
                    self.graph[loc1].append(loc2)
                    self.graph[loc2].append(loc1)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.graph:
            self.distances[start_loc] = self.bfs(start_loc, self.graph)

    def bfs(self, start_node, graph):
        """
        Performs Breadth-First Search to find shortest distances from start_node
        to all other nodes in the graph.
        Returns a dictionary mapping reachable nodes to their distance from start_node.
        Unreachable nodes are not included or have distance infinity.
        """
        distances = {node: float('inf') for node in graph}
        if start_node not in graph:
             # Start node is not in the graph of known locations, cannot reach anything
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Ensure node exists in graph keys before iterating neighbors
            if u in graph:
                for v in graph[u]:
                    if distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # Find Bob's current location
        bob_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_loc = get_parts(fact)[2]
                break
        # If Bob's location is not found, the state is likely invalid or unsolvable
        if bob_loc is None or bob_loc not in self.graph:
             return float('inf')

        # Check if Bob is carrying a spanner
        bob_carrying_spanner = False
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                bob_carrying_spanner = True
                break

        # Identify loose target nuts and their locations
        loose_target_nuts_locs = {} # {nut: loc}
        for nut in self.goal_nuts:
            # Check if the nut is loose in the current state
            if f"(loose {nut})" in state:
                # Find location of this nut
                nut_loc = None
                for fact in state:
                    if match(fact, "at", nut, "*"):
                        nut_loc = get_parts(fact)[2]
                        break
                # If a loose target nut is not found at any location, it's an invalid state?
                # Assuming nuts are always at locations if they exist.
                if nut_loc is not None and nut_loc in self.graph:
                     loose_target_nuts_locs[nut] = nut_loc
                else:
                     # Loose nut exists but has no known location, or location not in graph. Unsolvable.
                     return float('inf')


        # If no loose target nuts, the goal is reached for these nuts
        if len(loose_target_nuts_locs) == 0:
            return 0

        # Initialize heuristic cost
        h = 0

        # Cost for tightening each loose target nut
        h += len(loose_target_nuts_locs) # 1 action per nut

        # Set of locations Bob needs to visit (initially nut locations)
        locations_to_visit = set(loose_target_nuts_locs.values())

        # Current location for move cost calculation
        current_move_loc = bob_loc

        # If Bob is not carrying a spanner, he needs to get one
        if not bob_carrying_spanner:
            # Find usable spanners and their locations
            usable_spanner_locs = [] # List of locations
            for fact in state:
                if match(fact, "at", "*", "*"):
                    obj, loc = get_parts(fact)[1:]
                    # Check if the object at this location is a usable spanner
                    if f"(usable {obj})" in state:
                        usable_spanner_locs.append(loc)

            # Find the closest reachable usable spanner location from Bob's current location
            LocS_closest = None
            min_dist = float('inf')

            # Ensure Bob's location is a valid start node for distance lookup
            if bob_loc in self.distances:
                bob_distances = self.distances[bob_loc]
                for loc in usable_spanner_locs:
                    # Ensure spanner location is a valid target node for distance lookup
                    if loc in bob_distances:
                        dist = bob_distances[loc]
                        if dist < min_dist:
                            min_dist = dist
                            LocS_closest = loc

            # If a reachable usable spanner is found, add pickup cost and location to visit
            if LocS_closest is not None and min_dist != float('inf'):
                h += 1 # pickup action
                locations_to_visit.add(LocS_closest)
                # The greedy path will start from Bob's current location (bob_loc)
                # and potentially visit LocS_closest first if it's the closest
                # location in the set {LocS_closest} U {NutLocs}.
                # The greedy TSP calculation handles the starting point and visiting order.
            else:
                 # No reachable usable spanner, problem is likely unsolvable from here
                 return float('inf') # Return a large value indicating unsolvability

        # Calculate greedy TSP move cost to visit all required locations
        move_cost = 0
        # Use a list copy to allow modification during iteration
        temp_locations_to_visit = list(locations_to_visit)

        # The greedy path starts from Bob's current location (bob_loc)
        current_move_loc = bob_loc

        while temp_locations_to_visit:
            next_loc = None
            min_dist = float('inf')
            best_next_idx = -1

            # Find the closest location in the remaining set from the current move location
            if current_move_loc in self.distances: # Ensure current location is in the graph
                current_distances = self.distances[current_move_loc]
                for i, loc in enumerate(temp_locations_to_visit):
                    if loc in current_distances: # Ensure target location is in the graph
                        dist = current_distances[loc]
                        if dist < min_dist:
                            min_dist = dist
                            next_loc = loc
                            best_next_idx = i

            # If no reachable location is found in the remaining set, problem is unsolvable
            if next_loc is None or min_dist == float('inf'):
                 return float('inf') # Return a large value indicating unsolvability

            # Add distance to the move cost
            move_cost += min_dist
            # Move to the next location
            current_move_loc = next_loc
            # Remove the visited location
            temp_locations_to_visit.pop(best_next_idx)

        # Add the total estimated move cost to the heuristic
        h += move_cost

        return h
