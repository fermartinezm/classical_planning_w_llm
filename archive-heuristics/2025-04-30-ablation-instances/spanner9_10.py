from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It counts the remaining `tighten_nut` actions, the necessary `pickup_spanner`
    actions, and estimates the travel cost for the man to reach the required
    spanners and nuts.

    # Assumptions
    - Each loose nut requires one `tighten_nut` action.
    - Each `tighten_nut` action consumes one usable spanner.
    - The man can only carry one spanner at a time.
    - The man needs to pick up a new usable spanner for each nut, unless he is
      already carrying one at the start of the sequence for that nut.
    - Links between locations are bidirectional for travel distance calculation.

    # Heuristic Initialization
    - Extracts all location names from the static `link` facts.
    - Builds a graph representing the locations and links.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners currently on the ground and their locations.
    4. Check if the man is currently carrying a usable spanner.
    5. Count the number of loose nuts (`N_loose`). If zero, the goal is reached, heuristic is 0.
    6. The heuristic starts with `N_loose` (for the `tighten_nut` actions).
    7. Determine how many additional spanners the man needs to pick up from the ground.
       This is `max(0, N_loose - 1)` if the man is carrying a usable spanner,
       or `N_loose` if he is not. Let this be `P`.
    8. Add `P` to the heuristic (for the `pickup_spanner` actions).
    9. Estimate the travel cost. The man needs to travel from his current location
       to visit `P` spanner locations (to pick them up) and `N_loose` nut locations
       (to tighten nuts).
    10. Select the `P` usable spanners on the ground that are closest to the man's
        current location. Get their locations.
    11. Get the locations of all loose nuts.
    12. Calculate the sum of shortest path distances from the man's current location
        to each of the selected spanner locations and each of the loose nut locations.
    13. Add half of this total distance (integer division) to the heuristic as an
        estimate of the total travel cost. This simplifies the complex TSP-like
        problem of finding the optimal tour.
    14. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing
        shortest path distances between locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Collect all locations and build the graph from link facts
        self.graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Assume links are bidirectional

        self.locations = list(locations)

        # 2. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search from a start node to find shortest distances
        to all reachable nodes in the location graph.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            u = queue.popleft()

            if u in self.graph: # Check if node has neighbors
                for v in self.graph[u]:
                    if v not in visited:
                        visited.add(v)
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # 1. Identify man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Assuming there is only one man object, find its location
                # A more robust way would be to identify the man object first
                # based on type, but domain analysis suggests one man usually named 'bob'.
                # Let's find the object of type 'man' if possible, otherwise assume 'bob'.
                # A simpler way for this domain: find the 'at' predicate where the first arg is not a spanner or nut.
                # Or, rely on the fact that the problem instance will define the man object.
                # Let's find the object whose location is being described.
                # We need to know which object is the man. This info is not in state/static, but in task.
                # However, the heuristic only receives task in __init__.
                # A common pattern is to find the man object in the initial state or assume a name like 'bob'.
                # Let's find the object that is 'at' a location and is not a spanner or nut mentioned in goals/initial state.
                # Or, simply assume the first 'at' fact for a locatable that isn't a spanner/nut is the man.
                # A better way: the task object passed to __init__ has operators. Operators have parameters with types.
                # The 'walk' action has a parameter '?m - man'. We can get the man object name from an operator instance.
                # Let's assume the man object is always the first parameter of type 'man' in any operator.
                # This is tricky without access to task.objects or task.types.
                # Let's assume for this domain, the man object is the one whose location we track
                # and is not a spanner or nut. We can identify spanners/nuts by checking predicates like 'usable', 'tightened', 'loose'.
                obj_name = get_parts(fact)[1]
                is_spanner = any(match(f, "usable", obj_name) for f in state) or any(match(f, "carrying", "*", obj_name) for f in state)
                is_nut = any(match(f, "tightened", obj_name) for f in state) or any(match(f, "loose", obj_name) for f in state)
                if not is_spanner and not is_nut:
                     man_location = loc
                     man_name = obj_name
                     break # Found the man and his location

        if man_location is None:
             # This should not happen in a valid state for this domain
             return float('inf') # Or handle error

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

        # 3. Identify usable spanners on the ground and their locations
        usable_spanners_on_ground = {} # {spanner_name: location}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner_name = get_parts(fact)[1]
                # Check if it's on the ground (not carried)
                is_carried = any(match(f, "carrying", man_name, spanner_name) for f in state)
                if not is_carried:
                     # Find the location of this spanner
                     for loc_fact in state:
                         if match(loc_fact, "at", spanner_name, "*"):
                             usable_spanners_on_ground[spanner_name] = get_parts(loc_fact)[2]
                             break

        # 4. Check if man is carrying a usable spanner
        carrying_usable_spanner = False
        for fact in state:
             if match(fact, "carrying", man_name, "*"):
                  carried_spanner = get_parts(fact)[2]
                  if any(match(f, "usable", carried_spanner) for f in state):
                       carrying_usable_spanner = True
                  break # Assuming man carries at most one spanner

        # 5. Count loose nuts
        n_loose = len(loose_nuts)

        # If no loose nuts, goal is reached
        if n_loose == 0:
            return 0

        # 6. Heuristic starts with tighten actions
        h = n_loose

        # 7. Determine spanners to pick up
        spanners_to_pickup_count = max(0, n_loose - (1 if carrying_usable_spanner else 0))

        # 8. Add pickup actions
        h += spanners_to_pickup_count

        # 9-13. Estimate travel cost
        required_locations = []

        # Add locations of loose nuts
        required_locations.extend(loose_nuts.values())

        # Add locations of spanners to pick up (closest ones)
        if spanners_to_pickup_count > 0:
            # Sort usable spanners on ground by distance from man
            spanners_by_dist = sorted(
                usable_spanners_on_ground.items(),
                key=lambda item: self.dist[man_location].get(item[1], float('inf'))
            )
            # Select locations of the closest 'spanners_to_pickup_count' spanners
            for _, loc in spanners_by_dist[:spanners_to_pickup_count]:
                 required_locations.append(loc)

        # Calculate total distance from man's current location to all required locations
        total_dist_from_man = sum(
            self.dist[man_location].get(loc, float('inf')) for loc in required_locations
        )

        # Add half of the total distance as travel cost estimate
        # Handle cases where a location might be unreachable (dist is inf)
        if total_dist_from_man == float('inf'):
             return float('inf') # Problem is likely unsolvable from this state

        h += total_dist_from_man // 2 # Integer division

        # 14. Return the total heuristic value
        return h

