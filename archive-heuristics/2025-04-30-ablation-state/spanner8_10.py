from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError
        # Add a dummy task class for testing if needed
        # class DummyTask:
        #     def __init__(self, initial_state, goals, static):
        #         self.initial_state = frozenset(initial_state)
        #         self.goals = frozenset(goals)
        #         self.static = frozenset(static)
        #     def goal_reached(self, state):
        #          return self.goals <= state

        # Add a dummy node class for testing if needed
        # class DummyNode:
        #      def __init__(self, state):
        #           self.state = frozenset(state)


# Helper functions
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
    # Ensure parts has enough elements to match args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the total number of actions needed to tighten all
    required nuts. It sums the number of tighten actions, the number of spanner
    pickup actions, and an estimate of the walk actions required.

    # Assumptions:
    - The man can only carry one spanner at a time.
    - Links between locations are bidirectional.
    - The man's name is consistently used in facts (inferred or assumed).
    - All locations mentioned in initial state, goals, and links are relevant.

    # Heuristic Initialization
    - Extracts all location names from static facts (links) and initial/goal states.
    - Builds an adjacency list graph representing linked locations.
    - Computes all-pairs shortest path distances between all locations using BFS.
    - Identifies the set of nuts that need to be tightened based on goal conditions.
    - Infers the man's name (tries 'carrying' fact first, falls back to 'bob').

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all nuts that are currently 'loose' and are part of the goal. Store their locations.
    3. Count the total number of such loose nuts (`num_loose_nuts`).
    4. Count the number of usable spanners the man is currently 'carrying' (`num_carried_spanners`).
    5. Calculate the minimum number of 'pickup_spanner' actions needed: This is the number of loose nuts minus the number of spanners the man is already carrying (`pickups_needed = num_loose_nuts - num_carried_spanners`). Note: Since the man can only carry one, `num_carried_spanners` is 0 or 1.
    6. Estimate the 'walk' cost: This is estimated as the sum of shortest path distances from the man's current location to the location of *each* loose nut. This accounts for the travel needed to reach all target nut locations from the man's starting point in this state.
    7. The total heuristic value is the sum of:
       - `num_loose_nuts` (for the 'tighten_nut' actions).
       - `pickups_needed` (for the 'pickup_spanner' actions).
       - The estimated walk cost.
    8. Handle edge cases:
       - If `num_loose_nuts` is 0, the state is a goal state, heuristic is 0.
       - If the total number of usable spanners (carried + on ground) is less than `num_loose_nuts`, the problem is likely unsolvable from this state; return a large value.
       - If any required location is unreachable from the man's location, return a large value.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all locations and man's name

        # 1. Extract locations and build the graph from links
        self.locations = set()
        self.graph = {} # adjacency list

        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Assuming links are bidirectional

        # 2. Add locations from initial state and goals to ensure all relevant locations are included
        for fact in initial_state | self.goals:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 # Assuming any second argument to 'at' is a location for heuristic purposes
                 self.locations.add(loc)

        self.locations = list(self.locations) # Convert to list

        # Ensure all locations are keys in the graph dictionary, even if isolated
        for loc in self.locations:
             self.graph.setdefault(loc, [])


        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = {}
            # Initialize distances to infinity for all locations
            for loc in self.locations:
                 self.distances[start_node][loc] = float('inf')
            self.distances[start_node][start_node] = 0

            q = deque([(start_node, 0)])
            visited = {start_node}

            while q:
                current_node, dist = q.popleft()

                # If current_node is not in graph (isolated location), it has no neighbors
                for neighbor in self.graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

        # 4. Identify the set of nuts that need to be tightened (goal nuts)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # 5. Infer the man's name
        self.man_name = None
        # Try finding the object involved in a 'carrying' fact in the initial state
        for fact in initial_state:
             if match(fact, "carrying", "*", "*"):
                 self.man_name = get_parts(fact)[1]
                 break
        # Fallback assumption if not found via 'carrying'
        if self.man_name is None:
             # This is fragile and assumes 'bob' is the man's name if not carrying initially.
             # A robust solution requires parsing object types from PDDL.
             # We could also try finding the unique object at a location that isn't a spanner or nut type,
             # but we don't have type info here. Sticking with 'bob' as a common convention.
             self.man_name = 'bob' # Common name in examples


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                _, _, man_location = get_parts(fact)
                break

        # Should always find man's location in a valid state derived from a valid initial state
        if man_location is None:
             # This indicates an unexpected state structure
             return 1000000 # Treat as unsolvable

        # 2. Identify loose nuts that are goals and their locations
        loose_nuts = {} # {nut_name: location}
        current_usable_spanners = set() # Usable spanners currently existing (carried or on ground)

        for fact in state:
            # Check for loose nuts that are goal nuts
            if match(fact, "loose", "*"):
                nut_name = get_parts(fact)[1]
                if nut_name in self.goal_nuts:
                    # Find its location
                    # Assuming nut location is static and present in state if nut is loose
                    for loc_fact in state:
                        if match(loc_fact, "at", nut_name, "*"):
                            _, _, nut_location = get_parts(loc_fact)
                            loose_nuts[nut_name] = nut_location
                            break
            # Identify all usable spanners (on ground or carried)
            elif match(fact, "usable", "*"):
                 spanner_name = get_parts(fact)[1]
                 current_usable_spanners.add(spanner_name)
            elif match(fact, "carrying", self.man_name, "*"):
                 spanner_name = get_parts(fact)[2]
                 current_usable_spanners.add(spanner_name) # Carried spanners are usable until used

        num_loose_nuts = len(loose_nuts)

        # 8a. If no loose nuts that are goals, we are in a goal state
        if num_loose_nuts == 0:
            return 0

        # Count spanners carried by the man
        carried_spanners = {get_parts(fact)[2] for fact in state if match(fact, "carrying", self.man_name, "*")}
        num_carried_spanners = len(carried_spanners) # Should be 0 or 1 based on domain

        # 8b. Check if enough usable spanners exist in total
        num_usable_total = len(current_usable_spanners)
        if num_usable_total < num_loose_nuts:
             # Problem is likely unsolvable with remaining usable spanners
             return 1000000 # Large finite value

        # 5. Calculate number of pickups needed
        # Man needs one spanner per nut. If he carries one, he needs one less pickup.
        # Since he can only carry one, this is simply num_loose_nuts - num_carried_spanners.
        pickups_needed = num_loose_nuts - num_carried_spanners


        # 7. Heuristic calculation components:
        # - num_loose_nuts: cost for 'tighten_nut' actions (1 per nut)
        # - pickups_needed: cost for 'pickup_spanner' actions (1 per needed pickup)
        # - walk_cost: estimated cost for 'walk' actions

        # 6. Estimate walk cost: Sum of distances from man's current location to each loose nut location.
        walk_cost = 0
        for nut_name, nut_location in loose_nuts.items():
             # 8c. Ensure distance is computable (locations are in our graph/distances)
             if man_location in self.distances and nut_location in self.distances[man_location]:
                 dist = self.distances[man_location][nut_location]
                 if dist == float('inf'):
                      # Nut location is unreachable from man's location
                      return 1000000 # Unsolvable
                 walk_cost += dist
             else:
                  # Location not found in precomputed distances - indicates an issue
                  # This case should ideally be caught during initialization or problem parsing
                  return 1000000 # Treat as unsolvable

        # Total heuristic value
        total_cost = num_loose_nuts + pickups_needed + walk_cost

        return total_cost
