from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

# Assume Heuristic base class is available as in the example
# from heuristics.heuristic_base import Heuristic

# Need to define the base class if it's not provided in the execution environment
# For self-contained code, let's include a dummy base class
# In a real scenario, this would be imported from the planner's framework
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

# Helper functions from Logistics example
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
    if len(parts) != len(args) and '*' not in args:
         # Basic check for arity mismatch unless using wildcards.
         # Note: This check might be too strict if patterns are meant to match prefixes.
         # The zip handles the shorter sequence comparison.
         pass # Let zip handle length differences

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It considers the number of tightening actions, the number of spanner pickups
    required, and the travel cost to reach the first relevant location (either
    a nut location or a spanner location if a pickup is needed).

    # Assumptions:
    - Each spanner can only be used once for tightening.
    - The man can only carry one spanner at a time.
    - The graph of locations connected by 'link' predicates is connected, or
      at least the relevant locations (man's start, spanner locations, nut locations)
      are reachable from each other.
    - There are enough usable spanners in the initial state for all goal nuts
      in solvable problems.
    - There is exactly one man object in the domain, and it is present in the
      initial state with an '(at ?m ?l)' fact.

    # Heuristic Initialization
    - Identify all goal nuts from the task goals.
    - Identify the man's name from the initial state by finding a locatable
      object that is not a spanner or nut mentioned in the initial state.
    - Build a graph of locations based on 'link' predicates found in static facts
      and locations mentioned in the initial state.
    - Compute all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the set of loose nuts that are also goal conditions. Let this count be `NumNutsToTighten`. If this count is 0, the goal is reached, and the heuristic is 0.
    2. Initialize the heuristic value `h` with `NumNutsToTighten` (each requires one 'tighten_nut' action).
    3. Determine the man's current location from the state. If the man's location cannot be found (which implies an issue with the state or man identification), return a large value indicating an invalid/unsolvable state.
    4. Check if the man is currently carrying a spanner, and if so, if that spanner is usable according to the state.
    5. Identify the locations of all usable spanners currently on the ground (not carried by the man) from the state.
    6. Identify the locations of all loose goal nuts from the state.
    7. Calculate the number of spanner pickups needed: This is the number of loose goal nuts (`NumNutsToTighten`) minus 1 if the man is already carrying a usable spanner. If this value is negative, it's 0. Let this be `PickupsNeeded`.
    8. Check if the total number of usable spanners available (carried usable spanner + usable spanners on the ground) is less than `NumNutsToTighten`. If so, the problem is likely unsolvable from this state with the current set of usable spanners, return a large value.
    9. Add `PickupsNeeded` to `h` (each pickup is one 'pickup_spanner' action).
    10. Calculate the walking cost: The man needs to move to locations where he can perform the necessary actions. These locations include the sites of the loose goal nuts (for tightening) and, if pickups are needed, the locations of usable spanners on the ground (for picking up).
        - Create a set of required locations by combining the locations of loose goal nuts and, if `PickupsNeeded > 0`, the locations of usable spanners on the ground.
        - Estimate the walking cost as the shortest distance from the man's current location to *any* location in the set of required locations. This provides a lower bound on the initial travel needed to reach a relevant part of the problem space.
        - Add this minimum distance to `h`. If no required locations are reachable (and there are required locations), return a large value.
    11. Return the total calculated heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal nuts, man's name,
           building the location graph, and computing shortest paths."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to find man

        # 1. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            # Goal is typically (tightened ?n)
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

        # 2. Identify man's name (assuming only one man and he is 'at' a location initially)
        self.man_name = None
        # Collect all locatable objects mentioned in initial state facts
        initial_locatable_objects = set()
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 initial_locatable_objects.add(get_parts(fact)[1])
             if match(fact, "carrying", "*", "*"):
                 initial_locatable_objects.add(get_parts(fact)[1]) # Man
                 initial_locatable_objects.add(get_parts(fact)[2]) # Spanner

        # Collect all spanners and nuts mentioned in initial state facts
        spanners_and_nuts_in_init = set()
        for fact in self.initial_state:
             if match(fact, "usable", "*"):
                 spanners_and_nuts_in_init.add(get_parts(fact)[1])
             if match(fact, "loose", "*"):
                 spanners_and_nuts_in_init.add(get_parts(fact)[1])
             if match(fact, "tightened", "*"):
                 spanners_and_nuts_in_init.add(get_parts(fact)[1])
             if match(fact, "carrying", "*", "*"):
                 spanners_and_nuts_in_init.add(get_parts(fact)[2]) # Spanner

        # The man is a locatable object that is not a spanner or nut
        for obj in initial_locatable_objects:
            if obj not in spanners_and_nuts_in_init:
                self.man_name = obj
                break

        # 3. Build location graph and find all locations
        self.locations = set()
        self.graph = {} # Adjacency list {loc: [neighbor1, neighbor2, ...]}

        # Locations from link predicates (static)
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph.setdefault(l1, []).append(l2)
                self.graph.setdefault(l2, []).append(l1) # Links are bidirectional

        # Locations from initial state (where objects are initially)
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 self.locations.add(loc)
                 self.graph.setdefault(loc, []) # Ensure all locations are keys in graph

        # Ensure all locations found are in the graph keys
        for loc in self.locations:
             self.graph.setdefault(loc, [])

        # 4. Compute all-pairs shortest paths
        self.distances = {} # {(loc1, loc2): distance}
        for start_node in self.locations:
            self._bfs(start_node)

    def _bfs(self, start_node):
        """Perform BFS from start_node to find distances to all reachable nodes."""
        q = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify loose nuts that are goals
        loose_goal_nuts = {n for n in self.goal_nuts if f"(loose {n})" in state}
        num_nuts_to_tighten = len(loose_goal_nuts)

        # If all goal nuts are tightened, heuristic is 0.
        if num_nuts_to_tighten == 0:
            return 0

        # Initialize heuristic with the cost of tighten actions
        h = num_nuts_to_tighten

        # 3. Determine man's current location
        man_loc = None
        if self.man_name: # Ensure man's name was found during init
            for fact in state:
                if match(fact, "at", self.man_name, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        # If man's location is unknown, state is likely invalid or unhandled
        if man_loc is None:
             return 1000000 # Represents an unreachable or invalid state

        # 4. Check if man is carrying a usable spanner
        man_carrying_usable = False
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner = get_parts(fact)[2]
                if f"(usable {carried_spanner})" in state:
                    man_carrying_usable = True
                break # Assuming man carries at most one spanner

        # 5. Identify locations of usable spanners on the ground
        usable_spanner_locs = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "at", "*", "*") and f"(usable {get_parts(fact)[1]})" in state
        }

        # 6. Identify locations of loose goal nuts
        nut_locs = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "at", "*", "*") and get_parts(fact)[1] in loose_goal_nuts
        }

        # 7. Calculate spanner pickups needed
        num_spanners_needed = num_nuts_to_tighten
        num_spanners_available = (1 if man_carrying_usable else 0) + len(usable_spanner_locs)

        # 8. If not enough usable spanners exist (carried or on ground) for the remaining nuts
        if num_spanners_needed > num_spanners_available:
             return 1000000 # Unsolvable estimate

        pickups_needed = max(0, num_spanners_needed - (1 if man_carrying_usable else 0))
        h += pickups_needed # Add cost for pickup actions

        # 10. Calculate walking cost
        # The man needs to reach locations where actions can be performed.
        # These are nut locations (for tightening) and spanner locations (for picking up, if needed).
        required_locs = set(nut_locs)
        if pickups_needed > 0:
            required_locs.update(usable_spanner_locs)

        # Estimate walking cost as the minimum distance from the man's current location
        # to any of the required locations. This is a lower bound on the initial travel needed
        # to reach a relevant part of the problem space.
        min_dist_to_required = math.inf
        if required_locs:
            for loc in required_locs:
                # Use .get() with a default of infinity in case a required location is not in self.locations
                # or unreachable from man_loc (though BFS should handle reachability within the graph)
                dist = self.distances.get((man_loc, loc), math.inf)
                min_dist_to_required = min(min_dist_to_required, dist)

        if min_dist_to_required != math.inf:
            h += min_dist_to_required
        elif required_locs:
             # Required locations exist but are unreachable from man's location within the graph
             return 1000000 # Unsolvable estimate

        # 11. Return the total estimated cost
        return h
