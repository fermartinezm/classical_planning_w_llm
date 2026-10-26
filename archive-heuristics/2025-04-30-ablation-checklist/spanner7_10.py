from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in a 'heuristics' directory
# from heuristics.heuristic_base import Heuristic

# Mock Heuristic base class for standalone testing if needed
# In a real planning system, this would be provided.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

# Helper functions for parsing PDDL facts
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
    # Check if the number of parts matches the number of args, unless the pattern ends with a wildcard
    if len(parts) != len(args) and (not args or args[-1] != '*'):
         return False
    # Check if parts match args, allowing '*' wildcard
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for graph traversal
def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from start_node.

    graph: adjacency list {location: [neighbor1, neighbor2, ...]}
    start_node: the node to start the search from
    Returns: dictionary {location: distance}
    """
    # Ensure start_node is in the graph keys, even if isolated
    if start_node not in graph:
        graph[start_node] = [] # Add as isolated node

    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft() # Dequeue

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It sums the estimated costs for three main components:
    1. Tightening actions: One action per loose goal nut.
    2. Spanner pickup actions: One action per loose goal nut, adjusted if the man
       is already carrying a usable spanner.
    3. Walk actions: Estimated as the sum of shortest path distances from the
       man's current location to each unique location where a loose goal nut is situated.

    This heuristic is not admissible but aims to provide a good estimate for
    greedy best-first search by considering the number of tasks (nuts),
    required resources (spanners), and the travel distance to goal locations.

    # Heuristic Initialization
    - Identifies the set of nuts that are goals (need to be tightened).
    - Builds the location graph from 'link' facts for distance calculations.
    - Identifies the man object.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all usable spanners and their locations. Check if the man is carrying a usable spanner.
    3. Identify all nuts that are currently loose AND are goal nuts, along with their locations.
    4. If there are no loose goal nuts, the heuristic is 0.
    5. Calculate shortest path distances from the man's current location to all other locations using BFS on the location graph.
    6. Initialize heuristic value `h = 0`.
    7. Add the number of loose goal nuts to `h` (estimating the 'tighten_nut' actions).
    8. Estimate the number of 'pickup_spanner' actions needed: This is the number of loose goal nuts, minus one if the man is currently carrying a usable spanner (as he has one spanner ready for the first nut). Add this count (ensuring it's not negative) to `h`.
    9. Estimate the 'walk' actions: Calculate the sum of shortest path distances from the man's current location to each *unique* location where a loose goal nut is found. Add this sum to `h`. If any unique goal nut location is unreachable, the heuristic is infinity.
    10. Check if pickups are needed but no usable spanners exist anywhere. If so, return infinity.
    11. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts and building the location graph.
        """
        self.goals = task.goals
        self.static = task.static

        # Identify goal nuts from the goal conditions
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                self.goal_nuts.add(args[0])

        # Build the location graph (adjacency list) from link facts
        self.graph = {}
        self.all_locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Find the man object (assuming there's exactly one man)
        # We can infer the man object by looking for the object involved in 'carrying'
        # or the object at a location that isn't a spanner or nut.
        self.man_obj = None
        # Look for the object involved in a 'carrying' predicate in the initial state
        for fact in task.initial_state:
             if match(fact, "carrying", "*", "*"):
                 self.man_obj = get_parts(fact)[1]
                 break
        # If not carrying initially, try to find the object at a location that isn't a spanner/nut
        if self.man_obj is None:
             spanners_and_nuts = set()
             # Identify known spanners and nuts from initial state predicates
             for fact in task.initial_state:
                 if match(fact, "usable", "*"):
                      spanners_and_nuts.add(get_parts(fact)[1])
                 elif match(fact, "loose", "*"):
                      spanners_and_nuts.add(get_parts(fact)[1])
             # Find the object at a location that is not a known spanner or nut
             for fact in task.initial_state:
                  if match(fact, "at", "*", "*"):
                       obj = get_parts(fact)[1]
                       if obj not in spanners_and_nuts:
                            self.man_obj = obj
                            break

        # Ensure all locations mentioned in initial state are in the graph keys
        # This helps BFS not crash if a location is isolated.
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 loc = get_parts(fact)[2]
                 if loc not in self.graph:
                     self.graph[loc] = []
                 self.all_locations.add(loc)
        # Goal locations are where nuts are, which are already covered by initial state or links


    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state

        # 1. Identify man's current location
        man_location = None
        if self.man_obj:
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    man_location = get_parts(fact)[2]
                    break

        if man_location is None:
             # Man object not found or not located. This indicates a state issue.
             return float('inf')

        # 2. Identify usable spanner locations and if man is carrying one
        usable_spanner_locations = set()
        man_carrying_usable_spanner = False
        usable_spanners_in_state = set()

        for fact in state:
             if match(fact, "usable", "*"):
                 usable_spanners_in_state.add(get_parts(fact)[1])

        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in usable_spanners_in_state:
                    usable_spanner_locations.add(loc)
            elif self.man_obj and match(fact, "carrying", self.man_obj, "*"):
                 spanner_carried = get_parts(fact)[2]
                 if spanner_carried in usable_spanners_in_state:
                     man_carrying_usable_spanner = True


        # 3. Identify loose goal nuts and their locations
        loose_goal_nuts_info = [] # List of (nut_obj, location)
        for fact in state:
            if match(fact, "loose", "*"):
                nut_obj = get_parts(fact)[1]
                if nut_obj in self.goal_nuts:
                    # Find location of this loose goal nut
                    nut_location = None
                    for loc_fact in state:
                        if match(loc_fact, "at", nut_obj, "*"):
                            nut_location = get_parts(loc_fact)[2]
                            break
                    # If a loose goal nut doesn't have an 'at' predicate, it's a strange state.
                    # We'll ignore it or treat it as unreachable. Let's ignore for now.
                    if nut_location:
                        loose_goal_nuts_info.append((nut_obj, nut_location))


        # If all goal nuts are already tightened, heuristic is 0
        if not loose_goal_nuts_info:
            return 0

        # 4. Calculate distances from man's current location
        # Ensure man_location is in the graph keys before BFS
        # Add man_location to graph if it's an isolated location not in links
        if man_location not in self.graph:
             self.graph[man_location] = [] # Add as isolated node for BFS

        distances = bfs(self.graph, man_location)

        # 5. Compute heuristic value
        h = 0
        N = len(loose_goal_nuts_info)

        # Cost for tighten actions: 1 action per loose goal nut
        h += N

        # Cost for pickup actions: 1 action per spanner needed.
        # Man needs N spanners. If he starts carrying one usable, he needs N-1 pickups.
        pickups_needed = N
        if man_carrying_usable_spanner:
            pickups_needed -= 1
        h += max(0, pickups_needed) # Ensure cost is not negative

        # Check if necessary spanners are available *somewhere* if pickups are needed
        if pickups_needed > 0 and not usable_spanner_locations:
             # Need to pick up spanners, but no usable spanners exist anywhere
             return float('inf')

        # Cost for walk actions: Estimate as sum of distances to unique goal nut locations
        unique_nut_locations = {loc for nut, loc in loose_goal_nuts_info}
        for nut_loc in unique_nut_locations:
            d = distances.get(nut_loc, float('inf'))
            if d == float('inf'):
                # A goal nut location is unreachable from the man's current location
                return float('inf')
            h += d # Add walk cost to reach this unique location

        return h

