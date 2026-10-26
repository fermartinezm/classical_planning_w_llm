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

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for shortest path calculation using BFS
def shortest_path(graph, start, end):
    """Find the shortest path distance between two locations using BFS."""
    if start == end:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc == end:
            return dist
        # Ensure current_loc is a valid node in the graph before accessing neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # Return infinity if end is unreachable

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It sums the number of loose nuts, a cost for needing a spanner, and the travel
    cost for Bob to reach the closest loose nut.

    # Assumptions:
    - Bob can only carry one spanner at a time.
    - All loose nuts specified in the goal need to be tightened.
    - Spanners are usable and do not break (all spanners are usable).
    - The location graph defined by 'link' predicates is connected (or relevant parts are).
    - The primary travel cost is reaching the vicinity of the nuts.

    # Heuristic Initialization
    - Build the location graph based on 'link' predicates from static facts.
    - Identify all possible locations from initial state and static facts to build the graph nodes.
    - Store the set of nuts that are specified in the goal to be tightened.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of three main components:
    1.  **Tightening Actions:** The number of loose nuts that are part of the goal. Each requires one 'tighten' action.
    2.  **Spanner Acquisition:** If Bob is not currently carrying a spanner, he needs to pick one up. This adds a cost of 1 action (the 'pick' action). This cost is added only once if needed, regardless of the number of loose nuts, assuming he keeps the spanner.
    3.  **Travel Cost:** Bob needs to reach the location of at least one loose nut to begin tightening. The minimum travel cost for this is the shortest distance (number of 'move' actions) from Bob's current location to the nearest location containing a loose nut that needs tightening.

    The total heuristic value is the sum of these three components. If there are no loose nuts that are part of the goal, the heuristic is 0. If any required location (Bob's, or a loose nut's) is not in the graph or unreachable, the heuristic returns infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static # Store static facts if needed later (e.g., for 'usable')

        # Identify all locations from initial state and static facts
        all_locations = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                all_locations.add(parts[2])
        for fact in task.static:
             parts = get_parts(fact)
             if parts[0] == 'link':
                 all_locations.add(parts[1])
                 all_locations.add(parts[2])
             elif parts[0] == 'at': # Also check static 'at' facts
                 all_locations.add(parts[2])

        # Build the location graph from 'link' facts
        self.graph = {loc: set() for loc in all_locations}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations exist in the graph dictionary before adding links
                if loc1 in self.graph and loc2 in self.graph:
                    self.graph[loc1].add(loc2)
                    self.graph[loc2].add(loc1)


        # Store the set of nuts that need to be tightened (from goals)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Count loose nuts that are goals
        loose_goal_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}
        num_loose_goal_nuts = len(loose_goal_nuts)

        # If all goal nuts are tightened, heuristic is 0
        if num_loose_goal_nuts == 0:
            return 0

        total_cost = num_loose_goal_nuts # Base cost for tighten actions

        # Get Bob's current location
        bob_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_loc = get_parts(fact)[2]
                break
        if bob_loc is None or bob_loc not in self.graph:
             # Bob's location is unknown or not in the graph, problem likely unsolvable
             return float('inf')


        # 2. Check if Bob is carrying a spanner
        bob_carrying_spanner = any(match(fact, "carrying", "bob", "*") for fact in state)
        if not bob_carrying_spanner:
            total_cost += 1 # Cost to pick up a spanner


        # 3. Calculate minimum distance from Bob to any loose goal nut location
        loose_nut_locations_set = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in loose_goal_nuts and loc in self.graph: # Ensure location is in graph
                     loose_nut_locations_set.add(loc)

        min_dist_bob_to_nut_loc = float('inf')
        if loose_nut_locations_set: # Only calculate if there are loose nuts with locations in the graph
            for nut_loc in loose_nut_locations_set:
                 dist = shortest_path(self.graph, bob_loc, nut_loc)
                 min_dist_bob_to_nut_loc = min(min_dist_bob_to_nut_loc, dist)
        # If loose_nut_locations_set is empty but num_loose_goal_nuts > 0,
        # it means loose nuts exist but their locations are not in the graph,
        # which implies unsolvability. min_dist_bob_to_nut_loc remains inf.


        # Add the minimum travel cost
        if min_dist_bob_to_nut_loc == float('inf'):
             # Loose nuts exist but are unreachable from Bob's location
             return float('inf')
        total_cost += min_dist_bob_to_nut_loc


        return total_cost
