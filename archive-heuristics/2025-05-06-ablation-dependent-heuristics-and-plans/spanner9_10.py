from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace
    return fact.strip()[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Simple check: must have at least as many parts as non-wildcard args
    if len(parts) < len([arg for arg in args if arg != '*']):
         return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for BFS
def bfs(graph, start):
    """Computes shortest path distances from start to all reachable nodes in an unweighted graph."""
    distances = {node: float('inf') for node in graph}
    if start in graph: # Ensure start node exists in the graph
        distances[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            if u in graph: # Ensure node exists in graph keys
                for v in graph[u]:
                    if distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all required nuts.
    It considers the number of nuts still needing tightening, the number of spanners
    that need to be picked up, and an estimate of the movement cost based on
    distances from the man's current location to relevant task locations.

    # Assumptions:
    - There is only one man object.
    - Nuts do not move from their initial locations.
    - Spanners do not move unless carried by the man.
    - A spanner becomes unusable after tightening one nut.
    - The goal is to tighten a specific set of nuts.
    - The graph of locations is connected, or relevant locations are reachable from each other.

    # Heuristic Initialization
    - Identify the man object.
    - Build the graph of locations based on `link` facts and locations mentioned in initial state/goals.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify the set of nuts that need to be tightened in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the man object (done in __init__, assuming only one).
    2. Determine the man's current location from the state. If the man's location is unknown, the state is likely invalid or unsolvable, return infinity.
    3. Identify the set of nuts that are specified in the goal (`(tightened ?n)`) and are currently `(loose ?n)` in the current state. These are the nuts that still need tightening. Let this set be `LooseGoalNuts`.
    4. Count the number of nuts in `LooseGoalNuts` (`N_loose_goal_nuts`). If this count is 0, all goal nuts are tightened, and the heuristic value is 0.
    5. Identify usable spanners the man is currently carrying. Count them (`N_usable_carried`).
    6. Calculate the number of additional usable spanners the man needs to pick up to tighten all `N_loose_goal_nuts`: `N_pickups_needed = max(0, N_loose_goal_nuts - N_usable_carried)`.
    7. Identify usable spanners that are currently at locations (not carried by the man). Count them (`N_usable_at_locs`).
    8. Check for unsolvability: If the total number of usable spanners available (`N_usable_carried + N_usable_at_locs`) is less than the number of nuts that need tightening (`N_loose_goal_nuts`), the problem is unsolvable from this state. Return a large value (infinity).
    9. Calculate the base cost from non-movement actions: This includes one `tighten_nut` action for each loose goal nut (`N_loose_goal_nuts`) and one `pickup_spanner` action for each spanner that needs to be picked up (`N_pickups_needed`). Base cost = `N_loose_goal_nuts + N_pickups_needed`.
    10. Estimate the movement cost: The man needs to travel to the locations of the loose goal nuts to tighten them, and potentially to the locations of usable spanners to pick them up.
        - Find the locations of all nuts in `LooseGoalNuts`.
        - Find the locations of all usable spanners that are currently at locations (`UsableAtLocs`).
        - A simple, non-admissible estimate for movement cost is the sum of the shortest path distances from the man's current location to *each* location containing a loose goal nut, plus the sum of the shortest path distances from the man's current location to *each* location containing a usable spanner (that is not carried). This sums the "effort" to reach all relevant items/locations from the current position.
    11. The total heuristic value is the sum of the base cost and the estimated movement cost. If any required location is unreachable, the movement cost (and thus the total heuristic) will be infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Need initial state to find the man

        # 1. Identify the man object (assuming only one)
        # Find the object that is 'at' a location in the initial state and is not a spanner or nut
        spanners_in_init = {get_parts(f)[1] for f in initial_state if get_parts(f)[0] in ['usable', 'carrying']}
        nuts_in_init = {get_parts(f)[1] for f in initial_state if get_parts(f)[0] in ['loose', 'tightened']}
        self.man = None
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                 obj = parts[1]
                 if obj not in spanners_in_init and obj not in nuts_in_init:
                     self.man = obj
                     break
        # Fallback if man not found by checking spanner/nut predicates (less robust)
        if not self.man:
             for fact in initial_state:
                 parts = get_parts(fact)
                 if parts[0] == 'at' and len(parts) == 3:
                     self.man = parts[1] # Assume the first object 'at' a location is the man
                     break


        # 2. Build the graph of locations
        self.locations = set()
        self.location_graph = {} # Adjacency list

        # Add locations from link facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1) # Links are bidirectional

        # Add locations mentioned in initial state or goals even if not linked
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 self.locations.add(parts[2])
                 self.location_graph.setdefault(parts[2], []) # Ensure location exists in graph keys

        for goal in self.goals:
             parts = get_parts(goal)
             # Goals can be (at obj loc) or (tightened nut) etc. Only 'at' adds locations.
             if parts[0] == 'at' and len(parts) == 3:
                 self.locations.add(parts[2])
                 self.location_graph.setdefault(parts[2], [])

        # Ensure all locations are in the graph keys, even if they have no links
        for loc in self.locations:
             self.location_graph.setdefault(loc, [])


        # 3. Compute all-pairs shortest path distances
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = bfs(self.location_graph, start_node)

        # 4. Identify the set of nuts that need to be tightened in the goal state.
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'tightened'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 2. Get man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1] == self.man:
                man_location = parts[2]
                break

        if man_location is None:
             # Man is not at any location? Should not happen in valid states.
             return float('inf') # Indicate unsolvability from this state.

        # Check if man_location is a known location in our graph
        if man_location not in self.locations:
             return float('inf') # Man is in an unknown location


        # 3. Identify loose goal nuts in the current state
        loose_goal_nuts = {
            n for n in self.goal_nuts
            if f"(loose {n})" in state
        }
        n_loose_goal_nuts = len(loose_goal_nuts)

        # 4. If no loose goal nuts, we are in a goal state (or a state from which goals are already met)
        if n_loose_goal_nuts == 0:
            return 0

        # 5. Identify usable spanners carried by the man
        usable_carried = {
            s for fact in state if match(fact, "carrying", self.man, "*")
            for s in [get_parts(fact)[2]] # Extract spanner name
            if f"(usable {s})" in state
        }
        n_usable_carried = len(usable_carried)

        # 6. Calculate additional usable spanners needed
        n_pickups_needed = max(0, n_loose_goal_nuts - n_usable_carried)

        # 7. Identify usable spanners at locations
        usable_at_locs = {
            s for fact in state if match(fact, "at", "*", "*")
            for s in [get_parts(fact)[1]] # Extract object name
            if f"(usable {s})" in state and s not in usable_carried # Must be usable and not carried
        }
        n_usable_at_locs = len(usable_at_locs)

        # 8. Check for unsolvability
        if n_loose_goal_nuts > n_usable_carried + n_usable_at_locs:
            return float('inf') # Not enough usable spanners exist

        # 9. Calculate base cost (non-movement actions)
        base_cost = n_loose_goal_nuts + n_pickups_needed # tighten + pickup actions

        # 10. Estimate movement cost
        loose_goal_nut_locs = set()
        for n in loose_goal_nuts:
             for fact in state:
                 if match(fact, "at", n, "*"):
                     loose_goal_nut_locs.add(get_parts(fact)[2])
                     break # Assuming each nut is at only one location

        usable_spanner_locs = set()
        for s in usable_at_locs:
             for fact in state:
                 if match(fact, "at", s, "*"):
                     usable_spanner_locs.add(get_parts(fact)[2])
                     break # Assuming each spanner is at only one location if not carried


        movement_cost = 0
        # Add distance from man's current location to each loose goal nut location
        for loc in loose_goal_nut_locs:
            if man_location in self.dist and loc in self.dist[man_location]:
                 dist = self.dist[man_location][loc]
                 if dist == float('inf'): return float('inf') # Unreachable location
                 movement_cost += dist
            else:
                 # This location was not in our precomputed graph. Should not happen
                 # if __init__ correctly collected all locations.
                 return float('inf') # Location not in graph or unreachable

        # Add distance from man's current location to each usable spanner location (at a location)
        # This sums distances to ALL usable spanners at locations, which might be more than needed (N_pickups_needed).
        # For simplicity and efficiency, summing distances to all is a reasonable overestimate.
        for loc in usable_spanner_locs:
             if man_location in self.dist and loc in self.dist[man_location]:
                 dist = self.dist[man_location][loc]
                 if dist == float('inf'): return float('inf') # Unreachable location
                 movement_cost += dist
             else:
                 # This location was not in our precomputed graph.
                 return float('inf') # Location not in graph or unreachable


        # 11. Total heuristic value
        total_cost = base_cost + movement_cost

        return total_cost
