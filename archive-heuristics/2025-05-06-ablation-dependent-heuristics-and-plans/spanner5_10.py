import math
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available at this path
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node.
    Returns a dictionary mapping reachable nodes to their distance from start_node.
    """
    distances = {node: math.inf for node in graph}
    
    # Ensure start_node is a valid key in the graph dictionary
    if start_node not in graph:
        # If start_node is not a known location in the graph, it's isolated.
        # BFS from it can only reach itself with distance 0.
        # This case might happen if a location is mentioned in 'at' but not 'link'.
        # If start_node is not even in the initial graph keys (meaning it wasn't found in links/initial at),
        # the distances dict remains all inf, which is correct.
        # If start_node is a valid location but has no links, distances[start_node] = 0 is correct.
        if start_node in distances:
             distances[start_node] = 0
             # No neighbors to add to queue, BFS finishes immediately.
    else:
        # Standard BFS if start_node is in the graph and potentially has neighbors
        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()
            # Ensure current_node is a key in the graph, even if it has no neighbors
            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the number of nuts remaining, the number of spanners that need to be picked up,
    and the travel cost to reach the next required location (either a spanner or a nut).

    # Assumptions
    - There is only one man.
    - Each usable spanner can tighten exactly one nut.
    - The problem is solvable (i.e., there are enough usable spanners for all goal nuts).
      If not enough spanners, the heuristic returns infinity.
    - The locations form a graph where relevant locations are reachable.
    - Object names follow conventions (e.g., spanner*, nut*).

    # Heuristic Initialization
    - Extracts the set of goal nuts from the task goals.
    - Collects all unique locations mentioned in links and initial 'at' facts.
    - Builds a graph of locations based on `link` predicates.
    - Computes shortest path distances between all pairs of locations using BFS.
    - Identifies the name of the man object (heuristically, assuming one man).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Determine if the man is currently carrying a usable spanner.
    3. Identify all usable spanners on the ground and their locations.
    4. Identify all loose nuts and their locations.
    5. Filter the loose nuts to find those that are goal conditions (`loose goal nuts`).
    6. Count the number of loose goal nuts (`N_loose_goals`). If 0, the heuristic is 0 (goal state).
    7. Count the total number of usable spanners available (carried + on ground). If this count is less than `N_loose_goals`, the problem is likely unsolvable from this state, return infinity.
    8. Calculate the number of `pickup_spanner` actions needed: `N_pickups = max(0, N_loose_goals - (1 if man is carrying a usable spanner else 0))`.
    9. The base heuristic value is `N_loose_goals` (for tighten actions) + `N_pickups` (for pickup actions).
    10. Add travel cost:
        - Find the man's current location (`man_location`).
        - If the man is carrying a usable spanner: The next required location is a loose goal nut location. Find the minimum distance from `man_location` to any location of a loose goal nut. Add this minimum distance to the heuristic.
        - If the man is not carrying a usable spanner: The next required location is a usable ground spanner location. Find the minimum distance from `man_location` to any location of a usable ground spanner. Add this minimum distance to the heuristic.
        - If the minimum distance calculation results in infinity (e.g., no reachable required locations), the heuristic will be infinity, indicating unsolvability.
    11. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the location graph, and computing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Need initial state to find all locations and the man

        # 1. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

        # 2. Collect all unique locations
        locations = set()
        # Locations from links
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
        # Locations from initial 'at' facts (for objects that are locatable)
        # Assuming any object in an 'at' predicate in the initial state is at a valid location.
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)[1:]
                 locations.add(loc) # Add the location

        self.locations = list(locations) # Store as list

        # 3. Build adjacency list graph from (link l1 l2) facts
        self.graph = {loc: [] for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                # Ensure locations are known before adding links
                if l1 in self.graph and l2 in self.graph:
                    self.graph[l1].append(l2)
                    self.graph[l2].append(l1) # Links are bidirectional

        # 4. Compute shortest path distances between all pairs of locations using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(self.graph, start_loc)

        # 5. Identify the man object name (heuristic)
        self.man_name = None
        # Try finding the object of type 'man' from initial state 'at' facts
        # by excluding objects typically named spanner* or nut*.
        potential_men = set()
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Heuristic check based on typical naming conventions
                 if not (obj.startswith('spanner') or obj.startswith('nut')):
                     potential_men.add(obj)

        if len(potential_men) == 1:
            self.man_name = list(potential_men)[0]
        else:
             # Fallback: Look for an object involved in a 'carrying' predicate in the initial state
             for fact in initial_state:
                 if match(fact, "carrying", "*", "*"):
                     self.man_name = get_parts(fact)[1]
                     break
             # If still not found, the heuristic cannot identify the man.
             if self.man_name is None:
                 print("Warning: Could not determine man's name during initialization.")


    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if no path or location unknown."""
        if loc1 not in self.distances or loc2 not in self.locations:
             return math.inf
        return self.distances[loc1].get(loc2, math.inf)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If man's name wasn't determined during initialization, we can't proceed.
        if self.man_name is None:
             return math.inf

        # 1. Identify man's current location
        man_location = None
        carried_spanner = None
        is_spanner_usable = False
        usable_ground_spanners = [] # List of (spanner_name, location)
        loose_nuts = {} # Map nut_name to location

        # Find man's location and carried spanner from the current state
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
            elif match(fact, "carrying", self.man_name, "*"):
                carried_spanner = get_parts(fact)[2]
            # Also collect usable ground spanners and loose nuts here
            elif match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj.startswith('spanner'): # Heuristic type check
                     # Check if this spanner is usable and not carried by the man
                     is_carried_by_man = (carried_spanner == obj)
                     is_usable = f"(usable {obj})" in state
                     if not is_carried_by_man and is_usable:
                         usable_ground_spanners.append((obj, loc))
                 elif obj.startswith('nut'): # Heuristic type check
                     if f"(loose {obj})" in state:
                         loose_nuts[obj] = loc

        # If man_location is None, something is wrong with the state or problem
        if man_location is None:
             # This indicates the man object is not 'at' any location in the state, which is invalid.
             return math.inf

        # Check if the carried spanner is usable
        if carried_spanner and f"(usable {carried_spanner})" in state:
             is_spanner_usable = True

        # 5. Filter loose nuts to find loose goal nuts
        loose_goal_nuts = {n: loc for n, loc in loose_nuts.items() if n in self.goal_nuts}

        # 6. Count loose goal nuts. If 0, goal reached.
        N_loose_goals = len(loose_goal_nuts)
        if N_loose_goals == 0:
            return 0

        # 7. Count total usable spanners. Check solvability.
        N_usable_carried = 1 if is_spanner_usable else 0
        N_usable_ground = len(usable_ground_spanners)
        total_usable_spanners = N_usable_carried + N_usable_ground

        if N_loose_goals > total_usable_spanners:
            return math.inf # Unsolvable

        # 8. Calculate number of pickups needed
        N_pickups = max(0, N_loose_goals - N_usable_carried)

        # 9. Base heuristic value
        h = N_loose_goals + N_pickups

        # 10. Add travel cost
        travel_cost = math.inf # Initialize travel cost to infinity

        if N_usable_carried > 0:
            # Man has a spanner, needs to go to a nut
            min_dist_to_nut = math.inf
            if loose_goal_nuts: # Ensure there are nuts to go to
                for nut_loc in loose_goal_nuts.values():
                    dist = self.get_distance(man_location, nut_loc)
                    min_dist_to_nut = min(min_dist_to_nut, dist)
                travel_cost = min_dist_to_nut
            else:
                 # This case should be covered by N_loose_goals == 0 check earlier.
                 # If we are here, N_loose_goals > 0, so there are nuts.
                 pass # Logic seems sound.

        else: # Man does not have a usable spanner
            # Man needs a spanner, needs to go to a usable ground spanner
            min_dist_to_spanner = math.inf
            if usable_ground_spanners: # Ensure there are spanners to go to
                for spanner_loc in [loc for _, loc in usable_ground_spanners]:
                     dist = self.get_distance(man_location, spanner_loc)
                     min_dist_to_spanner = min(min_dist_to_spanner, dist)
                travel_cost = min_dist_to_spanner
            else:
                 # Man needs a spanner but none are available on the ground.
                 # This case should be caught by the total_usable_spanners check earlier.
                 # If N_loose_goals > 0 and total_usable_spanners == 0, it's unsolvable.
                 # If N_loose_goals > 0 and total_usable_spanners > 0 but N_usable_carried == 0 and N_usable_ground == 0, this is a contradiction.
                 # The check `if usable_ground_spanners:` handles the case where N_usable_ground is 0.
                 # If N_usable_carried is 0 and N_usable_ground is 0, total_usable_spanners is 0.
                 # If N_loose_goals > 0 and total_usable_spanners is 0, we return inf earlier.
                 # So, if we reach this else block and N_loose_goals > 0, it must be that N_usable_ground > 0.
                 # Therefore, min_dist_to_spanner will not be inf unless the spanners are unreachable.
                 pass # Logic seems sound.


        # If travel_cost is still infinity, it means required locations are unreachable
        # from the man's current location, which implies unsolvability.
        if travel_cost == math.inf:
             return math.inf

        h += travel_cost

        # 11. Return the total calculated heuristic value.
        return h
