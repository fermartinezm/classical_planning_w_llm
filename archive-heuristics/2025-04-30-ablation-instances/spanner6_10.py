from fnmatch import fnmatch
from collections import deque
import math # For infinity

# Assume Heuristic base is available from the framework
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, uncomment this dummy base class:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def find_fact_in_set(fact_set, *args):
     """
     Find and return the first fact in fact_set that matches the given pattern.
     Returns None if no match is found.
     """
     pattern_parts = args
     for fact in fact_set:
         if match(fact, *pattern_parts):
             return fact
     return None # Not found

def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all reachable nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {}
    queue = deque([start_node])
    distances[start_node] = 0
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Handle nodes with no outgoing links
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
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It sums the cost for each loose nut (1 for tighten + walk to nut) and adds
    the cost for acquiring necessary spanners (1 for pickup + walk to closest spanner).

    # Assumptions:
    - There is exactly one man.
    - Nut locations are static.
    - Spanners become unusable after one use for tightening a nut.
    - The problem is solvable (enough usable spanners exist in total).
    - Walk actions have a cost of 1 per link.
    - Pickup and Tighten actions have a cost of 1.

    # Heuristic Initialization
    - Build the location graph from static 'link' facts.
    - Identify all nuts from goal 'tightened' facts.
    - Store static locations of nuts from initial state 'at' facts.
    - Identify the man and all spanners from initial state facts ('carrying', 'usable', 'at').

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the goal is reached (no loose nuts). If so, heuristic is 0.
    2. Identify the man's current location from the state. If not found, return infinity (unsolvable).
    3. Run BFS from the man's current location on the location graph to compute shortest distances to all reachable locations.
    4. Identify all currently loose nuts from the state.
    5. Identify all currently usable spanners from the state (carried or on the ground).
    6. Check for unsolvability based on spanners: If the total number of usable spanners in the state
       (carried + on ground) is less than the number of loose nuts, return infinity.
    7. Initialize heuristic value `h` to 0.
    8. Add cost for each loose nut: For each loose nut, add 1 (for the tighten action)
       plus the shortest distance from the man's current location to the nut's static location.
       If any nut location is unreachable from the man's current location, return infinity.
    9. Calculate the number of additional spanners the man needs to pick up from the ground.
       This is `max(0, number of loose nuts - number of usable carried spanners)`.
    10. If pickups are needed (`num_pickups_needed > 0`):
        - Find all usable spanners currently on the ground and their locations from the state.
        - If there are no usable spanners on the ground, but pickups are needed, return infinity (should be caught by step 6, but double-check).
        - Find the minimum distance from the man's current location to any usable spanner on the ground.
        - If the location of the closest usable ground spanner is unreachable, return infinity.
        - Add the cost for these pickups: `number of pickups needed` (for the pickup actions)
          plus the `minimum distance found` (for the walk to the first spanner).
    11. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to find static nut locations and object names

        # Build location graph
        self.location_graph = {}
        self.all_locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Identify all nuts from goals
        self.all_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.all_nuts.add(nut)

        # Store static nut locations from initial state
        self.nut_locations = {}
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.all_nuts:
                     self.nut_locations[obj] = loc

        # Identify the man and all spanners
        self.man_name = None
        self.all_spanners = set()
        # Identify man and spanners from facts that uniquely type them
        for fact in self.initial_state:
            if match(fact, "carrying", "*", "*"):
                _, man, spanner = get_parts(fact)
                self.man_name = man # Assume one man
                self.all_spanners.add(spanner)
            elif match(fact, "usable", "*"):
                 _, spanner = get_parts(fact)
                 self.all_spanners.add(spanner)

        # If man_name wasn't found via 'carrying', try finding an 'at' fact
        # where the object is not a known nut or spanner.
        if self.man_name is None:
             for fact in self.initial_state:
                 if match(fact, "at", "*", "*"):
                      obj, loc = get_parts(fact)[1:]
                      if obj not in self.all_nuts and obj not in self.all_spanners:
                           self.man_name = obj
                           break # Assume one man

        # Add any spanners found via 'at' that weren't in 'carrying' or 'usable'
        # This assumes any 'locatable' object in 'at' that isn't the man or a nut is a spanner.
        # This is less robust, but necessary if 'usable'/'carrying' don't list all spanners initially.
        # Let's rely on 'usable' and 'carrying' as the primary way to find spanners, as they are typed.
        # If a spanner exists but is neither usable nor carried initially, it won't be found here.
        # However, the domain implies spanners are relevant if usable or carried.
        # Let's stick to spanners found via 'carrying' or 'usable' in the initial state.
        # If a spanner is only mentioned in an 'at' fact initially, but never in 'usable' or 'carrying',
        # it might be an irrelevant object of type spanner.

        # Basic check if essential objects were found
        if self.man_name is None:
             print("Warning: Could not identify the man in the initial state.")
             # Problem might be unsolvable, but heuristic should handle state where man is missing.
             # The __call__ method will return inf if man_loc is not found.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if goal is reached
        loose_nuts = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}
        if not loose_nuts:
            return 0 # Goal reached

        # 2. Identify man's current location
        man_loc = None
        if self.man_name: # Ensure man_name was identified in init
            man_at_fact = find_fact_in_set(state, "at", self.man_name, "*")
            if man_at_fact:
                man_loc = get_parts(man_at_fact)[2]

        if man_loc is None:
             # Man's location is unknown in this state (e.g., man_name not found or no 'at' fact for man)
             return float('inf') # Unsolvable from this state

        # 3. Run BFS from man's current location
        dist = bfs(self.location_graph, man_loc)

        # Check if man's location is a known location in the graph
        if man_loc not in self.all_locations:
             # Man is at an unknown location, cannot navigate
             return float('inf')

        # 4. Identify currently loose nuts (already done above)
        # 5. Identify currently usable spanners (carried or on ground)
        usable_carried_spanners = {
            get_parts(fact)[2] for fact in state
            if match(fact, "carrying", self.man_name, "*") and find_fact_in_set(state, "usable", get_parts(fact)[2])
        }
        usable_ground_spanners = {
            get_parts(fact)[1] for fact in state
            if match(fact, "at", "*", "*") and get_parts(fact)[1] in self.all_spanners and find_fact_in_set(state, "usable", get_parts(fact)[1])
        }

        # 6. Check for unsolvability based on spanners
        num_tightenings_needed = len(loose_nuts)
        total_usable_spanners_in_state = len(usable_carried_spanners) + len(usable_ground_spanners)

        if total_usable_spanners_in_state < num_tightenings_needed:
            return float('inf') # Unsolvable

        # 7. Initialize heuristic value
        h = 0

        # 8. Add cost for each loose nut
        for nut in loose_nuts:
            nut_loc = self.nut_locations.get(nut)
            if nut_loc is None:
                 # Nut location not found (should be static and in initial state)
                 return float('inf') # Problem definition issue?

            if nut_loc not in dist:
                # Nut location is unreachable from man's current location
                return float('inf')

            h += 1 # Cost for tighten_nut action
            h += dist[nut_loc] # Cost to walk from man_loc to nut_loc

        # 9. Calculate number of pickups needed
        num_pickups_needed = max(0, num_tightenings_needed - len(usable_carried_spanners))

        # 10. Add cost for spanner pickups if needed
        if num_pickups_needed > 0:
            # Find locations of usable ground spanners
            usable_ground_spanner_locations = {}
            for spanner in usable_ground_spanners:
                 spanner_at_fact = find_fact_in_set(state, "at", spanner, "*")
                 if spanner_at_fact:
                      usable_ground_spanner_locations[spanner] = get_parts(spanner_at_fact)[2]

            if len(usable_ground_spanner_locations) < num_pickups_needed:
                 # Need pickups, but not enough usable spanners on the ground.
                 # This implies total_usable_spanners_in_state < num_tightenings_needed,
                 # which should have been caught earlier. This is a redundant check but safe.
                 return float('inf')

            min_dist_to_ground_spanner = float('inf')
            for spanner, spanner_loc in usable_ground_spanner_locations.items():
                if spanner_loc not in dist:
                    # Usable spanner location is unreachable
                    return float('inf')
                min_dist_to_ground_spanner = min(min_dist_to_ground_spanner, dist[spanner_loc])

            if min_dist_to_ground_spanner == float('inf'):
                 # This can happen if usable_ground_spanner_locations is not empty but all are unreachable
                 return float('inf')

            h += num_pickups_needed # Cost for pickup_spanner actions
            h += min_dist_to_ground_spanner # Cost to walk to the closest spanner for the first pickup

        # 11. Return total heuristic value
        return h
