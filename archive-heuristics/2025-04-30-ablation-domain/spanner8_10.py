from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Utility function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Utility function to check if a PDDL fact matches a given pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a valid match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the cost of tightening each nut, the cost of acquiring a spanner
    if the man is not already carrying one, and the estimated movement cost for the
    man to visit all locations with loose nuts.

    # Assumptions
    - The man can only carry one spanner at a time (implicitly handled by checking if *any* spanner is carried).
    - Usable spanners, if needed, are available at some location and are reachable.
    - The locations form a connected graph based on 'link' predicates.
    - The man must visit each location with a loose nut to tighten it.
    - The movement cost between locations is the shortest path distance.
    - Action costs are uniform (cost 1 per action).

    # Heuristic Initialization
    - Build a graph of locations based on 'link' predicates found in static facts.
    - Compute all-pairs shortest path distances between locations using BFS on the graph.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information from the State:
       - Find the man's current location ('at bob ?loc').
       - Determine if the man is currently carrying any spanner ('carrying bob ?spanner').
       - Identify all nuts that are currently 'loose' and their locations ('loose ?nut', 'at ?nut ?loc').
       - Identify all spanners that are 'usable', not currently carried by the man, and their locations ('usable ?spanner', 'at ?spanner ?loc', not 'carrying bob ?spanner').

    2. Handle Goal State:
       - If there are no loose nuts, the goal is achieved for all nuts. Return 0.

    3. Initialize Heuristic Cost:
       - Start the heuristic cost with the total number of loose nuts. This accounts for the 'tighten' action needed for each loose nut.

    4. Account for Spanner Acquisition (if needed):
       - If the man is not currently carrying a spanner:
         - Find the minimum distance from the man's current location to any location containing a usable, non-carried spanner.
         - If no such spanner is reachable, the problem is likely unsolvable from this state; return infinity.
         - Add this minimum distance to the heuristic cost (representing movement to the spanner).
         - Add 1 to the heuristic cost (representing the 'pick' action).
         - Conceptually, the man is now at the spanner's location and has the spanner. Update the man's effective current location for subsequent calculations to this spanner location.

    5. Estimate Movement Cost to Visit Nuts:
       - Identify the set of unique locations where loose nuts are situated.
       - Starting from the man's current effective location (initial location, or spanner location if a spanner was picked up), greedily estimate the movement cost to visit all these loose nut locations.
       - Use a nearest-neighbor-like approach: repeatedly find the closest unvisited loose nut location from the current effective location, add the distance to the heuristic cost, and move the effective current location to the visited nut location.
       - Repeat until all loose nut locations have been visited.
       - If any remaining nut location is unreachable, return infinity.

    6. Final Heuristic Value:
       - The total heuristic value is the sum of the initial count of loose nuts, the cost to acquire a spanner (if needed), and the estimated movement cost to visit all nut locations.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and computing distances."""
        self.goals = task.goals
        static_facts = task.static

        # Build location graph
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        self.locations = list(locations) # Store locations list for easy iteration

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[(start_node, start_node)] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, dist = queue.popleft()

                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_node, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Extract Relevant Information from the State:
        man_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_loc = get_parts(fact)[2]
                break
        # If Bob's location is not in the state, something is wrong or unsolvable.
        if man_loc is None:
             return float('inf')

        carrying_spanner = any(match(fact, "carrying", "bob", "*") for fact in state)

        loose_nuts_locations = {} # nut -> location
        for fact in state:
            if match(fact, "loose", "*"):
                 nut = get_parts(fact)[1]
                 # Find location of this nut
                 for loc_fact in state:
                     if match(loc_fact, "at", nut, "*"):
                         loose_nuts_locations[nut] = get_parts(loc_fact)[2]
                         break
                 # Note: Assumes loose nuts always have an 'at' fact in valid states.

        usable_spanner_locations = set() # set of locations
        usable_spanners = set() # set of spanner names
        for fact in state:
             if match(fact, "usable", "*"):
                 usable_spanners.add(get_parts(fact)[1])

        for spanner in usable_spanners:
             # Check if this spanner is being carried by bob
             is_carried = f"(carrying bob {spanner})" in state
             if not is_carried:
                 # Find location of this usable, non-carried spanner
                 for loc_fact in state:
                     if match(loc_fact, "at", spanner, "*"):
                         usable_spanner_locations.add(get_parts(loc_fact)[2])
                         break
                 # Note: Assumes spanners always have an 'at' fact if not carried in valid states.


        # 2. Handle Goal State:
        if not loose_nuts_locations:
            return 0

        # 3. Initialize Heuristic Cost:
        h = len(loose_nuts_locations) # Cost for tighten actions

        # Determine the man's effective starting location for the sequence of tasks.
        current_man_loc = man_loc
        has_spanner = carrying_spanner

        # 4. Account for Spanner Acquisition (if needed):
        cost_to_get_spanner = 0
        if not has_spanner:
            if not usable_spanner_locations:
                # No usable spanners available and not carrying one. Unsolvable.
                return float('inf')

            # Find closest spanner location from current_man_loc
            min_dist_to_spanner = float('inf')
            closest_spanner_loc = None
            for loc_s in usable_spanner_locations:
                dist = self.distances.get((current_man_loc, loc_s), float('inf'))
                if dist == float('inf'):
                    # This spanner location is unreachable
                    continue
                if dist < min_dist_to_spanner:
                    min_dist_to_spanner = dist
                    closest_spanner_loc = loc_s

            if closest_spanner_loc is None or min_dist_to_spanner == float('inf'):
                 # Cannot reach any usable spanner
                 return float('inf')

            cost_to_get_spanner = min_dist_to_spanner + 1 # move + pick
            # Update effective location for subsequent travel cost calculation
            current_man_loc = closest_spanner_loc
            has_spanner = True # Conceptually, he now has a spanner

        h += cost_to_get_spanner

        # 5. Estimate Movement Cost to Visit Nuts:
        remaining_nut_locations = set(loose_nuts_locations.values())
        total_movement_cost_to_nuts = 0

        # Use a greedy approach: always go to the closest remaining nut location
        while remaining_nut_locations:
            min_dist_to_nut = float('inf')
            closest_nut_loc = None

            for loc_n in remaining_nut_locations:
                dist = self.distances.get((current_man_loc, loc_n), float('inf'))
                if dist == float('inf'):
                    # This nut location is unreachable
                    continue
                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
                    closest_nut_loc = loc_n

            if closest_nut_loc is None or min_dist_to_nut == float('inf'):
                 # Cannot reach any remaining nut location
                 return float('inf')

            total_movement_cost_to_nuts += min_dist_to_nut
            current_man_loc = closest_nut_loc # Man moves to this nut location
            remaining_nut_locations.remove(closest_nut_loc) # This nut location is now visited

        h += total_movement_cost_to_nuts

        # 6. Final Heuristic Value:
        return h
