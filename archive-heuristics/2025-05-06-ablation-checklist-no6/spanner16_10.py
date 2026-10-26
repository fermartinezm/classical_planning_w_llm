from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
# from heuristics.heuristic_base import Heuristic # Uncomment this line in the actual environment

# Define a dummy Heuristic base class for standalone testing if needed
# In a real planning environment, this would be provided.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost based on the number of nuts to tighten,
    spanners to pick up, and the maximum travel distance required
    to reach any necessary location (nut or spanner).

    Heuristic components:
    1. Number of loose nuts that are goal conditions (minimum tighten actions).
    2. Number of usable spanners that need to be picked up from the ground.
    3. Maximum shortest path distance from the man's current location to any
       location containing a loose goal nut or one of the necessary spanners
       to be picked up.

    Returns 0 for goal states, infinity for unsolvable states, and a finite
    estimate for solvable states.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances between locations.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Keep initial state to find nut names

        # 1. Extract all locations
        locations = set()
        # Locations from link facts
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
        # Locations from initial object placements
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 locations.add(loc)

        self.locations = list(locations)

        # 2. Build adjacency list for the location graph
        self.graph = {loc: [] for loc in self.locations}
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        self.distance = {}
        for start_loc in self.locations:
            self.distance[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distance[start_loc][start_loc] = 0

            while q:
                curr_loc, d = q.popleft()

                if curr_loc not in self.graph:
                     # This location was added from an 'at' fact but has no links.
                     # It's an isolated location. BFS from here only finds itself.
                     continue

                for neighbor in self.graph[curr_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distance[start_loc][neighbor] = d + 1
                        q.append((neighbor, d + 1))

        # Pre-identify all nut names from initial state or goals
        self.all_nut_names = set()
        for goal in self.goals:
             if match(goal, "tightened", "*"):
                 _, nut = get_parts(goal)
                 self.all_nut_names.add(nut)
        for fact in self.initial_state:
             if match(fact, "loose", "*"):
                 _, nut = get_parts(fact)
                 self.all_nut_names.add(nut)


    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state

        # 1. Identify loose nuts that are goal conditions
        goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                goal_nuts.add(nut)

        # A nut is a loose goal nut if it's a goal and it's currently loose.
        # If a goal nut is not 'tightened' in the state, we assume it's 'loose'.
        # This relies on the domain structure where nuts are either loose or tightened.
        tightened_nuts_in_state = {get_parts(fact)[1] for fact in state if match(fact, "tightened", "*")}
        loose_goal_nuts = {nut for nut in goal_nuts if nut not in tightened_nuts_in_state}

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # 2. Find man's current location and name
        man_name = None
        man_location = None

        # Find man name: Look for object in 'carrying' predicate first
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                _, m, s = get_parts(fact)
                man_name = m
                break

        # If man not carrying, find him by location (assume he's the only non-nut, non-spanner locatable)
        if man_name is None:
             for fact in state:
                 if match(fact, "at", "*", "*"):
                     obj, loc = get_parts(fact)[1:]
                     # Check if the object is not a nut and not a spanner
                     if obj not in self.all_nut_names and not match(fact, "at", "spanner*", "*"):
                          man_name = obj
                          break

        # Now find the location of the identified man
        for fact in state:
            if match(fact, "at", man_name, "*"):
                _, m, loc = get_parts(fact)
                man_location = loc
                break

        # If man_location is not found, the state is likely invalid/unreachable
        if man_location is None or man_location not in self.distance:
             return float('inf') # Unreachable state

        # 3. Count usable spanners carried by the man
        usable_spanners_carried = set()
        for fact in state:
            if match(fact, "carrying", man_name, "*"):
                _, m, s = get_parts(fact)
                # Check if this spanner is usable
                if f"(usable {s})" in state:
                    usable_spanners_carried.add(s)

        k_c_usable = len(usable_spanners_carried)

        # 4. Count usable spanners on the ground and their locations
        usable_spanners_ground = [] # Store as (spanner_name, location)
        for fact in state:
            if match(fact, "at", "spanner*", "*"): # Match spanners by name pattern
                _, s, loc = get_parts(fact)
                # Check if this spanner is usable
                if f"(usable {s})" in state:
                    usable_spanners_ground.append((s, loc))

        k_g_usable = len(usable_spanners_ground)

        # 5. Calculate number of pickups needed and check solvability
        K = len(loose_goal_nuts)
        total_usable_available = k_c_usable + k_g_usable

        if K > total_usable_available:
             return float('inf') # Unsolvable

        k_pickups = max(0, K - k_c_usable)


        # 6. Identify target locations
        # Locations of loose goal nuts
        nut_locations = set()
        for nut in loose_goal_nuts:
             for fact in state:
                 if match(fact, "at", nut, "*"):
                     _, n, loc = get_parts(fact)
                     nut_locations.add(loc)
                     break # Found location for this nut

        # Locations of spanners to pick up
        spanner_pickup_locations = set()
        if k_pickups > 0:
            # Sort usable ground spanners by distance from man's current location
            # Use .get(loc, float('inf')) for safety if a location is somehow not in distance map
            sorted_spanners = sorted(usable_spanners_ground,
                                     key=lambda item: self.distance[man_location].get(item[1], float('inf')))

            # Select the locations of the k_pickups nearest spanners
            picked_spanners_count = 0
            for s, loc in sorted_spanners:
                if picked_spanners_count < k_pickups:
                    spanner_pickup_locations.add(loc)
                    picked_spanners_count += 1
                else:
                    break # Got enough spanners

        all_target_locations = nut_locations | spanner_pickup_locations

        # 7. Calculate movement cost
        movement_cost = 0
        if all_target_locations: # Should be non-empty if K > 0
            # Find the maximum distance from the man's current location to any target location
            try:
                max_dist = 0
                for loc in all_target_locations:
                    dist = self.distance[man_location].get(loc, float('inf'))
                    if dist == float('inf'):
                         # A target location is unreachable from man's location
                         return float('inf') # Unsolvable
                    max_dist = max(max_dist, dist)
                movement_cost = max_dist

            except KeyError:
                 # Man's location not in distance map (should be caught earlier)
                 return float('inf')

        # 8. Calculate total heuristic value
        # Heuristic = (tighten actions) + (pickup actions) + (movement cost)
        heuristic_value = K + k_pickups + movement_cost

        return heuristic_value
