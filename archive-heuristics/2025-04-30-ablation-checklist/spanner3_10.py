import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions (reused from Logistics example, adapted)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_location_from_state(state, obj_name):
    """Find the location of a given object in the current state."""
    for fact in state:
        if match(fact, "at", obj_name, "*"):
            return get_parts(fact)[2]
    return None # Object not found at any location (e.g., carried)

def compute_shortest_paths(static_facts, initial_state):
    """
    Computes all-pairs shortest paths between locations using BFS.
    Assumes links are bidirectional.
    """
    locations = set()
    adjacency_list = collections.defaultdict(list)

    # Collect locations from link facts
    for fact in static_facts:
        if match(fact, "link", "*", "*"):
            l1, l2 = get_parts(fact)[1:]
            locations.add(l1)
            locations.add(l2)
            adjacency_list[l1].append(l2)
            adjacency_list[l2].append(l1) # Links are bidirectional

    # Collect locations from initial 'at' facts (in case some locations are isolated)
    for fact in initial_state:
         if match(fact, "at", "*", "*"):
             loc = get_parts(fact)[2]
             locations.add(loc)

    distances = {}
    for start_loc in locations:
        distances[start_loc] = {}
        queue = collections.deque([(start_loc, 0)])
        visited = {start_loc}
        distances[start_loc][start_loc] = 0

        while queue:
            current_loc, dist = queue.popleft()

            for neighbor in adjacency_list.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[start_loc][neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

    # Fill in unreachable locations with infinity
    for l1 in locations:
        for l2 in locations:
            if l2 not in distances[l1]:
                distances[l1][l2] = float('inf')

    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It considers the number of nuts to tighten, the number of spanners to pick up,
    and the estimated travel cost for the man to reach the necessary locations
    (spanners to pick up, nuts to tighten).

    # Heuristic Initialization
    - Computes all-pairs shortest paths between locations.
    - Identifies the man object (assumes there's only one man).
    - Identifies the fixed locations of all nuts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts that are part of the goal.
    3. Count the number of such nuts (`N_nuts`). If 0, goal is reached, heuristic is 0.
    4. Count usable spanners the man is currently carrying (`N_carried`).
    5. Count usable spanners available on the ground (`N_available`).
    6. Check if the total number of usable spanners (`N_carried + N_available`) is less than `N_nuts`. If so, the problem is unsolvable, return infinity.
    7. Calculate the number of additional usable spanners the man needs to pick up from the ground: `N_pickup = max(0, N_nuts - N_carried)`.
    8. The heuristic cost includes:
       - `N_nuts` tighten actions (cost 1 each).
       - `N_pickup` pickup actions (cost 1 each).
       - Estimated walk actions.
    9. Estimate walk actions: The man needs to visit the location of each loose goal nut, and the location of each of the `N_pickup` spanners he needs to pick up.
       - Identify the set of required locations: locations of all loose goal nuts + locations of all usable spanners on the ground (if `N_pickup > 0`).
       - Calculate the shortest distance from the man's current location to the *closest* required location (`min_dist_to_first`).
       - Estimate subsequent travel: Assume 1 walk action is sufficient to move between any two required locations after the first one is reached. This adds `max(0, |RequiredLocations| - 1)` to the walk cost.
       - Total walk cost = `min_dist_to_first + max(0, |RequiredLocations| - 1)`. Handle unreachable locations by returning infinity.
    10. Sum the costs: `N_nuts + N_pickup + walk_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        self.initial_state = task.initial_state # Keep initial state to find man and nut locs

        # 1. Identify all locations and compute shortest paths
        self.distances = compute_shortest_paths(task.static, task.initial_state)
        self.locations = list(self.distances.keys()) # Get the list of all known locations

        # 2. Identify the man object (assume first object of type man in initial state)
        self.man_name = None
        # A more robust way would parse the domain/instance objects section,
        # but given the input format, we infer from initial state and common naming.
        # Let's find the object at a location in the initial state that is likely the man.
        # We can assume the first object found with an 'at' predicate in the initial state
        # that isn't a spanner or nut is the man. Or, look for the object that is 'carrying'.
        # A simple approach: find the object at a location that is not a spanner or nut from goals/initial state.
        # Or, assume the man is the object involved in a 'carrying' predicate if one exists.
        # Let's try finding the object that is 'carrying' or the first 'at' object that isn't a known nut/spanner.
        nuts_in_goals = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}
        spanners_in_init = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at", "*", "*") and match(fact, "*", "spanner*")}

        for fact in task.initial_state:
            if match(fact, "carrying", "*", "*"):
                self.man_name = get_parts(fact)[1]
                break
        if not self.man_name:
             for fact in task.initial_state:
                 if match(fact, "at", "*", "*"):
                     obj_name = get_parts(fact)[1]
                     if obj_name not in nuts_in_goals and obj_name not in spanners_in_init:
                         self.man_name = obj_name
                         break

        if not self.man_name:
             # Fallback: Assume the first object in an 'at' predicate in initial state is the man
             # This is fragile but might work for simple instances
             for fact in task.initial_state:
                 if match(fact, "at", "*", "*"):
                     self.man_name = get_parts(fact)[1]
                     break

        if not self.man_name:
             print("Warning: Could not identify the man object.")
             # Heuristic might fail or return incorrect values if man is not found.
             # We could raise an error or return infinity later if man_loc is None.


        # 3. Identify fixed locations of nuts (nuts don't move unless carried, which isn't in domain)
        self.nut_locations = {}
        for fact in task.initial_state:
            if match(fact, "at", "*", "*"):
                obj_name, loc = get_parts(fact)[1:]
                if obj_name in nuts_in_goals: # Only track nuts that are goal conditions
                     self.nut_locations[obj_name] = loc


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Get man's current location
        man_loc = get_location_from_state(state, self.man_name)
        if man_loc is None:
             # Man is not at a location (e.g., problem state is invalid or man is carried - not possible in this domain)
             # Or man_name wasn't found in init state 'at' facts.
             # If man_name was not identified in __init__, this will also be None.
             # This state is likely unreachable or invalid for the heuristic.
             return float('inf')


        # 2. Identify loose nuts that are goal conditions
        loose_goal_nuts = {
            n for goal in self.goals
            if match(goal, "tightened", n) and f"(loose {n})" in state
        }
        N_nuts = len(loose_goal_nuts)

        # 3. If no loose goal nuts, goal is reached
        if N_nuts == 0:
            return 0

        # 4. Count usable spanners carried by the man
        carried_usable_spanners = {
            s for fact in state
            if match(fact, "carrying", self.man_name, s) and f"(usable {s})" in state
        }
        N_carried = len(carried_usable_spanners)

        # 5. Count usable spanners available on the ground
        available_usable_spanners = {
            s for fact in state
            if match(fact, "at", s, "*") and f"(usable {s})" in state and s not in carried_usable_spanners # Ensure not double counting carried ones
        }
        N_available = len(available_usable_spanners)

        # 6. Check if enough usable spanners exist in total
        if N_nuts > N_carried + N_available:
            # Problem is unsolvable from this state
            return float('inf')

        # 7. Calculate pickup cost
        N_pickup = max(0, N_nuts - N_carried)
        pickup_cost = N_pickup

        # 8. Calculate tighten cost
        tighten_cost = N_nuts

        # 9. Estimate walk cost
        required_locations = set()

        # Add locations of loose goal nuts
        for nut in loose_goal_nuts:
            if nut in self.nut_locations:
                 required_locations.add(self.nut_locations[nut])
            # else: nut not found in initial state locations? Should not happen in valid problems.

        # Add locations of usable spanners on the ground if pickups are needed
        if N_pickup > 0:
            for spanner in available_usable_spanners:
                 spanner_loc = get_location_from_state(state, spanner)
                 if spanner_loc: # Spanner must be at a location to be available
                     required_locations.add(spanner_loc)

        walk_cost = 0
        if required_locations:
            # Calculate distance to the closest required location
            min_dist_to_first = float('inf')
            reachable_required_locations = set()

            # Ensure man_loc is a valid key in distances (should be if found in state)
            if man_loc in self.distances:
                for loc in required_locations:
                    if loc in self.distances[man_loc]: # Check if reachable
                        dist = self.distances[man_loc][loc]
                        if dist != float('inf'):
                            min_dist_to_first = min(min_dist_to_first, dist)
                            reachable_required_locations.add(loc)

            if min_dist_to_first == float('inf'):
                 # No required location is reachable from the man's current location
                 return float('inf')

            # Estimate subsequent travel cost
            # If the man is already at one of the required locations, the first step cost is 0.
            # Otherwise, it's min_dist_to_first.
            # The number of *additional* locations to visit after reaching the first one.
            # If man_loc is one of the required locations, he still needs to visit the others.
            # Let's simplify: count how many *new* locations need visiting from the man's current spot.
            # If man is at L, and L is a required location, he still needs to visit |ReqLocs| - 1 others.
            # If man is at L, and L is NOT a required location, he needs to go to one (cost min_dist),
            # and then visit |ReqLocs| - 1 others.
            # The number of *distinct* required locations that are not the man's current location.
            num_distinct_required_not_at_man = len(reachable_required_locations - {man_loc})

            # The walk cost is the distance to the first required location, plus 1 action
            # for each subsequent distinct required location that needs visiting.
            # This is a simplified TSP-like estimate.
            walk_cost = min_dist_to_first + max(0, num_distinct_required_not_at_man - 1)


        # Total heuristic is the sum of estimated actions
        total_cost = tighten_cost + pickup_cost + walk_cost

        return total_cost

