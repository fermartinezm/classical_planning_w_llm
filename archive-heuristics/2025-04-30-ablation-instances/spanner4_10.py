from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def bfs(graph, start):
    """Compute shortest path distances from start node in an unweighted graph."""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, []):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions (tighten, pickup, walk)
    required to tighten all loose nuts. It considers the cost to get the man
    to locations where nuts are, and the cost to acquire a usable spanner
    if needed.

    # Assumptions
    - There is a single man agent.
    - Nuts are static (their location does not change).
    - Spanners become permanently unusable after one use for tightening a nut.
    - Spanners are initially either on the ground or carried by the man.
    - The man can carry multiple spanners.
    - The location graph is static and bidirectional.
    - The man agent can be identified by initially carrying a spanner or being the only locatable object that is not a nut or spanner.

    # Heuristic Initialization
    - Builds the location graph from static 'link' facts.
    - Computes all-pairs shortest paths on the location graph using BFS.
    - Identifies goal nuts from the task goals.
    - Determines the static location of each goal nut from the initial state.
    - Identifies all spanners and their initial ground locations/carried status from the initial state.
    - Identifies the man agent.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Count the number of loose nuts. If zero, the goal is reached, heuristic is 0.
    2. Initialize heuristic value with the number of loose nuts (representing the 'tighten_nut' actions).
    3. Find the man's current location and the set of usable spanners he is carrying.
    4. Find the current locations of all spanners on the ground and identify which ones are usable.
    5. Determine if the man needs a usable spanner (i.e., there are loose nuts remaining and he is not carrying any usable spanner).
    6. If a spanner is needed:
       a. Check if *any* usable spanner exists in the current state (carried by man or on ground). If not, the state is unsolvable; return infinity.
       b. Find the closest usable spanner currently on the ground from the man's current location.
       c. If no usable spanners are on the ground (but exist elsewhere, which shouldn't happen based on domain structure), the state is unsolvable; return infinity.
       d. Add the distance to the closest usable ground spanner plus 1 (for the 'pickup_spanner' action) to the heuristic. Update the man's effective location to the spanner's location for subsequent movement calculations.
    7. Identify the set of unique locations where loose nuts are located.
    8. Calculate the distance from the man's effective location (after potential spanner pickup) to the closest location containing a loose nut. Add this distance to the heuristic (representing the walk to the first required location).
    9. Estimate the cost of walking between the remaining locations with loose nuts. A simple estimate is the number of unique loose nut locations minus one (if there is more than one unique location). Add this value to the heuristic.
    10. Return the total accumulated heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        self.static = task.static
        initial_state = task.initial_state # Access initial state for static info

        # Build location graph and compute shortest paths
        self.location_graph = {}
        all_locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1) # Links are bidirectional
                all_locations.add(l1)
                all_locations.add(l2)

        self.shortest_paths = {}
        for loc in all_locations:
            self.shortest_paths[loc] = bfs(self.location_graph, loc)

        # Identify goal nuts and their (static) locations from initial state
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}
        self.nut_locations = {}
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.goal_nuts:
                    self.nut_locations[obj] = loc

        # Identify spanners based on initial state predicates and store their initial ground locations
        self.initial_spanners = set()
        for fact in initial_state:
            if match(fact, "usable", "*"):
                self.initial_spanners.add(get_parts(fact)[1])
            if match(fact, "carrying", "*", "*"):
                 self.initial_spanners.add(get_parts(fact)[2])

        self.initial_spanner_ground_locations = {}
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.initial_spanners:
                    self.initial_spanner_ground_locations[obj] = loc

        # Identify the man (assuming the object carrying spanners initially is the man)
        self.man_name = None
        for fact in initial_state:
            if match(fact, "carrying", "*", "*"):
                self.man_name = get_parts(fact)[1]
                break
        # Fallback if man is not carrying anything initially (less robust)
        if self.man_name is None:
             # Attempt to find a locatable object that isn't a known nut or spanner
             initial_locatables = {get_parts(fact)[1] for fact in initial_state if match(fact, "at", "*", "*")}
             all_initial_items = set(self.nut_locations.keys()) | self.initial_spanners
             potential_men = initial_locatables - all_initial_items
             if len(potential_men) == 1:
                 self.man_name = list(potential_men)[0]
             else:
                 # This heuristic might not work if the man cannot be identified.
                 # In a real planner, object types would be available.
                 print("Warning: Could not reliably identify the man agent.")
                 # Set to None, heuristic might fail later if man_name is needed.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find loose nuts
        loose_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}
        num_loose_nuts = len(loose_nuts)

        # If no loose nuts, goal is reached
        if num_loose_nuts == 0:
            return 0

        # Initialize heuristic with cost for tighten actions
        h = num_loose_nuts

        # Find man's current state
        # Check if man_name was successfully identified
        if self.man_name is None:
             # Cannot proceed without man_name
             return float('inf') # Indicate unsolvable or invalid setup

        # Find man's current location
        man_loc = None
        for fact in state:
             if match(fact, "at", self.man_name, "*"):
                 man_loc = get_parts(fact)[2]
                 break
        if man_loc is None:
             # Man must be somewhere if problem is valid
             return float('inf') # Indicate invalid state

        # Find spanners carried by man and usable spanners
        man_carried_spanners = set()
        current_usable = set()
        spanner_current_ground_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "carrying" and parts[1] == self.man_name:
                spanner = parts[2]
                # Check if this spanner is known from initial state (robustness)
                if spanner in self.initial_spanners:
                    man_carried_spanners.add(spanner)
            elif parts[0] == "usable":
                spanner = parts[1]
                 # Check if this spanner is known from initial state
                if spanner in self.initial_spanners:
                    current_usable.add(spanner)
            elif parts[0] == "at" and parts[1] in self.initial_spanners: # Check against known spanners
                 spanner_current_ground_locations[parts[1]] = parts[2]

        man_has_usable_spanner = any(s in current_usable for s in man_carried_spanners)

        effective_start_loc_for_nuts = man_loc
        cost_to_get_spanner_and_walk = 0

        # Cost to get a usable spanner if needed (i.e., if there are loose nuts remaining and man doesn't have one)
        if num_loose_nuts > 0 and not man_has_usable_spanner:
            # Check if any usable spanners exist at all in the current state
            all_usable_spanners_in_state = man_carried_spanners.intersection(current_usable) | {s for s in current_usable if s in spanner_current_ground_locations}
            if not all_usable_spanners_in_state:
                 # No usable spanners available anywhere, and man needs one. Unsolvable.
                 return float('inf')

            usable_spanners_on_ground = {s for s in current_usable if s in spanner_current_ground_locations}

            if not usable_spanners_on_ground:
                 # Man needs a spanner, but none are on the ground.
                 # Given the domain, the man can only pick up spanners from the ground.
                 # If he needs one and none are on the ground, it's unsolvable for him to get one.
                 return float('inf') # No usable spanners on the ground to pick up

            # Find the closest usable spanner on the ground from man's current location
            closest_spanner_dist = float('inf')
            closest_spanner_loc = None

            for spanner in usable_spanners_on_ground:
                spanner_loc = spanner_current_ground_locations.get(spanner) # Use .get for safety
                if spanner_loc is None: continue # Should not happen if logic is correct

                dist = self.shortest_paths.get(man_loc, {}).get(spanner_loc)
                if dist is not None and dist < closest_spanner_dist:
                     closest_spanner_dist = dist
                     closest_spanner_loc = spanner_loc

            if closest_spanner_loc is not None:
                cost_to_get_spanner_and_walk = closest_spanner_dist + 1 # Walk to spanner + pickup
                effective_start_loc_for_nuts = closest_spanner_loc # Man is now at spanner_loc
            else:
                 # Should not happen if usable_spanners_on_ground was not empty
                 return float('inf') # Error state or unsolvable

        # Add the cost to get the spanner (if needed)
        h += cost_to_get_spanner_and_walk

        # Cost to move man to nut locations
        loose_nut_locations = {self.nut_locations[nut] for nut in loose_nuts}

        # Find the closest loose nut location from the effective starting location
        min_dist_to_any_nut_loc = float('inf')
        if loose_nut_locations: # Only calculate if there are loose nuts (which is true if we are here)
            for nut_loc in loose_nut_locations:
                dist = self.shortest_paths.get(effective_start_loc_for_nuts, {}).get(nut_loc)
                if dist is not None and dist < min_dist_to_any_nut_loc:
                    min_dist_to_any_nut_loc = dist

        if min_dist_to_any_nut_loc != float('inf'):
             h += min_dist_to_any_nut_loc # Add cost to reach the first nut location
        else:
             # If there were nuts but none reachable from effective_start_loc
             if loose_nut_locations:
                 return float('inf')


        # Add cost for subsequent nut locations.
        # Estimate: number of additional unique locations to visit.
        num_unique_loose_nut_locations = len(loose_nut_locations)
        if num_unique_loose_nut_locations > 1:
            # Add estimated moves between the remaining locations.
            # A simple estimate is the number of remaining locations minus 1.
            h += num_unique_loose_nut_locations - 1 # Cost for moving between nut locations

        return h
