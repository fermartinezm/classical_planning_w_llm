from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def find_obj_location(state, obj_name):
    """Find the location of an object in the current state."""
    for fact in state:
        if match(fact, "at", obj_name, "*"):
            return get_parts(fact)[2]
    return None # Object is not located on the ground

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It sums the number of tighten actions, the number of spanner pickup actions,
    and the minimum travel distance to reach any required location (nut or spanner).

    # Heuristic Initialization
    - Infers the man object name, all spanner names, all nut names, and all location names
      from the initial state and goal facts.
    - Extracts link facts to build the location graph.
    - Computes all-pairs shortest paths between locations using BFS.
    - Stores the set of goal nuts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts that are goal conditions and their locations.
    3. If no loose goal nuts exist, the goal is reached, heuristic is 0.
    4. Identify all usable spanners, distinguishing between carried and on-ground spanners,
       and their locations.
    5. Calculate the number of spanners needed: the number of loose goal nuts minus
       the number of usable spanners the man is currently carrying. This is the minimum
       number of additional spanners the man must pick up from the ground.
    6. If the number of needed spanners is greater than the number of available usable
       spanners on the ground, the problem is likely unsolvable (or requires complex
       interactions not modeled), return infinity.
    7. Calculate the heuristic value:
       - Add the number of loose goal nuts (each requires one tighten action).
       - Add the number of spanners that need to be picked up (each requires one pickup action).
       - Identify the set of "required locations": the locations of all loose goal nuts,
         plus the locations of the N_to_pickup closest usable spanners on the ground.
       - Calculate the minimum shortest path distance from the man's current location
         to any location in the set of required locations. Add this distance to the heuristic.
       - If any required location is unreachable, return infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Infer object names and types from initial state and goals
        self.man_obj_name = None
        self.all_spanners = set()
        self.all_nuts = set()
        self.all_locations = set()
        self.links = set()

        locations_from_facts = set()

        # Collect objects and locations from initial state and static facts
        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()

        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == "carrying":
                # (carrying ?m - man ?s - spanner)
                potential_men.add(parts[1])
                potential_spanners.add(parts[2])
            elif parts[0] == "usable":
                 # (usable ?s - spanner)
                 potential_spanners.add(parts[1])
            elif parts[0] == "loose":
                 # (loose ?n - nut)
                 potential_nuts.add(parts[1])
            elif parts[0] == "at":
                 # (at ?o - locatable ?l - location)
                 obj, loc = parts[1], parts[2]
                 locations_from_facts.add(loc)
                 # Add obj to potential sets based on its name pattern if needed,
                 # but relying on specific predicates like 'carrying', 'usable', 'loose' is more reliable.

        # Collect nuts from goals
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "tightened":
                  # (tightened ?n - nut)
                  potential_nuts.add(parts[1])

        # Infer man: Find the object that is 'carrying' something.
        man_candidates = {get_parts(fact)[1] for fact in initial_state if match(fact, "carrying", "*", "*")}
        if len(man_candidates) == 1:
            self.man_obj_name = list(man_candidates)[0]
        else:
            # If no 'carrying' fact, or multiple, try to find an object in 'at' that isn't a known spanner/nut.
            all_spanners_set = potential_spanners
            all_nuts_set = potential_nuts

            at_objects = {get_parts(fact)[1] for fact in initial_state if match(fact, "at", "*", "*")}
            man_candidates_from_at = list(at_objects - all_spanners_set - all_nuts_set)

            if man_candidates:
                 self.man_obj_name = list(man_candidates)[0]
            elif len(man_candidates_from_at) >= 1:
                 # Fallback: pick the first object in an 'at' fact that isn't a known spanner or nut.
                 self.man_obj_name = man_candidates_from_at[0]
            # else: self.man_obj_name remains None. Problematic case, assuming valid problems have a man.


        # Collect all spanner and nut names found
        self.all_spanners = potential_spanners
        self.all_nuts = potential_nuts

        # Parse static facts for links and locations
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                self.links.add((l1, l2))
                self.links.add((l2, l1)) # Links are bidirectional
                locations_from_facts.add(l1)
                locations_from_facts.add(l2)

        # All locations are those found in link facts or 'at' facts in initial state.
        self.locations = list(locations_from_facts)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            while queue:
                (current_loc, dist) = queue.popleft()
                self.distances[start_loc][current_loc] = dist
                for l1, l2 in self.links:
                    neighbor = None
                    if l1 == current_loc and l2 not in visited:
                        neighbor = l2
                    elif l2 == current_loc and l1 not in visited:
                        neighbor = l1
                    if neighbor:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Store goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "tightened"}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find man's location
        man_location = find_obj_location(state, self.man_obj_name)
        if man_location is None:
             # Man must always be at a location in this domain.
             # If not found, something is wrong with the state representation or problem.
             # Return infinity as it's likely an unsolvable path.
             return float('inf')


        # Identify loose goal nuts and their locations
        loose_goal_nuts = [] # List of (nut_name, location)
        for nut_name in self.goal_nuts:
            if f"(loose {nut_name})" in state:
                nut_location = find_obj_location(state, nut_name)
                if nut_location:
                    loose_goal_nuts.append((nut_name, nut_location))
                # else: nut is loose but not located? Problematic state.

        # If no loose goal nuts, goal is reached
        if not loose_goal_nuts:
            return 0

        # Identify usable spanners (carried and on ground)
        usable_spanners_carried = [] # List of spanner_name
        usable_spanners_on_ground = [] # List of (spanner_name, location)

        for spanner_name in self.all_spanners:
            if f"(usable {spanner_name})" in state:
                if f"(carrying {self.man_obj_name} {spanner_name})" in state:
                    usable_spanners_carried.append(spanner_name)
                else:
                    spanner_location = find_obj_location(state, spanner_name)
                    if spanner_location:
                        usable_spanners_on_ground.append((spanner_name, spanner_location))
                    # else: usable spanner exists but is not carried and not on ground? Problematic.

        # Calculate heuristic components
        h = 0

        # 1. Cost for tighten actions: Each loose goal nut needs one tighten action.
        h += len(loose_goal_nuts)

        # 2. Cost for pickup actions: Man needs N_needed_spanners total. Has N_carried_usable.
        # Needs to pick up max(0, N_needed_spanners - N_carried_usable) more from ground.
        N_needed_spanners = len(loose_goal_nuts)
        N_carried_usable = len(usable_spanners_carried)
        N_to_pickup = max(0, N_needed_spanners - N_carried_usable)

        if N_to_pickup > 0:
            # Add cost for pickup actions
            h += N_to_pickup

            # Identify locations of the N_to_pickup closest usable ground spanners
            spanner_locations_needed = set()
            if usable_spanners_on_ground:
                distances_to_spanners = [] # List of (distance, spanner_name, location)
                for s_name, s_loc in usable_spanners_on_ground:
                     # Check if spanner location is reachable from man's location
                     if man_location in self.distances and s_loc in self.distances[man_location]:
                          distances_to_spanners.append((self.distances[man_location][s_loc], s_name, s_loc))
                     # else: spanner location unreachable? Problematic, handled below.

                if len(distances_to_spanners) < N_to_pickup:
                     # Not enough usable spanners on ground or reachable. Unsolvable.
                     return float('inf')

                distances_to_spanners.sort()
                picked_spanners_info = distances_to_spanners[:N_to_pickup]

                # Collect locations of these spanners for travel cost calculation
                spanner_locations_needed = {s_loc for dist, s_name, s_loc in picked_spanners_info}
            else:
                 # Need spanners but none available on ground. Unsolvable.
                 return float('inf')


        # 3. Cost for travel: Minimum distance from man to any required location.
        # Required locations are the locations of loose goal nuts and the locations of the N_to_pickup closest usable ground spanners.
        required_locations = {nut_loc for nut_name, nut_loc in loose_goal_nuts}
        if N_to_pickup > 0:
             required_locations.update(spanner_locations_needed)

        # Calculate minimum distance from man's current location to any required location
        min_dist_to_target = float('inf')
        for target_loc in required_locations:
            # Check if target location is reachable from man's location
            if man_location in self.distances and target_loc in self.distances[man_location]:
                min_dist_to_target = min(min_dist_to_target, self.distances[man_location][target_loc])
            else:
                 # Target location unreachable? Unsolvable.
                 return float('inf')

        # Add the minimum travel cost
        if min_dist_to_target != float('inf'):
            h += min_dist_to_target
        # else: No required locations? This should only happen if loose_goal_nuts is empty, which is handled at the start.

        return h

