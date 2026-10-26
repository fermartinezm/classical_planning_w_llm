from fnmatch import fnmatch
import collections
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper functions
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

# Dummy Heuristic base class definition if running standalone for testing
# In a real planning environment, this would be imported.
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It sums the estimated costs for:
    1. Performing the 'tighten_nut' action for each loose goal nut.
    2. Performing 'pickup_spanner' actions to acquire enough usable spanners.
    3. Moving the man to visit the necessary locations (spanner pickup spots and nut locations).

    # Assumptions:
    - The man can carry multiple spanners simultaneously.
    - Nuts are static (do not change location).
    - Links between locations are bidirectional.
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Extracts all locations and link relationships from the task definition to build a graph.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Identifies all goal nuts from the task's goal conditions.
    - Determines the static location of each nut from the initial state.
    - Identifies the man object and all spanner objects.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location. If not found, return infinity (invalid state).
    2. Identify all usable spanners, categorizing them as carried by the man or on the ground. Note the locations of spanners on the ground.
    3. Count the total number of usable spanners available (carried + on ground).
    4. Identify all goal nuts that are currently loose, and note their locations (precomputed).
    5. Count the number of loose goal nuts (`num_loose_needed`). If this is 0, the heuristic is 0 (goal achieved for nuts).
    6. If `num_loose_needed` is greater than the total number of usable spanners available, the problem is likely unsolvable from this state, return infinity.
    7. Count the number of usable spanners the man is currently carrying (`num_usable_spanners_carried`).
    8. Calculate the number of additional spanners the man needs to pick up from the ground: `num_spanners_to_pickup = max(0, num_loose_needed - num_usable_spanners_carried)`. Each pickup action acquires one spanner.
    9. The base heuristic cost is the sum of the number of tighten actions needed (`num_loose_needed`) and the number of pickup actions needed (`num_spanners_to_pickup`).
    10. Determine the set of locations the man *must* visit to perform necessary actions:
        - All locations of loose goal nuts (`required_nut_locations`).
        - The locations of the `num_spanners_to_pickup` usable spanners on the ground that are closest to the man's current location.
        Combine these into `locations_to_visit`.
    11. Estimate the movement cost:
        - If `locations_to_visit` is empty (only happens if `num_loose_needed` was 0, already handled), movement cost is 0.
        - Otherwise, calculate the shortest distance from the man's current location to the closest location in `locations_to_visit` (`min_dist_to_first_stop`). If the closest location is unreachable, return infinity.
        - Add an estimated cost for visiting the remaining locations, which is simply the number of remaining distinct locations (`len(locations_to_visit) - 1`). This is a simplified lower bound on the travel needed to connect the stops.
        - Total movement cost estimate = `min_dist_to_first_stop + (len(locations_to_visit) - 1)`.
    12. Add the estimated movement cost to the base heuristic cost.
    13. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals  # Goal conditions (set/frozenset of fact strings)
        static_facts = task.static  # Facts that are not affected by actions (frozenset of fact strings)
        initial_state = task.initial_state # Initial state facts (frozenset of fact strings)

        # 1. Parse locations from task.objects
        self.locations = [obj for obj, obj_type in task.objects.items() if obj_type == 'location']

        # 2. Parse link facts from task.static and build graph adjacency list
        adj = {loc: [] for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                if l1 in adj and l2 in adj: # Ensure locations are valid
                    adj[l1].append(l2)
                    adj[l2].append(l1) # Assuming links are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            while queue:
                current_loc, dist = queue.popleft()
                self.distances[start_loc][current_loc] = dist
                for neighbor in adj.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Handle unreachable locations (set distance to infinity)
        for l1 in self.locations:
            for l2 in self.locations:
                if l2 not in self.distances[l1]:
                    self.distances[l1][l2] = float('inf')

        # 4. Parse goal nuts from task.goals
        self.goal_nuts = set()
        # task.goals is a set/frozenset of goal facts (strings)
        for goal_fact_str in self.goals:
            predicate, *args = get_parts(goal_fact_str)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

        # 5. Parse initial nut locations from task.initial_state (nuts are static)
        self.nut_locations = {}
        # Need to identify all nut objects first
        all_nuts = [obj for obj, obj_type in task.objects.items() if obj_type == 'nut']
        for fact in initial_state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and args[0] in all_nuts:
                nut, loc = args
                self.nut_locations[nut] = loc

        # Identify the man object (assuming there's exactly one)
        man_objects = [obj for obj, obj_type in task.objects.items() if obj_type == 'man']
        if not man_objects:
             raise ValueError("No man object found in the problem definition.")
        self.man = man_objects[0]

        # Identify all spanner objects
        self.all_spanners = [obj for obj, obj_type in task.objects.items() if obj_type == 'spanner']


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify man's current location.
        man_location = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man location not found, problem state? Return infinity.
             return float('inf')

        # 2. Identify usable spanners (carried and on ground) and their locations.
        usable_spanners_carried = []
        usable_spanners_on_ground = [] # List of (spanner, location) tuples
        usable_spanners_available_count = 0

        for spanner in self.all_spanners:
            is_usable = f"(usable {spanner})" in state
            if is_usable:
                usable_spanners_available_count += 1
                is_carried = f"(carrying {self.man} {spanner})" in state
                if is_carried:
                    usable_spanners_carried.append(spanner)
                else:
                    # Find location if on ground
                    spanner_loc = None
                    for fact in state:
                        if match(fact, "at", spanner, "*"):
                            spanner_loc = get_parts(fact)[2]
                            break
                    if spanner_loc:
                        usable_spanners_on_ground.append((spanner, spanner_loc))
                    # Note: If usable but not carried and not at a location, it's an invalid state.
                    # We assume valid states for heuristic computation.


        # 3. Identify loose goal nuts and their locations.
        loose_goal_nuts_info = [] # List of (nut, location) tuples
        for nut in self.goal_nuts:
            if f"(loose {nut})" in state:
                # Get nut location (static, precomputed)
                nut_loc = self.nut_locations.get(nut)
                if nut_loc: # Ensure nut location was found in initial state
                    loose_goal_nuts_info.append((nut, nut_loc))
                # Note: If a goal nut's location wasn't in initial state, something is wrong with the problem definition.


        # 4. Count num_loose_needed. If 0, return 0.
        num_loose_needed = len(loose_goal_nuts_info)
        if num_loose_needed == 0:
            return 0 # Goal reached for all nuts

        # 5. Count num_usable_available. If insufficient, return infinity.
        # num_usable_available already computed in step 2.
        if num_loose_needed > usable_spanners_available_count:
            return float('inf') # Not enough usable spanners exist in the entire problem

        # 6. Count num_usable_spanners_carried.
        num_usable_spanners_carried = len(usable_spanners_carried)

        # 7. Calculate num_spanners_to_pickup.
        # The man needs num_loose_needed spanners in hand eventually.
        # He currently has num_usable_spanners_carried.
        # He needs to pick up the difference from the ground.
        num_spanners_to_pickup = max(0, num_loose_needed - num_usable_spanners_carried)

        # 8. Initialize heuristic h = num_loose_needed (tighten) + num_spanners_to_pickup (pickup).
        h = num_loose_needed + num_spanners_to_pickup

        # 9. Identify required_nut_locations.
        required_nut_locations = {loc for nut, loc in loose_goal_nuts_info}

        # 10. Identify available_spanner_locations (on ground).
        # usable_spanners_on_ground list contains (spanner, location)

        # 11. Determine the set of locations the man needs to visit (locations_to_visit).
        locations_to_visit = set(required_nut_locations)

        if num_spanners_to_pickup > 0:
            # Find the num_spanners_to_pickup usable spanners on the ground that are closest to man_location.
            # We need their locations.
            # Sort usable spanners on ground by distance from man_location.
            usable_spanners_on_ground_sorted = sorted(
                usable_spanners_on_ground,
                key=lambda item: self.distances[man_location].get(item[1], float('inf'))
            )

            # Add the locations of the required number of closest spanners to locations_to_visit.
            # Ensure we don't try to add more locations than available usable spanners on ground.
            for i in range(min(num_spanners_to_pickup, len(usable_spanners_on_ground_sorted))):
                 spanner, loc = usable_spanners_on_ground_sorted[i]
                 locations_to_visit.add(loc)


        # 12. Calculate movement cost if locations_to_visit is not empty.
        if locations_to_visit:
            # Calculate distance to the first stop
            min_dist_to_first_stop = float('inf')
            for loc in locations_to_visit:
                 dist = self.distances[man_location].get(loc, float('inf'))
                 min_dist_to_first_stop = min(min_dist_to_first_stop, dist)

            # If the closest location is unreachable, the state is unsolvable
            if min_dist_to_first_stop == float('inf'):
                 return float('inf')

            # Calculate number of subsequent stops
            num_subsequent_stops = len(locations_to_visit) - 1

            # Add movement cost estimate
            h += min_dist_to_first_stop + num_subsequent_stops
        # else: locations_to_visit is empty, movement cost is 0 (covered by h=0 case)


        # 13. Return h.
        return h
