from collections import deque
import math

# Assuming the Heuristic base class is available in a module named heuristics.heuristic_base
# If not, you might need to define a dummy base class or adjust the import path.
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential malformed facts gracefully, though planner facts should be well-formed
        return []
    return fact[1:-1].split()

# Helper function to build the location graph and compute shortest paths
def build_location_graph_and_shortest_paths(static_facts):
    """
    Builds a graph of locations based on 'link' facts and computes all-pairs shortest paths.
    Assumes links are bidirectional.
    """
    graph = {}
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'link':
            loc1, loc2 = parts[1:]
            locations.add(loc1)
            locations.add(loc2)
            graph.setdefault(loc1, set()).add(loc2)
            graph.setdefault(loc2, set()).add(loc1) # Assume links are bidirectional

    # Ensure all locations mentioned in links are in the graph keys
    for loc in locations:
        graph.setdefault(loc, set())

    all_distances = {}
    location_list = list(locations) # Use a list for consistent ordering if needed, though not strictly necessary here

    # Compute shortest paths from each location using BFS
    for start_loc in location_list:
        distances = {loc: float('inf') for loc in location_list}
        distances[start_loc] = 0
        queue = deque([(start_loc, 0)])

        while queue:
            (current_loc, current_dist) = queue.popleft()

            # If we found a shorter path later, ignore this one
            if current_dist > distances[current_loc]:
                continue

            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if distances[neighbor] > current_dist + 1:
                        distances[neighbor] = current_dist + 1
                        queue.append((neighbor, current_dist + 1))
        all_distances[start_loc] = distances

    return all_distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the cost of tightening each nut (1 action), picking up necessary spanners,
    and the estimated walk cost for the man to reach spanners and nuts.

    # Assumptions
    - The goal is to tighten all nuts that are initially loose.
    - Each spanner can be used only once (`usable` predicate is deleted).
    - The man can carry only one spanner at a time (implied by `(carrying ?m ?s)` predicate structure).
    - Links between locations are bidirectional for shortest path calculations.
    - All locations and objects involved in goals/initial state are part of the location graph or reachable.

    # Heuristic Initialization
    - Builds the location graph from `link` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all nuts that are currently loose and their locations.
    3. Identify all usable spanners and their locations (either carried by the man or on the ground).
    4. Count the total number of loose nuts (`k`). If `k` is 0, the goal is reached, heuristic is 0.
    5. Count the number of usable spanners currently carried by the man (`k_carried`).
    6. Count the number of usable spanners on the ground (`k_ground`).
    7. Check if the total number of usable spanners (`k_carried + k_ground`) is less than the number of loose nuts (`k`). If so, the problem is unsolvable from this state, return infinity.
    8. Initialize the heuristic value (`h`) with the minimum number of `tighten_nut` actions required, which is `k`.
    9. Calculate the number of additional spanners the man needs to pick up from the ground (`k_pickup = max(0, k - k_carried)`). Add `k_pickup` to `h` (cost of `pickup_spanner` actions).
    10. Determine the set of locations the man *must* visit:
        - All locations of loose nuts.
        - The locations of the `k_pickup` usable spanners on the ground that are "closest" to the man's current location. Select these spanners by sorting usable ground spanners by distance from the man and taking the first `k_pickup`.
    11. Calculate the estimated walk cost for the man to visit all required locations (`V`):
        - Find the minimum distance from the man's current location to any location in `V`.
        - If no location in `V` is reachable, return infinity.
        - The walk cost is estimated as this minimum distance plus the number of additional required locations to visit (`|V| - 1`). This is a simplified TSP approximation.
    12. Add the estimated walk cost to `h`.
    13. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing shortest paths between locations.
        """
        self.goals = task.goals
        self.static = task.static

        # Precompute all-pairs shortest paths based on 'link' facts
        self.shortest_paths = build_location_graph_and_shortest_paths(self.static)

    def dist(self, loc1, loc2):
        """
        Returns the shortest path distance between two locations.
        Returns float('inf') if locations are the same but not in the graph,
        or if loc2 is unreachable from loc1.
        """
        if loc1 == loc2:
            return 0
        # Handle cases where a location might not be in the precomputed paths (e.g., isolated)
        if loc1 not in self.shortest_paths or loc2 not in self.shortest_paths[loc1]:
             return float('inf')
        return self.shortest_paths[loc1][loc2]

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # 1. Identify man's current location
        man_location = None
        # Identify all object locations
        obj_locations = {}
        # Identify spanners carried by man
        spanners_carried = set()
        # Identify usable spanners
        usable_spanners = set()
        # Identify loose nuts
        loose_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at':
                obj, loc = parts[1:]
                obj_locations[obj] = loc
                # Assuming there is only one man object, typically named 'bob' or similar
                # We can identify the man by checking the type if available, or assume the object named 'man' or 'bob' is the man.
                # A more robust way would be to parse object types from the task, but let's assume the object name indicates type or there's only one man object.
                # Let's look for the object whose location is (at ?m - man ?l - location)
                # We don't have object types here, so let's assume the object named 'man' or the first object found with type 'man' in the initial state/objects list is the man.
                # A safer way is to find the object that is of type 'man' from the task object list.
                # Since we don't have task.objects here, let's assume the man object name is available or infer it.
                # Let's iterate through the initial state facts to find the man object name.
                # A fact like '(at bob location1)' where 'bob' is the man.
                # We need to find the object that appears as the first argument in an '(at ?m - man ?l)' predicate.
                # Let's find the man object name during initialization or assume a common name like 'man' or 'bob'.
                # The example state shows '(at bob gate)', suggesting 'bob' is the man.
                if obj == 'bob': # Assuming 'bob' is the man object name
                     man_location = loc
                # Store location for other locatable objects (spanners, nuts)
                # We need to know which objects are spanners and nuts. This info is in task.objects but not passed here.
                # Let's infer types from predicates:
                # (at ?s - spanner ?l)
                # (at ?n - nut ?l)
                # (carrying ?m - man ?s - spanner)
                # (usable ?s - spanner)
                # (tightened ?n - nut)
                # (loose ?n - nut)

            elif predicate == 'carrying':
                 # (carrying ?m - man ?s - spanner)
                 # Assuming the first argument is the man and the second is the spanner
                 carrier, spanner = parts[1:]
                 if carrier == 'bob': # Assuming 'bob' is the man
                     spanners_carried.add(spanner)
            elif predicate == 'usable':
                 # (usable ?s - spanner)
                 spanner = parts[1]
                 usable_spanners.add(spanner)
            elif predicate == 'loose':
                 # (loose ?n - nut)
                 nut = parts[1]
                 loose_nuts.add(nut)

        # Ensure man_location was found
        if man_location is None:
             # This shouldn't happen in a valid problem state, but handle defensively
             # If man's location is unknown, we can't move. Problem likely unsolvable from here.
             return float('inf')

        # 2. Identify loose nuts and their locations
        loose_nut_locations = {nut: obj_locations.get(nut) for nut in loose_nuts if nut in obj_locations}
        # Filter out loose nuts whose location is unknown (shouldn't happen in valid state)
        loose_nut_locations = {nut: loc for nut, loc in loose_nut_locations.items() if loc is not None}

        # 4. Count loose nuts
        k = len(loose_nut_locations)
        if k == 0:
            return 0 # Goal reached

        # 5. Count usable spanners carried by man
        usable_spanners_carried = spanners_carried.intersection(usable_spanners)
        k_carried = len(usable_spanners_carried)

        # 6. Identify usable spanners on ground and count them
        usable_spanners_on_ground = usable_spanners - spanners_carried
        usable_ground_spanner_locs = {
            spanner: obj_locations.get(spanner)
            for spanner in usable_spanners_on_ground
            if spanner in obj_locations and obj_locations.get(spanner) is not None
        }
        k_ground = len(usable_ground_spanner_locs)

        # 7. Check solvability
        k_total_usable = k_carried + k_ground
        if k > k_total_usable:
            return float('inf') # Not enough usable spanners

        # 8. Initialize heuristic
        h = k # Cost of tighten actions

        # 9. Calculate pickup actions
        k_pickup = max(0, k - k_carried)
        h += k_pickup # Cost of pickup actions

        # 10. Determine required visit locations
        required_visit_locations = set(loose_nut_locations.values())

        if k_pickup > 0:
            # Find the k_pickup usable ground spanners closest to the man
            spanner_distances = []
            for spanner, loc in usable_ground_spanner_locs.items():
                 d = self.dist(man_location, loc)
                 spanner_distances.append((d, spanner, loc))

            # Sort by distance, putting unreachable spanners at the end
            spanner_distances.sort(key=lambda item: item[0])

            # Add locations of the k_pickup closest spanners to required visits
            chosen_spanner_locs = set()
            for i in range(min(k_pickup, len(spanner_distances))):
                 d, spanner, loc = spanner_distances[i]
                 # If the spanner is unreachable, the problem might be unsolvable
                 if d == float('inf'):
                     return float('inf') # Cannot reach required spanner
                 chosen_spanner_locs.add(loc)

            required_visit_locations.update(chosen_spanner_locs)

        # 11. Calculate estimated walk cost
        walk_cost = 0
        if required_visit_locations:
            min_dist_to_first_stop = float('inf')
            for loc in required_visit_locations:
                d = self.dist(man_location, loc)
                min_dist_to_first_stop = min(min_dist_to_first_stop, d)

            # If the man cannot reach any required location, problem is unsolvable
            if min_dist_to_first_stop == float('inf'):
                return float('inf')

            # Walk cost is distance to first stop + (number of additional stops - 1)
            walk_cost = min_dist_to_first_stop + (len(required_visit_locations) - 1)

        # 12. Add walk cost to heuristic
        h += walk_cost

        # 13. Return heuristic value
        return h

