# Assume Heuristic base class is available as heuristics.heuristic_base.Heuristic
# from heuristics.heuristic_base import Heuristic

# If the base class is not provided, define a dummy one for testing/structure
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass

from collections import deque

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function for BFS
def bfs(start_loc, links, all_locations):
    """Computes shortest path distances from start_loc to all other locations."""
    distances = {loc: float('inf') for loc in all_locations}

    # Build adjacency list for graph considering only linked locations
    adj = {loc: [] for loc in all_locations}
    graph_locations = set()
    for l1, l2 in links:
        adj[l1].append(l2)
        adj[l2].append(l1) # Links are bidirectional
        graph_locations.add(l1)
        graph_locations.add(l2)

    if start_loc not in all_locations:
        # Start location is not even a known location object
        return distances # All distances remain inf

    if start_loc not in graph_locations:
         # Start location is a known location object but is isolated (not in graph)
         distances[start_loc] = 0 # Distance to itself is 0
         return distances # Cannot reach anything else in the graph

    # Start BFS from a location within the graph
    distances[start_loc] = 0
    queue = deque([start_loc])
    visited = {start_loc}

    while queue:
        current_loc = queue.popleft()

        for neighbor in adj.get(current_loc, []):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current_loc] + 1
                queue.append(neighbor)

    return distances

# Helper function to compute all-pairs shortest paths
def compute_shortest_paths(static_facts, all_locations):
    """Computes shortest path distances between all pairs of locations."""
    links = set()
    # Extract links from static facts
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'link':
            l1, l2 = parts[1], parts[2]
            links.add((l1, l2))

    all_distances = {}
    for start_loc in all_locations:
        all_distances[start_loc] = bfs(start_loc, links, all_locations)

    return all_distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the cost of walking the man to nut locations and spanner locations,
    picking up spanners, and tightening nuts. It uses a greedy approach, always
    prioritizing the closest available spanner and then the closest remaining nut.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Each nut requires one usable spanner, and a spanner becomes unusable after one use.
    - The man can only carry one spanner at a time.
    - The problem is solvable (enough usable spanners exist, locations are reachable).

    # Heuristic Initialization
    - Extracts all location objects from the task.
    - Computes the shortest path distances between all pairs of locations based on the 'link' predicates using BFS.
    - Identifies the man object name and the set of nuts that need to be tightened (from the goal).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that are currently 'loose' and are part of the goal, along with their locations. These are the nuts that still need tightening.
    2. If there are no such loose nuts, the heuristic is 0 (goal state).
    3. Identify the man's current location.
    4. Identify all spanners that are currently 'usable' and their locations (either on the ground or being carried by the man).
    5. Initialize the total estimated cost to 0.
    6. Check if the man is currently carrying a usable spanner.
    7. If the man is carrying a usable spanner:
       - This spanner can be used for the first nut.
       - Find the loose nut (from the goal set) closest to the man's current location.
       - Add the walk distance from the man's location to this nut's location to the cost.
       - Add 1 to the cost for the 'tighten_nut' action.
       - Update the man's current location to the nut's location.
       - Mark the carried spanner as used (conceptually, remove it from available usable spanners).
       - Remove the tightened nut from the list of remaining loose nuts.
    8. While there are still loose nuts remaining:
       - The man needs a new usable spanner.
       - Find the closest available usable spanner on the ground to the man's current location.
       - If no usable spanners are available on the ground, the state is likely unsolvable; return infinity.
       - Add the walk distance from the man's current location to the spanner's location to the cost.
       - Add 1 to the cost for the 'pickup_spanner' action.
       - Update the man's current location to the spanner's location.
       - Mark the picked-up spanner as used.
       - Now that the man has a spanner, find the closest remaining loose nut to the man's *current* location (where the spanner was picked up).
       - Add the walk distance from the man's current location to this nut's location to the cost.
       - Add 1 to the cost for the 'tighten_nut' action.
       - Update the man's current location to the nut's location.
       - Remove the tightened nut from the list of remaining loose nuts.
    9. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Extract all location objects from the task
        # Assuming task.objects is a dict mapping type names to lists of object names
        all_locations = set(task.objects.get('location', []))

        # Compute all-pairs shortest paths between locations
        self.all_distances = compute_shortest_paths(task.static, all_locations)

        # Store goal nuts for quick lookup
        self.goal_nuts = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'tightened' and len(parts) == 2:
                 self.goal_nuts.add(parts[1])

        # Find the man object name (assuming only one man)
        self.man_name = task.objects.get('man', [None])[0]
        # If task.objects doesn't contain 'man' or is empty, this will be None.
        # The heuristic will return inf if man_location cannot be found in __call__.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- State Parsing ---
        object_locations = {} # Map object_name to location_name
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 object_locations[obj_name] = loc_name

        loose_nuts_list = [] # List of (nut_name, location_name) for goal nuts that are loose
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'loose' and len(parts) == 2:
                nut_name = parts[1]
                # Only consider loose nuts that are part of the goal
                if nut_name in self.goal_nuts:
                    loc_name = object_locations.get(nut_name)
                    if loc_name:
                        loose_nuts_list.append((nut_name, loc_name))
                    # else: loose nut in goal but location unknown? Invalid state.

        # If no loose nuts that are in the goal, goal is reached
        if not loose_nuts_list:
            return 0

        man_location = object_locations.get(self.man_name)
        if man_location is None:
             # Man not at a location? Invalid state or unreachable.
             return float('inf')

        usable_spanners = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'usable' and len(parts) == 2:
                usable_spanners.add(parts[1])

        usable_spanners_on_ground_list = [] # List of (spanner_name, location_name)
        carried_usable_spanner = None # spanner_name or None

        spanner_being_carried = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and len(parts) == 3 and parts[1] == self.man_name:
                 spanner_being_carried = parts[2]
                 break # Man can only carry one spanner

        if spanner_being_carried in usable_spanners:
             carried_usable_spanner = spanner_being_carried

        for spanner_name in usable_spanners:
            if spanner_name != carried_usable_spanner: # Only consider usable spanners not carried
                 loc_name = object_locations.get(spanner_name)
                 if loc_name:
                     usable_spanners_on_ground_list.append((spanner_name, loc_name))
                 # else: usable spanner on ground but location unknown? Invalid state.

        # --- Heuristic Calculation ---
        cost = 0
        current_man_location = man_location
        remaining_loose_nuts = list(loose_nuts_list)
        available_usable_spanners_on_ground = list(usable_spanners_on_ground_list)
        man_has_usable_spanner = (carried_usable_spanner is not None)

        # Handle the spanner the man might be carrying first
        if man_has_usable_spanner:
            # Use this spanner for the closest loose nut
            if not remaining_loose_nuts:
                 # Should not happen if heuristic > 0 but no nuts remain
                 pass
            else:
                closest_nut_info = None
                min_dist = float('inf')
                for nut_info in remaining_loose_nuts:
                    nut_loc = nut_info[1]
                    dist = self.all_distances.get(current_man_location, {}).get(nut_loc, float('inf'))
                    if dist == float('inf'):
                         # This nut is unreachable from current location
                         continue
                    if dist < min_dist:
                        min_dist = dist
                        closest_nut_info = nut_info

                if closest_nut_info is None or min_dist == float('inf'):
                    # All remaining nuts are unreachable
                    return float('inf')

                # Walk to the nut
                cost += min_dist
                current_man_location = closest_nut_info[1]

                # Tighten the nut
                cost += 1
                remaining_loose_nuts.remove(closest_nut_info)
                man_has_usable_spanner = False # Spanner is used

        # Now, for every remaining loose nut, the man needs to get a spanner first.
        while remaining_loose_nuts:
            # Need to get a spanner
            if not available_usable_spanners_on_ground:
                # Not enough spanners available on the ground for remaining nuts
                return float('inf') # Unsolvable state

            # Find the closest available spanner on the ground
            closest_spanner_info = None
            min_dist_spanner = float('inf')
            for spanner_info in available_usable_spanners_on_ground:
                spanner_loc = spanner_info[1]
                dist = self.all_distances.get(current_man_location, {}).get(spanner_loc, float('inf'))
                if dist == float('inf'):
                    # This spanner is unreachable from current location
                    continue
                if dist < min_dist_spanner:
                    min_dist_spanner = dist
                    closest_spanner_info = spanner_info

            if closest_spanner_info is None or min_dist_spanner == float('inf'):
                 # All remaining spanners are unreachable
                 return float('inf')

            # Walk to the spanner
            cost += min_dist_spanner
            current_man_location = closest_spanner_info[1]

            # Pickup the spanner
            cost += 1
            available_usable_spanners_on_ground.remove(closest_spanner_info)

            # Now go to the nut
            # Find the closest remaining loose nut from the *current* location (where spanner was picked up)
            closest_nut_info = None
            min_dist_nut = float('inf')
            for nut_info in remaining_loose_nuts:
                nut_loc = nut_info[1]
                dist = self.all_distances.get(current_man_location, {}).get(nut_loc, float('inf'))
                if dist == float('inf'):
                    # This nut is unreachable from current location
                    continue
                if dist < min_dist_nut:
                    min_dist_nut = dist
                    closest_nut_info = nut_info

            if closest_nut_info is None or min_dist_nut == float('inf'):
                 # All remaining nuts are unreachable from spanner location
                 return float('inf')

            # Walk to the nut
            cost += min_dist_nut
            current_man_location = closest_nut_info[1]

            # Tighten the nut
            cost += 1
            remaining_loose_nuts.remove(closest_nut_info)

        return cost
