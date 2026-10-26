# from heuristics.heuristic_base import Heuristic # Assuming this base class exists
from collections import deque
# from fnmatch import fnmatch # Not strictly needed with get_parts

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Assuming Heuristic base class is defined elsewhere and provides __init__(self, task) and __call__(self, node)
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#         self.objects = task.objects # Assuming task provides objects with types
#         self.initial_state = task.init # Assuming task provides initial state facts

class spannerHeuristic: # Inherit from Heuristic if available, otherwise define as standalone
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It considers the man's current location, whether he is
    carrying a usable spanner, the locations of loose goal nuts, and the
    locations of usable spanners on the ground. It estimates the cost
    for the first nut based on reaching the nearest required resource
    (spanner or nut) and then adds a fixed cost for each subsequent nut,
    representing the actions needed to acquire another spanner, travel,
    and tighten.

    # Assumptions:
    - Links between locations are bidirectional.
    - The man can carry at most one spanner at a time.
    - The `tighten_nut` action consumes the spanner's usability and
      effectively makes it no longer carried, allowing the man to pick up
      another spanner for the next nut.
    - Sufficient usable spanners exist in the initial state to tighten all
      goal nuts (checked in initialization).
    - All locations mentioned in facts and links are part of a single
      connected graph, or relevant locations are reachable.

    # Heuristic Initialization
    - Extract all locations and links from static facts to build the
      connectivity graph.
    - Calculate All-Pairs Shortest Paths (APSP) between all locations
      using Breadth-First Search (BFS).
    - Identify all goal nuts from the task's goal conditions.
    - Count the total number of usable spanners available in the initial state
      to perform a basic solvability check.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Determine if the man is currently carrying a usable spanner.
    3. Identify all nuts that are currently loose and are part of the goal.
    4. Identify all spanners that are currently usable and are on the ground.
    5. If there are no loose goal nuts, the heuristic value is 0.
    6. Perform a basic solvability check: If the number of loose goal nuts
       exceeds the total number of usable spanners available in the initial
       state, return infinity (or a very large number).
    7. Calculate the heuristic value:
       - Initialize heuristic `h = 0`.
       - Estimate the cost for the *first* nut/trip:
         - If the man is carrying a usable spanner: Cost = shortest distance
           from the man's current location to the nearest loose goal nut location,
           plus 1 (tighten). If no loose goal nuts, this part is skipped (covered by step 5).
         - If the man is not carrying a usable spanner: Find the closest usable
           spanner location on the ground. If none exist, return infinity. Cost =
           shortest distance from the man's location to this spanner location,
           plus 1 (pickup), plus the shortest distance from the spanner location
           to the nearest loose goal nut location, plus 1 (tighten). If no loose
           goal nuts or no usable spanners on ground, return infinity (covered by checks).
       - Add this cost for the first trip to `h`.
       - Estimate the cost for the remaining `N_loose - 1` nuts/trips:
         - Each remaining nut requires acquiring a spanner, traveling to the nut,
           and tightening it. This involves a `pickup_spanner` action, travel
           from the previous nut location to a spanner, and travel from the
           spanner to the next nut location, and the `tighten_nut` action.
         - A simplified estimate for these remaining nuts is a fixed cost per nut.
           Based on typical plan structures (pickup, travel S->N, tighten, travel N->S),
           a cost of 4 actions per remaining nut (1 pickup + 1 tighten + ~2 travel)
           is used as an approximation. Add `max(0, N_loose - 1) * 4` to `h`.
    8. Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        # Store task details needed for heuristic calculation
        self.goals = task.goals
        self.static = task.static
        self.objects = task.objects # Assuming task provides objects with types
        self.initial_state = task.init # Assuming task provides initial state facts

        # Identify object types (assuming unique man)
        self.man = None
        for name, type in self.objects.items():
            if type == 'man':
                self.man = name
                break
        if self.man is None:
             # Handle case with no man, though domain implies one
             # print("Warning: No man object found in task.")
             pass # Avoid printing during heuristic initialization in a planner

        self.all_nuts = {name for name, type in self.objects.items() if type == 'nut'}
        self.all_spanners = {name for name, type in self.objects.items() if type == 'spanner'}
        self.all_locations = {name for name, type in self.objects.items() if type == 'location'}

        # Extract goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal) and get_parts(goal)[0] == 'tightened'}

        # Extract locations and links from static facts
        self.locations = set()
        self.links = {} # Adjacency list {loc: [neighbor1, neighbor2, ...]}

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.links.setdefault(loc1, []).append(loc2)
                self.links.setdefault(loc2, []).append(loc1) # Assume bidirectional links

        # Calculate All-Pairs Shortest Paths (APSP) using BFS
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                current_node, d = q.popleft()
                self.dist[start_node][current_node] = d
                if current_node in self.links:
                    for neighbor in self.links[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))

        # Count initial usable spanners for a basic solvability check
        self.initial_usable_spanners_count = 0
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'usable' and len(parts) == 1 and parts[1] in self.all_spanners:
                self.initial_usable_spanners_count += 1


    def get_distance(self, loc1, loc2):
        """Get shortest distance between two locations."""
        if loc1 is None or loc2 is None:
             return float('inf') # Cannot calculate distance if location is unknown
        if loc1 in self.dist and loc2 in self.dist[loc1]:
            return self.dist[loc1][loc2]
        # Return a large number to represent unreachable.
        return float('inf')

    def __call__(self, node):
        state = node.state

        # Find man's current location
        man_location = None
        carrying_spanner = None # Store the spanner object name if carried
        usable_spanners_in_state = set()
        loose_nuts_in_state = set()
        obj_locations = {} # {obj_name: location}

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                obj_locations[obj] = loc
                if obj == self.man:
                    man_location = loc
            elif predicate == 'carrying' and len(args) == 2:
                 carrier, carried_obj = args[0], args[1]
                 if carrier == self.man:
                     carrying_spanner = carried_obj
            elif predicate == 'usable' and len(args) == 1:
                usable_spanners_in_state.add(args[0])
            elif predicate == 'loose' and len(args) == 1:
                loose_nuts_in_state.add(args[0])

        # Identify loose goal nuts in the current state
        loose_goal_nuts = self.goal_nuts.intersection(loose_nuts_in_state)
        num_loose_goal_nuts = len(loose_goal_nuts)

        # If all goal nuts are tightened, heuristic is 0
        if num_loose_goal_nuts == 0:
            return 0

        # Basic solvability check: Not enough spanners ever existed
        # This is a necessary condition, but not sufficient (e.g., spanners unreachable)
        # However, for a non-admissible heuristic, this is acceptable for GBFS.
        if num_loose_goal_nuts > self.initial_usable_spanners_count:
             return float('inf')

        # Check if man location is known
        if man_location is None:
             # Man's location is unknown, cannot calculate travel costs.
             # This shouldn't happen in a valid state representation from a planner,
             # but handle defensively.
             return float('inf')


        h = 0

        # Determine if the man is carrying a usable spanner
        usable_carried_spanner = (carrying_spanner is not None) and (carrying_spanner in usable_spanners_in_state)

        # Locations of usable spanners available on the ground
        ground_usable_spanner_locs = {
            obj_locations[s] for s in usable_spanners_in_state
            if s in obj_locations and s != carrying_spanner # Spanner must be at a location and not carried
        }

        # Locations of loose goal nuts
        loose_goal_nut_locs = {obj_locations[n] for n in loose_goal_nuts if n in obj_locations}

        # --- Estimate cost for the first nut/trip ---
        cost_first_trip = float('inf')

        if usable_carried_spanner:
            # Man has spanner, needs to go to a nut.
            # Cost = Travel to nearest loose goal nut + 1 (tighten)
            if loose_goal_nut_locs:
                min_dist_M_N = float('inf')
                for loc in loose_goal_nut_locs:
                     min_dist_M_N = min(min_dist_M_N, self.get_distance(man_location, loc))
                if min_dist_M_N != float('inf'):
                    cost_first_trip = min_dist_M_N + 1 # Travel + Tighten
        else: # Not carrying a usable spanner
            # Man needs spanner, go to nearest usable spanner on ground, pickup, go to nearest nut, tighten.
            # Cost = Travel(M to S) + 1 (pickup) + Travel(S to N) + 1 (tighten)
            if ground_usable_spanner_locs and loose_goal_nut_locs:
                # Find the closest usable spanner location on the ground
                min_dist_M_S_ground_val = float('inf')
                closest_spanner_loc = None
                for loc in ground_usable_spanner_locs:
                     dist = self.get_distance(man_location, loc)
                     if dist < min_dist_M_S_ground_val:
                          min_dist_M_S_ground_val = dist
                          closest_spanner_loc = loc

                if closest_spanner_loc is not None:
                    # Find the nearest loose goal nut location from the closest spanner location
                    min_dist_S_N = float('inf')
                    for loc in loose_goal_nut_locs:
                         min_dist_S_N = min(min_dist_S_N, self.get_distance(closest_spanner_loc, loc))

                    if min_dist_M_S_ground_val != float('inf') and min_dist_S_N != float('inf'):
                         cost_first_trip = min_dist_M_S_ground_val + 1 + min_dist_S_N + 1 # Travel M->S + Pickup + Travel S->N + Tighten

        # If cost_first_trip is still infinity, it means the first step is impossible
        if cost_first_trip == float('inf'):
             return float('inf')

        h += cost_first_trip

        # --- Estimate cost for remaining nuts ---
        # Each remaining nut requires: pickup, travel to nut, tighten.
        # Simplified cost per remaining nut: 1 (pickup) + 1 (tighten) + ~2 (travel segments). Total = 4.
        # This assumes there are enough usable spanners on the ground for the remaining nuts.
        # The initial spanner count check helps, but doesn't guarantee reachability.
        h += max(0, num_loose_goal_nuts - 1) * 4

        return h
