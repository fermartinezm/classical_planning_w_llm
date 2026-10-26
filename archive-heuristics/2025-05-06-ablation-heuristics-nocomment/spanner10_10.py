import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated costs of the remaining high-level tasks: tightening
        each loose goal nut, picking up necessary spanners, and walking
        to the required locations. It counts the number of loose goal nuts
        (representing tighten actions), the number of additional usable
        spanners the man needs to pick up (representing pickup actions),
        and estimates the walking cost as the sum of shortest path distances
        from the man's current location to each location containing a loose
        goal nut or one of the closest available usable spanners needed.

    Assumptions:
        - Links between locations are bidirectional for walking.
        - All goal nuts are initially loose (or become loose if not in initial state, though domain doesn't show this). The heuristic only counts currently loose goal nuts.
        - There are enough usable spanners in the domain (either carried or at locations) to tighten all goal nuts. If not, the heuristic returns infinity.
        - The shortest path distance between any two linked locations is 1.
        - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
        1. Identify all locations from the task objects.
        2. Identify the man object from the task objects.
        3. Build an adjacency list representation of the location graph based on
           `(link ?l1 ?l2)` static facts, assuming bidirectionality.
        4. Compute all-pairs shortest paths between all locations using BFS
           starting from each location. Store these distances in a dictionary `self.dist[loc1][loc2]`.
        5. Identify the set of goal nuts from `task.goals`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state and the man's current location. If man's location is unknown or unreachable, return infinity.
        2. Identify all nuts that are goals (`(tightened nut_i)` in `task.goals`)
           and are currently loose (`(loose nut_i)` in the state). Count these
           as `N_loose_goals`. This contributes `N_loose_goals` to the heuristic
           (representing the `tighten_nut` actions). If `N_loose_goals` is 0, return 0.
        3. Find the locations of these loose goal nuts (`LooseNutLocations`). If any loose goal nut location is unknown or unreachable, return infinity.
        4. Identify usable spanners the man is currently carrying (`CarriedUsableSpanners`).
           Count these as `N_carrying_usable`.
        5. Identify all usable spanners currently at locations (`AvailableUsableSpannersAtLoc`) and their locations.
        6. Check if the total number of usable spanners available (carried + at locations) is sufficient for the remaining loose goal nuts. If `N_carrying_usable + |AvailableUsableSpannersAtLoc| < N_loose_goals`, return infinity (unsolvable regarding spanners).
        7. Calculate the number of additional usable spanners the man needs to
           pick up from locations: `NeededSpannersToPickUp = max(0, N_loose_goals - N_carrying_usable)`.
           This contributes `NeededSpannersToPickUp` to the heuristic (representing the `pickup_spanner` actions).
        8. If `NeededSpannersToPickUp > 0`, find the `NeededSpannersToPickUp` usable spanners at locations that are closest to the man's current location. Collect their locations into `ClosestSpannerLocations`. If not enough usable spanners are available at reachable locations, this step is constrained by the number available, but the unsolvable case is already handled in step 6.
        9. The set of target locations the man needs to visit is the union of
           `LooseNutLocations` and `ClosestSpannerLocations`.
        10. Estimate the walking cost (`walk_h`). This is the sum of shortest path distances from the man's current location to each distinct location in the set of target locations. If any target location is unreachable, return infinity.
           `walk_h = sum(self.dist[ManLoc][L] for L in TargetLocations)`
        11. The total heuristic value is `N_loose_goals + NeededSpannersToPickUp + walk_h`.
    """
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify all locations and the man object from task objects
        self.locations = set()
        self.man_obj = None
        self.goal_nuts = set()

        try:
            self.locations = {obj for obj, type_name in task.objects.items() if type_name == 'location'}
            # Assuming there is exactly one man object
            self.man_obj = next(obj for obj, type_name in task.objects.items() if type_name == 'man')
        except (AttributeError, StopIteration):
             # If task.objects is not available or types are unexpected, heuristic cannot proceed reliably.
             # This should ideally be handled by the planner setup ensuring task object structure.
             # For robustness, we could try inferring, but it's brittle. Returning a high value
             # or raising an error might be better if essential info is missing.
             # Let's assume task.objects is reliable as per problem description context.
             print("Error: Could not get locations or man object from task.objects.")
             # In a real scenario, might raise an exception or handle differently.
             # For this problem, we proceed assuming task.objects was populated correctly.
             pass # Continue, hoping inference or other parts work, though unlikely.


        # 3. Build adjacency list for location graph
        self.adj = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:3]
                # Only add links if both endpoints are identified locations
                if l1 in self.locations and l2 in self.locations:
                    self.adj[l1].add(l2)
                    self.adj[l2].add(l1) # Assuming bidirectional links

        # 4. Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

        # 5. Identify goal nuts
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                self.goal_nuts.add(nut)

    def _bfs(self, start_node):
        """Helper to perform BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not a known location, cannot compute distances
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.adj: # Check if node has neighbors in the graph
                for neighbor in self.adj[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state

        # 1. Get man's current location
        man_loc = None
        # Ensure self.man_obj was successfully identified in __init__
        if self.man_obj:
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        if man_loc is None or man_loc not in self.locations or man_loc not in self.dist:
             # Man's location not found, not a known location, or not in distance matrix
             return float('inf') # Invalid or unreachable state

        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts = set()
        nut_locations = {} # {nut: location}
        nut_current_status = {} # {nut: 'loose' or 'tightened'}
        nut_current_location = {} # {nut: location}

        # First pass to get all nut statuses and locations
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'loose' and len(parts) > 1:
                  nut_current_status[parts[1]] = 'loose'
             elif parts[0] == 'tightened' and len(parts) > 1:
                  nut_current_status[parts[1]] = 'tightened'
             elif match(fact, "at", "*", "*") and len(parts) > 2:
                  # Need to check if the object is a nut. Rely on goal_nuts set.
                  obj, loc = parts[1:3]
                  if obj in self.goal_nuts: # Assuming goal nuts are the only relevant nuts
                       nut_current_location[obj] = loc


        for nut in self.goal_nuts:
            # Check if it's a goal nut and is currently loose
            if nut_current_status.get(nut) == 'loose':
                 loose_goal_nuts.add(nut)
                 if nut in nut_current_location:
                     nut_locations[nut] = nut_current_location[nut]
                 else:
                     # Loose goal nut location not found - invalid state?
                     return float('inf') # Loose goal nut must have a location

        N_loose_goals = len(loose_goal_nuts)

        # If no loose goal nuts, goal is reached for nuts
        if N_loose_goals == 0:
            return 0

        # 3. Identify usable spanners carried by man
        carried_spanners = set()
        usable_spanners_set = set() # Set of spanner names that are usable
        spanner_current_location = {} # {spanner: location}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "carrying", self.man_obj, "*") and len(parts) > 2:
                carried_spanners.add(parts[2])
            elif parts[0] == 'usable' and len(parts) > 1:
                 usable_spanners_set.add(parts[1])
            elif match(fact, "at", "*", "*") and len(parts) > 2:
                 # Need to check if the object is a spanner. Rely on usable_spanners_set for candidates.
                 obj, loc = parts[1:3]
                 if obj in usable_spanners_set: # Assuming usable objects at locations are spanners
                      spanner_current_location[obj] = loc


        carried_usable_spanners = carried_spanners.intersection(usable_spanners_set)
        N_carrying_usable = len(carried_usable_spanners)

        # 5. Identify usable spanners at locations
        available_usable_spanners_at_loc = {
            spanner: loc for spanner, loc in spanner_current_location.items()
            if spanner in usable_spanners_set and spanner not in carried_spanners
        }

        # 6. Check if total usable spanners are sufficient
        total_usable_spanners = N_carrying_usable + len(available_usable_spanners_at_loc)
        if total_usable_spanners < N_loose_goals:
             return float('inf') # Unsolvable state regarding spanners

        # 7. Calculate needed spanners to pick up
        NeededSpannersToPickUp = max(0, N_loose_goals - N_carrying_usable)

        # 8. Find locations of needed spanners closest to man
        closest_spanner_locations = set()
        if NeededSpannersToPickUp > 0:
            available_spanner_distances = [] # List of (distance, spanner_name, spanner_location)
            for spanner, loc in available_usable_spanners_at_loc.items():
                 if man_loc in self.dist and loc in self.dist[man_loc] and self.dist[man_loc][loc] != float('inf'):
                    available_spanner_distances.append((self.dist[man_loc][loc], spanner, loc))
                 # else: spanner location not reachable, ignore this spanner

            # Sort by distance
            available_spanner_distances.sort()

            # Select the locations of the NeededSpannersToPickUp closest spanners
            picked_spanners_count = 0
            selected_spanners = set() # Keep track of spanners selected
            for dist, spanner, loc in available_spanner_distances:
                 if picked_spanners_count < NeededSpannersToPickUp and spanner not in selected_spanners:
                      closest_spanner_locations.add(loc)
                      selected_spanners.add(spanner)
                      picked_spanners_count += 1
                 if picked_spanners_count == NeededSpannersToPickUp:
                      break

            # Update NeededSpannersToPickUp based on how many reachable spanners we could actually identify
            NeededSpannersToPickUp = picked_spanners_count


        # 9. Calculate target locations
        target_locations = set(nut_locations.values()) # Locations of loose goal nuts
        target_locations.update(closest_spanner_locations) # Locations of needed spanners

        # 10. Estimate walking cost
        walk_h = 0
        # Ensure man's location is valid and reachable to itself (dist[man_loc] exists)
        if man_loc not in self.dist:
             return float('inf') # Should be caught earlier, but defensive check

        for loc in target_locations:
            if loc in self.dist[man_loc] and self.dist[man_loc][loc] != float('inf'):
                walk_h += self.dist[man_loc][loc]
            else:
                # Target location not reachable from man's location - unsolvable state
                return float('inf')


        # 11. Total heuristic value
        h = N_loose_goals + NeededSpannersToPickUp + walk_h

        return h
