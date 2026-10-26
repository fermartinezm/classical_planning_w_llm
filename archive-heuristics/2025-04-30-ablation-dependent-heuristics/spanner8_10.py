import math
from collections import deque

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated
    costs of necessary actions: tightening nuts, picking up spanners, and walking.
    It identifies the set of goal nuts that are currently loose. For each such nut,
    a 'tighten' action is required. If the man is not already carrying enough
    usable spanners for the remaining nuts, additional 'pickup' actions are required
    for available usable spanners. The walk cost is estimated as the cost to travel
    from the man's current location to a set of target locations, which include the
    locations of all untightened goal nuts and the locations of the closest available
    usable spanners needed. The walk cost is approximated using a star-graph
    structure centered at the target location closest to the man.

    Assumptions:
    - Links between locations are bidirectional.
    - Nut locations are static.
    - Spanners become unusable after one 'tighten_nut' action.
    - The heuristic assumes the man needs one usable spanner per untightened goal nut.
      It accounts for usable spanners the man is already carrying.
    - Object names starting with 'nut' are nuts, 'spanner' are spanners, and 'bob' is the man.
      This is based on the provided examples, as type information isn't directly
      available from the state/task objects in this format.

    Heuristic Initialization:
    1. Parse static facts (`task.static`) and initial state (`task.initial_state`)
       to identify all locations and build the location graph based on `link` facts.
       Assume links are bidirectional.
    2. Compute all-pairs shortest paths between all locations using BFS. Store
       these distances in a dictionary `self.dist[loc1][loc2]`.
    3. Identify the static locations of all nuts from the initial state.
       Store these in `self.nut_locations[nut_name]`.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set `U` of nuts that are in the task's goal but are currently `loose`
       in the given `state`. A nut is considered loose if the corresponding
       '(tightened nut_name)' fact is not present in the state.
    2. If `U` is empty, the goal is reached, return 0.
    3. Get the man's current location (`man_loc`) from the `state`.
    4. Identify the set of usable spanners the man is currently carrying.
    5. Identify the set of usable spanners available at locations (not carried).
    6. Calculate the total number of usable spanners available in the state
       (carried + at locations).
    7. If the number of untightened goal nuts (`|U|`) is greater than the total
       number of usable spanners available, the state is likely unsolvable,
       return `float('inf')`.
    8. Calculate the number of *additional* usable spanners the man needs to acquire:
       `NeededSpannersCount = max(0, |U| - |carried_usable_spanners|)`.
    9. Identify the target locations the man needs to visit:
       - `NutLocations`: The locations of all nuts in `U`.
       - `SpannerTargetLocations`: The locations of the `NeededSpannersCount`
         usable spanners (from those available at locations) that are closest
         to the man's current location.
       - `AllTargetLocations = NutLocations | SpannerTargetLocations`.
    10. Estimate the walk cost to visit all `AllTargetLocations` starting from `man_loc`.
        This is approximated by finding the target location closest to `man_loc`,
        calculating the distance to it, and then summing the distances from this
        closest target location to all other target locations. If `AllTargetLocations`
        is empty (only if `num_nuts` and `NeededSpannersCount` are both 0, which is
        handled by the goal check), the walk cost is 0.
    11. The total heuristic value is the sum of:
        - The estimated walk cost.
        - The number of untightened goal nuts (`|U|`, representing the 'tighten' actions).
        - The number of needed spanner pickups (`NeededSpannersCount`, representing the 'pickup' actions).
    """

    def __init__(self, task):
        self.task = task
        self.dist = {}
        self.nut_locations = {}
        self._precompute_distances()
        self._identify_static_nut_locations()

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into a list of strings."""
        # Remove parentheses and split by spaces
        return fact_str.strip('()').split()

    def _precompute_distances(self):
        """Build location graph and compute all-pairs shortest paths."""
        locations = set()
        adj = {}

        # Collect all locations and build adjacency list from links
        for fact_str in self.task.static:
            parsed = self._parse_fact(fact_str)
            if parsed[0] == 'link' and len(parsed) == 3:
                loc1, loc2 = parsed[1], parsed[2]
                locations.add(loc1)
                locations.add(loc2)
                adj.setdefault(loc1, set()).add(loc2)
                adj.setdefault(loc2, set()).add(loc1) # Assume bidirectional links

        # Also collect locations from initial state (e.g., where objects start)
        for fact_str in self.task.initial_state:
             parsed = self._parse_fact(fact_str)
             if parsed[0] == 'at' and len(parsed) == 3:
                 # The third argument is the location
                 locations.add(parsed[2])

        self.locations = list(locations)
        self.dist = {loc: {l: float('inf') for l in self.locations} for loc in self.locations}

        # Compute shortest paths using BFS from each location
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            # Use a local visited set for each BFS run
            visited = {start_node}
            self.dist[start_node][start_node] = 0

            while q:
                curr_loc, d = q.popleft()

                # Get neighbors from adjacency list, handle locations with no links
                neighbors = adj.get(curr_loc, set())

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[start_node][neighbor] = d + 1
                        q.append((neighbor, d + 1))

    def _identify_static_nut_locations(self):
        """Identify the static locations of nuts from the initial state."""
        # Nut locations do not change in this domain.
        # We find them in the initial state.
        for fact_str in self.task.initial_state:
            parsed = self._parse_fact(fact_str)
            if parsed[0] == 'at' and len(parsed) == 3:
                 obj, loc = parsed[1], parsed[2]
                 # Check if the object is a nut. Assume objects starting with 'nut' are nuts.
                 if obj.startswith('nut'):
                     self.nut_locations[obj] = loc

    def get_location(self, state, obj_name):
        """Find the current location of an object in the state."""
        # Nuts have static locations
        if obj_name in self.nut_locations:
            return self.nut_locations[obj_name]

        # For other objects (man, spanner), find their location in the current state
        for fact_str in state:
            parsed = self._parse_fact(fact_str)
            if parsed[0] == 'at' and len(parsed) == 3 and parsed[1] == obj_name:
                return parsed[2]
        return None # Object not found at any location (e.g., spanner being carried)

    def is_carrying(self, state, man_name, obj_name):
        """Check if the man is carrying the object."""
        return f'(carrying {man_name} {obj_name})' in state

    def is_usable(self, state, obj_name):
         """Check if the object (spanner) is usable."""
         return f'(usable {obj_name})' in state

    def __call__(self, state):
        """
        Compute the heuristic value for the given state.
        """
        # 1. Identify untightened goal nuts
        untightened_goal_nuts = set()
        for goal_fact_str in self.task.goals:
             parsed_goal = self._parse_fact(goal_fact_str)
             if parsed_goal[0] == 'tightened' and len(parsed_goal) == 2:
                 nut_name = parsed_goal[1]
                 # Check if the corresponding '(tightened nut_name)' fact is NOT in the current state
                 if goal_fact_str not in state:
                     untightened_goal_nuts.add(nut_name)

        num_nuts = len(untightened_goal_nuts)

        # 2. If goal is reached
        if num_nuts == 0:
            return 0

        # 3. Get man's current location
        man_name = None
        # Find the man's name (assume there's only one man based on examples, name starts with 'bob')
        for fact_str in self.task.initial_state:
             parsed = self._parse_fact(fact_str)
             if parsed[0] == 'at' and len(parsed) == 3:
                 if parsed[1].startswith('bob'):
                     man_name = parsed[1]
                     break # Found the man

        if man_name is None:
             # Should not happen in valid problems
             return float('inf') # Indicates an impossible state (no man)

        man_loc = self.get_location(state, man_name)
        if man_loc is None:
             # Man must be somewhere if problem is solvable
             return float('inf') # Indicates an impossible state (man not at any location)


        # 4. Identify carried usable spanners
        carried_usable_spanners = set()
        for fact_str in state:
             parsed = self._parse_fact(fact_str)
             if parsed[0] == 'carrying' and len(parsed) == 3 and parsed[1] == man_name:
                 spanner_name = parsed[2]
                 if self.is_usable(state, spanner_name):
                     carried_usable_spanners.add(spanner_name)

        # 5. Identify usable spanners at locations
        usable_spanners_at_loc = set() # Stores (spanner_name, location)
        for fact_str in state:
             parsed = self._parse_fact(fact_str)
             if parsed[0] == 'at' and len(parsed) == 3:
                 obj_name, loc = parsed[1], parsed[2]
                 # Assume objects starting with 'spanner' are spanners
                 if obj_name.startswith('spanner'):
                     if self.is_usable(state, obj_name):
                         usable_spanners_at_loc.add((obj_name, loc))

        # 6. Calculate total usable spanners
        num_usable_spanners_total = len(carried_usable_spanners) + len(usable_spanners_at_loc)

        # 7. Check solvability based on spanners
        if num_nuts > num_usable_spanners_total:
            return float('inf')

        # 8. Calculate needed spanner pickups
        # We need num_nuts usable spanners in total.
        # If the man is carrying k usable spanners, he needs num_nuts - k more.
        needed_spanners_count = max(0, num_nuts - len(carried_usable_spanners))

        # 9. Identify target locations
        nut_locations = {self.get_location(state, nut) for nut in untightened_goal_nuts}
        
        # Select the locations of the needed_spanners_count usable spanners at locations
        # that are closest to the man's current location.
        spanner_target_locations = set()
        if needed_spanners_count > 0:
            # Sort usable spanners at locations by distance from man_loc
            # Handle cases where man_loc or spanner_loc might not be in self.dist (disconnected graph)
            sorted_usable_spanners_at_loc = sorted(
                list(usable_spanners_at_loc),
                key=lambda sl_pair: self.dist.get(man_loc, {}).get(sl_pair[1], float('inf'))
            )
            # Take the locations of the top needed_spanners_count spanners
            # Ensure we don't try to access more spanners than available at locations
            spanner_target_locations = {loc for s, loc in sorted_usable_spanners_at_loc[:min(needed_spanners_count, len(sorted_usable_spanners_at_loc))]}

        all_target_locations = nut_locations | spanner_target_locations

        # 10. Estimate walk cost
        walk_cost = 0
        if all_target_locations:
            # Find the target location closest to the man
            closest_target_loc = None
            min_dist_to_closest_target = float('inf')
            for target_loc in all_target_locations:
                 d = self.dist.get(man_loc, {}).get(target_loc, float('inf'))
                 if d < min_dist_to_closest_target:
                     min_dist_to_closest_target = d
                     closest_target_loc = target_loc

            # If closest_target_loc is still None, it means all targets are unreachable
            if closest_target_loc is None or min_dist_to_closest_target == float('inf'):
                 return float('inf') # Unreachable targets mean unsolvable state

            walk_cost = min_dist_to_closest_target

            # Add distances from the closest target to all other targets
            for target_loc in all_target_locations:
                if target_loc != closest_target_loc:
                    # Check if the distance from closest_target_loc to target_loc is finite
                    d = self.dist.get(closest_target_loc, {}).get(target_loc, float('inf'))
                    if d == float('inf'):
                         return float('inf') # Unreachable target means unsolvable state
                    walk_cost += d


        # 11. Total heuristic
        # Cost = Walk cost + Tighten actions + Pickup actions
        # Each untightened nut requires one tighten action.
        # NeededSpannersCount is the number of pickup actions required.
        heuristic_value = walk_cost + num_nuts + needed_spanners_count

        return heuristic_value
