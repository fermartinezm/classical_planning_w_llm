from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Basic check for number of parts, allowing for wildcards
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the domain-dependent heuristic class
# class spannerHeuristic(Heuristic): # Uncomment and inherit if using a base class
class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the cost of tightening each nut, picking up necessary spanners,
    and the estimated walking distance for the man to reach all required locations
    (nut locations and locations of spanners that need to be picked up).

    # Assumptions
    - The man is the only agent performing actions.
    - Nuts stay in their initial locations.
    - Spanners become unusable after one use but remain carried.
    - Links between locations are bidirectional.
    - The graph of locations connected by links is relatively small, allowing for BFS precomputation.
    - The man can carry multiple spanners.

    # Heuristic Initialization
    - Precomputes shortest path distances between all pairs of locations using BFS based on the static `link` facts.
    - Identifies all spanner objects in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners.
    4. Identify which spanners the man is currently carrying.
    5. Count the number of loose nuts remaining (`N_loose_remaining`).
    6. Count the number of usable spanners the man is carrying (`N_carried_usable`).
    7. Count the total number of usable spanners available in the state (carried or on the ground).
    8. Check for unsolvability: If `N_loose_remaining` is greater than the total number of usable spanners available, the goal is unreachable. Return infinity.
    9. If `N_loose_remaining` is 0, the goal is reached. Return 0.
    10. Calculate the estimated cost for tightening actions: This is simply `N_loose_remaining` (one action per nut).
    11. Calculate the estimated cost for pickup actions: The man needs `N_loose_remaining` usable spanners in total. He currently has `N_carried_usable`. He needs to pick up `N_spanners_to_pickup = max(0, N_loose_remaining - N_carried_usable)` more usable spanners from the ground. This costs `N_spanners_to_pickup` pickup actions.
    12. Calculate the estimated walking cost: The man must visit all locations where loose nuts are (`NutLocs`). If he needs to pick up spanners (`N_spanners_to_pickup > 0`), he must also visit `N_spanners_to_pickup` locations where usable spanners are available on the ground (`UsableSpannerLocsOnGround`).
        - Identify `NutLocs`.
        - Identify `UsableSpannerLocsOnGround`.
        - Determine the set of required locations to visit (`RequiredLocations`): Start with `NutLocs`. If `N_spanners_to_pickup > 0`, add the `N_spanners_to_pickup` locations from `UsableSpannerLocsOnGround` that are *not* already in `NutLocs` and are closest to the man's current location.
        - Estimate the walking cost using a Nearest Neighbor TSP approximation on `RequiredLocations`, starting from the man's current location.
    13. The total heuristic value is the sum of the tightening cost, pickup cost, and walking cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying objects.
        """
        self.goals = task.goals  # Goal conditions
        self.static_facts = task.static # Static facts

        # 1. Precompute distances between all locations using BFS
        self.locations = set()
        links = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                # Store links bidirectionally
                links.add((l1, l2))
                links.add((l2, l1))

        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            queue = [(start_loc, 0)]
            visited = {start_loc}
            while queue:
                current_loc, dist = queue.pop(0)
                self.distances[start_loc][current_loc] = dist
                for l1, l2 in links:
                    if l1 == current_loc and l2 not in visited:
                        visited.add(l2)
                        queue.append((l2, dist + 1))
                    # No need for the l2 == current_loc check if links already added bidirectionally

        # Ensure all pairs have a distance (infinity if unreachable)
        for l1 in self.locations:
            for l2 in self.locations:
                if l2 not in self.distances[l1]:
                    self.distances[l1][l2] = float('inf')

        # 2. Identify all spanner objects (used for unsolvability check)
        # We can find spanners by looking at 'usable', 'carrying', or 'at' facts in initial/static state
        all_facts = task.initial_state | task.static_facts
        self.all_spanners = set()
        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] == 'usable':
                 self.all_spanners.add(parts[1])
             elif parts[0] == 'carrying' and len(parts) == 3: # (carrying man spanner)
                 self.all_spanners.add(parts[2])
             elif parts[0] == 'at' and len(parts) == 3: # (at locatable location)
                 # Check if the locatable is a spanner type. This requires domain parsing.
                 # A simpler heuristic-specific approach: assume objects starting with 'spanner' are spanners.
                 if parts[1].startswith('spanner'):
                     self.all_spanners.add(parts[1])

        # Identify the man object (assuming there's only one man and it's named 'bob' or similar, or find it from initial state)
        # A robust way is to find the object of type 'man' from the domain/instance definition.
        # Assuming for this domain, the man is the object in the (at ?m - man ?l - location) initial fact.
        self.man = None
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                 # This is fragile, relies on type information not in fact string.
                 # A better way: parse task.objects which should have type info.
                 # Since task.objects is not exposed in the provided Task class,
                 # let's assume the man is the object in the initial (at ?m ...) fact
                 # that is not a spanner or nut. Or, assume a standard name like 'bob'.
                 # Let's assume the first object in an (at ...) fact that isn't a spanner/nut is the man.
                 obj_name = parts[1]
                 if not obj_name.startswith('spanner') and not obj_name.startswith('nut'):
                     self.man = obj_name
                     break
        if self.man is None:
             # Fallback: Look for 'carrying' facts
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if parts[0] == 'carrying' and len(parts) == 3:
                     self.man = parts[1]
                     break

        # Store goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # 1. Identify man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man's location is unknown? Should not happen in valid states.
             # Or maybe man is not in the state if problem is malformed?
             # Assume man is always located somewhere.
             pass # man_location should be found

        # 2. Identify loose nuts and their locations
        loose_nuts = {parts[1] for fact in state if match(fact, "loose", "*")}
        # We only care about loose nuts that are goal nuts
        loose_nuts_to_tighten = loose_nuts.intersection(self.goal_nuts)

        # 9. If N_loose_remaining is 0, the goal is reached. Return 0.
        N_loose_remaining = len(loose_nuts_to_tighten)
        if N_loose_remaining == 0:
            return 0

        nut_locations = {}
        for nut in loose_nuts_to_tighten:
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_locations[nut] = get_parts(fact)[2]
                    break
        NutLocs = set(nut_locations.values())


        # 3. Identify usable spanners in the current state
        usable_spanners_in_state = {parts[1] for fact in state if match(fact, "usable", "*")}

        # 4. Identify spanners carried by the man
        carried_spanners = {parts[2] for fact in state if match(fact, "carrying", self.man, "*")}

        # 6. Count usable spanners carried
        carried_usable_spanners = usable_spanners_in_state.intersection(carried_spanners)
        N_carried_usable = len(carried_usable_spanners)

        # 7. Count usable spanners on the ground
        usable_spanners_on_ground = usable_spanners_in_state - carried_spanners
        N_usable_on_ground = len(usable_spanners_on_ground)

        # 8. Check for unsolvability
        total_usable_spanners_available = N_carried_usable + N_usable_on_ground
        if N_loose_remaining > total_usable_spanners_available:
             # Problem is likely unsolvable from this state (not enough usable spanners exist)
             return float('inf')


        # 10. Calculate tighten cost
        tighten_cost = N_loose_remaining

        # 11. Calculate pickup cost
        N_spanners_to_pickup = max(0, N_loose_remaining - N_carried_usable)
        pickup_cost = N_spanners_to_pickup

        # 12. Calculate walking cost
        # Identify locations of usable spanners on the ground
        usable_spanner_locs_on_ground = set()
        if N_spanners_to_pickup > 0:
             for spanner in usable_spanners_on_ground:
                 for fact in state:
                     if match(fact, "at", spanner, "*"):
                         usable_spanner_locs_on_ground.add(get_parts(fact)[2])
                         break

        # Determine the set of required locations to visit
        RequiredLocations = set(NutLocs)
        spanner_locations_to_add = []

        if N_spanners_to_pickup > 0:
            # Find candidate spanner locations on the ground that are not already nut locations
            candidate_spanner_locs = list(usable_spanner_locs_on_ground - RequiredLocations)
            # Sort candidates by distance from ManLoc
            # Handle cases where man_location might not have distances computed to all locations
            candidate_spanner_locs.sort(key=lambda l: self.distances.get(man_location, {}).get(l, float('inf')))

            # Add the closest N_spanners_to_pickup locations (that are not already nut locations)
            count = 0
            for loc in candidate_spanner_locs:
                 spanner_locations_to_add.append(loc)
                 count += 1
                 if count >= N_spanners_to_pickup:
                     break

        RequiredLocations.update(spanner_locations_to_add)

        # Calculate walking cost using Nearest Neighbor TSP approximation
        CurrentLoc = man_location
        RemainingTargets = list(RequiredLocations)
        walking_cost = 0

        # Handle case where man_location might not be in self.distances keys (e.g., initial state location not linked)
        if CurrentLoc not in self.distances:
             # If man is at a location not part of the linked graph, he can't move.
             # If there are loose nuts, this state is likely unsolvable unless nuts are at his location.
             # If nuts are at his location, he still needs spanners.
             # If RequiredLocations is not empty and man cannot move to any of them, it's unsolvable.
             can_reach_any_target = False
             for target in RemainingTargets:
                 if self.distances.get(CurrentLoc, {}).get(target, float('inf')) != float('inf'):
                     can_reach_any_target = True
                     break
             if not can_reach_any_target and N_loose_remaining > 0:
                  return float('inf') # Cannot reach any required location

             # If man is at an isolated location but targets are also there, walking cost is 0 within that location set.
             # The NN loop below will handle this if distances are 0.
             pass # Proceed with NN calculation

        while RemainingTargets:
            NearestLoc = None
            MinDist = float('inf')

            # Find nearest target from CurrentLoc
            for target in RemainingTargets:
                # Use .get for safety in case CurrentLoc or target is not in distances keys
                dist = self.distances.get(CurrentLoc, {}).get(target, float('inf'))
                if dist < MinDist:
                    MinDist = dist
                    NearestLoc = target

            if NearestLoc is None:
                # This happens if RemainingTargets is not empty but all targets are unreachable from CurrentLoc
                # This case should ideally be caught by the unsolvability check earlier if N_loose_remaining > 0
                # But as a safeguard:
                if N_loose_remaining > 0:
                     return float('inf') # Cannot reach required locations
                else:
                     # Should not happen if N_loose_remaining is 0
                     break # Should have returned 0 already

            walking_cost += MinDist
            CurrentLoc = NearestLoc
            RemainingTargets.remove(NearestLoc)

        # 13. Total heuristic value
        total_cost = tighten_cost + pickup_cost + walking_cost

        return total_cost

