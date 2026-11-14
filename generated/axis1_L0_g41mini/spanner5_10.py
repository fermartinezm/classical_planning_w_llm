from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The pattern to match, wildcards `*` allowed.
    - Returns True if the fact matches the pattern else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Summary:
    ----------
    This heuristic estimates the minimal number of actions required to tighten all nuts 
    specified in the goal from the current state. It accounts for the need to:
      - Move the man between locations (using the connectivity of locations),
      - Pick up usable spanners if the man is not carrying one,
      - Tighten loose nuts by having the man carry a usable spanner and be co-located with the nut.

    Assumptions:
    ----------
    - There is exactly one man.
    - Tightening a nut requires the man to be carrying a usable spanner at the same location as the nut.
    - Spanners become unusable immediately after use; thus, each tightening requires a usable spanner available or obtained.
    - The man can carry only one spanner at a time.
    - Walking costs one action per step between linked locations.
    - Picking up a spanner costs one action.
    - Tightening a nut costs one action.
    - The static 'link' facts form a connectivity graph; shortest walking paths can be computed efficiently.

    Heuristic Initialization:
    -------------------------
    - Extract all locations and build a bidirectional graph from 'link' static facts.
    - Extract the man object.
    - Extract all nuts required to be tightened in the goal.
    - Precompute shortest distances (number of walk actions) between all locations using BFS.
    
    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify:
       - The man's current location.
       - Which spanners are usable and their locations.
       - Which nuts still need tightening.
       - Which spanner (if any) the man is currently carrying.

    2. For each loose nut that must be tightened:
       a) Estimate the minimal walking distance:
          - From the man's current location or from last nut tightened,
          - To a location where a usable spanner is available (or is currently carried),
          - Then to the nut's location.
       b) Sum up:
          - The walking cost to reach the spanner if not carried,
          - The cost to pick up the spanner if needed (1 action),
          - The walking cost to the nut,
          - The tightening action cost (1 action).
       c) Mark the spanner used as now unusable and remove it from the usable spanner set.
       d) Update the man's location to the nut's location.

    3. If the man initially carries a usable spanner, use it for the first nut without pickup cost.

    4. Sum the estimated costs for all nuts to produce the heuristic estimate.

    This heuristic is not guaranteed to be admissible but aims to be informative and efficiently computable.
    """

    def __init__(self, task):
        """Initialize heuristic data structures."""
        self.goals = task.goals
        self.static = task.static

        # Parse all link facts to build undirected graph of locations
        self.graph = {}
        self.locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1)

        self.locations = list(self.locations)

        # Precompute shortest distance (number of walk actions) between all pairs of locations using BFS
        self.distance = {loc: self._bfs_distances(loc) for loc in self.locations}

        # Identify the man (should be exactly one)
        self.man = None
        # Since the domain typing info is not directly available, we infer man from state during heuristic call.

        # Extract the set of nuts required to be tightened in the goal:
        # The goals are expected to be facts like (tightened nutX)
        self.nuts_to_tighten = set()
        for fact in self.goals:
            if match(fact, "tightened", "*"):
                _, nut = get_parts(fact)
                self.nuts_to_tighten.add(nut)

    def _bfs_distances(self, start):
        """Compute shortest walking distances from start location to all other locations."""
        from collections import deque

        distances = {loc: float('inf') for loc in self.locations}
        distances[start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            for neighbor in self.graph.get(current, ()):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute heuristic estimate as the minimal number of actions to tighten all remaining nuts.

        Returns:
            int: Estimated number of actions (walk, pickup, tighten) to reach the goal.
        """
        state = node.state

        # Find the man object (assumed single) and his current location
        man = None
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                # Heuristic assumes one man object present (e.g., named 'bob')
                # Check if this obj is a man by checking if (carrying obj *) facts exist or by heuristic assumption
                if any(match(f, "carrying", obj, "*") for f in state) or man is None:
                    # Let's assign the first found as the man since domain has single man
                    man = obj
                    man_location = loc
                    # We assume the first 'at' matched for man is reliable, so break here
                    break
        # Defensive fallback if above fails (should not happen in valid states)
        if man is None or man_location is None:
            # No man found, heuristically return 0 if goal reached else a fallback large value
            return 0 if self.goals <= state else 1000

        # Identify which spanner the man is carrying (if any)
        spanner_carried = None
        for fact in state:
            if match(fact, "carrying", man, "*"):
                _, _, _, spanner = get_parts(fact)
                spanner_carried = spanner
                break

        # Collect all usable spanners and their locations (excluding the one carried)
        usable_spanners = {}
        for fact in state:
            if match(fact, "usable", "*"):
                _, spanner = get_parts(fact)
                # Find location of spanner if not carried by man
                if spanner != spanner_carried:
                    for loc_fact in state:
                        if match(loc_fact, "at", spanner, "*"):
                            _, _, loc = get_parts(loc_fact)
                            usable_spanners[spanner] = loc
                            break

        # Collect nuts that still need tightening (must appear loose in state and are in goals)
        loose_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                _, nut = get_parts(fact)
                if nut in self.nuts_to_tighten:
                    loose_nuts.add(nut)

        # Collect locations of these nuts
        nut_locations = {}
        for nut in loose_nuts:
            for fact in state:
                if match(fact, "at", nut, "*"):
                    _, _, loc = get_parts(fact)
                    nut_locations[nut] = loc
                    break

        # Early exit: if no nuts left to tighten, heuristic is 0
        if not loose_nuts:
            return 0

        # Heuristic calculation:
        # For each loose nut, estimate cost of:
        #   walk to a usable spanner (or use carried one),
        #   pickup spanner if needed,
        #   walk to nut,
        #   tighten nut
        #
        # We greedily pick the next nut minimizing incremental cost.
        #
        # Approximate approach:
        # Starting at man's location and carried spanner state,
        # iteratively pick the next nut with minimal incremental action cost,
        # updating man location and carried spanner availability,
        # consuming the spanner from usable_spanners.

        # Action costs
        WALK_COST = 1
        PICKUP_COST = 1
        TIGHTEN_COST = 1

        # Current state in heuristic calculation
        current_location = man_location
        carried_spanner_usable = spanner_carried is not None
        usable_spanners_available = usable_spanners.copy()
        remaining_nuts = loose_nuts.copy()

        total_cost = 0

        while remaining_nuts:
            best_cost = float('inf')
            best_nut = None
            best_spanner_loc = None
            best_need_pickup = True  # True if we must pick up spanner to tighten this nut

            # For each nut find minimal cost to tighten it next
            for nut in remaining_nuts:
                nut_loc = nut_locations[nut]

                # Consider two scenarios:
                # 1) Using carried usable spanner if available
                cost_using_carried = None
                if carried_spanner_usable:
                    # Walk from current location to nut location + tighten cost
                    walk_dist = self.distance[current_location][nut_loc]
                    cost_using_carried = walk_dist * WALK_COST + TIGHTEN_COST

                # 2) Use one of the usable spanners on the map (must walk to spanner, pickup, then walk to nut, then tighten)
                cost_using_spanner = None
                spanner_candidate = None
                for spanner, spanner_loc in usable_spanners_available.items():
                    dist_to_spanner = self.distance[current_location][spanner_loc]
                    dist_spanner_to_nut = self.distance[spanner_loc][nut_loc]
                    cost = dist_to_spanner * WALK_COST + PICKUP_COST + dist_spanner_to_nut * WALK_COST + TIGHTEN_COST
                    if cost_using_spanner is None or cost < cost_using_spanner:
                        cost_using_spanner = cost
                        spanner_candidate = spanner

                # Pick best scenario
                if cost_using_carried is not None and cost_using_carried <= (cost_using_spanner if cost_using_spanner is not None else float('inf')):
                    # Use carried spanner
                    if cost_using_carried < best_cost:
                        best_cost = cost_using_carried
                        best_nut = nut
                        best_spanner_loc = None  # no pickup needed
                        best_need_pickup = False
                elif cost_using_spanner is not None:
                    # Use spanner from map
                    if cost_using_spanner < best_cost:
                        best_cost = cost_using_spanner
                        best_nut = nut
                        best_spanner_loc = spanner_candidate
                        best_need_pickup = True

            # Update heuristic state
            total_cost += best_cost
            remaining_nuts.remove(best_nut)
            current_location = nut_locations[best_nut]

            # Update spanner carried state
            if best_need_pickup:
                # After pickup and use, spanner becomes unusable, remove from available
                del usable_spanners_available[best_spanner_loc]
                carried_spanner_usable = False
            else:
                # Used carried spanner => now unusable
                carried_spanner_usable = False

        return int(total_cost)
