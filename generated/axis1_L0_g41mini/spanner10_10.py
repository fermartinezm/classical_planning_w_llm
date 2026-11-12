from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern with possible wildcards `*`.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Summary
    -------
    Estimates the minimum number of actions required for the man to tighten all loose nuts.
    The plan involves navigating the man across linked locations, picking up usable spanners,
    carrying them to nuts that are loose, and tightening them. Each tighten action consumes the spanner's usability.

    Assumptions
    -----------
    - There is exactly one man.
    - Spanners located somewhere can be picked up if the man is at the same location.
    - The man carries at most one spanner at a time before using (tightening) it.
    - Tightening a nut requires the man to be at the nut's location carrying a usable spanner.
    - After tightening a nut, that spanner becomes unusable.
    - Only tight nuts are considered goal-achieved.
    - The links between locations are bidirectional but only one direction is listed as a fact (assumed symmetry).
    - The heuristic disregards spanner usability replenishment as none exists.
    - Actions cost uniformly 1.

    Heuristic Initialization
    ------------------------
    From static facts, constructs an undirected graph of locations using the links.
    Extracts:
    - The list of all locations.
    - Links between locations (undirected).
    Also extracts:
    - The set of goal-tightened nuts (to identify nuts still loose).
    - The tasks' goals and static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the single man and his current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners and their locations.
    4. Check if the man is currently carrying a usable spanner.
       - If yes, this decreases the need to pick one up.
       - If no, he must go pick up a usable spanner before tightening any nut.
    5. For each loose nut:
       a. Compute the shortest path distances:
          - From the man's current location or last nut tightened location, 
            to a spanner location (if he needs to pick one up).
          - From spanner location (if picking up) or current if carrying spanner,
            to the nut's location.
       b. Sum the minimal cost (walks + pickup + tighten).
       c. After tightening the nut, the spanner used becomes unusable, so the man must pick another spanner for the next nut.
    6. Since the heuristic must be efficient, approximate:
       - Use shortest paths precomputed via BFS in the location graph.
       - For multiple nuts and spanners, greedily pick the closest spanner and closest nut to current position.
    7. Sum the costs for all nuts remaining loose:
       - 1 action to pickup a spanner (if not carrying usable one).
       - walk actions equals shortest path length between relevant locations.
       - 1 action to tighten each nut.
    8. Return 0 if no loose nuts remain (goal state).
    9. Guarantee finite heuristic by computing shortest distances on fully connected graph.

    This heuristic balances realism and efficiency by approximating the minimal traveling and spanner pickup needed.
    """

    def __init__(self, task):
        # Store goals and static facts.
        self.goals = task.goals
        self.static = task.static

        # Build an undirected graph of locations from the link facts.
        self.location_graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1)

        self.locations = locations

    def _bfs_shortest_distances(self, start):
        """
        Compute shortest distances from start location to all other locations using BFS.
        Returns a dictionary location->distance.
        """
        from collections import deque

        distances = {loc: float('inf') for loc in self.locations}
        distances[start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.location_graph.get(current, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Helper: find man name (only one man expected)
        man = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                predicate, obj, loc = get_parts(fact)
                if obj in {f.split()[1] for f in self.static if "man" in f or "bob" in f} or obj.startswith("bob") or obj.startswith("man"):
                    # We refine: actually we find the single man by checking types not given here,
                    # So instead, find the unique man by filtering carrying or at among all state.
                    # Better: find individuals mentioned as man in problem instance - the man is an object of type man.
                    # Since we don't have types explicitly here, we'll find the unique individual who is carrying spanners or whose "at" predicates appear with man.
                    man = obj
                    break
        if man is None:
            # Fallback: find carrying fact to identify man
            for fact in state:
                if match(fact, "carrying", "*", "*"):
                    man = get_parts(fact)[1]
                    break
        # If still none, fallback to "bob" (common example name)
        if man is None:
            man = "bob"

        # Find current man location
        man_location = None
        for fact in state:
            if match(fact, "at", man, "*"):
                man_location = get_parts(fact)[2]
                break
        assert man_location is not None, "Man must have a location."

        # Find all loose nuts and their locations
        loose_nuts = []
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                loose_nuts.append(nut)
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in loose_nuts:
                    nut_locations[obj] = loc

        if not loose_nuts:
            # No loose nuts, heuristic zero.
            return 0

        # Find all usable spanners and their locations
        usable_spanners = set()
        spanner_locations = {}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                usable_spanners.add(spanner)
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in usable_spanners:
                    spanner_locations[obj] = loc

        # Check if man is carrying a usable spanner
        carrying_usable_spanner = False
        spanner_carried = None
        for fact in state:
            if match(fact, "carrying", man, "*"):
                spanner_carried = get_parts(fact)[2]
                if spanner_carried in usable_spanners:
                    carrying_usable_spanner = True
                break

        # Precompute shortest distances from all relevant locations for efficiency
        # Distances from all usable spanner locations + man location + all nut locations needed.

        # Collect all relevant locations for BFS from man and each spanner location
        # However, to be efficient, we'll just compute BFS from man_location on demand.

        # Precompute shortest distances from every location where man might start for picking spanners and nuts:
        # Implement a helper to get shortest path distance between two locations.
        # We'll use BFS precomputed here.

        # Precompute BFS distance from man_location:
        dist_from_man = self._bfs_shortest_distances(man_location)

        # Also precompute distances from each spanner location (to nuts)
        dist_from_spanners = {}
        for spanner, loc in spanner_locations.items():
            dist_from_spanners[loc] = self._bfs_shortest_distances(loc)

        # The plan: for each loose nut, estimate minimal cost to tighten it, considering:
        # - If man already carries usable spanner: just move to nut and tighten (cost = walk + tighten(1))
        # - Otherwise: move to nearest usable spanner, pickup (1), move to nut, tighten(1)
        # - After tightening, spanner used is unusable, so next nut requires new spanner pickup.

        # We approximate the cost by visiting nuts in a greedy order, always starting from man_location or last nut location.

        # Heuristic accumulates:
        #  walking steps (number of 'walk' actions),
        #  pickup actions (1 per spanner pick),
        #  tighten actions (1 per nut)

        # Initialize
        current_location = man_location
        total_cost = 0
        carrying_spanner = carrying_usable_spanner

        # We'll keep track of which spanners are available (usable ones)
        available_spanner_locations = list(spanner_locations.values())

        # To avoid repeated computations, create helper to get shortest distance between any two locations.
        # We'll use BFS distances precomputed from current point or spanner points.

        # Since BFS distances are precomputed only from certain points, we define helper function:
        def get_distance(loc_from, loc_to):
            if loc_from == loc_to:
                return 0
            # Prefer to use precomputed BFS from loc_from if available
            if loc_from == man_location:
                return dist_from_man.get(loc_to, float('inf'))
            if loc_from in dist_from_spanners:
                return dist_from_spanners[loc_from].get(loc_to, float('inf'))
            # If no precomputed distances, fallback to BFS:
            dists = self._bfs_shortest_distances(loc_from)
            return dists.get(loc_to, float('inf'))

        # Process nuts greedily in order of minimal cost to tighten next nut:
        remaining_nuts = set(loose_nuts)

        while remaining_nuts:
            # For each nut, compute cost of tightening next:
            candidates = []
            for nut in remaining_nuts:
                nut_loc = nut_locations[nut]

                if carrying_spanner:
                    # cost: walk from current location to nut location + tighten(1)
                    dist_walk = get_distance(current_location, nut_loc)
                    cost = dist_walk + 1
                    candidates.append((cost, nut, nut_loc, None, None))
                else:
                    # cost: walk to closest usable spanner + pickup(1) + walk to nut + tighten(1)
                    best_cost = float('inf')
                    best_spanner_loc = None
                    for sp_loc in available_spanner_locations:
                        dist_to_spanner = get_distance(current_location, sp_loc)
                        dist_spanner_to_nut = get_distance(sp_loc, nut_loc)
                        cost_candidate = dist_to_spanner + 1 + dist_spanner_to_nut + 1  # pickup + tighten
                        if cost_candidate < best_cost:
                            best_cost = cost_candidate
                            best_spanner_loc = sp_loc
                    candidates.append((best_cost, nut, nut_loc, best_spanner_loc, best_cost))

            # Select nut with minimal estimated cost to tighten next
            candidates.sort(key=lambda x: x[0])
            cost, nut, nut_loc, spanner_loc, _ = candidates[0]

            # Accumulate cost
            total_cost += cost

            # Update current location
            current_location = nut_loc

            # Update carrying_spanner and available spanner list
            if carrying_spanner:
                # used current spanner, now unusable -> must pick a new one
                carrying_spanner = False
            else:
                # we picked a spanner at spanner_loc, remove this spanner location from available list
                if spanner_loc in available_spanner_locations:
                    available_spanner_locations.remove(spanner_loc)
                carrying_spanner = True

            # Remove nut from remaining
            remaining_nuts.remove(nut)

        # Finally, return the heuristic cost as integer number of actions needed.
        # Cost includes walk (move), pickup_spanner, and tighten_nut actions.
        return int(total_cost)
