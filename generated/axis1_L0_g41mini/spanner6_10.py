from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob gate)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    ----------
    Estimates the number of actions still required to tighten all loose nuts in the state.
    To tighten a nut, the man must pick up a usable spanner, go to the nut's location, and perform a tighten action.
    The heuristic sums the minimal effort needed per remaining loose nut considering:
    - Moving/spanning distances between the man, spanners, and nuts.
    - The necessity to pick up spanners (if not currently carried or usable).
    Since the problem involves travel in a linked graph of locations, the heuristic uses shortest paths
    computed on the static "link" graph to estimate travel costs efficiently.

    Assumptions:
    ------------
    - Only one man ("man") is acting in the domain.
    - Each tightening requires a usable spanner. After use, the spanner becomes unusable.
    - The man can carry at most one spanner at a time (given problem definition and actions).
    - The heuristic does not need to be admissible but should minimize node expansions.
    - All links are bidirectional as no directionality is specified.
    - Movement costs per link and actions (pickup, tighten) are counted as 1 action each.

    Heuristic Initialization:
    --------------------------
    - Extract from static facts the bidirectional connectivity of locations ("link" facts) to form a graph.
    - Identify all spanners and their usable status.
    - Identify all nuts that need tightening.
    - Precompute shortest-path distances between all locations using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1) Identify:
       - The man's current location.
       - The spanners that are usable and their locations or if the man is currently carrying one usable spanner.
       - The nuts that are loose and their locations.
    2) Precompute shortest paths (or compute on demand) between relevant locations to estimate movement costs.
    3) For each loose nut:
       - Evaluate the minimal "cost" to:
         i) if carrying a usable spanner: move from man's location to nut location + tighten (1 action)
         ii) else pick up the nearest usable spanner (pickup + move) then move to nut location and tighten.
       - Each pickup_spanner costs: 1 pickup action + walking between man location and spanner location.
       - Each walk costs are number of moves (edges).
       - Tighten action costs 1.
    4) Since spanners become unusable after use, assume the man picks a different spanner for each nut.
       Sum the heuristic costs for all loose nuts, assuming optimal selection (i.e., minimal pairing).
    5) This heuristic ignores that the man must move between nuts in sequence but captures cost trends well.
    6) Return 0 if no loose nuts remain (goal condition).

    Efficient computability:
    -------------------------
    - Use BFS-based shortest paths on the static connected graph for distance computations.
    - Cache distances in a dictionary to avoid recomputation per heuristic call.

    """

    def __init__(self, task):
        # Save goal facts
        self.goals = task.goals
        # Extract static link facts (edges)
        static_facts = task.static

        # Build bidirectional adjacency map of locations
        self.graph = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1)

        # Extract all locations reachable (for safety)
        self.locations = set(self.graph.keys())

        # Cache for shortest path distances
        self._distance_cache = {}

    def _bfs_distance(self, start, goal):
        """
        Compute shortest distance (in moves) between two locations using BFS on the static graph.

        Cache results in a dictionary to ensure efficiency.
        If no path found, return a large number (e.g., 1000) indicating unreachable.
        """
        if (start, goal) in self._distance_cache:
            return self._distance_cache[(start, goal)]
        if start == goal:
            self._distance_cache[(start, goal)] = 0
            return 0
        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor == goal:
                    self._distance_cache[(start, goal)] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        self._distance_cache[(start, goal)] = 1000  # Arbitrary large number for unreachable
        return 1000

    def __call__(self, node):
        state = node.state

        # 1) Identify the man (only one assumed)
        men = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # The domain declares "man" type -- try to identify man(s)
                # In examples there's one 'bob', we pick first man fact:
                # We'll identify man by presence of 'man' in name or by matching carrying fact
                # As domain has only one man, we pick any object carrying or at.
                # To be certain, check carrying or at for man type:
                # We rely on "carrying ?m ?s" facts for man names:
                # We'll gather all "man" candidates from "carrying" or "at" facts of man objects.

                # We'll detect man names as those that occur as first argument in carrying.
                # Collect possible men names from carrying facts:
                pass

        # Extract man name(s) from carrying facts (since only man can carry)
        men = {get_parts(fact)[1] for fact in state if match(fact, "carrying", "*", "*")}
        # If no man carrying found (should not happen), fall back to any "at ?man ?loc" where ?man is man
        if not men:
            men = {get_parts(fact)[1] for fact in state if match(fact, "at", "*", "*")}
        # Take the first man name, assuming exactly one man
        man = next(iter(men))

        # Man location
        man_location = None
        for fact in state:
            if match(fact, "at", man, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
            # Unsafe fallback
            man_location = ""  # no location known

        # 2) Identify usable spanners and their locations or if man is carrying usable spanner
        spanner_locations = {}
        usable_spanners = set()
        spanners_carried_by_man = set()
        for fact in state:
            if match(fact, "at", "*", "*"):  # possible spanner location
                obj, loc = get_parts(fact)[1:]
                # We must confirm obj is a spanner
                # domain types not explicitly listed at runtime, but spanner name prefixes may be used
                # We rely on usable facts to identify spanners
                # For safety we gather all (usable ?s) spanners first
                pass
            if match(fact, "usable", "*"):
                s = get_parts(fact)[1]
                usable_spanners.add(s)
            if match(fact, "carrying", man, "*"):
                s = get_parts(fact)[2]
                spanners_carried_by_man.add(s)

        # Map spanners to locations they are at (excluding those carried)
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Only consider spanners that are currently usable and not carried
                if obj in usable_spanners:
                    spanner_locations[obj] = loc

        # 3) Identify all loose nuts and their locations
        loose_nuts = set()
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                loose_nuts.add(nut)
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Store location of nuts (could be needed)
                if obj in loose_nuts:
                    nut_locations[obj] = loc

        if not loose_nuts:
            return 0  # goal state

        # Heuristic components:
        # Actions cost:
        WALK_COST = 1
        PICKUP_COST = 1
        TIGHTEN_COST = 1

        # For each loose nut, estimate minimal cost to:
        #   get usable spanner (either carried or by walking + pickup)
        #   move to nut location
        #   tighten nut

        # If man carries a usable spanner, we do not pay pickup + walk to spanner cost for first nut
        # But note each spanner becomes unusable after one tighten => need one spanner per nut.

        # We simplify and:
        # For first nut, if man carries usable spanner, use it directly.
        # For other nuts, must pick up different spanners from spanner_locations.

        # For each spanner location, precompute walk distance from man_location (used repeatedly)
        dist_man_to_spanner = {}
        for s, loc in spanner_locations.items():
            dist_man_to_spanner[s] = self._bfs_distance(man_location, loc)

        # For each nut location, precompute distance to nut location from spanner pickup loc or man_location
        # We'll also need distance from spanner locations to nut locations.

        # We'll also precompute distances from all spanner locations to each nut location,
        # so we can estimate (walk to spanner) + (walk spanner->nut_location) + pickup + tighten.

        # Since man moves after each nut to next spanner, conservatively sum independent costs.

        # Strategy:
        # For each loose nut:
        #   Find min over spanners s usable:
        #      if s is carried usable spanner (only possible for one nut), cost is walk(man_loc->nut_loc) + tighten
        #      else cost = walk(man_loc-> s_loc) + pickup + walk(s_loc -> nut_loc) + tighten
        # Sum costs for all nuts, assuming always optimal spanner choice.

        # If man carries multiple usable spanners (possible?), use one for first nuts accordingly.
        # In example domain only one spanner can be carried at once (no precondition allowing multiple), so max 1 carried.

        # Maintain a set of spanners available for pickups (usable and not carried by man)
        spanners_for_pickup = set(spanner_locations.keys())

        total_cost = 0

        # Flag to signal if the man currently carries a usable spanner to use once (for one nut)
        carried_usable_spanner_available = len(spanners_carried_by_man & usable_spanners) > 0

        for nut in loose_nuts:
            nut_loc = nut_locations[nut]

            best_cost = float("inf")

            # Option 1: Use carried usable spanner (only for 1 nut)
            if carried_usable_spanner_available:
                walk_to_nut = self._bfs_distance(man_location, nut_loc)
                cost = walk_to_nut * WALK_COST + TIGHTEN_COST
                if cost < best_cost:
                    best_cost = cost
                # After using it once, carried usable spanner is no longer available
                carried_usable_spanner_available = False
                # To be conservative, assume man stays at nut_loc after tightening
                man_location = nut_loc
                continue  # next nut

            # Option 2: Pick up a usable spanner (not carried) before moving to nut and tighten
            # Evaluate all spanners for pickup
            for spanner, sp_loc in spanner_locations.items():
                # if spanner already assigned to previous nut (naively, we ignore assigning unique spanners)
                # But each spanner becomes unusable after use -- assume infinite spanners for heuristic
                # So we do not remove spanners after reuse.
                # We'll consider all spanners available for every nut.

                dist_man_to_s = self._bfs_distance(man_location, sp_loc)
                dist_s_to_nut = self._bfs_distance(sp_loc, nut_loc)
                cost = dist_man_to_s * WALK_COST + PICKUP_COST + dist_s_to_nut * WALK_COST + TIGHTEN_COST
                if cost < best_cost:
                    best_cost = cost

            total_cost += best_cost
            # For heuristic, assume man moves to nut location after tightening
            man_location = nut_loc

        return int(total_cost)
