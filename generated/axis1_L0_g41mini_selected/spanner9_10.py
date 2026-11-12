from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    Summary
    --------
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It takes into account the following action costs:
     - walking between locations (1 action per move along a link)
     - picking up a spanner (1 action)
     - tightening a single nut (1 action)
    The heuristic is not necessarily admissible but aims to be informative and efficient.

    Assumptions
    -----------
    - There is exactly one man who tightens nuts.
    - The man can only carry one or more spanners (the domain allows multiple spanners carried simultaneously).
    - Each spanner can only be used once per tightening action (becomes unusable after tightening one nut).
    - A nut can only be tightened if the man is at the same location as the nut and carrying a usable spanner.
    - Moving along a link edge always costs exactly 1 action.
    - Carrying multiple spanners allows tightening multiple nuts consecutively without picking up another spanner.
    - The heuristic ignores the order and concurrency constraints by estimating a minimal approximate cost.

    Heuristic Initialization
    -------------------------
    - Extract the unique man name.
    - Extract all nuts and spanners.
    - Parse static "link" facts to build a bidirectional graph of locations.
    - Precompute shortest distances between all pairs of locations using Floyd-Warshall.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the current location of the man.
    2. Identify the locations and usability of all spanners:
       - Which spanners are usable and where (either at some location or being carried).
       - Which spanners the man is currently carrying and usable.
    3. Identify all loose nuts and their locations.
       If there are no loose nuts, heuristic is 0 (goal state).

    4. The heuristic estimates how many nuts remain to be tightened (the number of loose nuts).

    5. To tighten each nut, the man must:
       - Be at the nut's location.
       - Carry a usable spanner.

    6. The heuristic estimates the minimal walking cost to:
       - Collect required spanners if not already carried.
       - Travel to loosen nuts one by one for tightening.

    7. Since spanners become unusable after one use, at least as many usable spanners must be picked up or carried as loose nuts.

    8. The heuristic computes:
       - The minimal cost to gather enough usable spanners by traveling from the man's location or current carried spanners.
       - The minimal cost to travel between nuts for tightening.
       - The total actions: moving cost + picking up spanner actions + tightening nut actions.

    9. The heuristic assumes one spanner is consumed per nut tightened, and picking up a spanner costs one action each.
       Tightening a nut costs 1 action each.

    10. The heuristic ignores complex ordering and assumes the man can sequence moves optimally.

    11. Output is the sum of:
        - Minimum travel distances for spanner pickups + nuts visits,
        - Number of pickups,
        - Number of tighten actions,
        which gives a reasonable estimate of cost to reach the goal.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all locations from all link facts (build undirected graph)
        self.links = set()
        self.locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:3]
                self.links.add((loc1, loc2))
                self.links.add((loc2, loc1))
                self.locations.add(loc1)
                self.locations.add(loc2)
        self.locations = sorted(self.locations)

        # Build adjacency list
        self.adj = {loc: set() for loc in self.locations}
        for (l1, l2) in self.links:
            self.adj[l1].add(l2)

        # Precompute shortest paths between all locations via Floyd-Warshall
        self.shortest_dist = self._floyd_warshall(self.locations, self.adj)

        # Extract man name (assumed unique)
        self.man = None
        for fact in task.initial_state:
            if fact.startswith("(at "):
                parts = get_parts(fact)
                obj = parts[1]
                if obj in task.facts and obj.lower() in ["man", "bob", "alice"]:
                    # Not safe to assume obj lowercases; better to check type from domain
                    # Instead, from domain file we know 'bob' is the man in examples.
                    # So, heuristic assumes exactly one man -- heuristic scans for all (at ?m ?l)
                    # where ?m is of type man, but we only get names from facts strings.
                    # Here we fallback to find in goal conditions or known:
                    pass
        # Alternative: Use given from task: find any object typed man
        # We rely on the heuristic called with node (current state).
        # So delay assignment of man until heuristic call.

    def _floyd_warshall(self, locations, adj):
        # Initialize distances: 0 diagonal, 1 for neighbors, inf else
        inf = float("inf")
        dist = {loc: {loc2: inf for loc2 in locations} for loc in locations}
        for loc in locations:
            dist[loc][loc] = 0
            for neigh in adj[loc]:
                dist[loc][neigh] = 1
        # Floyd-Warshall main loop
        for k in locations:
            for i in locations:
                di = dist[i]
                dik = di[k]
                if dik == inf:
                    continue
                dk = dist[k]
                for j in locations:
                    new_dist = dik + dk[j]
                    if new_dist < di[j]:
                        di[j] = new_dist
        return dist

    def __call__(self, node):
        state = node.state

        # Helper: find the man and his current location
        # From domain: only one man of type 'man' exists, find (at ?m ?l)
        man = None
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                # Find man by checking 'carrying' or 'at man loc' facts. From domain, man appears in carrying facts.
                if any(match(f, "carrying", obj, "*") or match(f, "carrying", "*", obj) for f in state) or obj in self._find_all_men(state):
                    man = obj
                    man_location = loc
                    break
        if man is None:
            # Fallback: pick any object that occurs in carrying or at; assume it's man if no better info
            candidates = set()
            for fact in state:
                if fact.startswith("(carrying "):
                    parts = get_parts(fact)
                    candidates.add(parts[1])  # man name 
            for c in candidates:
                for fact in state:
                    if match(fact, "at", c, "*"):
                        man = c
                        man_location = get_parts(fact)[2]
                        break
                if man is not None:
                    break
        if man is None:
            # Cannot find man, heuristic returns 0 as safe fallback
            return 0

        # Extract location of all nuts and their status (loose or tightened)
        nuts = set()
        loose_nuts = {}
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                n = get_parts(fact)[1]
                loose_nuts[n] = True
                nuts.add(n)
            elif match(fact, "tightened", "*"):
                n = get_parts(fact)[1]
                nuts.add(n)
        # Get nuts locations
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:3]
                if obj in loose_nuts:
                    nut_locations[obj] = loc

        # If no loose nuts, heuristic is 0 (goal reached)
        if not loose_nuts:
            return 0

        # Extract spanner info: usable status, location or carried
        spanner_locations = {}
        usable_spanners = set()
        carried_spanners = set()
        # All spanners known by domain are of type spanner, but we know from facts they appear as (at s l) or (carrying m s)
        for fact in state:
            if match(fact, "usable", "*"):
                s = get_parts(fact)[1]
                usable_spanners.add(s)
            elif match(fact, "at", "*", "*"):
                s = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                # Check if s is spanner by usage in domain, assume all spanners appear in usable facts or carrying or at.
                # Only store usable spanners location for pickup
                if s in usable_spanners:
                    spanner_locations[s] = loc
            elif match(fact, "carrying", man, "*"):
                s = get_parts(fact)[2]
                if s in usable_spanners:
                    carried_spanners.add(s)

        # Count how many usable spanners are currently carried
        num_spanners_carried = len(carried_spanners)

        # Count how many loose nuts remain
        num_loose_nuts = len(loose_nuts)

        # If enough usable spanners are carried for all loose nuts, no pickup needed
        spanners_needed_to_pick = max(0, num_loose_nuts - num_spanners_carried)

        # Locations where usable spanners are available for pickup (not carried)
        pickup_locations = [loc for s, loc in spanner_locations.items() if s not in carried_spanners]

        # For heuristic estimate:
        # 1) Distance from man location to spanner pickups (to get required spanners)
        #    Estimate minimal cumulative cost to pick all needed spanners by shortest paths
        # 2) Distance from last spanner pickup (or man location if no pickup needed) to nuts locations in some order
        # 3) Sum costs:
        #    - walking along links (each distance unit counts as 1 action)
        #    - picking up each spanner: 1 action each
        #    - tightening each nut: 1 action each

        # If no spanners needed, just estimate travel cost between man and nuts
        # Otherwise, estimate a minimum cost to walk to spanner pickups and then nuts.

        # Simplifications / approximations for speed:
        # - For pickups: assume sum of minimal distances from man to closest needed spanners
        # - For nuts: sum minimal travel from closest nut to all other nuts (greedy approx)

        d = self.shortest_dist

        # Calculate pickup travel cost:

        # If spanners are needed:
        if spanners_needed_to_pick > 0 and pickup_locations:
            # Greedy approach: pick closest spanner, then from there next closest, etc.

            # Start from man location, find min distance sum to get spanners
            # We try a greedy heuristic: at each step, pick closest remaining spanner

            remaining = set(pickup_locations)
            current_pos = man_location
            pickup_travel_cost = 0
            for _ in range(spanners_needed_to_pick):
                # Find closest spanner pickup location
                closest_loc = None
                closest_dist = float("inf")
                for loc in remaining:
                    dist_to_loc = d[current_pos].get(loc, float("inf"))
                    if dist_to_loc < closest_dist:
                        closest_dist = dist_to_loc
                        closest_loc = loc
                if closest_loc is None or closest_dist == float("inf"):
                    # No reachable spanners; heuristic returns a large finite number
                    return 1000 + num_loose_nuts * 10
                pickup_travel_cost += closest_dist
                current_pos = closest_loc
                remaining.remove(closest_loc)
            # After pickups, current_pos is at last spanner picked

        else:
            # No pickup needed, start traveling from man location
            pickup_travel_cost = 0
            current_pos = man_location

        # Calculate cost to tighten nuts: sum minimal travel cost to visit all nuts starting from current_pos
        # Use greedy nearest neighbor visit heuristic (not the exact TSP)

        nut_locs = list(nut_locations.values())
        if not nut_locs:
            # No loose nuts location found (should not happen if loose nuts exist); return 0
            return 0

        loose_nut_locations = nut_locs

        remaining_nuts = set(loose_nut_locations)
        nut_travel_cost = 0
        pos = current_pos
        while remaining_nuts:
            closest_nut = None
            closest_dist = float("inf")
            for loc in remaining_nuts:
                dist_to_nut = d[pos].get(loc, float("inf"))
                if dist_to_nut < closest_dist:
                    closest_dist = dist_to_nut
                    closest_nut = loc
            if closest_nut is None or closest_dist == float("inf"):
                # Unreachable nut; return large heuristic
                return 1000 + num_loose_nuts * 10
            nut_travel_cost += closest_dist
            pos = closest_nut
            remaining_nuts.remove(closest_nut)

        # Total action cost:
        # walking steps: pickup_travel_cost + nut_travel_cost
        # picking up spanners: spanners_needed_to_pick actions
        # tightening each nut: num_loose_nuts actions (1 per nut)

        total_estimated_cost = pickup_travel_cost + nut_travel_cost + spanners_needed_to_pick + num_loose_nuts

        # Ensure heuristic is zero only if goal is reached (no loose nuts)
        if num_loose_nuts == 0:
            return 0

        # Return integer heuristic estimate
        return int(total_estimated_cost)

    def _find_all_men(self, state):
        men = set()
        for fact in state:
            parts = get_parts(fact)
            if len(parts) >= 2:
                if parts[0] == "carrying":  # format: (carrying man spanner)
                    men.add(parts[1])
        return men
