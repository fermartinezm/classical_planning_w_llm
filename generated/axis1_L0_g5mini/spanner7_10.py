from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'spanner' domain. It estimates the
    number of actions required to achieve all goal (tightened <nut>) facts
    from a given state. The heuristic models the necessary sequence of
    actions for each loose nut: (walks to a usable spanner if necessary),
    (pickup_spanner), (walk to the nut), (tighten_nut). Because spanners
    become unusable after a single use and the planner allows carrying
    multiple spanners, the heuristic assigns distinct usable spanners to
    distinct nuts (each spanner can be used at most once). The total
    estimate is the minimal sum of per-nut costs under that assignment.

    Assumptions
    -----------
    - Each usable spanner can be used at most once (tighten_nut makes the
      spanner unusable).
    - The planner allows carrying multiple spanners simultaneously; carried
      spanners are immediately available at the man's current location
      (no pickup action required).
    - Walk actions follow the directed 'link' graph specified in static facts.
    - If there are fewer usable spanners than nuts that must be tightened,
      the state is considered (for heuristic purposes) very far from the
      goal (we return a large finite number).
    - The heuristic does not attempt to perfectly sequence the man's moves
      across multiple nuts (that would be TSP-like). Instead, it computes
      the minimal assignment cost of spanners to nuts, where each assigned
      spanner cost includes travel from the man's current location to the
      spanner (if not carried), the pickup (if not carried), travel from
      spanner to nut, and one tighten action.

    Heuristic Initialization
    ------------------------
    - Extract goal nuts from task.goals (facts of form "(tightened <nut>)").
    - Build a directed adjacency list from static 'link' facts to compute
      shortest path distances between locations via BFS.
    - No other static facts are required.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the goal is already reached (all tightened nuts in the state),
       return 0.
    2. Parse the current state to determine:
       - The man's location (prefer the subject of any '(carrying ...)'
         fact; otherwise, pick an object that occurs in 'at' facts but
         does not appear as a spanner or nut).
       - Locations of nuts (from '(at <nut> <loc>)').
       - Locations of spanners (from '(at <spanner> <loc>)'); if a spanner
         is currently carried, treat its location as the man's location.
       - Which spanners are usable (from '(usable <spanner>)').
       - Which spanners are currently carried (from '(carrying <man> <spanner>)').
    3. Build a cost matrix cost[nut_i][spanner_j] where the cost is:
       - dist(man_loc, spanner_loc)
         + pickup_cost (1 if spanner is not already carried, else 0)
         + dist(spanner_loc, nut_loc)
         + tighten_cost (1)
       Distances are shortest-path lengths on the directed 'link' graph;
       if no path exists, use a large value.
    4. Solve the minimum-cost assignment that assigns a distinct usable
       spanner to each loose nut. This is done with a small dynamic
       programming (bitmask) algorithm. If there are fewer usable spanners
       than nuts, return a large finite cost.
    5. Return the minimal assignment cost as the heuristic estimate.
    """

    def __init__(self, task):
        """Extract goals and static link graph from the task."""
        self.goals = task.goals
        # Extract directed links from static facts.
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                if a not in self.adjacency:
                    self.adjacency[a] = []
                self.adjacency[a].append(b)

    def bfs_distances_from(self, source):
        """
        Compute shortest-path distances from source to all reachable locations
        using directed edges in self.adjacency. Returns a dict loc->distance.
        """
        dist = {source: 0}
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.adjacency.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        """
        Compute heuristic estimate for the node (node.state is a frozenset of facts).
        Returns an integer cost estimate.
        """
        state = node.state

        # Quick goal check: heuristic is 0 for goal states.
        if self.goals <= state:
            return 0

        # Parse state facts.
        at_facts = {}           # object -> location (for objects that are at a location)
        usable_spanners = set() # spanners that are usable
        carried_by = {}         # spanner -> man who carries it (if any)
        carrying_subjects = set()  # set of subjects that appear in carrying facts
        nuts_loose = set()      # nuts that are loose (loose facts)
        nuts_tightened = set()  # nuts that are tightened (tightened facts)

        # Collect facts and identify types by predicate appearance.
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                at_facts[obj] = loc
            elif pred == "usable" and len(parts) >= 2:
                usable_spanners.add(parts[1])
            elif pred == "carrying" and len(parts) >= 3:
                man = parts[1]; sp = parts[2]
                carried_by[sp] = man
                carrying_subjects.add(man)
            elif pred == "loose" and len(parts) >= 2:
                nuts_loose.add(parts[1])
            elif pred == "tightened" and len(parts) >= 2:
                nuts_tightened.add(parts[1])

        # Determine which goal nuts still need to be tightened.
        goal_nuts = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "tightened" and len(parts) >= 2:
                nut = parts[1]
                if goal not in state:
                    goal_nuts.append(nut)

        # If no nuts need tightening, return 0.
        if not goal_nuts:
            return 0

        # Determine the man's identity and location.
        man = None
        man_loc = None
        # Prefer subject from carrying facts if any (reliable).
        if carrying_subjects:
            man = next(iter(carrying_subjects))
            man_loc = at_facts.get(man, None)
        else:
            # Heuristic: choose an object that appears at a location but is not
            # a spanner (in usable_spanners) nor a nut (in nuts_loose/nuts_tightened).
            spanners_candidates = set(usable_spanners)
            nuts_candidates = set(nuts_loose) | set(nuts_tightened)
            for obj, loc in at_facts.items():
                if obj not in spanners_candidates and obj not in nuts_candidates:
                    man = obj
                    man_loc = loc
                    break
            # Fallback: if none found, try to pick any object that has an at-fact
            if man is None and at_facts:
                man, man_loc = next(iter(at_facts.items()))

        # If still no man or location known, we cannot compute distances reliably.
        if man is None or man_loc is None:
            # Return a conservative finite estimate.
            return 10**6

        # Build locations set for distance computations: include all locations
        locations = set()
        # From adjacency keys and values
        for a, vs in self.adjacency.items():
            locations.add(a)
            for v in vs:
                locations.add(v)
        # From state at facts second argument values
        for obj, loc in at_facts.items():
            locations.add(loc)

        # Precompute distances from man and from every spanner location we will need.
        # We'll compute BFS from man_loc and from each spanner location encountered below.
        dist_from_man = self.bfs_distances_from(man_loc)

        # Build list of usable spanners (we only consider spanners that are currently usable).
        # A spanner is usable if it appears in usable_spanners.
        usable_list = list(usable_spanners)

        # If there are fewer usable spanners than nuts to tighten, return large cost.
        if len(usable_list) < len(goal_nuts):
            return 10**6

        # Determine spanner locations considering carried spanners are at man_loc.
        spanner_loc = {}
        for sp in usable_list:
            if sp in carried_by:
                # carried by someone (likely the man), location is that person's location
                carrier = carried_by[sp]
                sp_loc = at_facts.get(carrier, man_loc)  # default to man's loc if unknown
            else:
                sp_loc = at_facts.get(sp, None)
            if sp_loc is None:
                # If we lack a concrete location, assign a placeholder unreachable location
                sp_loc = "__unknown_loc__" + sp
            spanner_loc[sp] = sp_loc

        # Nut locations (from at facts). If a nut has no at-fact, heuristic returns large.
        nut_loc = {}
        for nut in goal_nuts:
            loc = at_facts.get(nut, None)
            if loc is None:
                # If we cannot find nut location, consider it unreachable for heuristic.
                return 10**6
            nut_loc[nut] = loc

        # Precompute BFS distances from any spanner location we need (and from man we already have).
        dist_from_spanner_loc = {}
        for sp in usable_list:
            loc = spanner_loc[sp]
            if loc == man_loc:
                # we can reuse dist_from_man
                dist_from_spanner_loc[sp] = dist_from_man
            else:
                dist_from_spanner_loc[sp] = self.bfs_distances_from(loc)

        INF = 10**6

        # Build cost matrix: rows = nuts (len N), cols = usable spanners (len M)
        nuts = list(goal_nuts)
        spanners = usable_list
        N = len(nuts)
        M = len(spanners)
        cost = [[INF for _ in range(M)] for _ in range(N)]

        for i, nut in enumerate(nuts):
            nloc = nut_loc[nut]
            for j, sp in enumerate(spanners):
                slog = spanner_loc[sp]
                # distance from man to spanner location
                d1 = dist_from_man.get(slog, INF)
                # if spanner is carried by the man, pickup cost is 0 and d1 should be 0
                pickup_cost = 0 if (sp in carried_by and carried_by[sp] == man) else 1
                # distance from spanner location to nut location
                d2 = dist_from_spanner_loc[sp].get(nloc, INF)
                # tighten cost is 1
                if d1 >= INF or d2 >= INF:
                    total = INF
                else:
                    total = d1 + pickup_cost + d2 + 1
                cost[i][j] = total

        # Solve minimum assignment of M spanners to N nuts (N <= M) using DP over bitmasks.
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(i, mask):
            """
            Minimal cost to assign spanners to nuts[i:].
            mask indicates which spanners are already used (bit j set means spanner j used).
            """
            if i == N:
                return 0
            best = INF
            for j in range(M):
                if not (mask & (1 << j)):
                    c = cost[i][j]
                    if c >= INF:
                        continue
                    val = c + dp(i + 1, mask | (1 << j))
                    if val < best:
                        best = val
            return best

        assignment_cost = dp(0, 0)
        if assignment_cost >= INF:
            return 10**6
        # Return integer estimate
        return int(assignment_cost)