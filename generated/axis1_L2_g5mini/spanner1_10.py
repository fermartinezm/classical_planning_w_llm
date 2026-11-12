from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Return the components of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (du770mtw) domain.

    This heuristic estimates the number of actions required to achieve all goal
    predicates of the form (ne6sl557 <wqid8mog>). In this domain a goal fact
    (ne6sl557 ris) is achieved by applying the action f7y6nzq3 which requires:
      - a du770mtw object cz such that (ndg963qz cz) is true (a reusable resource
        for a single application because f7y6nzq3 deletes ndg963qz cz),
      - a pt4b08k4 object n that is linked to cz via (fqyrzl30 n cz) (this link
        can be created earlier using action jql8txb3 whenever n and cz are co-located),
      - the pt4b08k4 object n to be located at the same mz6r1d30 location as the
        wqid8mog ris (both must have (qkdx0xe2 <obj> <loc>) at the same loc),
      - of6wn8m4 ris must be true prior to applying f7y6nzq3.
    Relevant actions:
      - nm6lzp2x moves a pt4b08k4 object along a directed edge (lpmls56r edge)
        between mz6r1d30 nodes (cost = number of nm6lzp2x applications).
      - jql8txb3 creates (fqyrzl30 n cz) when n and cz are co-located (cost = 1).
      - f7y6nzq3 performs the final goal-producing action (cost = 1).

    Assumptions
    -----------
    - The only dynamic way to change pt4b08k4 locations is via nm6lzp2x along
      directed lpmls56r edges; therefore movement cost equals the directed
      graph distance along lpmls56r.
    - du770mtw objects (cz) and wqid8mog objects (ris) do not move.
    - of6wn8m4 for a ris must be true in the current state for the goal to be
      achievable (there is no operator that sets of6wn8m4 true).
    - jql8txb3 requires both cz and n to be co-located at the time of linking.
      Once linked, fqyrzl30 remains until used by f7y6nzq3.
    - Each ndg963qz cz can be used at most once to satisfy one goal because
      f7y6nzq3 deletes ndg963qz cz. The heuristic therefore assigns distinct
      cz objects to goals (greedy assignment).

    Heuristic Initialization
    ------------------------
    - Extract the set of goal objects (the wqid8mog names) from task.goals.
    - Extract the static directed graph of mz6r1d30 nodes from lpmls56r facts
      in task.static. The adjacency is stored to compute shortest directed
      distances for pt4b08k4 movements.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic proceeds as follows:

    1. Quick check: If all goals are already satisfied (all (ne6sl557 <ris>)
       facts are present in the current state) the heuristic returns 0.

    2. Parse the state to build:
       - loc_of: mapping from any object to its current mz6r1d30 location according
         to (qkdx0xe2 <obj> <loc>) facts.
       - of_set: set of wqid8mog objects for which (of6wn8m4 <ris>) holds.
       - linked: mapping cz -> set(n) for existing (fqyrzl30 n cz) links.
       - ndg_set: set of du770mtw objects cz for which (ndg963qz cz) holds.

    3. Precompute directed shortest path distances on the lpmls56r graph (BFS)
       between all relevant mz6r1d30 nodes encountered in the static facts and
       in the state's qkdx0xe2 facts.

    4. For each unsatisfied goal ris:
       - If of6wn8m4 ris is not true or ris has no qkdx0xe2 location then mark
         the goal as currently unreachable (very large cost).
       - Otherwise compute minimal per-cz cost for all available cz in ndg_set:
         * Option 1 (existing link): If there exists n with fqyrzl30 n cz,
           cost = distance(loc_of[n], loc_of[ris]) + 1 (the +1 is for f7y6nzq3).
         * Option 2 (create link): If cz has a qkdx0xe2 location cz_loc,
           choose any candidate pt n (an object that can move, heuristically
           extracted from qkdx0xe2 facts but excluding known du770mtw/wqid8mogs),
           cost = distance(loc_of[n], cz_loc) + 1 (jql8txb3) +
                  distance(cz_loc, loc_of[ris]) + 1 (f7y6nzq3).
       - The minimal feasible cost over all cz gives the minimal cost to achieve
         that ris using a specific cz.

    5. Because each cz can be used only once (ndg963qz is deleted by
       f7y6nzq3), the heuristic assigns cz resources to unsatisfied goals using
       a simple greedy matching: repeatedly pick the (goal, cz) pair with the
       smallest per-pair cost, assign them, sum costs, and remove that goal and
       cz from consideration. This yields a fast, informative estimate (not
       necessarily optimal but effective for guidance).

    6. If any goal remains unassigned (no feasible cz) then return a large
       finite penalty proportional to the number of remaining goals.

    Notes on costs returned
    -----------------------
    - The heuristic returns an integer estimate of the number of actions.
    - The value is 0 for goal states.
    - The heuristic is finite for states that appear solvable under the domain's
      static connectivity; otherwise it returns a large finite penalty.
    """

    def __init__(self, task):
        # Store raw goals (strings) and parse goal objects for ne6sl557 facts.
        self.goals = set(task.goals)
        self.goal_objs = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "ne6sl557" and len(parts) >= 2:
                self.goal_objs.append(parts[1])

        # Extract directed adjacency graph for mz6r1d30 nodes from static lpmls56r facts.
        self.adj = defaultdict(set)  # directed graph: src -> set(dsts)
        self.nodes = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "lpmls56r" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj[src].add(dst)
                self.nodes.add(src)
                self.nodes.add(dst)

        # Large finite penalty for unreachable goals.
        self.INF_COST = 1000

        # Cache for shortest path distances between mz nodes: {src: {dst: dist}}
        self._dist_cache = {}

    def _bfs_distances_from(self, src):
        """Return dict of directed shortest distances from src to all reachable nodes."""
        if src in self._dist_cache:
            return self._dist_cache[src]
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[src] = dist
        return dist

    def _directed_distance(self, src, dst):
        """Return directed distance from src to dst along lpmls56r graph or math.inf."""
        if src == dst:
            return 0
        if src not in self.nodes:
            return math.inf
        dist_from_src = self._bfs_distances_from(src)
        return dist_from_src.get(dst, math.inf)

    def __call__(self, node):
        state = node.state

        # 1) If all goals satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse relevant predicates from state.
        loc_of = {}  # object -> mz location from qkdx0xe2 facts
        of_set = set()  # objects with of6wn8m4
        linked = defaultdict(set)  # cz -> set of n with fqyrzl30 n cz
        ndg_set = set()  # set of cz with ndg963qz cz

        # collect objects appearing in various positions to heuristically identify movable pt4b08k4 candidates
        appears_as_fqyrzl30_second = set()  # cz seen as second arg in fqyrzl30
        appears_as_of = set()  # wqid8mog (first arg in of6wn8m4)
        appears_as_ne6 = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "qkdx0xe2" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                loc_of[obj] = loc
                self.nodes.add(loc)  # ensure node considered in graph
            elif pred == "of6wn8m4" and len(parts) >= 2:
                of_set.add(parts[1])
                appears_as_of.add(parts[1])
            elif pred == "fqyrzl30" and len(parts) >= 3:
                n, cz = parts[1], parts[2]
                linked[cz].add(n)
                appears_as_fqyrzl30_second.add(cz)
            elif pred == "ndg963qz" and len(parts) >= 2:
                ndg_set.add(parts[1])
            elif pred == "ne6sl557" and len(parts) >= 2:
                appears_as_ne6.add(parts[1])

        # Candidate movable pt4b08k4 objects:
        # Heuristic rule: any object that has a qkdx0xe2 location and that is not
        # identified as a du770mtw (cz candidates appear in second-arg of fqyrzl30
        # and in ndg_set) and not a wqid8mog (appear in of_set or ne6sl557).
        movable_candidates = []
        for obj, loc in loc_of.items():
            # Exclude cz and ris like objects
            if obj in ndg_set or obj in appears_as_fqyrzl30_second:
                # likely a du770mtw -> skip as mover candidate
                continue
            if obj in of_set or obj in appears_as_ne6:
                # likely a wqid8mog -> skip
                continue
            # otherwise treat as candidate pt4b08k4 mover
            movable_candidates.append(obj)

        # If no movable candidates but some links exist, include linked n's as candidates
        if not movable_candidates:
            for cz, nset in linked.items():
                for n in nset:
                    if n in loc_of:
                        movable_candidates.append(n)

        # Helper: compute minimal cost to satisfy one goal using a particular cz.
        def cost_for_goal_and_cz(ris, cz):
            # Preconditions: ris must have location and of6wn8m4
            if ris not in loc_of or ris not in of_set:
                return math.inf

            ris_loc = loc_of[ris]

            best = math.inf

            # Option A: use existing link fqyrzl30 n cz (no jql8txb3 needed)
            for n in linked.get(cz, ()):
                if n not in loc_of:
                    continue
                n_loc = loc_of[n]
                d = self._directed_distance(n_loc, ris_loc)
                if d is math.inf:
                    continue
                # d moves by nm6lzp2x plus 1 for f7y6nzq3
                cost = d + 1
                if cost < best:
                    best = cost

            # Option B: create a link by jql8txb3 at cz location then move n to ris
            # jql8txb3 requires cz to have a qkdx0xe2 location where n can be moved to
            if cz in loc_of:
                cz_loc = loc_of[cz]
                for n in movable_candidates:
                    # n must have a location to move from
                    if n not in loc_of:
                        continue
                    n_loc = loc_of[n]
                    d1 = self._directed_distance(n_loc, cz_loc)
                    if d1 is math.inf:
                        continue
                    # after jql8txb3 we still need to move the same n (now linked) to ris_loc
                    d2 = self._directed_distance(cz_loc, ris_loc)
                    if d2 is math.inf:
                        continue
                    # d1 moves to cz_loc, +1 for jql8txb3, d2 moves to ris_loc, +1 for f7y6nzq3
                    cost = d1 + 1 + d2 + 1
                    if cost < best:
                        best = cost

            return best

        # Build cost matrix: for each unsatisfied goal and for each available cz
        unsat_goals = [g for g in self.goal_objs if f"(ne6sl557 {g})" not in state]
        if not unsat_goals:
            return 0

        if not ndg_set:
            # No cz resources available at all -> impossible in current state
            return self.INF_COST * len(unsat_goals)

        # Precompute per-pair costs
        pair_costs = []
        for ris in unsat_goals:
            for cz in ndg_set:
                c = cost_for_goal_and_cz(ris, cz)
                if c != math.inf:
                    pair_costs.append((c, ris, cz))

        # If no feasible pairs at all, return large penalty
        if not pair_costs:
            return self.INF_COST * len(unsat_goals)

        # Greedy matching: repeatedly pick smallest-cost pair, remove that ris and cz
        pair_costs.sort()
        total_cost = 0
        used_cz = set()
        assigned_goals = set()

        for c, ris, cz in pair_costs:
            if ris in assigned_goals or cz in used_cz:
                continue
            # assign
            assigned_goals.add(ris)
            used_cz.add(cz)
            total_cost += int(c)

            if len(assigned_goals) == len(unsat_goals):
                break

        # If some goals remain unassigned, add large penalty for each leftover
        remaining = len(unsat_goals) - len(assigned_goals)
        if remaining > 0:
            total_cost += self.INF_COST * remaining

        return int(total_cost)