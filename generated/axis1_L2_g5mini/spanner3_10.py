from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (du770mtw) domain.

    The main goal of this domain is to produce ne6sl557 facts for certain
    wqid8mog objects (these are the planning goals). The only action that
    produces ne6sl557 is `f7y6nzq3`. To be applicable for a specific
    wqid8mog object R (the goal object) `f7y6nzq3` requires:
      - of6wn8m4 R          (R is "available"),
      - qkdx0xe2 R T        (R is at some mz6r1d30 node T),
      - ndg963qz C          (a du770mtw object C with ndg flag),
      - fqyrzl30 P C        (a pt4b08k4 object P linked to C),
      - qkdx0xe2 P T        (P is at the same mz6r1d30 T as R).

    The `fqyrzl30` link can be created by action `jql8txb3` if both:
      - qkdx0xe2 C T and qkdx0xe2 P T hold.
    The pt4b08k4 objects can be moved along a directed graph of mz6r1d30 nodes
    using action `nm6lzp2x` when a static `lpmls56r` edge exists between nodes.
    Each move along one `lpmls56r` edge costs one application of `nm6lzp2x`.

    The heuristic estimates the number of actions needed to produce the missing
    ne6sl557 facts. It reasons about:
      - which ndg-enabled C (du770mtw) objects are usable (ndg963qz true),
      - whether an existing fqyrzl30 already satisfies the linking,
      - how many nm6lzp2x moves are needed to bring some P to the required T,
      - and the two terminal actions jql8txb3 (create fqyrzl30) and f7y6nzq3
        (produce ne6sl557).

    Assumptions
    -----------
    - qkdx0xe2 relations encode the current "position" of objects (object -> mz6r1d30).
    - lpmls56r static facts form a directed adjacency relation between mz6r1d30
      nodes; nm6lzp2x moves a pt4b08k4 object one step along an lpmls56r edge.
    - Only f7y6nzq3 produces the goal predicate ne6sl557.
    - of6wn8m4 for a goal object must be true before applying f7y6nzq3; if it
      is not true in the current state then the goal is considered unreachable
      for the purpose of the heuristic (a large penalty is returned).
    - cz2xr (du770mtw) resources with ndg963qz can be consumed once (f7y6nzq3
      deletes ndg963qz). The heuristic models that consumption by assigning
      each available cz2xr to at most one missing goal (greedy assignment).
    - pt4b08k4 objects are identified from state facts (first argument of
      fqyrzl30 or of qkdx0xe2 entries not belonging to other recognized types).
      This detection is robust in practice for the instances considered.

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts (task.goals).
    - Extract the static directed graph of mz6r1d30 nodes from lpmls56r facts
      (task.static). This graph is used to compute shortest (directed) distances
      between mz6r1d30 nodes for nm6lzp2x move costs.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. From the current state, compute:
       - which goals (ne6sl557 ...) are already satisfied; ignore them.
       - qk_map: mapping object -> mz6r1d30 for every qkdx0xe2 fact.
       - fq_map: mapping pt4b08k4 object -> set of du770mtw objects for every
         fqyrzl30 fact (these are existing links).
       - ndg_set: set of du770mtw objects that have ndg963qz.
       - of_set: set of wqid8mog objects that currently have of6wn8m4.
    2. Determine the set of missing goal objects R (each R is a wqid8mog for
       which ne6sl557 R is not yet in the state).
       - If there are no missing goals, return 0.
    3. For each missing goal object R:
       - Confirm R has a qkdx0xe2 R T (its mz6r1d30 T). If not present, treat
         the goal as unreachable (large penalty).
       - Gather candidate C (du770mtw) objects that both:
         a) are in ndg_set (ndg963qz true), and
         b) have qkdx0xe2 C T (C is currently at the same T as R).
         Only such C can be used: jql8txb3 requires qkdx0xe2 C T, and f7y6nzq3
         additionally requires ndg963qz C.
       - For each candidate C compute the minimal cost to enable f7y6nzq3:
         i) If there exists a pt P such that
               fqyrzl30 P C is already true AND qkdx0xe2 P T holds,
            then f7y6nzq3 can be applied immediately (cost = 1).
         ii) Otherwise, find a pt P that can be moved to T using nm6lzp2x
             along the directed lpmls56r graph. The move cost is the number of
             steps (edges) along the directed path from P's current node to T.
             After moving P to T, one jql8txb3 action (cost 1) creates fqyrzl30,
             and one f7y6nzq3 action (cost 1) produces the goal.
             So total cost in this case is (distance_moves + 2).
         iii) If no P can reach T, C is not usable for R.
       - If no candidate C is usable for R (or R lacks of6wn8m4), consider the
         goal unreachable and assign a large penalty.
    4. Resolve the competition for limited C resources: each ndg-enabled C can
       produce at most one goal (f7y6nzq3 consumes ndg963qz). Use a greedy
       minimal-pairing assignment: repeatedly pick the (R,C) pair with the
       smallest cost, assign C to R, add cost, and remove R and C from
       consideration. This yields a reasonable approximate minimal total
       cost while keeping the heuristic efficient to compute.
    5. If any remaining R cannot be assigned to a C, add a large penalty per
       remaining goal to the total heuristic. Return the summed cost.

    The heuristic is not admissible: it approximates the true number of actions
    and intentionally simplifies some reuse/resource interactions to remain
    efficient.
    """

    def __init__(self, task):
        # Store goal conditions as provided by the task (set of fact strings).
        self.goals = task.goals
        # Static facts: used to build the directed graph for lpmls56r edges.
        static_facts = task.static

        # Build directed adjacency for mz6r1d30 nodes from lpmls56r static facts.
        # Each lpmls56r A B means nm6lzp2x can move a pt object from A to B.
        self.adj = defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "lpmls56r" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj[src].add(dst)

    def _bfs_distance(self, start, target):
        """
        Compute shortest directed distance from start to target in the lpmls56r graph.
        Returns an integer distance (number of edges) or None if unreachable.
        """
        if start == target:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr == target:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # Quick goal check: 0 if all goals are satisfied in state.
        if self.goals <= state:
            return 0

        # Parse the state into convenient structures.
        qk_map = {}  # object -> mz6r1d30 location from qkdx0xe2 facts
        fq_map = defaultdict(set)  # pt -> set of cz2xr from fqyrzl30 facts
        ndg_set = set()  # set of cz2xr with ndg963qz
        of_set = set()   # set of wqid8mog objects with of6wn8m4
        ne_set = set()   # set of wqid8mog with ne6sl557 (already satisfied)

        # Populate the maps/sets.
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "qkdx0xe2" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                qk_map[obj] = loc
            elif pred == "fqyrzl30" and len(parts) >= 3:
                p, c = parts[1], parts[2]
                fq_map[p].add(c)
            elif pred == "ndg963qz" and len(parts) >= 2:
                ndg_set.add(parts[1])
            elif pred == "of6wn8m4" and len(parts) >= 2:
                of_set.add(parts[1])
            elif pred == "ne6sl557" and len(parts) >= 2:
                ne_set.add(parts[1])

        # Determine which goal objects still need to be achieved.
        # Goals are stored as fact strings like "(ne6sl557 p7oo9n0w)" in self.goals.
        missing_goals = []
        for gfact in self.goals:
            gparts = get_parts(gfact)
            if gparts and gparts[0] == "ne6sl557" and len(gparts) >= 2:
                obj = gparts[1]
                if obj not in ne_set:  # not yet true in state
                    missing_goals.append(obj)

        if not missing_goals:
            return 0

        # Build candidate cz2xr objects: those with ndg963qz and a qkdx0xe2 location.
        cz_candidates = [c for c in ndg_set if c in qk_map]

        # If there are no cz candidates but goals remain, penalize strongly.
        if not cz_candidates:
            # Use a large but finite penalty.
            return 1000 * len(missing_goals)

        # Gather pt candidates: objects that might be moved.
        # Heuristic identifies pt-like objects as those appearing as first arg in fqyrzl30
        # or as first arg in qkdx0xe2 but not being known cz or goal/wqid8mog objects.
        pts = set(fq_map.keys())
        for obj in list(qk_map.keys()):
            if obj not in pts and obj not in cz_candidates and obj not in of_set and obj not in ne_set:
                # It's likely a pt4b08k4 (or at least movable); include it as potential mover.
                pts.add(obj)

        # Build cost matrix: for each missing goal R, for each candidate C that has the same T,
        # compute the minimal cost to produce ne6sl557 R using C.
        # cost(R,C) = 1 if a P exists with fqyrzl30 P C and qkdx0xe2 P T already AND of6wn8m4 R present.
        # Otherwise cost = distance(P_current_T -> T) + 2  (nm6lzp2x moves + jql8txb3 + f7y6nzq3).
        INF = 10**6
        # Precompute distances from each pt current location to all target nodes lazily as needed.
        # We'll compute per (pt, target) using BFS method above.
        # Map of (pt, target) -> distance or None
        dist_cache = {}

        # For fast lookup: map cz -> its T (qk_map)
        cz_to_T = {c: qk_map[c] for c in cz_candidates}

        # For each missing goal, compute candidate costs to usable czs.
        goal_to_costs = {}
        for R in missing_goals:
            # Verify R has of6wn8m4 and qkdx0xe2 location; otherwise unreachable.
            if R not in of_set:
                # cannot apply f7y6nzq3 on this R; unreachable under current assumptions.
                goal_to_costs[R] = {}
                continue
            if R not in qk_map:
                goal_to_costs[R] = {}
                continue
            T = qk_map[R]
            costs_for_R = {}
            # Consider only cz with same T (they must share T to satisfy jql8txb3 precondition)
            for C in cz_candidates:
                if cz_to_T.get(C) != T:
                    continue
                best_cost = INF
                # Option 1: existing fqyrzl30 P C and P is at T now -> cost 1 (just apply f7y6nzq3)
                for P in fq_map:
                    if C in fq_map[P] and qk_map.get(P) == T:
                        # All preconditions for f7y6nzq3 are satisfied (given qk_map[R]==T and of_set)
                        best_cost = min(best_cost, 1)
                        break
                if best_cost == 1:
                    costs_for_R[C] = 1
                    continue
                # Option 2: move some P to T and then do jql8txb3 + f7y6nzq3
                for P in pts:
                    if P not in qk_map:
                        continue
                    key = (P, T)
                    if key in dist_cache:
                        d = dist_cache[key]
                    else:
                        d = self._bfs_distance(qk_map[P], T)
                        dist_cache[key] = d
                    if d is None:
                        continue
                    # Total cost: d (nm6lzp2x moves) + 1 (jql8txb3) + 1 (f7y6nzq3)
                    total = d + 2
                    if total < best_cost:
                        best_cost = total
                if best_cost < INF:
                    costs_for_R[C] = best_cost
            goal_to_costs[R] = costs_for_R

        # Now assign cz resources to goals greedily using minimal available pair each iteration.
        remaining_goals = set(missing_goals)
        remaining_czs = set(cz_candidates)
        total_cost = 0

        while remaining_goals:
            # Find minimal cost pair (R,C)
            best_pair = None
            best_pair_cost = INF
            for R in list(remaining_goals):
                costs_for_R = goal_to_costs.get(R, {})
                for C, cost in costs_for_R.items():
                    if C in remaining_czs and cost < best_pair_cost:
                        best_pair_cost = cost
                        best_pair = (R, C)
            if best_pair is None:
                # No assignable remaining goal: penalize remainder.
                total_cost += 1000 * len(remaining_goals)
                break
            R, C = best_pair
            total_cost += best_pair_cost
            remaining_goals.remove(R)
            remaining_czs.remove(C)

        return int(total_cost)