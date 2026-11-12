from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (ud3uf11z). The domain
    contains two kinds of objects (nxbd1hc3 and t4rdxh32). A marker predicate
    (omggbn9z <loc>) denotes a currently active location and can be moved
    along directed edges (acovodrf <from> <to>) using movement operators.
    For each object p of type nxbd1hc3 the goal is (a7sb56om p). To achieve
    (a7sb56om p) the domain requires an intermediate fact (f2xo7hb6 p)
    produced by performing an action at a location w where (kut2ksnk p w)
    holds and the marker is at w. A second action requires the marker to be
    at a location w' where (klw11kjf p w') holds and (f2xo7hb6 p) is true.
    Therefore each goal object typically requires (possibly) moving the marker
    and then applying one or two domain actions.

    Assumptions
    -----------
    - Each action (moving the marker, applying diwlmgg9 or y7nik9q2) has unit
      cost. Thus the heuristic measures the number of actions.
    - klw11kjf facts are commonly static, while kut2ksnk facts may appear in
      the current state and can be removed by actions. Therefore kut2ksnk is
      read from the current state; klw11kjf is read from static facts and the
      current state (to be robust).
    - The marker (omggbn9z) may appear at multiple locations; the heuristic
      chooses the best marker location to start from.
    - The heuristic is not required to be admissible; it aims to be informative
      and computationally cheap.

    Heuristic Initialization
    ------------------------
    - Extract the directed adjacency graph of t4rdxh32 locations from static
      facts of the form (acovodrf <src> <tgt>).
    - Extract static klw11kjf relations (klw11kjf <p> <loc>) into a mapping
      for quick lookup.
    - Parse the goal set to obtain the set of goal objects (those that require
      (a7sb56om <p>)).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state) the heuristic proceeds as follows:

    1. Quick goal check:
       - If all goal facts (a7sb56om <p>) are already present in the state,
         return 0.

    2. Gather dynamic facts from the state:
       - Current marker locations: all `loc` such that (omggbn9z loc) is in state.
       - Current kut2ksnk relations for objects: (kut2ksnk p loc) as they may
         change during the plan.
       - Current f2xo7hb6 facts: which objects already have the intermediate
         fact set.
       - Also include any klw11kjf facts that appear in the state in addition
         to the static ones.

    3. Distance computation:
       - The marker moves along directed edges (acovodrf). The heuristic uses
         shortest-path distances (in number of marker-move actions) between
         locations as computed by BFS on the adjacency graph. BFS is computed
         on demand and cached for each source location to keep the cost small.

    4. Per-goal cost estimate:
       - For each goal object p not yet satisfying (a7sb56om p), compute the
         minimal local cost to transform the current state so that the
         (a7sb56om p) can be achieved, assuming independent handling of each
         object (i.e., ignoring possible savings due to reusing moves across
         different objects).
       - Two cases:
         a) If (f2xo7hb6 p) is already true: we only need to have the marker
            at some klw location w' (where klw11kjf p w' holds) and execute
            y7nik9q2. Cost = min_over_marker_locs distance(marker -> w') + 1.
         b) If (f2xo7hb6 p) is false: we must first perform diwlmgg9 at some
            kut location w (where kut2ksnk p w is currently true) and then
            perform y7nik9q2 at some klw location w'. Cost =
            min_{marker, w, w'} [dist(marker -> w) + 1 (diwlmgg9) +
                                 dist(w -> w') + 1 (y7nik9q2)].
            Note: if w == w' the dist(w -> w') term is zero and both actions
            can be executed without an intermediate marker move.

    5. Aggregation:
       - Sum the per-object minimal costs for all unsatisfied goal objects.
         This yields the heuristic estimate. The heuristic therefore favors
         states that reduce per-object distances and already possess
         intermediate facts (f2xo7hb6) or final facts (a7sb56om).

    Notes on robustness and finiteness:
    - If no marker is present or no path exists to required locations, the
      heuristic returns a large finite value to guide the search away from
      hopeless nodes while remaining finite for solvable states (where paths
      and markers exist).
    """

    def __init__(self, task):
        """
        Extract static information needed for heuristic computation.

        Parameters
        ----------
        task : Task
            The planning task object provided by the planner. We extract:
            - self.adj: directed adjacency mapping for locations from acovodrf
            - self.klw_static: mapping p -> set(loc) from static klw11kjf facts
            - self.goals_objs: set of nxbd1hc3 objects that must satisfy a7sb56om
        """
        self.goals = task.goals  # set of goal fact strings
        self.static = task.static

        # Build directed adjacency graph of t4rdxh32 locations from acovodrf
        self.adj = defaultdict(list)  # src -> [tgt, ...]
        self.locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "acovodrf" and len(parts) >= 3:
                src, tgt = parts[1], parts[2]
                self.adj[src].append(tgt)
                self.locations.add(src)
                self.locations.add(tgt)

        # Extract static klw11kjf relations (may also appear in state; we will
        # merge them at runtime)
        self.klw_static = defaultdict(set)  # p -> set(loc)
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "klw11kjf" and len(parts) >= 3:
                p, loc = parts[1], parts[2]
                self.klw_static[p].add(loc)
                self.locations.add(loc)

        # Extract goal objects that require (a7sb56om p)
        self.goal_objects = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "a7sb56om" and len(parts) >= 2:
                self.goal_objects.add(parts[1])

        # Cache BFS distance maps per source location to avoid repeated BFS.
        self._bfs_cache = {}

    def _bfs_distances(self, src):
        """Return a dict {node: distance} with shortest distances (in edges)
        from src to all reachable nodes using directed adjacency. Cache results."""
        if src in self._bfs_cache:
            return self._bfs_cache[src]
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._bfs_cache[src] = dist
        return dist

    def __call__(self, node):
        """
        Compute an estimate (integer) of the number of actions required to
        reach a goal state from the current node.state.

        The heuristic returns 0 iff the node.state satisfies all goal facts.
        """
        state = node.state  # frozenset of fact strings

        # If goal reached, heuristic = 0
        if self.goals <= state:
            return 0

        # Build dynamic information from the current state
        # Marker locations (locations that currently have the marker)
        marker_locs = {
            get_parts(fact)[1]
            for fact in state
            if get_parts(fact)[0] == "omggbn9z" and len(get_parts(fact)) >= 2
        }

        # Current f2xo7hb6 set (objects that already have the intermediate fact)
        has_f2 = {
            get_parts(fact)[1]
            for fact in state
            if get_parts(fact)[0] == "f2xo7hb6" and len(get_parts(fact)) >= 2
        }

        # Current kut2ksnk relations (may change over time) p -> set(loc)
        kut_map = defaultdict(set)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "kut2ksnk" and len(parts) >= 3:
                p, loc = parts[1], parts[2]
                kut_map[p].add(loc)
                self.locations.add(loc)

        # Merge klw11kjf: static ones plus any klw facts present in the current state.
        klw_map = defaultdict(set, {p: set(locs) for p, locs in self.klw_static.items()})
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "klw11kjf" and len(parts) >= 3:
                p, loc = parts[1], parts[2]
                klw_map[p].add(loc)
                self.locations.add(loc)

        # If there are no marker locations, assign a large but finite penalty so
        # the heuristic is finite. (Solvable states should typically have a marker.)
        INF = 10**6
        if not marker_locs:
            return INF

        total_cost = 0

        # For each goal object not yet satisfied compute minimal local cost
        for p in self.goal_objects:
            goal_fact = f"(a7sb56om {p})"
            if goal_fact in state:
                continue  # already satisfied

            best_cost = INF

            # Candidate klw locations for the y7nik9q2 action
            klw_locs = klw_map.get(p, set())
            # Candidate kut locations for the diwlmgg9 action (from current state)
            kut_locs = kut_map.get(p, set())

            # Case A: intermediate fact already present -> only y7nik9q2 required
            if p in has_f2:
                # For each possible klw location, choose best marker -> klw distance
                for klw in klw_locs:
                    for m in marker_locs:
                        # compute distance from marker m to klw
                        if m == klw:
                            d = 0
                        else:
                            dmap = self._bfs_distances(m)
                            d = dmap.get(klw, None)
                        if d is None:
                            continue
                        cost = d + 1  # moves to bring marker + y7nik9q2
                        if cost < best_cost:
                            best_cost = cost

            # Case B: intermediate fact not present -> diwlmgg9 then y7nik9q2
            # Need a kut location now (from kut_locs). klw_locs may be same or different.
            if kut_locs:
                for kut in kut_locs:
                    for klw in klw_locs or {kut}:
                        for m in marker_locs:
                            # distance marker -> kut
                            if m == kut:
                                d1 = 0
                            else:
                                dmap1 = self._bfs_distances(m)
                                d1 = dmap1.get(kut, None)
                            if d1 is None:
                                continue

                            # distance kut -> klw
                            if kut == klw:
                                d2 = 0
                            else:
                                dmap2 = self._bfs_distances(kut)
                                d2 = dmap2.get(klw, None)
                            if d2 is None:
                                continue

                            cost = d1 + 1 + d2 + 1  # move to kut, diwlmgg9, move to klw, y7nik9q2
                            if cost < best_cost:
                                best_cost = cost

            # If best_cost still INF, we set a finite large penalty to keep heuristic finite.
            if best_cost >= INF:
                best_cost = INF

            total_cost += best_cost

            # Early exit: if total is already huge, return INF to avoid unnecessary work.
            if total_cost >= INF:
                return INF

        # Return integer heuristic value
        return int(total_cost)