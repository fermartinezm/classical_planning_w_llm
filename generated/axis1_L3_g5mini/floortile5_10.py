from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact string by removing parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the `spanner` (secret_domain) planning domain. The domain's
    goal facts are of the form (tightened-2 <locatable> <location>). To achieve
    such a fact an operator must (a) have a nut that is loose-2 at the goal
    location and (b) have a `tightened` relation for some nut at a neighboring
    locatable (a predecessor in the static/dynamic relation graph). In
    addition, many operators require the `link` predicate to be true at the
    target locatable before producing `tightened-2`. Operators that propagate
    `tightened` move it along edges defined by the static relations
    `loose`, `carrying`, and `usable`.

    This heuristic estimates the number of actions still required to satisfy
    all `tightened-2` goals by:
    - counting unsatisfied tightened-2 goals and, for each,
    - estimating the minimal number of propagation steps to bring a `tightened`
      instance to the right predecessor locatable (d_tight),
    - estimating the minimal number of moves to bring the `link` predicate to
      the goal locatable (d_link),
    - adding 1 for the final action that produces the `tightened-2` fact.
    The heuristic sums those per-goal estimates.

    Assumptions
    -----------
    - Movement of the `tightened` relation is possible along edges induced by
      facts with predicates `loose`, `carrying`, and `usable`. For a static
      fact "(loose A B)" a single propagation action can move `tightened` from
      B to A.
    - A `tightened-2` fact for locatable G and location L requires a nut that
      is (loose-2 ?nut L) in the current state (or initial state). If no
      nut is loose-2 at that location the heuristic will produce a large cost
      (treating the goal as hard to reach).
    - The `link` predicate must be true at the goal locatable when the final
      action creating `tightened-2` is applied. Moving the `tightened` relation
      tends to also affect `link`, but modeling that exactly is complex; we
      therefore separately estimate a minimal move distance for `link` and add
      it to the propagation distance. The heuristic is allowed to be
      non-admissible, so it may overestimate in cases where `tightened`
      propagation already moves the `link`.

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts (task.goals).
    - Extract static relation edges from the static facts available in the
      constructor (task.static). From static facts with predicates `loose`,
      `carrying`, and `usable`, build a directed adjacency graph where a fact
      "(pred A B)" induces an edge B -> A (meaning `tightened` can be moved
      from B to A in one propagation action).
    - The graph will be augmented on-the-fly in __call__ by including dynamic
      facts (from the current state) that also define the same predicates.
    """

    def __init__(self, task):
        # store goals and static facts for quick access
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Build adjacency graph from static facts.
        # For predicates 'loose', 'carrying', 'usable' a fact "(pred A B)"
        # allows propagating 'tightened' from B to A in one step.
        self.static_adjacency = {}
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("loose", "carrying", "usable"):
                # parts = [pred, dest, src] so add edge src -> dest
                if len(parts) >= 3:
                    dest, src = parts[1], parts[2]
                    self.static_adjacency.setdefault(src, set()).add(dest)

        # Precompute set of locatables mentioned in static to bound BFS if needed
        self.all_static_locatables = set()
        for fact in self.static:
            parts = get_parts(fact)
            # collect arguments that look like locatables (best-effort)
            for arg in parts[1:]:
                self.all_static_locatables.add(arg)

    def _bfs_min_distance(self, sources, targets, adjacency, undirected=False, max_limit=None):
        """
        Generic BFS returning minimal distance from any source to any target.

        - sources: iterable of starting node names (strings)
        - targets: set of target node names
        - adjacency: dict mapping node -> set(neighbor nodes) for directed edges
                     (edge u->v means neighbor v is reachable from u)
        - undirected: if True, treat adjacency as undirected
        - max_limit: optional cutoff; if reached return a large number
        Returns an integer distance (0 if any source in targets) or a large int
        when no target is reachable.
        """
        if not sources or not targets:
            return float("inf")
        sources = set(sources)
        if sources & targets:
            return 0
        visited = set(sources)
        q = deque((s, 0) for s in sources)
        limit = max_limit if max_limit is not None else (len(self.all_static_locatables) + 50)
        while q:
            node, dist = q.popleft()
            if dist >= limit:
                return float("inf")
            # neighbors in directed sense
            for nbr in adjacency.get(node, ()):
                if nbr in targets:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, dist + 1))
            if undirected:
                # also follow incoming edges (treat edges as undirected)
                # compute incoming by scanning adjacency (cheap enough for these sizes)
                for src, nbrs in adjacency.items():
                    if node in nbrs and src not in visited:
                        if src in targets:
                            return dist + 1
                        visited.add(src)
                        q.append((src, dist + 1))
        return float("inf")

    def __call__(self, node):
        """
        Compute heuristic value for `node`.

        Step-By-Step Thinking for Computing Heuristic
        --------------------------------------------
        1. If the state already satisfies all goals return 0.
        2. Build the current adjacency graph by combining static adjacency
           (constructed in __init__) with dynamic facts in the current state
           of the predicates `loose`, `carrying`, and `usable`. For a fact
           "(pred A B)" (pred in that set) an edge B -> A is added, because
           domain operators can propagate `tightened` from B to A.
        3. Identify current positions of:
           - all `tightened` facts: these provide source nodes where a
             `tightened` instance currently resides (one or more nuts may be
             tightened at different locatables). For a fact "(tightened nut X)"
             we take X as a source.
           - all `link` facts: these mark locatables where the tool is
             currently available.
           - all `loose-2` facts in the current state (we need at least one nut
             that is loose-2 at the goal location to produce tightened-2 there).
        4. For each goal "(tightened-2 G LOC)" that is not yet satisfied:
           a) Determine predecessor nodes P such that there exists a fact
              "(loose G P)" in the combined (static + current) facts. Any such
              P is a node where a `tightened` instance must be located prior to
              applying the final action that creates (tightened-2 G LOC).
           b) Compute d_tight = minimal propagation steps to get `tightened`
              from any current tightened node to any P (BFS on directed graph).
           c) Compute d_link = minimal steps to bring a `link` instance to G
              (BFS on the undirected version of the same graph from any current
              link node to G). The heuristic treats link movement as separate
              from `tightened` propagation; this may overestimate but is
              informative.
           d) If no nut is loose-2 at LOC in the current state, treat the goal
              as hard/unreachable by assigning a large penalty.
           e) The per-goal estimate is d_tight + d_link + 1 (1 for the final action
              that produces (tightened-2 G LOC)).
        5. Sum the per-goal estimates and return that sum as heuristic value.

        The heuristic always returns a finite number for solvable instances
        where the graph connects tightened/link sources to the relevant nodes.
        When necessary facts are missing it returns a large cost to discourage
        exploration of that branch.
        """
        state = set(node.state)

        # Quick check: if already at goal return 0
        if self.goals <= state:
            return 0

        # Build combined adjacency: start from static and add dynamic facts.
        adjacency = {k: set(v) for k, v in self.static_adjacency.items()}

        # Also collect all locatables seen in current state for BFS bounds.
        dynamic_locatables = set()
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            for arg in parts[1:]:
                dynamic_locatables.add(arg)
            if pred in ("loose", "carrying", "usable") and len(parts) >= 3:
                dest, src = parts[1], parts[2]
                adjacency.setdefault(src, set()).add(dest)

        # Current tightened positions (locatables)
        tightened_positions = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "tightened" and len(parts) >= 3:
                # "(tightened nut locatable)" -> parts[2] is locatable
                tightened_positions.add(parts[2])

        # Current link positions (locatables)
        link_positions = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "link" and len(parts) >= 2:
                link_positions.add(parts[1])

        # Current loose-2 facts mapping location -> set(nuts)
        loose2_at_location = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "loose-2" and len(parts) >= 3:
                nut, loc = parts[1], parts[2]
                loose2_at_location.setdefault(loc, set()).add(nut)

        # As a fallback if static contains loose-2 facts (rare), include them
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "loose-2" and len(parts) >= 3:
                nut, loc = parts[1], parts[2]
                loose2_at_location.setdefault(loc, set()).add(nut)

        total_cost = 0
        # bounds for BFS (if graph small, limit can be small)
        max_limit = max(1, len(self.all_static_locatables | dynamic_locatables) + 20)

        # Precompute an undirected adjacency view for link movement
        undirected_adj = {}
        for u, nbrs in adjacency.items():
            undirected_adj.setdefault(u, set()).update(nbrs)
            for v in nbrs:
                undirected_adj.setdefault(v, set()).add(u)

        # For each goal not satisfied compute estimate and sum
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "tightened-2" or len(parts) < 3:
                # ignore unexpected goal types (shouldn't happen)
                continue
            goal_locatable, goal_location = parts[1], parts[2]
            # If already satisfied skip
            if goal in state:
                continue

            # Check there is at least one nut loose-2 at the required location.
            if goal_location not in loose2_at_location or not loose2_at_location[goal_location]:
                # No nut currently loose-2 at the goal location: assign large penalty.
                # Use a value dependent on problem size but finite.
                total_cost += max_limit * 3
                continue

            # Determine predecessor nodes P where loose goal_locatable P exists
            # Use both state and static for loose facts
            predecessors = set()
            # dynamic loose facts
            for fact in state:
                p = get_parts(fact)
                if p[0] == "loose" and len(p) >= 3 and p[1] == goal_locatable:
                    predecessors.add(p[2])
            # static loose facts
            for fact in self.static:
                p = get_parts(fact)
                if p[0] == "loose" and len(p) >= 3 and p[1] == goal_locatable:
                    predecessors.add(p[2])

            # If no predecessor is defined, it's unlikely we can produce tightened-2 there.
            if not predecessors:
                # assign large penalty
                total_cost += max_limit * 2
                continue

            # Compute minimal propagation distance to any predecessor from any current tightened_pos
            d_tight = self._bfs_min_distance(
                sources=tightened_positions,
                targets=predecessors,
                adjacency=adjacency,
                undirected=False,
                max_limit=max_limit,
            )

            # Compute minimal link movement distance to goal_locatable from any current link pos
            d_link = self._bfs_min_distance(
                sources=link_positions,
                targets={goal_locatable},
                adjacency=undirected_adj,
                undirected=False,
                max_limit=max_limit,
            )

            # If either unreachable, inflate cost
            if d_tight == float("inf"):
                d_tight = max_limit * 2
            if d_link == float("inf"):
                d_link = max_limit

            # 1 action for the final operator that produces (tightened-2 ...)
            per_goal = int(d_tight + d_link + 1)
            total_cost += per_goal

        # Return finite integer heuristic
        return int(total_cost)