from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions required to achieve all
    (ne6sl557 <wqid8mog>) goal facts. In this domain a goal fact
    (ne6sl557 R) is produced by a single operator (f7y6nzq3) that requires:
      - an available processor/unit (a du770mtw object with ndg963qz),
      - a pt object (pt4b08k4) "attached" to a du770mtw (a fqyrzl30 fact),
      - both that pt and the target W (wqid8mog) to be at the same mz6r1d30
        location (qkdx0xe2 <obj> <loc>),
      - and the W object to be "ready" (of6wn8m4).
    The pt objects can be moved along static edges (lpmls56r facts) by
    nm6lzp2x, one edge per action. Attaching a pt to a du requires the
    jql8txb3 operator (one action), unless the pt is already attached.
    Finally, applying f7y6nzq3 to mark a wqid8mog as done costs one action.
    The heuristic therefore reasons about moving pt objects along the
    static location graph, optional attachment, and the final processing step.

    Assumptions
    -----------
    - Static facts of the form (lpmls56r A B) describe directed reachable
      edges between mz6r1d30 locations. Moving a pt along one such edge
      requires exactly one action (nm6lzp2x).
    - Objects that appear as subjects of qkdx0xe2 but not in ndg963qz
      (du770mtw) and not in of6wn8m4 (wqid8mog) are treated as pt4b08k4
      candidates (these are the movable parts).
    - Each ndg963qz (du770mtw available) can be used at most once (f7y6nzq3
      deletes ndg963qz). If there are fewer available du770mtw than
      remaining goals the state is treated as (heuristically) expensive
      by returning a large finite cost.
    - The heuristic is not guaranteed to be admissible; it aims to be
      informative and cheaply computable.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
      - the set of goal objects (the wqid8mog instances that must become ne6sl557),
      - the directed location graph from static lpmls56r facts,
      - nothing else is assumed static beyond lpmls56r.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already true in the state, return 0.
    2. Parse the state to obtain:
       - positions: mapping obj -> location for every qkdx0xe2 fact,
       - attached map: pt -> du from fqyrzl30 facts,
       - available processors: set of du objects from ndg963qz facts,
       - ready set: objects from of6wn8m4 facts (these must be true for f7y6nzq3).
    3. Identify pt candidates as objects that appear in qkdx0xe2 but are
       not in the sets of available du units and not in the ready set
       (i.e., not the wqid8mog targets).
    4. For each goal object G that is not yet achieved:
       - Determine G's current location (must exist as a qkdx0xe2 fact
         otherwise the goal is treated as expensive).
       - Greedily assign a distinct pt candidate P to G (so different
         goals prefer different pts). For each candidate compute:
           a) the shortest directed distance d (number of lpmls56r edges)
              from P's current location to G's location using BFS.
           b) if P is already attached to a du that is still available,
              then attaching cost is 0; otherwise attaching will cost 1
              (jql8txb3).
           c) final processing by f7y6nzq3 costs 1 action.
           Total estimated cost for matching P->G is: d + attach_cost + 1.
       - Choose the minimal-cost assignment available. Remove the chosen P
         from the pool and consume (decrement) one available du resource
         (if P provided an already-attached du, that particular du is
         consumed; otherwise we assume one available du will be used).
    5. If at any point there are not enough available du units for the
       remaining goals, return a large finite cost (heuristically signaling
       infeasibility).
    6. Sum the per-goal estimates and return the total as the heuristic
       value.

    Notes on design choices:
    - The heuristic uses shortest-path distances on the static lpmls56r
      graph to estimate the number of nm6lzp2x moves required to bring a
      pt to a goal's location.
    - It prefers to use already-attached pts since that avoids the
      jql8txb3 attach cost.
    - The algorithm is greedy in assigning pts to goals (fast and usually
      effective); it purposely keeps computation cheap to be suitable for
      guiding greedy best-first search.
    """

    def __init__(self, task):
        # goals: set of goal fact strings like "(ne6sl557 obj)"
        self.goals = set(task.goals)
        # Build directed graph of mz6r1d30 locations from static lpmls56r facts.
        self.location_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "lpmls56r" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.location_graph.setdefault(src, []).append(dst)
                # ensure nodes exist even if no outgoing edges
                self.location_graph.setdefault(dst, [])

        # A conservative large finite cost for impossible / very expensive cases.
        self.LARGE_COST = 1000

    def _bfs_distance(self, start, target):
        """
        Return the shortest directed distance (number of edges) from start
        to target in the location_graph. If start==target returns 0.
        If no path exists returns None.
        """
        if start == target:
            return 0
        if start not in self.location_graph or target not in self.location_graph:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nb in self.location_graph.get(node, ()):
                if nb == target:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # Quick check: if goal already reached, heuristic is zero.
        # task.goals are strings e.g. "(ne6sl557 obj)"; goal has been reached if all these are in state
        if self.goals <= state:
            return 0

        # Parse current state
        positions = {}      # obj -> location from qkdx0xe2
        attached = {}       # pt -> du from fqyrzl30
        available_du = set()  # du objects from ndg963qz
        ready_set = set()   # wqid8mog objects from of6wn8m4

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "qkdx0xe2" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                positions[obj] = loc
            elif pred == "fqyrzl30" and len(parts) >= 3:
                pt, du = parts[1], parts[2]
                attached[pt] = du
            elif pred == "ndg963qz" and len(parts) >= 2:
                available_du.add(parts[1])
            elif pred == "of6wn8m4" and len(parts) >= 2:
                ready_set.add(parts[1])

        # which goal objects still need processing (objects, not fact strings)
        remaining_goals = []
        for gfact in self.goals:
            parts = get_parts(gfact)
            if parts and parts[0] == "ne6sl557" and len(parts) >= 2:
                obj = parts[1]
                if gfact not in state:
                    remaining_goals.append(obj)

        # If no goals to achieve, return 0.
        if not remaining_goals:
            return 0

        # Basic feasibility check: there must be at least as many available du units
        # as goals (each f7y6nzq3 consumes one). If not, return LARGE_COST.
        if len(available_du) < len(remaining_goals):
            return self.LARGE_COST

        # Identify pt candidates: objects that have a qkdx0xe2 position and are
        # not in available_du and not in ready_set (i.e., not the wqid8mog targets).
        pt_candidates = {
            obj for obj in positions.keys() if obj not in available_du and obj not in ready_set
        }

        # If there are no pt candidates but goals remain, heuristically expensive.
        if not pt_candidates:
            return self.LARGE_COST

        total_cost = 0
        # Copy sets so we can consume them while assigning
        pts_available = set(pt_candidates)
        du_available = set(available_du)

        # Greedy: for each goal pick the pt that gives smallest estimated cost.
        # We process goals in arbitrary order; a better ordering could help but
        # this keeps the computation cheap.
        for goal_obj in remaining_goals:
            # Goal must be in positions (qkdx0xe2). If not, treat as expensive.
            goal_loc = positions.get(goal_obj)
            if goal_loc is None:
                return self.LARGE_COST

            best_cost = None
            best_pt = None
            best_used_du = None  # which du would be consumed (if attached)

            for pt in list(pts_available):
                pt_loc = positions.get(pt)
                if pt_loc is None:
                    # No known location for this pt candidate -> skip
                    continue
                dist = self._bfs_distance(pt_loc, goal_loc)
                if dist is None:
                    # cannot reach goal location from this pt
                    continue

                # If pt is already attached to a du that is still available,
                # we can avoid attach step
                attached_du = attached.get(pt)
                if attached_du and attached_du in du_available:
                    attach_cost = 0
                    used_du = attached_du
                else:
                    # We will need to attach (jql8txb3) at cost 1 and consume
                    # any available du resource (we don't model moving a du).
                    attach_cost = 1
                    used_du = None  # will consume a generic du

                # final processing cost
                final_cost = 1

                cost = dist + attach_cost + final_cost

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_pt = pt
                    best_used_du = used_du

            # If we could not find any pt that can reach the goal location, treat as expensive.
            if best_cost is None:
                return self.LARGE_COST

            # Assign chosen pt to this goal: consume resources
            total_cost += best_cost
            pts_available.discard(best_pt)
            # consume du resource
            if best_used_du:
                # consume that specific du
                du_available.discard(best_used_du)
            else:
                # consume any one du
                if du_available:
                    # remove an arbitrary element
                    du_available.pop()
                else:
                    # no du left -> infeasible
                    return self.LARGE_COST

        return total_cost