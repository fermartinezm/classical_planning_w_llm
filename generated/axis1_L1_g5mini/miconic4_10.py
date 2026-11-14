from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (ud3uf11z).

    The domain contains two kinds of objects: "p" objects (type nxbd1hc3) and
    "f" objects (type t4rdxh32). There is exactly one "active" f node indicated
    by the predicate (omggbn9z fX). To achieve (a7sb56om p), a two-step
    procedure is required in general:
      1) At some f where (kut2ksnk p f) holds and where that f is active,
         apply the action diwlmgg9 which yields (f2xo7hb6 p).
      2) At some f' where (klw11kjf p f') holds and where that f' is active,
         apply the action y7nik9q2 which consumes (f2xo7hb6 p) and produces (a7sb56om p).

    Omggbn9z (active f) can be moved along a directed graph of f-nodes using
    actions qs8pdrqj / i0veo5kv; the edges of that graph are given by the
    static predicate (acovodrf f1 f2).

    This heuristic estimates the number of actions remaining by:
    - For each goal object p that is not yet accomplished, compute the minimal
      sequence of actions assuming independence:
        * If (f2xo7hb6 p) already holds: distance(current_active_f, some target f')
          + 1 (for the y7nik9q2 action).
        * Otherwise: distance(current_active_f, some source f) + 1 (diwlmgg9)
                   + distance(source f, some target f') + 1 (y7nik9q2).
      Distances are shortest-path lengths (number of move actions) in the
      directed graph given by acovodrf edges. If multiple valid source/target
      nodes exist, the nearest combination (minimum total) is chosen.
    - The heuristic value is the sum of the per-p minimal estimates. This is
      optimistic with respect to reuse of movement between different p's (we
      assume we can "teleport" back to the current active f for each p), which
      usually yields an admissible/optimistic estimate but is not required to
      be admissible for greedy best-first search.

    Assumptions
    -----------
    - acovodrf edges are static (appear in task.static) and define the directed
      graph over f-nodes.
    - klw11kjf predicates may be static (typical) but this heuristic also
      checks the current state for any dynamic klw11kjf facts.
    - kut2ksnk facts are typically dynamic (in the current state) and are used
      to determine which source f nodes are currently available for diwlmgg9.
    - There is at least one active f node (omggbn9z ...) in solvable states.
    - If a necessary source/target or path is missing (i.e., the goal is
      provably unreachable from the given state), the heuristic returns a large
      finite constant to indicate difficulty (rather than infinity).

    Heuristic Initialization
    ------------------------
    The constructor extracts from the task:
    - The set of goal predicates to be achieved (task.goals).
    - The directed adjacency graph of f-nodes from static acovodrf facts.
    - The static mapping of klw11kjf relations (p -> set of target f-nodes).
      Dynamic klw11kjf facts (present in the current state) are also considered
      during the heuristic computation.
    """

    def __init__(self, task):
        """Extract static structures from the task for efficient reuse."""
        # Goal conditions (set of fact strings that must hold).
        self.goals = task.goals

        # Build adjacency list for f-nodes from static acovodrf facts.
        self.adj = {}  # node -> set(neighbours)
        self.all_f_nodes = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "acovodrf" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.all_f_nodes.add(a)
                self.all_f_nodes.add(b)
                self.adj.setdefault(a, set()).add(b)

        # Static klw11kjf mapping: p -> set(target_f)
        self.static_klw = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "klw11kjf" and len(parts) >= 3:
                p, fnode = parts[1], parts[2]
                self.static_klw.setdefault(p, set()).add(fnode)

        # Precompute adjacency nodes existence (ensure keys exist for BFS).
        for f in list(self.all_f_nodes):
            self.adj.setdefault(f, set())

        # A large constant used when the heuristic detects an unreachable goal.
        self.UNREACHABLE_COST = 10000

    def _bfs_distances(self, start):
        """
        Compute shortest-path distances (in number of move actions) from start
        to all reachable f-nodes using the directed adjacency self.adj.
        Returns a dict node->distance.
        """
        if start is None:
            return {}
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        """
        Compute the heuristic estimate for node.state.

        Step-By-Step Thinking for Computing Heuristic
        ---------------------------------------------
        1. If all goal facts are already true in the current state, return 0.

        2. Identify the currently active f-node by searching for (omggbn9z fX)
           in the state. If none exists and movement is required, we treat the
           problem as temporarily unreachable (return a large finite cost).

        3. Extract from the current state:
           - dynamic_kut: mapping p -> set(source_f) for facts (kut2ksnk p f).
           - dynamic_klw: mapping p -> set(target_f) for facts (klw11kjf p f).
           - f2xo_set: set of p for which (f2xo7hb6 p) already holds.
           - achieved: set of p for which (a7sb56om p) already holds.

           Note: klw11kjf targets are taken as the union of static and dynamic
           klw facts.

        4. For each goal object p not yet achieved:
           - If f2xo7hb6 p holds: we only need to move to some valid target f'
             (one where klw11kjf p f' holds) and apply y7nik9q2: cost =
             dist(current_active, f') + 1.
           - Otherwise: we must first visit some source f (where kut2ksnk p f
             holds) to apply diwlmgg9, then some target f' (where klw11kjf p f'
             holds) to apply y7nik9q2:
             cost = dist(current_active, source) + 1 + dist(source, target) + 1.
           - Distances are taken as shortest paths in the acovodrf graph.
           - If no source or no target or no connecting path exists for a
             required step, the state is treated as (for heuristic purposes)
             unreachable and a large finite cost is returned.

        5. Sum the per-goal minimal costs and return that sum. The heuristic is
           optimistic and does not attempt to schedule an optimal visiting order
           across multiple goals; it treats goals independently, which keeps
           computation cheap and effective as a guidance heuristic for greedy
           best-first search.
        """
        state = node.state

        # If all goals satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Find current active f-node (omggbn9z fX).
        current_active = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "omggbn9z" and len(parts) >= 2:
                current_active = parts[1]
                break

        # Extract dynamic kut2ksnk facts (sources), dynamic klw11kjf facts (targets),
        # and which p already have f2xo7hb6 or a7sb56om.
        dynamic_kut = {}   # p -> set(source f)
        dynamic_klw = {}   # p -> set(target f) (dynamic portion)
        f2xo_set = set()
        achieved = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "kut2ksnk" and len(parts) >= 3:
                p, fnode = parts[1], parts[2]
                dynamic_kut.setdefault(p, set()).add(fnode)
                self.all_f_nodes.add(fnode)
                self.adj.setdefault(fnode, set())
            elif pred == "klw11kjf" and len(parts) >= 3:
                p, fnode = parts[1], parts[2]
                dynamic_klw.setdefault(p, set()).add(fnode)
                self.all_f_nodes.add(fnode)
                self.adj.setdefault(fnode, set())
            elif pred == "f2xo7hb6" and len(parts) >= 2:
                f2xo_set.add(parts[1])
            elif pred == "a7sb56om" and len(parts) >= 2:
                achieved.add(parts[1])

        # Combine static and dynamic klw targets.
        klw_map = {}
        # Start from static
        for p, s in self.static_klw.items():
            klw_map.setdefault(p, set()).update(s)
        # Add dynamic ones
        for p, s in dynamic_klw.items():
            klw_map.setdefault(p, set()).update(s)

        # Precompute distances from the current active f to all f-nodes.
        distances_from_active = self._bfs_distances(current_active)

        total_cost = 0

        # For each goal fact that must hold (we expect goals are of form '(a7sb56om p)'),
        # extract the object names from self.goals.
        goal_ps = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "a7sb56om" and len(parts) >= 2:
                goal_ps.append(parts[1])
            else:
                # If goals use different predicates, treat them conservatively by
                # returning a large cost (this domain's goals are expected to be a7sb56om).
                return self.UNREACHABLE_COST

        for p in goal_ps:
            if p in achieved:
                continue  # already satisfied

            # If p already has f2xo7hb6, only need to reach a klw11kjf target and apply y action.
            if p in f2xo_set:
                targets = klw_map.get(p, set())
                if not targets:
                    return self.UNREACHABLE_COST
                # Find nearest reachable target from current_active.
                best = None
                for t in targets:
                    d = distances_from_active.get(t)
                    if d is None:
                        continue
                    cost = d + 1  # moves + apply y7nik9q2
                    if best is None or cost < best:
                        best = cost
                if best is None:
                    # No reachable target from current active.
                    return self.UNREACHABLE_COST
                total_cost += best
            else:
                # Need to first apply diwlmgg9 at some source (kut2ksnk in current state),
                # then y7nik9q2 at some target (klw11kjf).
                sources = dynamic_kut.get(p, set())
                targets = klw_map.get(p, set())
                if not sources or not targets:
                    return self.UNREACHABLE_COST

                best = None
                # For each candidate source, compute dist(current_active -> source),
                # then dist(source -> target). Precompute BFS from source when needed.
                for s in sources:
                    d1 = distances_from_active.get(s)
                    if d1 is None:
                        # Active cannot reach this source; skip it
                        continue
                    # BFS from s
                    dist_from_s = self._bfs_distances(s)
                    for t in targets:
                        d2 = dist_from_s.get(t)
                        if d2 is None:
                            continue
                        cost = d1 + 1 + d2 + 1  # move to source + diwlmgg9 + move to target + y7nik9q2
                        if best is None or cost < best:
                            best = cost
                if best is None:
                    # No combination of source/target reachable
                    return self.UNREACHABLE_COST
                total_cost += best

        # Finally, return the summed estimate.
        return int(total_cost)