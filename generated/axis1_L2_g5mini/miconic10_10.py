from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Return components of a fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (unkhrp2l) domain.

    This domain models a single movable token (predicate v0yy74n7 on ql6b2h31 nodes),
    a set of objects of type jxhpkqr5 that can be in two possible states (pxch7pi8 <-> eacj8ftl),
    and a final achieved state (guby8wxu) for objects. Movement of the token between
    ql6b2h31 nodes is possible along edges given by the static predicate pvhp8m0p.
    Two actions transfer token across an edge in either direction (so the effective
    movement graph is undirected). The two-step object processing is:
      - makpz2tc: at a node where pxch7pi8(object, node) and token at node -> create eacj8ftl(object)
      - jg64qouf: if eacj8ftl(object) and token at N and pahbmn6t(object, N) holds -> create guby8wxu(object)

    Assumptions
    -----------
    - There is a single token (v0yy74n7) in the domain. If absent, the heuristic returns a large value.
    - The static facts include pvhp8m0p edges and pahbmn6t object->node associations.
    - pxch7pi8 and eacj8ftl and guby8wxu are dynamic (in the state).
    - If an object has no pxch7pi8 and no eacj8ftl and no way to reach its pahbmn6t node, the goal for that object
      is considered unreachable and the heuristic returns a large value.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
    - collects the set of goal facts (task.goals).
    - extracts the static movement graph from pvhp8m0p facts and builds an undirected adjacency map.
    - extracts the mapping(s) from objects to their associated node(s) from pahbmn6t static facts
      (these are the node(s) where jg64qouf can be applied to produce guby8wxu).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic computes a compact estimate of the number of actions
    required to achieve all goal guby8wxu(object) facts that are not yet true. The heuristic
    is deliberately approximate but aims to be informative:

    1. If all goals are satisfied in the given state -> heuristic = 0.

    2. Identify the current token location (v0yy74n7 node). If no token is present, return a large value.

    3. Identify, per object:
       - whether guby8wxu(object) is already true (then cost 0 for that object),
       - whether eacj8ftl(object) is currently true,
       - whether pxch7pi8(object, node) is currently true and at which node.

    4. For each remaining object (those missing guby8wxu), there are two ways to achieve guby8wxu:
       A) If eacj8ftl(object) is true: move the token to any node N with pahbmn6t(object, N) and apply jg64qouf.
          Cost estimate = distance(token_location, N) + 1 (one jg64qouf action).
       B) If pxch7pi8(object, M) is true: move the token to M, apply makpz2tc (1 action) to get eacj8ftl,
          then move the token to N where pahbmn6t(object, N) and apply jg64qouf (1 action).
          Cost estimate = distance(token, M) + 1 + distance(M, N) + 1.

       Distances are computed on the undirected graph induced by static pvhp8m0p facts
       using BFS. If a required distance is unreachable, that option is considered infeasible.

    5. Because token movement is shared among objects, the heuristic applies a greedy routing
       approximation: starting from the current token location, repeatedly pick the object
       whose completion (by one of the two methods) yields the smallest incremental cost,
       add that cost to the total, and update the token location to the node where jg64qouf
       finishes (a pahbmn6t node). Repeat until all objects are handled or an object is
       unreachable (in which case return a large value).

    6. The heuristic returns the accumulated total cost (number of actions estimated).
       The returned value is finite for solvable states under the assumptions above and
       equals 0 only for goal states.
    """

    def __init__(self, task):
        # Goals (set of fact strings)
        self.goals = task.goals

        # Build an undirected adjacency map from static pvhp8m0p facts:
        # pvhp8m0p A B -> add undirected edge A <-> B (movement allowed in both directions
        # via the two movement operators).
        self.adj = {}
        self.pahb_map = {}  # object -> set(nodes) from static pahbmn6t
        static_facts = task.static
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "pvhp8m0p" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)
            elif pred == "pahbmn6t" and len(parts) >= 3:
                obj, node = parts[1], parts[2]
                self.pahb_map.setdefault(obj, set()).add(node)

        # Cache for BFS distances to avoid repeated searches
        self._dist_cache = {}

    def _bfs_distances(self, start):
        """Return a dict node->distance from start using undirected adj graph (BFS)."""
        if start in self._dist_cache:
            return self._dist_cache[start]
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[start] = dist
        return dist

    def __call__(self, node):
        state = node.state

        # If goals already satisfied -> zero
        if self.goals <= state:
            return 0

        # Find current token location
        token_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "v0yy74n7":
                token_loc = parts[1]
                break
        if token_loc is None:
            # No token -> actions that require token cannot be applied; return large value.
            return INF

        # Identify goal objects (objects that must have guby8wxu in the goal)
        goal_objects = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "guby8wxu" and len(parts) >= 2:
                goal_objects.append(parts[1])

        # Dynamic information from state
        eac_set = set()
        pxch_map = {}  # object -> node (pxch7pi8 object node)
        guby_set = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "eacj8ftl" and len(parts) >= 2:
                eac_set.add(parts[1])
            elif pred == "pxch7pi8" and len(parts) >= 3:
                obj, node = parts[1], parts[2]
                # pxch should be unique per object in well-formed instances; keep one if multiple
                pxch_map.setdefault(obj, node)
            elif pred == "guby8wxu" and len(parts) >= 2:
                guby_set.add(parts[1])

        # Remaining objects we still need to achieve guby for
        remaining = [o for o in goal_objects if o not in guby_set]
        if not remaining:
            return 0

        total_cost = 0
        current_token = token_loc

        # Greedy servicing of remaining objects to capture token movement sharing approximately.
        while remaining:
            best_obj = None
            best_cost = INF
            best_goal_node = None

            # Precompute BFS distances from current_token
            dist_from_current = self._bfs_distances(current_token)

            for obj in remaining:
                # Possible goal nodes for this object (from static pahbmn6t)
                goal_nodes = self.pahb_map.get(obj)
                if not goal_nodes:
                    # No known node where jg64qouf can be applied -> unreachable
                    continue

                # For each candidate goal node evaluate the best option (use eac or create it from pxch)
                for goal_node in goal_nodes:
                    # Option A: eacj8ftl already present -> move token to goal_node + jg64qouf
                    cost_a = INF
                    if obj in eac_set:
                        d = dist_from_current.get(goal_node, INF)
                        if d < INF:
                            cost_a = d + 1  # move actions (d) + jg64qouf

                    # Option B: pxch present at some node M -> move token to M, makpz2tc (1),
                    # move token M->goal_node, jg64qouf (1)
                    cost_b = INF
                    if obj in pxch_map:
                        px_node = pxch_map[obj]
                        # distance current -> px_node (may require computing BFS from current)
                        d1 = dist_from_current.get(px_node, None)
                        if d1 is None:
                            # maybe px_node equals current_token? else compute distances from current already done
                            d1 = self._bfs_distances(current_token).get(px_node, INF)
                        # distance px_node -> goal_node: BFS from px_node
                        dist_px = self._bfs_distances(px_node)
                        d2 = dist_px.get(goal_node, INF)
                        if d1 < INF and d2 < INF:
                            cost_b = d1 + 1 + d2 + 1  # move to px + makpz + move to goal + jg64qouf

                    # Choose the better option for this goal_node
                    node_best_cost = min(cost_a, cost_b)
                    if node_best_cost < best_cost:
                        best_cost = node_best_cost
                        best_obj = obj
                        best_goal_node = goal_node

            if best_cost >= INF:
                # At least one remaining object appears unreachable -> return large value
                return INF

            # Apply (conceptually) the chosen plan for best_obj:
            total_cost += best_cost
            # After jg64qouf the token will be at best_goal_node
            current_token = best_goal_node
            # Remove that object from remaining
            remaining.remove(best_obj)
            # Update dynamic sets to reflect that we conceptually achieved guby for best_obj.
            # This matters if eac/pxch statuses are used by later planning for other objects.
            eac_set.discard(best_obj)
            pxch_map.pop(best_obj, None)
            guby_set.add(best_obj)
            # Clear distance cache for new token location computations (we keep cached BFS results,
            # but they are keyed by source; it's fine to keep them)

        return total_cost