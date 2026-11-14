from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (unkhrp2l). The domain models a single
    mobile token (predicate v0yy74n7 at a location of type ql6b2h31), objects of type
    jxhpkqr5 that must reach a target location (predicate guby8wxu for success), and
    small pickup/drop mechanics:
      - pxch7pi8 OBJ LOC  : an object OBJ is available at LOC (must be picked up)
      - makpz2tc (when v0 at LOC and pxch7pi8 OBJ LOC) creates eacj8ftl OBJ (picks it up)
      - jg64qouf (when v0 at LOC' and eacj8ftl OBJ and pahbmn6t OBJ LOC') creates guby8wxu OBJ (delivers it)
      - pvhp8m0p LOC1 LOC2 are directed movement links for v0.

    The heuristic estimates the number of actions remaining to achieve all goal guby8wxu facts.

    Assumptions
    -----------
    - There is a single mobile token indicated by predicate v0yy74n7 (if multiple exist, the
      heuristic uses any one of them; domain instances typically have exactly one).
    - Movement cost between adjacent ql6b2h31 locations (connected by pvhp8m0p) is 1 action per edge.
    - Picking an available object at its location (makpz2tc) costs 1 action.
    - Delivering a held object to its target (jg64qouf) costs 1 action.
    - If an object is already "picked up" (predicate eacj8ftl OBJ is true) we don't need the pick action.
    - The mapping from each object to its delivery location (pahbmn6t OBJ LOC) and the graph
      of movement links (pvhp8m0p) are static and available in task.static.
    - If an object has neither pxch7pi8 nor eacj8ftl in the current state, the heuristic treats
      it as unreachable and assigns a large finite penalty.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.goals: the set of goal facts (task.goals).
    - self.target_of: mapping from object -> target location (from static pahbmn6t facts).
    - self.adj: adjacency mapping (directed) between locations (from static pvhp8m0p facts).
    These static data structures are used to compute shortest path distances between locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already present in the state, return 0.
    2. Determine the current location of the mobile token (the unique fact (v0yy74n7 LOC)).
       If none exists, treat as unreachable (large penalty).
    3. For each goal object OBJ that does not yet satisfy (guby8wxu OBJ):
       a) Look up its required delivery location T = pahbmn6t OBJ LOC (from static).
          If no such mapping exists, treat as unreachable (large penalty).
       b) Check the current state for either:
          - eacj8ftl OBJ  (object already picked up), or
          - pxch7pi8 OBJ S  (object available at location S to pick up).
          If neither exists -> unreachable (large penalty).
       c) Compute directed shortest-path distances using the static adjacency:
          - If eacj8ftl is present: cost = distance(v0_loc, T) + 1 (for jg64qouf).
          - If pxch7pi8 at S is present: cost = distance(v0_loc, S) + 1 (makpz2tc)
                                         + distance(S, T) + 1 (jg64qouf).
          If any required distance is not reachable in the movement graph, treat as unreachable.
    4. Sum the per-object costs to produce the final heuristic value. (This treats each goal
       independently and therefore may overcount travel when a single mobile token can serve
       several objects in a clever sequence. The heuristic is intentionally simple and fast
       to compute; it remains informative because it accounts for movement and mandatory pickup/deliver actions.)
    5. For unreachable subproblems a large finite penalty (LARGE = 10000) is added so the
       heuristic is finite but strongly discourages paths that cannot reach the goal.

    Notes on Design Choices
    -----------------------
    - BFS is used to compute shortest path distances on the directed graph of ql6b2h31 locations.
    - The heuristic is zero exactly when all goal facts are already satisfied.
    - The heuristic is fast: adjacency and object->target mapping are precomputed from static facts;
      BFS distances are computed on demand (fast for typical instance sizes).
    """

    LARGE = 10000

    def __init__(self, task):
        # Goal facts (strings like "(guby8wxu obj)").
        self.goals = task.goals

        # Build mapping object -> target location from static pahbmn6t facts.
        self.target_of = {}
        # Build directed adjacency list from static pvhp8m0p facts.
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "pahbmn6t" and len(parts) >= 3:
                obj = parts[1]
                loc = parts[2]
                self.target_of[obj] = loc
            elif pred == "pvhp8m0p" and len(parts) >= 3:
                a = parts[1]
                b = parts[2]
                self.adj.setdefault(a, []).append(b)

    def bfs_distance(self, start, goal):
        """Return shortest directed distance (number of edges) from start to goal or None if unreachable."""
        if start == goal:
            return 0
        if start not in self.adj:
            # there may still be nodes with no outgoing edges; if start has no outgoing edges and start != goal => unreachable
            # but it might be reachable only if start == goal which is handled above
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for neigh in self.adj.get(node, []):
                if neigh in visited:
                    continue
                if neigh == goal:
                    return dist + 1
                visited.add(neigh)
                q.append((neigh, dist + 1))
        return None

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If all goal facts are present, heuristic is 0.
        if self.goals <= state:
            return 0

        # Determine current location of the mobile token v0yy74n7.
        v0_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "v0yy74n7" and len(parts) >= 2:
                v0_loc = parts[1]
                break
        if v0_loc is None:
            # No mobile token present in the state => treat as unreachable
            return self.LARGE

        total_cost = 0

        # Pre-index dynamic state facts for quick lookup.
        has_guby = {get_parts(f)[1] for f in state if get_parts(f)[0] == "guby8wxu"}
        has_eac = {get_parts(f)[1] for f in state if get_parts(f)[0] == "eacj8ftl"}
        # map pxch7pi8 object -> source location (if present)
        pxch_location = {}
        for f in state:
            parts = get_parts(f)
            if parts[0] == "pxch7pi8" and len(parts) >= 3:
                obj = parts[1]
                loc = parts[2]
                pxch_location[obj] = loc

        # Extract goal objects that must achieve guby8wxu
        goal_objects = [get_parts(g)[1] for g in self.goals if get_parts(g)[0] == "guby8wxu"]

        for obj in goal_objects:
            if obj in has_guby:
                continue  # already satisfied

            # Determine target location for this object
            target = self.target_of.get(obj)
            if target is None:
                # No static mapping to a target location -> unreachable in domain model
                return self.LARGE

            # If object already picked up (eacj8ftl), only need to move to target and apply jg64qouf.
            if obj in has_eac:
                dist = self.bfs_distance(v0_loc, target)
                if dist is None:
                    return self.LARGE
                # 1 for the delivery action jg64qouf
                total_cost += dist + 1
                # assume v0 will be at target after finishing this object's delivery; keep v0_loc unchanged
                # (we do not thread v0's position between objects to keep heuristic cheap)
                continue

            # Otherwise, object must be picked up from its pxch location first.
            if obj in pxch_location:
                src = pxch_location[obj]
                d1 = self.bfs_distance(v0_loc, src)
                d2 = self.bfs_distance(src, target)
                if d1 is None or d2 is None:
                    return self.LARGE
                # 1 for makpz2tc (pick), 1 for jg64qouf (deliver)
                total_cost += d1 + 1 + d2 + 1
                continue

            # No pxch or eac for this object -> unreachable (cannot create eac without pxch)
            return self.LARGE

        return total_cost