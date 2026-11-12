from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Split a PDDL fact string like '(pred a b)' into ['pred','a','b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (ud3uf11z).
    The domain models agents/items of type `nxbd1hc3` (call them "items") that must
    reach the predicate (a7sb56om ?p), and two kinds of item-location relations:
    - (kut2ksnk ?p ?loc): an item can be "picked up" at location ?loc via action diwlmgg9,
      which yields the temporary predicate (f2xo7hb6 ?p).
    - (klw11kjf ?p ?loc): an item can be "finished" at location ?loc via action y7nik9q2,
      which requires (f2xo7hb6 ?p) and (omggbn9z ?loc) to be true.
    Locations of type t4rdxh32 are connected via static facts (acovodrf ?l1 ?l2) and
    an active location indicator (omggbn9z ?loc) denotes the agent's current location.
    There are two move actions that move the active location along acovodrf edges.

    The heuristic estimates the number of (primitive) actions required to achieve
    all outstanding (a7sb56om ?p) goals from a given state. It computes, for each
    item that is not yet completed, a shortest individual plan cost:
      - move(s) to a node where the item can be picked up (kut2ksnk),
      - one pick action (diwlmgg9),
      - move(s) from pickup node to a finishing node (klw11kjf),
      - one finish action (y7nik9q2).
    If the item already holds (f2xo7hb6 ?p), the pick action is omitted.
    The heuristic sums these per-item minimal costs (ignoring opportunities to
    combine moves across multiple items). This is not admissible but is cheap
    and effective in guiding greedy best-first search.

    Assumptions
    -----------
    - acovodrf facts are static and describe an (undirected) connectivity graph
      between location nodes of type t4rdxh32. The heuristic treats edges as
      bidirectional (both move operators allow moving along either direction).
    - klw11kjf facts (finish locations for items) are considered static if present
      in task.static; they may also appear in the state.
    - kut2ksnk (pickup locations) are taken from the current state (they may be
      removed after pickup).
    - There is exactly one active location (omggbn9z ...) in the current state;
      if none is present, the heuristic assumes no reachable progress and returns
      a large finite cost.
    - If a required pickup or finish location does not exist or is disconnected,
      the heuristic returns a large finite penalty for that item (treating it as
      effectively unreachable).

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Extract the set of goal facts (self.goals).
    - Parse static facts to build:
      * adjacency dictionary for the location graph from acovodrf facts.
      * a mapping of items -> set of finishing locations from klw11kjf facts found
        in static facts.
    - Record a large penalty constant used for unreachable items.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given node (state):
    1. If all goal facts are already present in the state, return 0.
    2. Identify the current active location by finding the fact (omggbn9z X).
       If none exists, return a large penalty (no moves are possible).
    3. Build:
       - For each item p, the set of pickup locations Up from facts (kut2ksnk p u)
         present in the current state.
       - For each item p, the set of finish locations Vp from both static facts
         (extracted at initialization) and any klw11kjf facts present in the state.
       - The set of items already having (f2xo7hb6 p) in the current state.
    4. Precompute shortest-path distances between all locations in the known
       location graph using BFS (treating edges bidirectionally).
    5. For each item p that is not yet finished:
       - If p already has (f2xo7hb6 p): compute min_v distance(current_loc, v) + 1
         (the +1 is the y7nik9q2 action to finish at v).
       - Else: compute min_{u in Up, v in Vp} distance(current_loc, u) + 1 (diwlmgg9)
               + distance(u, v) + 1 (y7nik9q2).
       - If any required distance is missing (disconnected) or Up/Vp empty,
         use the large penalty for that item.
    6. Sum the costs over all unfinished items and return the sum as heuristic value.
    """

    def __init__(self, task):
        # Goal facts (set of strings)
        self.goals = task.goals

        # Build adjacency for the location graph from static facts "acovodrf a b".
        self.adjacency = {}
        self.all_locations = set()
        static = task.static
        for fact in static:
            parts = get_parts(fact)
            if parts and parts[0] == "acovodrf" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.all_locations.add(a)
                self.all_locations.add(b)
                self.adjacency.setdefault(a, set()).add(b)
                self.adjacency.setdefault(b, set()).add(a)

        # Extract klw11kjf facts from static: finishing locations for items.
        # Map: item -> set(locations)
        self.static_finish_locs = {}
        for fact in static:
            parts = get_parts(fact)
            if parts and parts[0] == "klw11kjf" and len(parts) >= 3:
                p, loc = parts[1], parts[2]
                self.static_finish_locs.setdefault(p, set()).add(loc)
                self.all_locations.add(loc)

        # Large penalty for unreachable items (finite but large).
        self.UNREACHABLE_COST = 1000

    def _bfs_distances_from(self, src):
        """Return a dict mapping node->distance from src using BFS on self.adjacency.
           If src is unknown (not in adjacency), return {src: 0} (isolated)."""
        if src not in self.adjacency:
            # If isolated node (no neighbors known), it can reach itself only.
            return {src: 0}
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adjacency.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # 1) If goal already reached, return 0.
        if self.goals <= state:
            return 0

        # 2) Identify current active location (omggbn9z X)
        current_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "omggbn9z" and len(parts) >= 2:
                current_loc = parts[1]
                break
        if current_loc is None:
            # No active location known -> cannot move. Return large cost.
            return self.UNREACHABLE_COST

        # 3) Collect pickup locations (kut2ksnk) from current state,
        #    finish locations (klw11kjf) from both state and static,
        #    and which items already have f2xo7hb6.
        pickup_locs = {}  # item -> set(loc)
        finish_locs = {}  # item -> set(loc)
        have_f2 = set()
        have_a7 = set()

        # Initialize finish_locs with static info
        for p, locs in self.static_finish_locs.items():
            finish_locs.setdefault(p, set()).update(locs)
            self.all_locations.update(locs)

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "kut2ksnk" and len(parts) >= 3:
                p, loc = parts[1], parts[2]
                pickup_locs.setdefault(p, set()).add(loc)
                self.all_locations.add(loc)
            elif pred == "klw11kjf" and len(parts) >= 3:
                p, loc = parts[1], parts[2]
                finish_locs.setdefault(p, set()).add(loc)
                self.all_locations.add(loc)
            elif pred == "f2xo7hb6" and len(parts) >= 2:
                have_f2.add(parts[1])
            elif pred == "a7sb56om" and len(parts) >= 2:
                have_a7.add(parts[1])
            elif pred == "acovodrf" and len(parts) >= 3:
                # In case some acovodrf facts are present in the state (rare),
                # include them in adjacency as well.
                a, b = parts[1], parts[2]
                self.adjacency.setdefault(a, set()).add(b)
                self.adjacency.setdefault(b, set()).add(a)
                self.all_locations.update([a, b])

        # 4) Precompute distances from all relevant locations (including current loc)
        #    We'll compute BFS from each location that appears in pickups, finishes or current.
        locations_to_expand = set(self.all_locations)
        locations_to_expand.add(current_loc)
        # Ensure adjacency has entries for isolated nodes to avoid KeyErrors in BFS.
        for loc in locations_to_expand:
            self.adjacency.setdefault(loc, set())

        distances = {}
        for loc in locations_to_expand:
            distances[loc] = self._bfs_distances_from(loc)

        # 5) For each item required by goals, compute minimal individual cost.
        total_cost = 0
        # Extract goal items (objects in (a7sb56om p))
        goal_items = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "a7sb56om" and len(parts) >= 2:
                goal_items.append(parts[1])

        for p in goal_items:
            # If already achieved skip
            if p in have_a7:
                continue

            best = float("inf")

            # Candidate finish locations for p
            Vl = finish_locs.get(p, set())
            # If no finish locations known at all, can't finish -> penalty
            if not Vl:
                total_cost += self.UNREACHABLE_COST
                continue

            # Case A: already have f2xo7hb6, just need to move to some finish location and apply y7nik9q2
            if p in have_f2:
                for v in Vl:
                    d = distances.get(current_loc, {}).get(v)
                    if d is None:
                        continue
                    cost = d + 1  # move(s) + one y7nik9q2
                    if cost < best:
                        best = cost
            else:
                # Need pickup then finish: find pickup locations from current state
                Ul = pickup_locs.get(p, set())
                if not Ul:
                    # No pickup location available in current state -> unreachable
                    total_cost += self.UNREACHABLE_COST
                    continue
                for u in Ul:
                    d0 = distances.get(current_loc, {}).get(u)
                    if d0 is None:
                        continue
                    for v in Vl:
                        d1 = distances.get(u, {}).get(v)
                        if d1 is None:
                            continue
                        cost = d0 + 1 + d1 + 1  # move to u, diwlmgg9, move u->v, y7nik9q2
                        if cost < best:
                            best = cost

            if best == float("inf"):
                # No reachable combo found
                total_cost += self.UNREACHABLE_COST
            else:
                total_cost += int(best)

        return int(total_cost)