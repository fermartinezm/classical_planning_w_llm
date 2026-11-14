from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens inside a PDDL fact string, without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (e2zu8y93) problem.
    The domain contains:
      - objects that can be at locations: spanners (predicate u4srx1nb for spanner),
        nuts (u4srx1nb for nut), and actors (u4srx1nb for actors),
      - a predicate yud7t3vb(actor, spanner) meaning actor holds a spanner,
      - g0brvc06(spanner) meaning the spanner is in a usable ("good") state,
      - bu1om0m1(nut) meaning a nut is loose,
      - g59iwyxf(nut) meaning a nut is tightened (the goal),
      - kxokb5hx loc1 loc2 facts in static information describe directed
        adjacency between locations (edges used by the move action).
    The heuristic estimates the number of actions required to tighten all
    nuts that are not yet in the goal state (g59iwyxf). It reasons about:
      - distances along the directed location graph (number of move actions),
      - the need to pick up a good spanner (moi6wnfu),
      - and the tightening action itself (kjsqfni8).
    The heuristic is not required to be admissible; it tries to be informative
    while remaining cheap to compute.

    Assumptions
    -----------
    - A nut is tightened by a single kjsqfni8 action; the preconditions require:
        actor and nut at same location, actor holds a spanner, spanner is good,
        nut is loose.
    - Picking up a spanner (moi6wnfu) requires the actor and the spanner to be
      at the same location and costs 1 action.
    - Moving the actor between adjacent locations (kxokb5hx edges) costs 1
      per edge and is accomplished by the n9sbfmrs action.
    - Tightening costs 1 action (kjsqfni8).
    - After tightening a nut the spanner's g0brvc06 predicate is removed (spanner
      is no longer "good") — the heuristic ignores resource contention between
      multiple nuts (optimistic, i.e., non-admissible).
    - If no usable (good) spanner is available, the heuristic returns a large
      finite value (to keep the value finite for unsolvable configurations).

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
      - the set of goal facts (task.goals) so we can detect when a nut goal
        is already achieved,
      - the directed adjacency graph of locations from the static facts
        (kxokb5hx facts in task.static).
    The adjacency graph is stored as a dictionary mapping a location to the set
    of directly reachable successor locations (directed edges).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each nut that is not yet satisfied by the goal (i.e., there is no
    g59iwyxf(nut) in the current state), estimate the minimal number of actions
    required to make g59iwyxf(nut) true. The estimate for a single nut is
    computed as follows:

    1. Identify candidate usable spanners:
       - A spanner is considered usable if g0brvc06(spanner) is present in the state.
       - The spanner's current location is:
           - a location L if there is u4srx1nb spanner L in the state, or
           - the holder's location if some yud7t3vb holder spanner is present.
         (If neither information is present the spanner is treated as unreachable.)

    2. Identify actors:
       - We regard as actors the objects that appear in u4srx1nb facts and are
         not classified as spanners or nuts (this matches typical encoding:
         actors are distinct objects like 'bob').
       - An actor's current location is given by u4srx1nb actor Loc.

    3. Compute directed shortest-path distances between locations using BFS
       on the adjacency graph extracted from static facts. Distances are the
       number of move actions required.

    4. For each combination of (actor, usable_spanner), compute the optimistic
       cost to tighten the nut:
       - If the actor currently holds the usable spanner (yud7t3vb actor sp),
         cost = distance(actor_loc, nut_loc) + 1   # move to nut + tighten
       - Otherwise, cost = distance(actor_loc, sp_loc) + 1               # move to spanner + pick
                               + distance(sp_loc, nut_loc) + 1        # move to nut + tighten
       - If a required distance is unreachable (no path), treat that option
         as infinite.
       - The minimal cost over all (actor, usable_spanner) pairs is taken as the
         estimate for that nut.

    5. The heuristic value is the sum of these per-nut minimal estimates for all
       nuts not yet in the goal. This is optimistic across nuts (it ignores the
       fact that a spanner used for one nut may become unusable for another),
       which makes the heuristic aggressive (non-admissible) but generally
       informative for greedy best-first search.

    Implementation notes:
    - The code carefully parses PDDL facts using get_parts(...) which strips
      parentheses and splits on whitespace (satisfying the requirement to
      ignore surrounding brackets).
    - If any nut cannot be reached or no usable spanner is available for that
      nut, the heuristic returns a large finite number (100000) instead of infinity.
    """

    def __init__(self, task):
        # Store goal facts
        self.goals = set(task.goals)
        # Build directed adjacency graph from static facts (kxokb5hx)
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "kxokb5hx" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj.setdefault(src, set()).add(dst)
                # ensure nodes exist even if no outgoing edges
                self.adj.setdefault(dst, set())

        # Cache for BFS distance computations: map start -> dict(target -> dist)
        self._dist_cache = {}

    def _bfs_distances_from(self, start):
        """Return a dict mapping location -> shortest distance from start (directed)."""
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

    def _distance(self, src, dst):
        """Return shortest path length from src to dst (directed). Return None if unreachable."""
        if src == dst:
            return 0
        dist_map = self._bfs_distances_from(src)
        return dist_map.get(dst, None)

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goal facts hold, heuristic is zero.
        if self.goals <= state:
            return 0

        # Helper data structures extracted from the current state.
        # spanner_good: set of spanners that currently have g0brvc06
        spanner_good = {get_parts(f)[1] for f in state if get_parts(f)[0] == "g0brvc06"}

        # nuts already tightened (goal-like facts) and loose nuts
        nuts_tight = {get_parts(f)[1] for f in state if get_parts(f)[0] == "g59iwyxf"}
        nuts_loose = {get_parts(f)[1] for f in state if get_parts(f)[0] == "bu1om0m1"}

        # all nuts that appear in location facts or in nut predicates
        nuts = set()
        for f in state:
            parts = get_parts(f)
            if parts[0] in ("g59iwyxf", "bu1om0m1"):
                nuts.add(parts[1])
            if parts[0] == "u4srx1nb":
                subj = parts[1]
                # if subject is known via nut predicates add heuristically later
                # we will not assume classification here

        # Map actor -> location from u4srx1nb facts (actors are those u4srx1nb objects not in spanner_good/nuts sets)
        obj_locations = {}  # object -> location (direct u4srx1nb facts)
        for f in state:
            parts = get_parts(f)
            if parts[0] == "u4srx1nb" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                obj_locations[obj] = loc

        # Who holds which spanner? yud7t3vb actor spanner facts.
        holder_of_spanner = {}
        for f in state:
            parts = get_parts(f)
            if parts[0] == "yud7t3vb" and len(parts) >= 3:
                actor, sp = parts[1], parts[2]
                holder_of_spanner[sp] = actor

        # Identify spanner candidates (only those that are currently "good")
        spanners = set(spanner_good)

        # Identify actors: any object appearing in u4srx1nb that is not known to be a spanner or a nut
        # and also any object appearing as first arg in yud7t3vb (an actor holding a spanner).
        actors = set()
        for obj in obj_locations:
            if obj not in spanners and obj not in nuts:
                actors.add(obj)
        # Also include actors that appear in yud7t3vb first position
        for f in state:
            parts = get_parts(f)
            if parts[0] == "yud7t3vb":
                actors.add(parts[1])

        # If no actors are found, we cannot proceed sensibly: return large value
        if not actors:
            return 100000

        # Gather nut set as all objects that appear in goal facts (g59iwyxf) or loose facts
        # If the task goal contains g59iwyxf facts, use them to know which nuts are relevant
        goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == "g59iwyxf"}
        # If no explicit goal nuts in the task (rare), fallback to nuts detected in state
        if not goal_nuts:
            goal_nuts = set(nuts) | nuts_loose | nuts_tight

        total_cost = 0
        BIG = 100000

        # For each nut that is in goal_nuts but not yet tightened, estimate cost to tighten it
        for nut in goal_nuts:
            if ("(g59iwyxf " + nut + ")") in state or nut in nuts_tight:
                # already achieved
                continue

            # Nut must be loose to be tightened (kjsqfni8 precondition). If not loose and not tightened, we treat as unreachable.
            if nut not in nuts_loose:
                # Some encodings might allow making it loose first, but there is no action for that in this domain.
                return BIG

            # Determine nut location
            nut_loc = obj_locations.get(nut, None)
            if nut_loc is None:
                # If the nut has no explicit u4srx1nb fact, it's unreachable in current encoding
                return BIG

            # Consider all usable spanners (only those that are good)
            if not spanners:
                # No usable spanners in the state => cannot tighten any nuts
                return BIG

            best_cost_for_nut = None

            for sp in spanners:
                # Determine spanner's current location: either directly in obj_locations, or at the location of its holder.
                if sp in obj_locations:
                    sp_loc = obj_locations[sp]
                    sp_held_by = None
                elif sp in holder_of_spanner:
                    sp_held_by = holder_of_spanner[sp]
                    sp_loc = obj_locations.get(sp_held_by, None)
                else:
                    # No location info for this spanner; treat as unreachable
                    continue

                # For each actor compute cost of sequence to tighten nut using this spanner
                for actor in actors:
                    actor_loc = obj_locations.get(actor, None)
                    if actor_loc is None:
                        # Actor without a known location; skip actor
                        continue

                    # If actor already holds this spanner
                    if holder_of_spanner.get(sp) == actor:
                        # cost = moves from actor_loc to nut_loc + 1 (tighten)
                        d = self._distance(actor_loc, nut_loc)
                        if d is None:
                            continue
                        cost = d + 1
                    else:
                        # actor must move to sp_loc, pick (1), then move to nut_loc, then tighten (1)
                        if sp_loc is None:
                            continue
                        d1 = self._distance(actor_loc, sp_loc)
                        if d1 is None:
                            continue
                        d2 = self._distance(sp_loc, nut_loc)
                        if d2 is None:
                            continue
                        cost = d1 + 1 + d2 + 1  # move to sp + pick + move to nut + tighten

                    if best_cost_for_nut is None or cost < best_cost_for_nut:
                        best_cost_for_nut = cost

            if best_cost_for_nut is None:
                # No viable (actor, spanner) combination found for this nut -> unreachable
                return BIG

            total_cost += best_cost_for_nut

        # Ensure heuristic is finite
        return int(total_cost)