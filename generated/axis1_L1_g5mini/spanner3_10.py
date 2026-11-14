from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (tightening nuts with spanners).

    This heuristic estimates the number of actions required to achieve all
    g59iwyxf (tightened) goals for nuts. The domain contains:
    - u4srx1nb X L    : object X is at location L
    - yud7t3vb P S    : person P has spanner S
    - g0brvc06 S      : spanner S is working (can be used to tighten once)
    - bu1om0m1 N      : nut N is untight
    - g59iwyxf N      : nut N is tightened (goal)
    - kxokb5hx A B    : directed adjacency A -> B between locations
    - n9sbfmrs        : moves a person along adjacency (one step per application)
    - moi6wnfu        : person picks up a spanner at the same location (1 action)
    - kjsqfni8        : tighten a nut when person has spanner and spanner is working (1 action);
                       it consumes the spanner's working status (spanner breaks)

    Assumptions
    -----------
    - Persons are inferred as objects that appear as the first argument of
      yud7t3vb (possession) or as an object in a u4srx1nb fact that is not
      recognized as a spanner or a nut.
    - Each working spanner (g0brvc06) can be used at most once (it breaks on use).
    - Persons can carry multiple spanners (domain allows multiple yud7t3vb facts
      for the same person). The heuristic will prefer using spanners already
      possessed by the chosen person when beneficial.
    - If an object (person, spanner, nut) location is not reachable by the
      adjacency graph, the heuristic penalizes such plans with a large finite cost.
    - The heuristic is not required to be admissible; it aims to produce
      informative finite estimates quickly.

    Heuristic Initialization
    ------------------------
    - Store the goal facts passed in the task (self.goals).
    - Extract the directed adjacency graph between locations from the task's
      static facts: collect edges for facts whose predicate is kxokb5hx.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already true in the state, return 0.
    2. Identify the set of nuts that still need tightening (goals not yet satisfied).
    3. Parse the state:
       - location_of[obj] from u4srx1nb facts,
       - possession facts yud7t3vb -> mapping spanner -> holder and holder -> set(spanners),
       - set of working spanners from g0brvc06 facts,
       - set of tightened nuts from g59iwyxf facts.
    4. Build a directed adjacency graph (from static facts). Use BFS to compute
       shortest path lengths (number of move actions) between locations.
    5. Prepare a list of available working spanners and their current place:
       - If a spanner is possessed by the chosen person it is immediately available
         to use (no pick action needed).
       - Otherwise the person must travel to the spanner's location, perform a
         pick action (moi6wnfu), then travel to the nut location and tighten it.
    6. Greedy assignment loop:
       - Choose a single representative person (if multiple persons are present the
         heuristic picks one deterministically: the first it discovers).
       - Repeatedly pick the outstanding nut with the minimal incremental cost
         using any remaining available working spanner. The cost model:
           * If the chosen spanner is already possessed by the current person:
               cost = distance(person_loc, nut_loc) + 1   # move steps + tighten
           * If the spanner is unpossessed:
               cost = distance(person_loc, spanner_loc)   # move to spanner
                      + 1                                # pick up (moi6wnfu)
                      + distance(spanner_loc, nut_loc)   # move to nut
                      + 1                                # tighten (kjsqfni8)
         Distances are computed via BFS on the adjacency graph; if unreachable,
         a large finite penalty is used for that option.
       - After assigning a spanner to a nut, remove that spanner from the pool
         (it breaks on use) and set the current person location to the nut's
         location (tightening leaves the person there). Continue until all
         nuts have been assigned or no working spanners remain.
    7. If not enough working spanners are available to cover all pending nuts,
       the heuristic returns a large finite cost to reflect difficulty/unsolvability.
    8. The heuristic returns the sum of the chosen incremental costs.

    Implementation remarks
    ----------------------
    - The heuristic is intentionally greedy (not globally optimal) to keep
      computation fast while producing informative guidance for greedy
      best-first search.
    - BFS on the adjacency graph is efficient because the number of locations
      is typically small compared to the total search space.
    """

    def __init__(self, task):
        # store the goal facts
        self.goals = task.goals
        # store static facts and build adjacency graph from kxokb5hx facts
        self.static = task.static
        self.adj = {}  # directed adjacency: src -> set(dst)
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "kxokb5hx" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj.setdefault(src, set()).add(dst)

    def _bfs_distance(self, src, dst):
        """Return the directed shortest-path length from src to dst using adjacency self.adj.
        If src == dst returns 0. If dst is unreachable returns None."""
        if src == dst:
            return 0
        if src not in self.adj:
            return None
        q = deque()
        q.append((src, 0))
        visited = {src}
        while q:
            node, d = q.popleft()
            for nb in self.adj.get(node, ()):
                if nb == dst:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        return None

    def __call__(self, node):
        """Compute the heuristic estimate for the node.state (a frozenset of fact strings)."""
        state = node.state

        # If all goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Helper containers
        location_of = {}         # object -> location (from u4srx1nb facts)
        possessed_by = {}        # spanner -> person (from yud7t3vb)
        possessions = {}         # person -> set(spanners)
        working_spanners = set() # spanners that have g0brvc06
        tightened_nuts = set()   # nuts already tightened (g59iwyxf)
        untight_nuts = set()     # nuts that are untight (from goals or bu1om0m1)
        untight_by_state = set() # nuts that are currently untight in state (bu1om0m1)

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "u4srx1nb" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                location_of[obj] = loc
            elif pred == "yud7t3vb" and len(parts) >= 3:
                person, spanner = parts[1], parts[2]
                possessed_by[spanner] = person
                possessions.setdefault(person, set()).add(spanner)
            elif pred == "g0brvc06" and len(parts) >= 2:
                working_spanners.add(parts[1])
            elif pred == "g59iwyxf" and len(parts) >= 2:
                tightened_nuts.add(parts[1])
            elif pred == "bu1om0m1" and len(parts) >= 2:
                untight_by_state.add(parts[1])

        # Determine which nuts remain to be tightened according to the goals.
        pending_nuts = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "g59iwyxf" and len(parts) >= 2:
                nut = parts[1]
                if nut not in tightened_nuts:
                    pending_nuts.append(nut)

        # If no pending nuts, return 0 (safety; should have been caught above)
        if not pending_nuts:
            return 0

        # Build set of all known spanners (those that appear as working or are possessed or have locations).
        known_spanners = set(working_spanners) | set(possessed_by.keys())
        for obj in location_of:
            # some spanners appear only in location facts; include them conservatively
            # (they will be useful only if they are also working)
            if obj not in known_spanners:
                # we cannot be certain whether the object is a spanner, but this is harmless
                # since only working spanners (g0brvc06) can be used.
                pass

        # Identify person(s). Heuristic: persons are those that appear in possession facts
        # as holders, or those objects in location_of that are not nuts and not known spanners.
        persons = set(possessed_by.values())
        if not persons:
            # fallback: any object in location_of that is not a nut and not a working spanner is a candidate person
            candidate_persons = set(location_of.keys()) - working_spanners - set(tightened_nuts) - set(pending_nuts)
            # choose those that are not known nuts (if any)
            persons = candidate_persons

        # Pick a representative current person location.
        current_person_loc = None
        if persons:
            # Prefer a person who has possessions (so that we can use possessed spanners first).
            chosen_person = None
            for p in persons:
                if p in possessions:
                    chosen_person = p
                    break
            if chosen_person is None:
                chosen_person = next(iter(persons))
            current_person_loc = location_of.get(chosen_person, None)

        # Build pool of available working spanners and their status (possessed or at location).
        # Only working spanners (g0brvc06 present) can be used to tighten once.
        pool = {}
        for s in working_spanners:
            holder = possessed_by.get(s, None)
            loc = None
            if holder is None:
                loc = location_of.get(s, None)
            pool[s] = {"holder": holder, "loc": loc}

        # Greedy assignment of spanners to pending nuts.
        total_cost = 0
        LARGE_PENALTY = 1000  # finite but large penalty used when something is unreachable or missing
        # copy of pool keys to modify
        available_spanners = set(pool.keys())

        # If there are no working spanners at all, return large cost (unsolvable unless no pending nuts).
        if not available_spanners:
            return LARGE_PENALTY * len(pending_nuts)

        # Work on a local mutable list of pending nuts (so we can reorder)
        remaining_nuts = list(pending_nuts)

        # For deterministic behavior sort lists
        remaining_nuts.sort()

        # Greedy loop: assign the cheapest available spanner for the current person location
        for _ in range(len(remaining_nuts)):
            if not remaining_nuts:
                break

            # For the current step, find the nut-spanner pair (nut in remaining_nuts, spanner in available_spanners)
            # that yields the smallest incremental cost given current_person_loc.
            best = None  # (cost, nut, spanner, new_person_loc)
            for nut in remaining_nuts:
                nut_loc = location_of.get(nut, None)
                if nut_loc is None:
                    # If nut location unknown, treat as unreachable -> penalize
                    continue
                for sp in list(available_spanners):
                    info = pool.get(sp)
                    if info is None:
                        continue
                    holder = info["holder"]
                    sp_loc = info["loc"]
                    # If spanner is possessed by someone else (not our chosen person), then unless we
                    # allow multi-person plans, we skip it (we don't reason about coordinating other agents).
                    # However, if we can identify that another person exists and has a location and could
                    # do the job, we could include that. To keep heuristic fast, only use spanners possessed
                    # by our chosen person or unpossessed spanners.
                    if holder is not None and current_person_loc is not None:
                        # If holder is not our chosen person, we still allow it but must consider
                        # the cost for that holder to move to the nut themselves. For speed we skip
                        # this complexity and only accept spanners possessed by (or unpossessed w.r.t our person).
                        # Accept it only if holder is the chosen person (i.e., current_person_loc corresponds to holder)
                        # or if no person was identified (current_person_loc is None) we'll allow using it
                        if current_person_loc is not None:
                            # detect if holder's location equals current_person_loc (i.e., same agent)
                            holder_loc = location_of.get(holder, None)
                            if holder_loc != current_person_loc:
                                # skip this spanner as not immediately available to our representative person
                                continue

                    # compute cost
                    cost = 0
                    # If spanner is currently possessed by current person (holder matches chosen person),
                    # no pick action is needed; only moves to nut (if any) and tighten.
                    if holder is not None and current_person_loc is not None and location_of.get(holder, None) == current_person_loc:
                        # cost = moves from current_person_loc to nut_loc + tighten
                        dist_p_n = self._bfs_distance(current_person_loc, nut_loc)
                        if dist_p_n is None:
                            cost = LARGE_PENALTY
                        else:
                            cost = dist_p_n + 1
                        new_person_loc = nut_loc
                    else:
                        # spanner unpossessed (or we have no identified person): person must reach spanner,
                        # pick it, then go to nut and tighten.
                        # If we have a person location known, use that as start; else start at spanner loc.
                        # Distances: person -> spanner, spanner -> nut
                        if sp_loc is None:
                            # no location info for spanner: penalize
                            cost = LARGE_PENALTY
                            new_person_loc = nut_loc
                        else:
                            if current_person_loc is None:
                                # no identified person: assume start at spanner (optimistic) to avoid huge penalty
                                dist_p_s = 0
                            else:
                                dist_p_s = self._bfs_distance(current_person_loc, sp_loc)
                            dist_s_n = self._bfs_distance(sp_loc, nut_loc)
                            if dist_p_s is None or dist_s_n is None:
                                cost = LARGE_PENALTY
                            else:
                                # pick and tighten each cost 1
                                cost = dist_p_s + 1 + dist_s_n + 1
                            new_person_loc = nut_loc

                    # record candidate best
                    if best is None or cost < best[0]:
                        best = (cost, nut, sp, new_person_loc)

            if best is None:
                # no feasible assignment found for remaining nuts -> penalize heavily
                total_cost += LARGE_PENALTY * len(remaining_nuts)
                break

            # apply best assignment
            cost, chosen_nut, chosen_sp, new_loc = best
            total_cost += cost
            # remove chosen nut and spanner from the pools
            remaining_nuts.remove(chosen_nut)
            if chosen_sp in available_spanners:
                available_spanners.remove(chosen_sp)
            # update current person location
            current_person_loc = new_loc

            # if no spanners remain but nuts still remain, penalize and break
            if not available_spanners and remaining_nuts:
                total_cost += LARGE_PENALTY * len(remaining_nuts)
                break

        return int(total_cost)