from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the token list of a fact string '(pred a b ...)' -> ['pred','a','b',...]"""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the secret_domain (e2zu8y93) that estimates the number of actions
    required to achieve all goal nut predicates (g59iwyxf <nut>). The domain models
    an agent that can walk along locations (kxokb5hx edges), pick up spanners
    (moi6wnfu), and use spanners to tighten nuts (kjsqfni8). Each spanner that is
    usable is marked by (g0brvc06 <spanner>) and can be used once (the action
    removes this predicate). Picking up a spanner requires agent and spanner to
    be colocated and costs one action; using a spanner to tighten a nut is one
    action; moving the agent along one kxokb5hx edge costs one action.

    Assumptions
    -----------
    - Goal facts are of the form (g59iwyxf <nut>).
    - Spanners that can be used have the fact (g0brvc06 <spanner>).
    - Spanners can be either on the ground (u4srx1nb <spanner> <loc>) or
      carried by an agent (yud7t3vb <agent> <spanner>); if carried, effective
      location is the agent's location (u4srx1nb <agent> <loc>).
    - Nuts have a fixed location in the state via (u4srx1nb <nut> <loc>).
    - The topology of locations (kxokb5hx facts) is static and directed; moving
      along each kxokb5hx arc corresponds to one move action.
    - There may be multiple spanners and multiple nuts. Each usable spanner can
      be consumed to tighten at most one nut.
    - The heuristic may underestimate in some situations (it is not required to
      be admissible), but it aims to be informative and cheaply computable.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
    - Stores the goal nut names extracted from task.goals.
    - Builds a directed adjacency graph of locations from task.static facts
      having predicate kxokb5hx.
    - Keeps a reference to the full set of static facts (task.static) and the
      goals (task.goals) for quick checks.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state) we compute an estimate as follows:

    1. If all goals are already satisfied in the state, return 0.

    2. Parse the state's facts to gather:
       - loc_of[obj] for facts (u4srx1nb obj loc).
       - carried_by[spanner] = agent for facts (yud7t3vb agent spanner).
       - agents' locations via loc_of[agent] when agent is carried_by some spanner.
       - usable_spanners: those spanners appearing in (g0brvc06 spanner).
       - remaining goal nuts: those nuts from self.goal_nuts whose
         (g59iwyxf nut) fact is not yet in the state.

    3. For each usable spanner compute its effective location:
       - If carried_by[spanner] = agent and agent location known, spanner_loc =
         agent_loc.
       - Else, if (u4srx1nb spanner loc) exists, spanner_loc = that loc.
       - Otherwise treat spanner as unavailable (penalize if needed).

    4. For each remaining nut compute the cheapest spanner assignment cost:
       - Use BFS on the static location graph to compute the shortest number of
         move actions between spanner location and nut location.
       - pick_cost = 0 if the spanner is already carried by an agent; else 1.
       - tighten_cost = 1 (the kjsqfni8 action).
       - total_pair_cost = moves_distance + pick_cost + tighten_cost.

    5. Perform a greedy assignment of spanners to nuts:
       - Repeatedly pick the remaining nut/spanner pair with the smallest pair
         cost, add that cost to the heuristic total, and remove that spanner and
         nut from further consideration.
       - If some nuts remain but no usable spanners are left, assign a large
         finite penalty per remaining nut (reflects that the plan requires
         additional spanners that are not present); this keeps the heuristic
         finite for all states.

    6. Return the summed cost as the heuristic estimate.

    Notes
    -----
    - BFS distances are cached for efficiency.
    - The heuristic intentionally uses a greedy matching for speed; this is not
      guaranteed optimal but is efficient and informative for guiding greedy
      best-first search.
    """

    def __init__(self, task):
        # Save goals and static facts
        self.goals = task.goals
        self.static = task.static

        # Extract the set of goal nut names (from predicates g59iwyxf)
        self.goal_nuts = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "g59iwyxf" and len(parts) >= 2:
                self.goal_nuts.append(parts[1])

        # Build adjacency graph from static kxokb5hx facts (directed edges).
        self.adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "kxokb5hx" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, []).append(b)

        # BFS distance cache between locations
        self._dist_cache = {}

    def _distance(self, src, dst):
        """Return directed shortest-path length from src to dst using static graph.
        If unreachable, return a large sentinel (INF_DIST)."""
        INF_DIST = 10**6
        if src == dst:
            return 0
        if (src, dst) in self._dist_cache:
            return self._dist_cache[(src, dst)]
        # BFS
        if src not in self.adj:
            self._dist_cache[(src, dst)] = INF_DIST
            return INF_DIST
        q = deque()
        q.append((src, 0))
        visited = {src}
        while q:
            node, d = q.popleft()
            for neigh in self.adj.get(node, ()):
                if neigh in visited:
                    continue
                if neigh == dst:
                    self._dist_cache[(src, dst)] = d + 1
                    return d + 1
                visited.add(neigh)
                q.append((neigh, d + 1))
        self._dist_cache[(src, dst)] = INF_DIST
        return INF_DIST

    def __call__(self, node):
        state = node.state

        # If goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse state facts
        loc_of = {}         # object -> location from (u4srx1nb obj loc)
        carried_by = {}     # spanner -> agent from (yud7t3vb agent spanner)
        usable_spanners = set()  # spanners with (g0brvc06 spanner)
        goal_achieved = set()   # set of nuts already g59iwyxf in state

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "u4srx1nb" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                loc_of[obj] = loc
            elif pred == "yud7t3vb" and len(parts) >= 3:
                agent, sp = parts[1], parts[2]
                carried_by[sp] = agent
            elif pred == "g0brvc06" and len(parts) >= 2:
                usable_spanners.add(parts[1])
            elif pred == "g59iwyxf" and len(parts) >= 2:
                goal_achieved.add(parts[1])

        # Identify which goal nuts are still to be tightened
        remaining_nuts = [nut for nut in self.goal_nuts if nut not in goal_achieved]
        if not remaining_nuts:
            return 0

        # Compute effective locations for spanners that are usable
        spanner_locations = {}  # spanner -> location (or None if unknown)
        for sp in usable_spanners:
            if sp in carried_by:
                agent = carried_by[sp]
                agent_loc = loc_of.get(agent)
                spanner_locations[sp] = agent_loc  # may be None if agent location unknown
            else:
                sp_loc = loc_of.get(sp)
                spanner_locations[sp] = sp_loc  # may be None if not located

        # Prepare list of available spanners (those with a known effective location or carried by an agent)
        available_spanners = list(spanner_locations.keys())

        total_cost = 0

        # Greedy assignment: repeatedly pair the nut and spanner with minimal pair cost
        # Make copies to mutate
        nuts_left = list(remaining_nuts)
        spanners_left = set(available_spanners)

        # If there are no usable spanners at all, return a finite large estimate
        if not spanners_left:
            # Finite penalty per nut: assume pick+tighten but no movement known (heavy penalty)
            PENALTY_PER_NUT = 100
            return len(nuts_left) * PENALTY_PER_NUT

        while nuts_left and spanners_left:
            best = None  # (cost, nut, spanner, details)
            for nut in nuts_left:
                nut_loc = loc_of.get(nut)
                # If nut location unknown, treat as heavy cost
                if nut_loc is None:
                    # Heavily penalize unknown nut location
                    for sp in spanners_left:
                        c = 200
                        if best is None or c < best[0]:
                            best = (c, nut, sp)
                    continue

                for sp in spanners_left:
                    sp_loc = spanner_locations.get(sp)
                    # If spanner location unknown, assign heavy cost
                    if sp_loc is None:
                        # possible carried by agent whose location unknown or not located
                        pair_cost = 200
                    else:
                        # movement cost along static graph
                        move_cost = self._distance(sp_loc, nut_loc)
                        # If unreachable, large penalty
                        if move_cost >= 10**6:
                            pair_cost = 500
                        else:
                            pick_cost = 0 if sp in carried_by else 1
                            tighten_cost = 1
                            pair_cost = move_cost + pick_cost + tighten_cost
                    if best is None or pair_cost < best[0]:
                        best = (pair_cost, nut, sp)
            if best is None:
                break
            cost, chosen_nut, chosen_sp = best
            total_cost += cost
            # remove chosen nut and chosen spanner
            try:
                nuts_left.remove(chosen_nut)
            except ValueError:
                pass
            spanners_left.discard(chosen_sp)

        # If nuts remain but no spanners left, assign finite penalty per remaining nut
        if nuts_left:
            PENALTY_PER_NUT = 100
            total_cost += len(nuts_left) * PENALTY_PER_NUT

        # Heuristic must be finite integer
        return int(total_cost)