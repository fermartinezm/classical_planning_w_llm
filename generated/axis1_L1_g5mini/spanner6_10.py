from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic tailored for the secret_domain (e2zu8y93). The
    domain models an agent that must use spanners to tighten nuts. Spanners are
    objects that can be picked up and carried by the agent; using (tightening)
    a spanner on a nut consumes the spanner's usability. Movement between
    locations is governed by a directed adjacency predicate.

    The heuristic estimates the number of actions (moves, picks, tightenings)
    required to reach the goal state in which specific nuts are tightened
    (predicate g59iwyxf). It builds a cheap approximate route for the single
    agent that sequentially obtains (or uses carried) spanners and tightens
    the remaining nuts. The estimate accounts for travel distances on the
    location graph and the necessary pick/tighten actions.

    Assumptions
    -----------
    - There is at least one agent (an object that appears in u4srx1nb facts but
      is not identified as a spanner or a nut).
    - The goals of interest are all of the form (g59iwyxf <nut>).
    - Movement along locations is along directed edges given by the static
      kxokb5hx facts. Each edge has unit cost (one move action).
    - Picking up a spanner (moi6wnfu) costs 1 action, tightening a nut
      (kjsqfni8) costs 1 action. Each movement along one adjacency edge costs 1.
    - A spanner can be consumed (made unusable) when used; therefore each nut
      requires a usable spanner (g0brvc06 true for that spanner at the time of use).
    - The agent may carry multiple spanners (domain allows yud7t3vb without a
      capacity restriction), but a spanner must be usable (g0brvc06) to serve
      as the tool for tightening.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts:
    - The set of goal nut names (from task.goals looking for g59iwyxf facts).
    - The directed adjacency graph between location names from the static facts
      (kxokb5hx facts). The graph is stored as a mapping from location -> list
      of successor locations.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state the heuristic proceeds as follows:

    1. If all goal facts are already satisfied in the provided state, return 0.

    2. Identify which nuts still require tightening:
       - Use the set of goal nuts (from initialization) and remove those nuts
         already having the g59iwyxf fact in the current state.

    3. Collect current world information:
       - Agent location: choose one object that appears in u4srx1nb facts but
         that is not a known spanner (g0brvc06) or nut (bu1om0m1 / g59iwyxf).
         If multiple such objects exist, the heuristic uses the first as the
         acting agent.
       - Spanners: for each spanner object (identified by presence of
         g0brvc06 in the state), determine whether it is currently carried
         (yud7t3vb agent spanner) and compute its effective location:
         either the location it lies at (u4srx1nb spanner loc) or the agent's
         location if carried.
       - Nuts: for each nut still to be tightened, determine its current
         location from u4srx1nb facts.

    4. Precompute shortest path distances (BFS) on the directed location
       graph between all relevant locations (agent location, spanner locations,
       nut locations). Distances are the number of move actions (n9sbfmrs).

    5. Build an approximate sequential plan greedily:
       - Start with current_location = agent_location.
       - While there remain nuts to be tightened:
         * Consider all usable spanners (those with g0brvc06 in the state and
           not already consumed by the heuristic plan) and all remaining nuts.
         * For each pair compute travel_cost = dist(current_location, spanner_loc)
           + dist(spanner_loc, nut_loc). If the spanner is already carried by
           the chosen agent, dist(current_location, spanner_loc) will be 0
           (and the pick action is not required).
         * Select the pair with minimal travel_cost.
         * Add to the heuristic total:
             - travel_cost (movement actions),
             - pick_cost = 0 if the selected spanner is currently carried by
               the agent, else 1 (moi6wnfu),
             - tighten_cost = 1 (kjsqfni8).
         * Mark the selected spanner consumed (cannot be used again) and remove
           the nut from the remaining set. Set current_location = nut_loc.
       - If at any point no feasible (reachable) pair exists, return a large
         finite value to indicate difficulty (INF). For solvable states with
         a connected location graph this remains finite.

    6. Return the accumulated action count as the heuristic estimate.

    The greedy pairing is a computationally cheap approximation that captures
    travel effort, pickup cost, and the 1-action tightening step per nut while
    respecting that each tightening consumes a usable spanner. This often
    guides search effectively while remaining efficient to compute.
    """

    def __init__(self, task):
        # Store goal facts (strings) to quickly test for goal states.
        self.goals = set(task.goals)

        # Extract goal nut names from goals (predicate g59iwyxf <nut>).
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "g59iwyxf" and len(parts) >= 2:
                self.goal_nuts.add(parts[1])

        # Build directed adjacency graph from static facts kxokb5hx loc1 loc2.
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "kxokb5hx" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj.setdefault(src, []).append(dst)

        # For convenience, also ensure nodes with no outgoing edges appear in adj
        # (BFS will still handle absent keys).
        for src in list(self.adj.keys()):
            for dst in self.adj[src]:
                self.adj.setdefault(dst, [])

    def _bfs_distance(self, start, goal):
        """Return shortest directed distance between two location names or INF."""
        if start == goal:
            return 0
        if start not in self.adj:
            # If start not present in adj we still allow starting from it if it has no outgoing edges.
            # Represent as isolated node: no path unless start == goal handled above.
            return INF
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, d = dq.popleft()
            for succ in self.adj.get(node, []):
                if succ == goal:
                    return d + 1
                if succ not in visited:
                    visited.add(succ)
                    dq.append((succ, d + 1))
        return INF

    def __call__(self, node):
        state = node.state

        # If goals are already all satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse current facts for quick lookup.
        facts = list(state)

        # Identify which nuts are required by the goal and still not satisfied.
        # A nut is considered satisfied if (g59iwyxf <nut>) is present in state.
        tightened_nuts = {get_parts(f)[1] for f in facts if get_parts(f)[0] == "g59iwyxf" and len(get_parts(f)) >= 2}
        nuts_to_tighten = list(self.goal_nuts - tightened_nuts)

        if not nuts_to_tighten:
            return 0

        # Determine all currently usable spanners: those that have (g0brvc06 spanner)
        usable_spanners = set(get_parts(f)[1] for f in facts if get_parts(f)[0] == "g0brvc06" and len(get_parts(f)) >= 2)

        # Map spanner -> (is_carried_by_agent_bool, effective_location)
        # effective_location is a location name string (not an agent name).
        spanner_carried = {}
        # First detect any yud7t3vb facts: (yud7t3vb agent spanner)
        carried_by = {}
        for f in facts:
            parts = get_parts(f)
            if parts[0] == "yud7t3vb" and len(parts) >= 3:
                agent, sp = parts[1], parts[2]
                carried_by[sp] = agent

        # Next build object location map from u4srx1nb facts: (u4srx1nb object location)
        obj_loc = {}
        for f in facts:
            parts = get_parts(f)
            if parts[0] == "u4srx1nb" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                obj_loc[obj] = loc

        # Identify an agent: pick an object that appears as subject in u4srx1nb but is
        # not identified as a spanner or a nut. This is a heuristic choice;
        # typically there's a single agent (e.g., 'bob').
        agent = None
        for f in facts:
            parts = get_parts(f)
            if parts[0] == "u4srx1nb" and len(parts) >= 3:
                subj = parts[1]
                # If subj is not a spanner and not a known nut (goal or tightened), treat as agent
                if subj not in usable_spanners and subj not in self.goal_nuts and subj not in tightened_nuts:
                    agent = subj
                    break
        # If no agent found by that rule, try to find any object that appears as a carrier in yud7t3vb
        if agent is None and carried_by:
            # take the first carrier
            agent = next(iter(carried_by.values()))

        # If still None, we cannot reason about agent location; return large but finite heuristic
        if agent is None or agent not in obj_loc:
            return INF

        agent_loc = obj_loc.get(agent)

        # Build spanner info (only consider usable spanners, they are needed).
        spanner_info = {}
        for sp in usable_spanners:
            # Determine if carried by the selected agent
            carried = (sp in carried_by and carried_by[sp] == agent)
            if carried:
                effective_loc = agent_loc
            else:
                # If spanner on the ground, use its location from obj_loc
                effective_loc = obj_loc.get(sp)
            # If no known location for this spanner, skip it (unusable for planning)
            if effective_loc is None:
                continue
            spanner_info[sp] = (carried, effective_loc)

        # Build nut -> location mapping for nuts that must still be tightened.
        nut_loc = {}
        for nut in nuts_to_tighten:
            # nut location should appear in obj_loc
            if nut in obj_loc:
                nut_loc[nut] = obj_loc[nut]
            else:
                # If no location info, cannot reach; return INF
                return INF

        # If not enough usable spanners to cover remaining nuts, it may still be doable
        # if some spanners are not listed in g0brvc06 but exist (broken spanners cannot be used).
        if len(spanner_info) == 0:
            return INF

        # Greedy sequential assignment: choose next (spanner, nut) that minimizes travel from current.
        current_loc = agent_loc
        total_cost = 0
        available_spanners = dict(spanner_info)  # spanner -> (carried_bool, loc)
        remaining_nuts = set(nut_loc.keys())

        # Precompute distances between all relevant locations to avoid repeated BFS:
        relevant_locations = set(available_spanners[s][1] for s in available_spanners) | set(nut_loc[n] for n in nut_loc) | {agent_loc}
        dist_cache = {}
        rel_list = list(relevant_locations)
        for a in rel_list:
            for b in rel_list:
                if (a, b) not in dist_cache:
                    dist_cache[(a, b)] = self._bfs_distance(a, b)

        while remaining_nuts:
            best = None
            best_cost = INF
            best_sp = None
            best_nu = None
            best_pick_cost = 1

            for sp, (carried, sp_loc) in available_spanners.items():
                for nu in remaining_nuts:
                    n_loc = nut_loc[nu]
                    # distances
                    d1 = dist_cache.get((current_loc, sp_loc), self._bfs_distance(current_loc, sp_loc))
                    d2 = dist_cache.get((sp_loc, n_loc), self._bfs_distance(sp_loc, n_loc))
                    if d1 >= INF or d2 >= INF:
                        continue
                    # pick cost is 0 if the spanner is already carried by the acting agent
                    pick_cost = 0 if carried and sp_loc == current_loc else 1
                    cost = d1 + d2 + pick_cost
                    if cost < best_cost:
                        best_cost = cost
                        best = (sp_loc, n_loc)
                        best_sp = sp
                        best_nu = nu
                        best_pick_cost = pick_cost

            if best is None:
                # no reachable pair found
                return INF

            # Add movement cost + pick + tighten cost (tighten always costs 1)
            total_cost += best_cost + 1  # +1 for the tighten action

            # Update state of the heuristic plan:
            # - consume the selected spanner (cannot be used again)
            # - remove nut from remaining
            # - agent ends up at the nut location
            available_spanners.pop(best_sp, None)
            remaining_nuts.remove(best_nu)
            current_loc = nut_loc[best_nu]

            # If there are no spanners left but nuts remain, impossible within model
            if not available_spanners and remaining_nuts:
                return INF

            # Ensure distances to new current_loc are in cache
            if current_loc not in relevant_locations:
                # compute distances from current_loc to other relevant locations
                for r in rel_list:
                    dist_cache[(current_loc, r)] = self._bfs_distance(current_loc, r)
                    dist_cache[(r, current_loc)] = self._bfs_distance(r, current_loc)
                relevant_locations.add(current_loc)
                rel_list.append(current_loc)

        return int(total_cost)