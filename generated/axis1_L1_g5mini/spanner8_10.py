from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Match a fact against a simple pattern (supports exact matching only here).
    pattern elements must match exactly the corresponding fact parts.
    """
    parts = get_parts(fact)
    if len(parts) < len(pattern):
        return False
    return all(pat == "*" or pat == part for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the secret_domain (e2zu8y93). The domain models a single agent
    (a person/tool-user) who can move along directed location edges, pick up
    spanners, and use a spanner once on a nut to make the nut reach the goal
    predicate (g59iwyxf). Each spanner that currently has predicate `g0brvc06`
    can be used exactly once to fix one nut. Movement between adjacent locations
    is represented by the predicate kxokb5hx and the action n9sbfmrs; picking up
    a spanner is represented by the action moi6wnfu; using a spanner on a nut is
    represented by the action kjsqfni8.

    The heuristic estimates the number of actions still required to achieve all
    goal facts of the form (g59iwyxf <nut>), by greedily assigning available
    usable spanners to remaining nuts and summing:
      - movement actions (shortest-path length along kxokb5hx edges),
      - one pick action (moi6wnfu) when the spanner is not already held by the agent,
      - one use action (kjsqfni8) to fix the nut.

    Assumptions
    -----------
    - There is (practically) a single agent (person) performing actions. If
      multiple agent-like objects are present the heuristic picks one that is
      present in u4srx1nb facts and not classified as a spanner or nut.
    - Spanners that have the predicate (g0brvc06 <s>) are usable exactly once.
    - A spanner that is currently held by the agent is represented by the
      predicate (yud7t3vb <agent> <spanner>) and does not have a u4srx1nb fact;
      its effective location equals the agent's current location.
    - Movement edges (kxokb5hx) are directed and each traversal costs 1 action.
    - Picking up a spanner costs 1 action (moi6wnfu). Using a spanner (kjsqfni8)
      costs 1 action.
    - If no usable spanners remain or a nut is unreachable, the heuristic returns
      a large finite value (indicating difficult / unsolvable from this state).

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
      - The set of goal facts (self.goals) to detect goal states quickly.
      - The directed adjacency graph between locations from static facts of the
        form (kxokb5hx locA locB).
    No other static information is required for the greedy assignment.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goal facts are satisfied in the current state, return 0.
    2. Parse the current state to determine:
       - Which nuts are already fixed (have g59iwyxf) and which remain to fix.
       - The current location of each nut via (u4srx1nb <nut> <loc>).
       - Which spanners are still usable (have g0brvc06).
       - The location of each spanner via (u4srx1nb <spanner> <loc>) or if the
         spanner is held by the agent via (yud7t3vb <agent> <spanner>).
       - The agent's current location via (u4srx1nb <agent> <loc>) (the heuristic
         chooses an object that is neither a usable spanner nor a known nut).
    3. Compute shortest-path distances between locations using BFS on the
       directed adjacency graph (each edge has cost 1, representing one move).
    4. Greedily fix nuts one by one:
       - At each step, pick the pair (usable spanner, remaining nut) that
         minimizes: dist(agent_loc, spanner_loc) + pick_cost + dist(spanner_loc, nut_loc) + use_cost,
         where pick_cost is 1 if the spanner is not already held by the agent,
         otherwise 0; use_cost is always 1.
       - Add that cost to the total, mark the nut fixed and the spanner used,
         and set agent_loc = nut_loc (the agent ends up at the nut after use).
    5. If any remaining nut cannot be matched to any usable spanner nor reached
       via the adjacency graph, return a large finite heuristic (e.g., 9999).
    6. Return the summed cost as the heuristic estimate.
    """

    def __init__(self, task):
        # store goals (set of goal facts)
        self.goals = task.goals
        # build directed adjacency from static facts: (kxokb5hx A B)
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "kxokb5hx" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, []).append(b)
                # Ensure nodes exist even if no outgoing edges
                self.adj.setdefault(b, self.adj.get(b, []))

    def _bfs_dist(self, start):
        """
        Compute shortest distances from start location to all reachable locations.
        Returns a dict: location -> distance (int). If start is unknown, returns {}.
        """
        if start not in self.adj:
            return {}
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # 1) If goal satisfied, heuristic = 0
        if self.goals <= state:
            return 0

        # Parse state to extract information
        # Nuts: those that appear in g59iwyxf (done) or bu1om0m1 (still intact)
        nuts_done = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == "g59iwyxf"}
        nuts_all = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] in ("g59iwyxf", "bu1om0m1")}
        # Nuts needing fix = those in nuts_all but not in nuts_done
        nuts_to_fix = {n for n in nuts_all if n not in nuts_done}

        # Locations of nuts (u4srx1nb <obj> <loc>)
        obj_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "u4srx1nb" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                obj_locations[obj] = loc

        # Spanners that are still usable: predicate g0brvc06 <spanner>
        usable_spanners = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == "g0brvc06"}

        # Determine which spanners are currently held by which agent: yud7t3vb <agent> <spanner>
        held_by = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "yud7t3vb" and len(parts) >= 3:
                agent, spanner = parts[1], parts[2]
                held_by[spanner] = agent

        # Determine candidate agent(s): objects that have u4srx1nb but are not classified as usable spanners or nuts.
        candidate_agents = []
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "u4srx1nb" and len(parts) >= 3:
                obj = parts[1]
                if obj not in usable_spanners and obj not in nuts_all:
                    candidate_agents.append(obj)
        # If none found, but there is an object that appears as first arg in yud7t3vb, use that
        if not candidate_agents:
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "yud7t3vb":
                    candidate_agents.append(parts[1])
        # If still none, fallback: pick any object that has u4srx1nb
        if not candidate_agents:
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "u4srx1nb":
                    candidate_agents.append(parts[1])
        if not candidate_agents:
            # No agent found: this is abnormal, return large estimate
            return 9999
        # Choose the first candidate as the agent we model
        agent = candidate_agents[0]
        agent_loc = obj_locations.get(agent, None)
        if agent_loc is None:
            # Agent has no explicit location (abnormal), return large estimate
            return 9999

        # Build spanner locations: if held_by, location equals holder's location
        spanner_locations = {}
        for sp in list(usable_spanners):
            if sp in obj_locations:
                spanner_locations[sp] = obj_locations[sp]
            elif sp in held_by:
                holder = held_by[sp]
                holder_loc = obj_locations.get(holder)
                if holder_loc is not None:
                    spanner_locations[sp] = holder_loc
                else:
                    # holder has no location? treat as unknown (skip)
                    spanner_locations[sp] = None
            else:
                # spanner neither at a location nor held -> treat as unavailable
                spanner_locations[sp] = None

        # Nut locations
        nut_locations = {}
        for n in nuts_to_fix:
            # If nut location present
            if n in obj_locations:
                nut_locations[n] = obj_locations[n]
            else:
                nut_locations[n] = None

        # Quick infeasible checks
        if not nuts_to_fix:
            return 0
        if not usable_spanners:
            # No usable spanners left to fix nuts
            return 9999

        # Precompute BFS distances from any location encountered on demand
        dist_cache = {}

        def distance(a, b):
            """Return shortest directed distance between locations a->b or None if unreachable."""
            if a is None or b is None:
                return None
            if a == b:
                return 0
            if a not in dist_cache:
                dist_cache[a] = self._bfs_dist(a)
            return dist_cache[a].get(b)

        # Greedy assignment: at each step pick (spanner, nut) minimizing incremental cost
        remaining_spanners = set(sp for sp in usable_spanners if spanner_locations.get(sp) is not None)
        remaining_nuts = set(nuts_to_fix)
        total_cost = 0
        current_loc = agent_loc

        PICK_COST = 1  # moi6wnfu
        USE_COST = 1   # kjsqfni8

        while remaining_nuts:
            best = None  # tuple (cost, spanner, nut, sp_loc, nut_loc, pick_needed)
            for sp in list(remaining_spanners):
                sp_loc = spanner_locations.get(sp)
                if sp_loc is None:
                    continue
                for nut in list(remaining_nuts):
                    nut_loc = nut_locations.get(nut)
                    if nut_loc is None:
                        continue
                    # dist from current agent location to spanner
                    dist1 = distance(current_loc, sp_loc)
                    # dist from spanner to nut
                    dist2 = distance(sp_loc, nut_loc)
                    if dist1 is None or dist2 is None:
                        continue
                    pick_needed = 0
                    # If spanner is already held by agent (held_by[sp] == agent), no pick needed.
                    if held_by.get(sp) == agent:
                        pick_needed = 0
                        # In that case dist1 must be 0 (agent and spanner at same loc); allow it anyway.
                    else:
                        pick_needed = PICK_COST
                    cost = dist1 + pick_needed + dist2 + USE_COST
                    if best is None or cost < best[0]:
                        best = (cost, sp, nut, sp_loc, nut_loc, pick_needed)
            if best is None:
                # No feasible pair found for remaining nuts -> unreachable or no spanner connectivity
                return 9999
            cost, sp_chosen, nut_chosen, sp_loc, nut_loc, pick_needed = best
            total_cost += cost
            # update state for greedy planning: agent ends up at nut location, spanner consumed
            current_loc = nut_loc
            remaining_spanners.remove(sp_chosen)
            remaining_nuts.remove(nut_chosen)
            # If any spanners were located at the same location as the chosen spanner and would be
            # considered moved due to agent movement we don't model that; greedy approximation.

        return int(total_cost)