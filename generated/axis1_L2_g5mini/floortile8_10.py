from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (hi4l5zmb). The domain's goals
    are facts of the form (cb6fw2xd <loc> <item>). Producing such a goal fact
    requires:
      - some agent/actor of type k9b7lknr (predicate ypt9mrgu indicates agent location)
      - the agent to have a mapping qp2t6u4s agent <item> (can be changed with one action
        if a target <item> is available as fezojtj3)
      - the agent to be located at an adjacent position <stq> such that a static relation
        of type ga4l9do0 or s4jrpk2b holds between the goal location <loc> and that <stq>
      - the goal location <loc> must be "free" (fguw2142 <loc>) before the final operator
        that creates (cb6fw2xd ...) is applied.

    This heuristic estimates the (small) number of actions needed to produce each missing
    cb6fw2xd goal fact. It is not admissible but is designed to provide an informative
    and cheaply-computable estimate.

    Assumptions
    -----------
    - Movement of agents (ypt9mrgu) occurs along adjacency edges determined by any of the
      static relations: vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0. For heuristic purposes we
      treat these relations as undirected adjacency.
    - Changing an agent's qp2t6u4s mapping to a desired qekv4465 value can be done in one
      action (wvswfzmn) provided the desired qekv4465 appears in a static fezojtj3 fact
      and the agent currently has some qp2t6u4s value.
    - Making a location free (fguw2142 ...) is approximated by requiring one extra action
      if the location is currently not free. This is a coarse approximation.
    - The final operator that creates cb6fw2xd is counted as one action.
    - Agents and their current positions and qp2t6u4s mappings are taken from the current
      (dynamic) state. Static reachability information is extracted from task.static.

    Heuristic Initialization
    ------------------------
    In the constructor we extract the following static structures from the planning task:
    - self.goals: set of goal fact strings (for quick goal-reached check).
    - self.goal_pairs: list of parsed (loc, item) tuples for all cb6fw2xd goals.
    - self.adjacency: adjacency map (undirected) between nh1sy8v9 locations. Any static
      fact with predicate in {vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0} yields an edge.
    - self.allowed_target_neighbors: for each goal location `loc` the set of candidate
      stq positions (those `stq` for which ga4l9do0 loc stq or s4jrpk2b loc stq is true).
    - self.fezo_items: set of qekv4465 constants that appear under fezojtj3 in static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each missing goal (cb6fw2xd loc item) we compute the minimal estimated cost to
    achieve that goal (independently of other goals) as follows:

    1. If (cb6fw2xd loc item) is already present in the node's state, cost = 0.

    2. Identify candidate neighbor positions stq for `loc` by consulting the static
       relations ga4l9do0 and s4jrpk2b. If none exist, we treat the goal as currently
       unreachable and assign a large finite cost.

    3. For each agent `en` known in the current state (we detect agents by scanning
       current ypt9mrgu and qp2t6u4s facts), compute:
       a) qp_cost: 0 if the agent already has qp2t6u4s en item in the state;
          otherwise 1 if the desired `item` exists among static fezojtj3 (meaning a
          single wvswfzmn can change an agent's qp2t6u4s to the target), otherwise
          treat as impossible for this agent.
       b) move_cost: shortest-path distance (in hops) from the agent's current location
          to any candidate stq, using the adjacency graph. If unreachable, this agent
          cannot serve the goal (skip).
       c) free_cost: 0 if fguw2142 loc is true in the current state, else 1 (approx).
       d) final_cost: 1 for the final operator (kkqx0eg7 or wbsgr2yj) that produces
          (cb6fw2xd loc item).

       Candidate total = qp_cost + move_cost + free_cost + final_cost.

       We take the minimum candidate total over all agents `en`.

    4. If no agent yields a finite candidate total, we assign a large fallback cost
       for that particular goal (so heuristic remains finite).

    5. The heuristic value returned for the state is the sum of the per-goal minima.
       Finally, if the planner state is not a goal state but the computed heuristic is 0
       (rare due to approximations), we return 1 to ensure that only true goal states
       yield heuristic 0.

    Notes on Complexity and Guidance
    --------------------------------
    - The heuristic does simple BFS shortest-path computations on a static graph and
      inspects a few facts from the current state. It is therefore efficient.
    - The heuristic is deliberately optimistic in a number of respects (e.g., it
      counts only one action to free a location), which tends to make it informative
      for greedy best-first search (it tries to estimate necessary steps without
      overcounting intricate interactions between different goals).
    """

    def __init__(self, task):
        # Goals (set of goal fact strings)
        self.goals = task.goals

        # Parse goal pairs (loc, item) from cb6fw2xd goals
        self.goal_pairs = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "cb6fw2xd" and len(parts) == 3:
                loc, item = parts[1], parts[2]
                self.goal_pairs.append((loc, item))

        # Static facts
        static_facts = task.static

        # Build undirected adjacency between nh1sy8v9 location symbols
        self.adjacency = {}
        def add_edge(a, b):
            self.adjacency.setdefault(a, set()).add(b)
            self.adjacency.setdefault(b, set()).add(a)

        # Candidate neighbors for a goal location `loc` are stq with ga4l9do0 loc stq or s4jrpk2b loc stq
        self.allowed_target_neighbors = {}

        # Items available via fezojtj3 (qekv4465 constants)
        self.fezo_items = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in ("vjcqgdgn", "s4jrpk2b", "vom4pfig", "ga4l9do0"):
                # expecting binary relation between nh1sy8v9 nodes
                if len(parts) >= 3:
                    a, b = parts[1], parts[2]
                    add_edge(a, b)
                    # If this is ga4l9do0 or s4jrpk2b we may add neighbor relation for allowed targets
                    if pred == "ga4l9do0":
                        # ga4l9do0 ?stq0j ?nb9v4 in domain file; but in static we saw ga4l9do0 <stq> <nb>
                        # To be robust, we accept relation in either order and store candidate stq for nb
                        # store both interpretations: treat parts[1] as potential stq and parts[2] as potential loc
                        stq, nb = parts[1], parts[2]
                        self.allowed_target_neighbors.setdefault(nb, set()).add(stq)
                    if pred == "s4jrpk2b":
                        stq, nb = parts[1], parts[2]
                        self.allowed_target_neighbors.setdefault(nb, set()).add(stq)
            elif pred == "fezojtj3" and len(parts) >= 2:
                self.fezo_items.add(parts[1])

        # Ensure every goal loc has an entry (possibly empty)
        for loc, _ in self.goal_pairs:
            self.allowed_target_neighbors.setdefault(loc, set())

        # A large finite cost used when something looks unreachable
        self._BIG_COST = 1000

    def _bfs_distance_to_any(self, start, targets):
        """
        BFS on the static adjacency graph to compute the shortest distance (in hops)
        from start node to any node in the `targets` set. Returns integer distance or
        None if unreachable.
        """
        if start in targets:
            return 0
        if start not in self.adjacency:
            return None
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adjacency.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in targets:
                    return dist + 1
                visited.add(nbr)
                queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If the node already satisfies all goals, heuristic is 0
        if self.goals <= state:
            return 0

        # Parse dynamic facts of interest from the current state
        # - agent locations: ypt9mrgu en loc
        # - agent qp mappings: qp2t6u4s en item
        # - which locations are free: fguw2142 loc
        agent_location = {}   # en -> loc
        agent_qp = {}         # en -> item
        free_locations = set()
        cb6_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "ypt9mrgu" and len(parts) >= 3:
                en, loc = parts[1], parts[2]
                agent_location[en] = loc
            elif pred == "qp2t6u4s" and len(parts) >= 3:
                en, item = parts[1], parts[2]
                agent_qp[en] = item
            elif pred == "fguw2142" and len(parts) >= 2:
                free_locations.add(parts[1])
            elif pred == "cb6fw2xd" and len(parts) >= 3:
                # already achieved goal
                cb6_in_state.add((parts[1], parts[2]))

        total = 0

        # Gather all agents we can consider: those that appear in location or qp facts
        agents = set(agent_location.keys()) | set(agent_qp.keys())

        # If there are no agents, we cannot make progress; return a large cost
        if not agents:
            # but produce finite heuristic
            return len([1 for (loc, item) in self.goal_pairs if (loc, item) not in cb6_in_state]) * self._BIG_COST

        for loc, item in self.goal_pairs:
            if (loc, item) in cb6_in_state:
                continue  # already satisfied

            # Candidate neighbors (stq) that allow creation of cb6fw2xd at `loc`
            candidate_stqs = self.allowed_target_neighbors.get(loc, set())
            if not candidate_stqs:
                # No static neighbor allows the creation; mark large cost
                total += self._BIG_COST
                continue

            best_for_goal = None

            for en in agents:
                # qp cost
                if agent_qp.get(en) == item:
                    qp_cost = 0
                else:
                    # If the desired item is among fezo items, we assume one action suffices to set agent's qp
                    if item in self.fezo_items and en in agent_qp:
                        qp_cost = 1
                    elif item in self.fezo_items and agents:
                        # We could change some agent's qp and then use that agent; still costs 1
                        qp_cost = 1
                    else:
                        # cannot set qp to desired item for this agent
                        continue

                # move cost: distance from agent_location[en] to any candidate stq
                en_loc = agent_location.get(en)
                if en_loc is None:
                    # agent has no known location in state; skip this agent
                    continue
                dist = self._bfs_distance_to_any(en_loc, candidate_stqs)
                if dist is None:
                    # unreachable
                    continue
                move_cost = dist  # one movement action per hop

                # free cost: if loc is free already, zero; else approximate one action needed to free it
                free_cost = 0 if loc in free_locations else 1

                # final operator cost to create cb6fw2xd
                final_cost = 1

                candidate_cost = qp_cost + move_cost + free_cost + final_cost

                if best_for_goal is None or candidate_cost < best_for_goal:
                    best_for_goal = candidate_cost

            if best_for_goal is None:
                # No agent could serve this goal under the approximations
                total += self._BIG_COST
            else:
                total += best_for_goal

        # Ensure heuristic is > 0 for non-goal states
        if total == 0:
            return 1
        return total