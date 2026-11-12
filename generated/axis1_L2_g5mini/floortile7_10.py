from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string without the surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic tailored to the secret_domain (hi4l5zmb).
    The main goal in this domain is to produce cb6fw2xd facts of the form
    (cb6fw2xd <nh-node> <qekv-resource>) by using agents of type k9b7lknr
    (predicate ypt9mrgu) that must be (a) located at an nh1sy8v9 node which
    is connected to the target nh1sy8v9 node via specific static relations
    (ga4l9do0 or s4jrpk2b), and (b) have their qp2t6u4s resource set to the
    target qekv value. This heuristic estimates the number of actions
    required to reach all required cb6fw2xd goals.

    Assumptions
    -----------
    - Static relation facts (vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0) define a
      connectivity graph over nh1sy8v9 nodes. A movement action can move an
      agent from node X to node Y in one step if there is a static relation
      (REL Y X) (this mirrors the operators' preconditions in the domain).
    - Any agent (k9b7lknr) that already has some qp2t6u4s binding can change
      its qp2t6u4s value to a target qekv in at most one action (wvswfzmn),
      provided the target qekv is available as a fezojtj3 resource (these are
      extracted from static facts). If an agent has no qp2t6u4s binding at
      all in the current state, we conservatively assume it can be assigned a
      qp2 value in two actions (finite but more expensive).
    - We ignore resource contention (e.g., fguw2142 tokens) and subtle
      interactions between simultaneous goals. The heuristic is therefore
      optimistic and not necessarily admissible, but tends to be informative.
    - Only the operators that add cb6fw2xd are considered as goal-achieving:
      kkqx0eg7 and wbsgr2yj. Both require the agent to be at a neighbor stq
      such that either ga4l9do0 or s4jrpk2b (respectively) holds for (nb stq).
      The heuristic therefore looks for such stq nodes.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - self.goals: the goal facts (task.goals).
    - self.neighbor_graph: a mapping node -> set(neighbors) where a single
      movement step can move an agent from node to any neighbor. This graph
      is built from all four relation predicates (vjcqgdgn, s4jrpk2b,
      vom4pfig, ga4l9do0) found among static facts. For a static fact
      (REL A B) we add an edge B -> A (matching domain operator directions).
    - self.goal_connection_map: mapping target_nb -> set(stq) capturing the
      particular stq nodes that can be used to achieve cb6fw2xd for that
      target_nb. This is built from static facts of type ga4l9do0 and
      s4jrpk2b (the two relation types used by the cb6fw2xd-producing ops).
    - self.fezo_resources: set of qekv values that are available via
      (fezojtj3 ...) static facts (these determine valid qp2 targets).
    """

    def __init__(self, task):
        # Goals and static facts.
        self.goals = task.goals
        static_facts = task.static

        # neighbor_graph[u] contains v if an agent at u can move to v in one operator step.
        # We add an edge B -> A for every static fact (REL A B) with REL in the four movement relations.
        self.neighbor_graph = defaultdict(set)
        # For computing required precondition nodes for cb6fw2xd, we collect nodes stq
        # such that (ga4l9do0 nb stq) or (s4jrpk2b nb stq) holds.
        self.goal_connection_map = defaultdict(set)
        # fezojtj3 resources (valid qp2 targets).
        self.fezo_resources = set()

        movement_rels = {"vjcqgdgn", "s4jrpk2b", "vom4pfig", "ga4l9do0"}

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in movement_rels and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # static (REL A B) allows a move from B to A (one operator).
                self.neighbor_graph[b].add(a)
            if pred == "fezojtj3" and len(parts) >= 2:
                self.fezo_resources.add(parts[1])
            # Collect nodes that can be used to satisfy cb6fw2xd preconditions:
            if pred == "ga4l9do0" and len(parts) >= 3:
                nb, stq = parts[1], parts[2]
                self.goal_connection_map[nb].add(stq)
            if pred == "s4jrpk2b" and len(parts) >= 3:
                nb, stq = parts[1], parts[2]
                self.goal_connection_map[nb].add(stq)

    def _bfs_distance_to_any(self, start, targets):
        """
        Compute shortest path length (in number of moves) from start node to
        any node in targets using the self.neighbor_graph. Returns None if no
        path exists.
        """
        if start in targets:
            return 0
        if not targets:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            u, d = q.popleft()
            for v in self.neighbor_graph.get(u, ()):
                if v in visited:
                    continue
                if v in targets:
                    return d + 1
                visited.add(v)
                q.append((v, d + 1))
        return None

    def __call__(self, node):
        """
        For every goal of the form (cb6fw2xd NB TLF), estimate the minimal
        number of actions needed to produce that fact and sum these estimates.

        Step-By-Step Thinking for Computing Heuristic
        --------------------------------------------
        1. If the entire goal set is already satisfied in the node.state, return 0.
        2. Parse the current state to extract:
           - agent locations: from (ypt9mrgu agent loc)
           - agent qp2 bindings: from (qp2t6u4s agent tlf)
           - present cb6fw2xd facts (to ignore already achieved goals)
        3. For each yet-unachieved cb6fw2xd goal (nb, tlf):
           - Determine possible 'stq' nodes that can be used for creating the cb6fw2xd,
             i.e., nodes stq where either (ga4l9do0 nb stq) or (s4jrpk2b nb stq) is true.
           - For each agent 'en' in the current state:
               a) qp_cost:
                  - 0 if agent already has qp2t6u4s en tlf
                  - 1 if agent has some qp2t6u4s but not the target (we can swap via wvswfzmn)
                    and the target tlf is available in static as fezojtj3
                  - 2 otherwise (conservative finite penalty for assigning a qp2)
               b) move_cost: BFS distance from en's current location to any stq in the
                  chosen stq set (each step corresponds to a movement operator).
               c) Total candidate cost = qp_cost + move_cost + 1 (the final op that creates cb6fw2xd).
           - The heuristic contribution for the goal is the minimum candidate cost over agents.
           - If no agent can reach a required stq, assign a large finite penalty for that goal.
        4. Sum contributions across all goals and return the sum as heuristic estimate.

        Notes
        -----
        - We treat movement along the graph as a simple shortest-path problem using
          the static connectivity. We deliberately ignore token constraints (fguw2142)
          and possible conflicts between goals, producing an optimistic but fast
          estimate suitable for greedy best-first search guidance.
        """
        state = node.state

        # 1) Quick check for solved state.
        if self.goals <= state:
            return 0

        # 2) Parse state for useful information.
        agent_location = {}  # agent -> nh-node (from ypt9mrgu)
        agent_qp = defaultdict(set)  # agent -> set of qp2 values
        achieved_cb = set()  # cb6fw2xd facts already present

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "ypt9mrgu" and len(parts) >= 3:
                en, loc = parts[1], parts[2]
                agent_location[en] = loc
            elif pred == "qp2t6u4s" and len(parts) >= 3:
                en, tlf = parts[1], parts[2]
                agent_qp[en].add(tlf)
            elif pred == "cb6fw2xd" and len(parts) >= 3:
                achieved_cb.add((parts[1], parts[2]))

        total_cost = 0
        LARGE_PENALTY = 100  # finite but large cost for unreachable goals

        # Evaluate each goal cb6fw2xd.
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] != "cb6fw2xd" or len(parts) < 3:
                # ignore non-cb6fw2xd goals (there typically aren't any in this domain),
                # but keep a conservative cost 0 if the goal is not of this pattern.
                continue
            nb, tlf = parts[1], parts[2]
            if (nb, tlf) in achieved_cb:
                continue  # already satisfied

            # Candidate stq nodes that can be used to apply the cb6fw2xd-producing ops.
            stq_targets = self.goal_connection_map.get(nb, set())

            best_for_goal = None

            # If there are no connection nodes known from static facts, we cannot apply
            # the cb6fw2xd operators (conservative large penalty).
            if not stq_targets:
                total_cost += LARGE_PENALTY
                continue

            # Try each agent to find minimal cost to achieve this goal with that agent.
            # If there are no agents recorded in the state, give a large penalty.
            if not agent_location and not agent_qp:
                total_cost += LARGE_PENALTY
                continue

            # Consider all agents known either by location or qp2 bindings.
            all_agents = set(agent_location.keys()) | set(agent_qp.keys())

            for en in all_agents:
                # qp cost
                if tlf in agent_qp.get(en, set()):
                    qp_cost = 0
                elif self.fezo_resources and tlf in self.fezo_resources and agent_qp.get(en):
                    # agent has some qp2 and we can swap it to the target tlf in 1 action
                    qp_cost = 1
                elif self.fezo_resources and tlf in self.fezo_resources:
                    # no qp2 present for agent but target is a valid resource: be conservative (2)
                    qp_cost = 2
                else:
                    # target tlf not available as fezo resource; make this option expensive
                    qp_cost = 3

                # move cost: distance from agent's current location to any allowed stq.
                en_loc = agent_location.get(en)
                if en_loc is None:
                    # agent has no known location; assume a small penalty to discover it
                    move_cost = 2
                else:
                    dist = self._bfs_distance_to_any(en_loc, stq_targets)
                    if dist is None:
                        # unreachable by movement via static edges
                        move_cost = None
                    else:
                        move_cost = dist

                if move_cost is None:
                    continue  # this agent cannot reach required stq set
                candidate_cost = qp_cost + move_cost + 1  # +1 for the producing operator
                if best_for_goal is None or candidate_cost < best_for_goal:
                    best_for_goal = candidate_cost

            if best_for_goal is None:
                # No agent could reach required stq -> large penalty
                total_cost += LARGE_PENALTY
            else:
                total_cost += best_for_goal

        return int(total_cost)