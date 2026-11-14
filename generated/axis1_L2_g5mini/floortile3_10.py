from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic tailored for the secret_domain (hi4l5zmb). The primary goal in
    the provided instances is to produce facts of the form
    (cb6fw2xd <node> <target>). Those facts are created by actions that
    require:
      - an 'en' object with a qp2t6u4s mapping to the desired <target>,
      - the 'en' to be physically (ypt9mrgu) at a specific neighboring site
        <stq> of the goal node <node> (the static relations ga4l9do0 or
        s4jrpk2b relate <node> and <stq>),
      - the goal node <node> to currently have the fluent fguw2142 (a "free"
        marker).
    The heuristic estimates the number of actions required to obtain each
    missing cb6fw2xd goal by summing:
      - the minimal cost to ensure an en has the required qp2t6u4s mapping
        (0 if already mapped, 1 if an existing en can switch mapping using
        the wvswfzmn action),
      - the minimal number of move-actions to bring such an en to a suitable
        neighboring site <stq> (BFS distance on a graph built from the static
        movement relations),
      - one final action to create the cb6fw2xd fact (kkqx0eg7 or wbsgr2yj),
      - plus a small extra cost if the goal node currently lacks fguw2142
        (optimistically counted as +1 to represent arranging a free marker).
    The heuristic is intentionally optimistic (non-admissible) but fast to
    compute and informative for guiding greedy best-first search.

    Assumptions
    -----------
    - The only way to create (cb6fw2xd NB TLF) in the domain is via operators
      that require:
        * some en with (qp2t6u4s en TLF),
        * en located at some STQ that satisfies a static neighbor relation to NB
          (we consider ga4l9do0 and s4jrpk2b as valid relations for this purpose),
        * (fguw2142 NB) true at the time of applying the final operator.
    - An en's mapping (qp2t6u4s) can be changed in one action (wvswfzmn)
      provided the action's static preconditions (fezojtj3 on the destination
      mapping) are available. We detect feasible remappings by checking static
      fezojtj3 facts.
    - Movement of en objects can be approximated by shortest paths on an
      undirected graph built from the static predicates
      {s4jrpk2b, vjcqgdgn, vom4pfig, ga4l9do0}. Each edge is assumed to be
      traversable by one move action (these correspond to the domain's swap
      movement operators). This is an optimistic abstraction.
    - We do not simulate exact consumption of fguw2142 tokens or mappings when
      multiple goals compete for the same resources. The heuristic is therefore
      admissibly optimistic (it may underestimate the true cost) but remains
      informative and efficiently computable.

    Heuristic Initialization
    ------------------------
    - Extract all cb6fw2xd goals from the task goals and store them.
    - Build an undirected adjacency graph among nh1sy8v9 nodes from static
      facts with predicates s4jrpk2b, vjcqgdgn, vom4pfig, ga4l9do0.
      This graph is used to compute shortest path lengths (movement costs).
    - Build a mapping of "acceptable stq neighbors" for each potential goal
      node NB from static ga4l9do0 and s4jrpk2b facts. These are the STQ
      positions where an en must be located to enable the final cb6fw2xd
      producing action for node NB.
    - Remember which tlfqu names are valid remapping targets from static
      fezojtj3 facts (those are possible destinations of the wvswfzmn action).

    Step-By-Step Thinking for Computing Heuristic (for a given state)
    -----------------------------------------------------------------
    1. If all goals are already satisfied in the state, return 0.
    2. Parse the current state to extract:
       - qp2_map: en -> current tlf (from facts qp2t6u4s en tlf),
       - en_pos: en -> node where ypt9mrgu en node holds (en's position),
       - free_nodes: set of nodes that currently hold (fguw2142 node).
    3. For each missing goal (cb6fw2xd NB TLF):
       a) If (cb6fw2xd NB TLF) already in the state, contribute 0.
       b) Determine mapping_cost:
          - 0 if some en already has qp2t6u4s en TLF,
          - 1 if there exists an en with any qp2t6u4s mapping (it can be
            remapped in one wvswfzmn action to TLF provided TLF is a valid
            fezojtj3 static target),
          - a large finite penalty if no en with a qp2t6u4s mapping exists or
            TLF is not a valid remapping target (heuristic remains finite but
            large).
       c) Determine movement_cost:
          - Consider all candidate en that can produce the mapping (either
            already mapped to TLF or mappable).
          - For each such en, compute the BFS distance from en_pos[en] to any
            STQ in the precomputed acceptable neighbor set for NB. The BFS is
            performed on the undirected adjacency graph from initialization.
          - Use the minimum such distance across candidate ens. If unreachable,
            treat as large cost.
       d) Determine free_node_cost:
          - If NB is currently in free_nodes, 0 else +1 (optimistic cost to
            make NB free).
       e) The estimated cost for this particular missing goal is:
            mapping_cost + movement_cost + 1 (final cb6 action) + free_node_cost
          where movement_cost of 0 is allowed when en is already at a valid STQ.
    4. Sum the estimates for all missing cb6 goals and return the sum. If any
       sub-goal is detected unreachable under the optimistic assumptions, the
       heuristic will return a large finite number instead of infinity.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Predicates that allow movement between nh1sy8v9 nodes. We treat them
        # as undirected edges for a fast optimistic shortest-path calculation.
        movement_predicates = {"s4jrpk2b", "vjcqgdgn", "vom4pfig", "ga4l9do0"}

        # Predicates that define acceptable (NB, STQ) pairs for producing cb6fw2xd.
        neighbor_predicates_for_cb6 = {"ga4l9do0", "s4jrpk2b"}

        # Valid remapping destination tokens (fezojtj3 ZUB86).
        self.valid_tlf_targets = set()

        # Build adjacency graph among nh1sy8v9 nodes.
        self.adj = {}  # node -> set(neighbor nodes)
        # For each NB, which STQ nodes are acceptable as en positions for final op.
        self.acceptable_stq_for_nb = {}  # nb -> set(stq)

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in movement_predicates and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # Undirected edge for optimistic reachability.
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)
            if pred in neighbor_predicates_for_cb6 and len(parts) >= 3:
                nb, stq = parts[1], parts[2]
                self.acceptable_stq_for_nb.setdefault(nb, set()).add(stq)
            if pred == "fezojtj3" and len(parts) >= 2:
                self.valid_tlf_targets.add(parts[1])

        # Extract cb6fw2xd goals explicitly (list of (nb, tlf) pairs).
        self.cb6_goals = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "cb6fw2xd" and len(parts) >= 3:
                self.cb6_goals.append((parts[1], parts[2]))

        # A finite large cost used when something is (optimistically) unavailable.
        self.large_cost = 1000

    def _bfs_distance_to_any(self, start, targets):
        """
        Breadth-first search on the undirected adjacency graph to find the
        minimal number of move-steps from start to any node in targets.
        Returns:
          - integer distance >= 0 if reachable,
          - None if unreachable or start/targets invalid.
        """
        if start is None or targets is None or not targets:
            return None
        if start in targets:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in targets:
                    return dist + 1
                visited.add(nbr)
                queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # 1) If all goals satisfied -> 0
        if self.goals <= state:
            return 0

        # Parse relevant fluents from the current state:
        qp2_map = {}      # en -> tlf (qp2t6u4s en tlf)
        en_pos = {}       # en -> node (ypt9mrgu en node)
        free_nodes = set()  # nodes with fguw2142

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "qp2t6u4s" and len(parts) >= 3:
                en, tlf = parts[1], parts[2]
                qp2_map[en] = tlf
            elif pred == "ypt9mrgu" and len(parts) >= 3:
                en, nodepos = parts[1], parts[2]
                en_pos[en] = nodepos
            elif pred == "fguw2142" and len(parts) >= 2:
                free_nodes.add(parts[1])

        total_estimate = 0

        # For each cb6 goal, compute optimistic cost to achieve it.
        for nb, tlf in self.cb6_goals:
            goal_fact = f"(cb6fw2xd {nb} {tlf})"
            if goal_fact in state:
                continue  # already satisfied

            # 1) Mapping cost: 0 if some en already mapped to tlf; 1 if some en can be remapped.
            mapping_cost = self.large_cost
            candidate_ens = set()

            # If any en currently has desired mapping, prefer that en(s).
            for en, current_tlf in qp2_map.items():
                if current_tlf == tlf:
                    mapping_cost = 0
                    candidate_ens.add(en)
            # If none currently mapped, see if remapping is possible in one action.
            if mapping_cost != 0:
                # Need an en with any qp2 mapping and the tlf being a valid fezojtj3 target.
                if self.valid_tlf_targets and (tlf in self.valid_tlf_targets):
                    if qp2_map:
                        mapping_cost = 1
                        candidate_ens.update(qp2_map.keys())
                else:
                    # If tlf is not a valid remap target and no en already maps to it,
                    # then we mark the subgoal as expensive (but finite).
                    mapping_cost = self.large_cost

            # 2) Movement cost: find minimal BFS distance from any candidate en position
            # to any acceptable STQ neighbor of NB.
            movement_cost = None
            targets = self.acceptable_stq_for_nb.get(nb, set())
            if not targets:
                # No acceptable STQ for this NB according to static facts -> unreachable
                movement_cost = None
            else:
                # For each candidate en, compute BFS distance.
                best = None
                for en in candidate_ens:
                    start = en_pos.get(en)
                    if start is None:
                        # If en has no known position, skip it.
                        continue
                    dist = self._bfs_distance_to_any(start, targets)
                    if dist is None:
                        continue
                    if best is None or dist < best:
                        best = dist
                movement_cost = best

            # If no candidate en was usable/found or movement unreachable, set movement to large.
            if movement_cost is None:
                movement_cost_val = self.large_cost
            else:
                movement_cost_val = movement_cost

            # 3) Free node cost: final operator requires fguw2142 on NB.
            free_cost = 0 if nb in free_nodes else 1

            # 4) Final operator cost: 1
            final_op_cost = 1

            # Sum for this goal (optimistic).
            this_goal_cost = mapping_cost + movement_cost_val + free_cost + final_op_cost

            # If any component was flagged as large_cost, this_goal_cost will be large.
            total_estimate += this_goal_cost

            # Quick exit if estimate becomes very large to avoid overflow.
            if total_estimate >= self.large_cost:
                return self.large_cost

        return int(total_estimate)