from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (hi4l5zmb) domain.

    This heuristic estimates the number of actions required to achieve all
    goal facts of the form (cb6fw2xd <location> <resource>), which are the
    typical goals in this domain. The heuristic treats each goal fact
    independently and sums per-goal estimated costs. Each per-goal estimate
    accounts for:
      - positioning an appropriate agent (an object of type k9b7lknr) at a
        neighboring source location (stq0j) from which the "produce" action
        (kkqx0eg7 or wbsgr2yj) can be applied,
      - ensuring the agent has the correct resource mapping (qp2t6u4s),
      - ensuring the target location has the marker (fguw2142),
      - the final production action itself.

    Assumptions
    -----------
    - Goal facts that matter are of the form (cb6fw2xd <loc> <tlfqu>).
      Other goal predicates are ignored by this heuristic (there are none in
      the provided instances).
    - Static adjacency relations (vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0) are
      used to build a directed movement graph. An adjacency fact
      (pred A B) means there exists a directed edge B -> A for movement.
      (This follows the domain actions where movement preconditions are of
      the form (pred target current).)
    - Agents (k9b7lknr) are located using ypt9mrgu facts in the state and can
      be moved along the adjacency graph. We ignore the complex interplay
      with the fguw2142 marker for movement reachability and treat movement
      cost simply as the shortest graph distance to a producer-adjacent node.
      This simplification keeps the heuristic efficient to compute.
    - If an agent does not currently have the desired qp2t6u4s mapping, but
      the target resource has a fezojtj3 fact (available resource), we
      estimate one additional action (wvswfzmn) to change the agent's
      mapping. If no mapping or no fezojtj3 is present, we treat the option
      as infeasible for that agent.
    - If the target location lacks fguw2142, we pessimistically estimate one
      additional action to position the marker there (in practice this may
      require multiple moves but this keeps the heuristic simple and fast).

    Heuristic Initialization
    ------------------------
    During initialization the heuristic extracts and organizes:
      - the set of goal facts (self.goals),
      - static adjacency information (used to build a directed graph),
      - which adjacency relations serve as producer neighbors (ga4l9do0 and
        s4jrpk2b are required by the producing actions),
      - the set of resources that are marked fezojtj3 (available targets
        for wvswfzmn).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node (with current state):
    1. If all goal facts are already true in the state, return 0.
    2. Parse the state to identify:
       - current locations of agents (from ypt9mrgu facts),
       - current resource mapping of agents (from qp2t6u4s facts),
       - which locations presently have the fguw2142 marker,
       - which cb6fw2xd goal facts are already satisfied.
    3. For each unsatisfied goal (cb6fw2xd target_loc resource):
       a. Identify the set of producer-adjacent source nodes stq0j such that a
          producer action can be applied (i.e., static facts of the form
          (ga4l9do0 target_loc stq0j) or (s4jrpk2b target_loc stq0j)).
       b. For each agent (en18p) available in the current state:
          i. Compute the shortest directed distance (number of movement actions)
             from the agent's current location to any of the producer-adjacent
             source nodes using the adjacency graph built from static facts.
          ii. Compute the cost to obtain the required qp2t6u4s mapping:
              - 0 if the agent already maps to the required resource,
              - 1 if the resource has a fezojtj3 static fact and the agent
                currently has some qp2t6u4s mapping (we assume wvswfzmn can
                be applied to change the mapping),
              - infeasible otherwise.
          iii. fguw_cost = 0 if the target location currently has fguw2142,
               else 1 (we assume at least one extra action will be needed).
          iv. Add 1 for the producer action itself (kkqx0eg7 or wbsgr2yj).
          v. Sum movement + qp + fguw + 1 to get candidate cost for this agent.
       c. The per-goal cost is the minimum candidate cost across agents. If no
          agent provides a finite candidate, a large penalty value is used.
    4. Sum the per-goal costs for all unsatisfied goals and return this sum.
    """

    def __init__(self, task):
        # Store goals and static facts.
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Build a directed adjacency graph from the four movement-related
        # adjacency predicates. For a static fact of the form
        #   (pred A B)
        # we add an edge B -> A because movement operators expect
        #   (pred target current) and move the agent from current to target.
        self.adj_graph = {}  # node -> set(neighbor_nodes)
        self.producer_neighbors = {}  # target_loc -> set(stq0j nodes usable for production)
        self.available_tlfqu = set()  # resources with fezojtj3

        # adjacency predicate names that can be used for movement
        movement_preds = {"vjcqgdgn", "s4jrpk2b", "vom4pfig", "ga4l9do0"}
        # predicates that allow using a source node as the producer-adjacent node
        producer_preds = {"ga4l9do0", "s4jrpk2b"}

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "fezojtj3" and len(parts) >= 2:
                self.available_tlfqu.add(parts[1])

            if pred in movement_preds and len(parts) >= 3:
                a, b = parts[1], parts[2]  # (pred A B) -> edge B -> A
                if b not in self.adj_graph:
                    self.adj_graph[b] = set()
                self.adj_graph[b].add(a)
                # Ensure nodes exist in graph (so BFS won't KeyError for isolated nodes).
                if a not in self.adj_graph:
                    self.adj_graph[a] = set()

                # If this adjacency can be used as the producer adjacency
                # (i.e., an agent at B can produce at A), record it.
                if pred in producer_preds:
                    if a not in self.producer_neighbors:
                        self.producer_neighbors[a] = set()
                    self.producer_neighbors[a].add(b)

        # Ensure producer_neighbors keys exist for nodes even if empty.
        for node in list(self.adj_graph.keys()):
            if node not in self.producer_neighbors:
                self.producer_neighbors.setdefault(node, set())

        # A fallback large cost for infeasible / very expensive situations.
        self.INF_COST = 1000

    def _bfs_distance_to_targets(self, start, targets):
        """
        BFS on the directed adjacency graph from start node to the nearest node
        in the targets set. Returns integer distance (>= 0) or None if unreachable.
        """
        if start in targets:
            return 0
        if start not in self.adj_graph:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nbr in self.adj_graph.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in targets:
                    return dist + 1
                visited.add(nbr)
                q.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # 1) If all goals are already reached, return 0.
        if self.goals <= state:
            return 0

        # Parse current state into useful maps.
        fguw_set = set()  # locations that currently have fguw2142
        en_locations = {}  # en18p -> location (from ypt9mrgu)
        en_qp = {}  # en18p -> tlfqu (from qp2t6u4s)
        cb_present = set()  # cb6fw2xd facts present

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "fguw2142" and len(parts) >= 2:
                fguw_set.add(parts[1])
            elif pred == "ypt9mrgu" and len(parts) >= 3:
                en = parts[1]
                loc = parts[2]
                en_locations[en] = loc
            elif pred == "qp2t6u4s" and len(parts) >= 3:
                en = parts[1]
                res = parts[2]
                en_qp[en] = res
            elif pred == "cb6fw2xd" and len(parts) >= 3:
                cb_present.add((parts[1], parts[2]))

        total_cost = 0

        # Consider each goal fact that requires creating (cb6fw2xd loc res).
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "cb6fw2xd" or len(parts) < 3:
                # This heuristic focuses on cb6fw2xd goals. If other goal types
                # appear, conservatively add a small penalty (1) per such goal
                # unless already satisfied.
                if goal not in state:
                    total_cost += 1
                continue

            target_loc, target_res = parts[1], parts[2]
            if (target_loc, target_res) in cb_present:
                continue  # already achieved in current state

            # Determine the set of producer-adjacent source nodes (stq0j)
            # that allow applying the producing action for this target.
            producer_sources = self.producer_neighbors.get(target_loc, set())
            if not producer_sources:
                # If no static producer adjacency exists for this target, mark
                # it infeasible by assigning a large cost.
                total_cost += self.INF_COST
                continue

            # If there are no agents available to perform production, infeasible.
            if not en_locations:
                total_cost += self.INF_COST
                continue

            best_for_goal = None

            # Try to minimize cost by picking the best agent for this goal.
            for en, en_loc in en_locations.items():
                # movement cost: steps to reach any source in producer_sources
                move_dist = self._bfs_distance_to_targets(en_loc, producer_sources)
                if move_dist is None:
                    # No path to any required producer source for this agent.
                    continue

                # qp cost: ensure agent maps to target_res (qp2t6u4s en target_res)
                if en_qp.get(en) == target_res:
                    qp_cost = 0
                else:
                    # If target resource is available for wvswfzmn, we can change
                    # agent's mapping in one action, provided agent currently has
                    # some qp2t6u4s mapping (wvswfzmn requires an existing mapping).
                    if target_res in self.available_tlfqu and en in en_qp:
                        qp_cost = 1
                    else:
                        # Agent cannot readily be prepared for this resource.
                        continue

                # fguw2142 presence at target location.
                fguw_cost = 0 if target_loc in fguw_set else 1

                # +1 for the producing action itself (kkqx0eg7 or wbsgr2yj).
                candidate_cost = move_dist + qp_cost + fguw_cost + 1

                if best_for_goal is None or candidate_cost < best_for_goal:
                    best_for_goal = candidate_cost

            if best_for_goal is None:
                # No agent provided a feasible plan to produce this cb6fw2xd.
                total_cost += self.INF_COST
            else:
                total_cost += best_for_goal

        return int(total_cost)