from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (hi4l5zmb) domain.
    The domain models agents (k9b7lknr) that can move between locations (nh1sy8v9)
    using several adjacency predicates and that carry or can switch resources
    (qekv4465). The planning goal is to establish facts of the form
    (cb6fw2xd <location> <resource>) at certain locations. These are generated
    by actions that require:
      - an agent to be at a particular neighbor location of the goal location,
      - the goal location to be "free" (predicate fguw2142 true),
      - the agent to have the required resource (qp2t6u4s <agent> <resource>).
    Movement actions move agents along directed adjacency facts (four different
    adjacency predicates exist), and a short action can change the agent's
    resource type given that fezojtj3 marks that resource as available.
    This heuristic estimates how many actions (roughly) are required to create
    all missing cb6fw2xd goal facts from the current state.

    Assumptions
    -----------
    - Movements along adjacency facts each cost 1 action (we treat all move
      operators uniformly).
    - Changing an agent's resource with the wvswfzmn action costs 1 action,
      but is only possible if the agent already has some qp2t6u4s mapping
      (this reflects the domain: wvs requires an existing qp2t6u4s).
    - Creating the cb6fw2xd fact (the goal-producing operator) costs 1 action.
    - If the target location is currently not free (fguw2142 false), we add
      a small penalty (1) to reflect the need to free that location (moving a
      blocking agent). We do not attempt to fully reason about sequences to
      free locations; this is an informed approximation.
    - We ignore interference between different goals (i.e., the same agent may
      be counted for multiple goals). This makes the heuristic fast and
      usually informative for greedy best-first search even though it is not
      admissible.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - The set of goal (cb6fw2xd) requirements as (location, resource) pairs.
    - Static adjacency relations for the four movement predicates:
        vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0.
      For each static fact of the form (pred X Y) we add a directed edge Y -> X
      to the movement graph because domain actions allow moving from Y to X
      when the static fact (X Y) exists.
    - The set of known resources that are declared (fezojtj3 facts) from static
      facts (these allow resource changes).
    The graph built this way gives, for any source location, the set of
    locations reachable in one move (ignoring occupancy constraints).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state we compute the heuristic as follows:
    1. If all goal facts are already present in the state, return 0.
    2. Parse the state to obtain:
       - For each agent (from ypt9mrgu facts), its current location.
       - For each agent, its current resource (qp2t6u4s facts), if any.
       - The set of currently free locations (fguw2142 facts).
       - The set of already achieved cb6fw2xd facts.
    3. For each missing goal (location L, resource R) compute an estimated
       minimal cost to produce that fact:
       a) Identify the set of allowed agent positions (stq) from static facts:
          these are the second-arguments `stq` in any static adjacency fact
          whose first argument equals the goal location L (i.e. facts of the
          form (P L stq) where P is one of the movement predicates). An agent
          must be at one such stq to perform the goal-producing action.
       b) For each agent, compute the shortest number of moves (in the static
          movement graph) from the agent's current location to any of those
          allowed stq positions (BFS). If no path exists, that agent cannot
          reach the required position and is ignored for this goal.
       c) Compute the resource-change cost:
          - 0 if the agent already has qp2t6u4s with resource R.
          - 1 if the agent has some qp2t6u4s but not R (we can switch using
            wvswfzmn, provided R is a valid resource in static fezojtj3).
          - very large (agent not usable) if the agent has no qp2t6u4s mapping.
       d) If the goal location L is currently not free (no fguw2142 L in
          state), add a small penalty (+1) to reflect the need to free it.
       e) The total estimated cost for that agent is:
          move_distance + resource_change_cost + 1 (goal-producing action) + free_penalty.
       f) The estimated cost for the goal is the minimum over all agents.
       g) If no agent can achieve the goal (no reachable agent with a usable
          resource), set the cost for this goal to a large constant.
    4. The heuristic value is the sum of the per-goal minimal estimates.
       (This is a simple aggregation that keeps computation cheap while
       providing useful guidance. It is not admissible but tends to point
       search toward locations where an agent can quickly produce goals.)
    """

    def __init__(self, task):
        # Save goal cb6fw2xd facts as (location, resource) pairs
        self.goals = set()
        for g in task.goals:
            parts = get_parts(g)
            if parts and parts[0] == "cb6fw2xd" and len(parts) == 3:
                loc, res = parts[1], parts[2]
                self.goals.add((loc, res))

        # Movement-related static predicates: when a static fact (P X Y) exists,
        # an agent at Y can move to X (edge Y -> X).
        self.move_predicates = {"vjcqgdgn", "s4jrpk2b", "vom4pfig", "ga4l9do0"}

        # Build directed movement graph from static facts: node -> set(neighbors)
        self.move_graph = {}
        # Also, for each location L we want to quickly obtain required agent
        # positions stq for generating cb6fw2xd at L. These are the second args
        # in static facts where first == L.
        self.required_agent_positions = {}

        self.available_resources = set()  # from fezojtj3 static facts

        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in self.move_predicates and len(parts) == 3:
                x, y = parts[1], parts[2]
                # add edge y -> x (agent can move from y to x)
                self.move_graph.setdefault(y, set()).add(x)
                # record that to act on location x you may need an agent at y
                self.required_agent_positions.setdefault(x, set()).add(y)
            elif pred == "fezojtj3" and len(parts) == 2:
                self.available_resources.add(parts[1])

        # Precompute list of all movement nodes (locations)
        # Ensure nodes present for BFS even if they have no outgoing edges
        all_nodes = set(self.move_graph.keys())
        for targets in self.move_graph.values():
            all_nodes.update(targets)
        for n in all_nodes:
            self.move_graph.setdefault(n, set())

        # A large cost used to indicate an unreachable/impossible goal in the
        # current state approximation.
        self.INF_COST = 1000

    def _bfs_distance(self, start, targets):
        """
        Compute shortest distance in the movement graph from start to any node in targets.
        Returns integer distance (0 if start in targets), or self.INF_COST if no path.
        """
        if start in targets:
            return 0
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, dist = dq.popleft()
            for neigh in self.move_graph.get(node, ()):
                if neigh in visited:
                    continue
                if neigh in targets:
                    return dist + 1
                visited.add(neigh)
                dq.append((neigh, dist + 1))
        return self.INF_COST

    def __call__(self, node):
        state = node.state

        # 1) If all goals already satisfied, return 0
        # Convert state's cb6fw2xd facts into set of pairs for quick check
        satisfied_goals = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "cb6fw2xd" and len(parts) == 3:
                satisfied_goals.add((parts[1], parts[2]))
        if self.goals <= satisfied_goals:
            return 0

        # 2) Parse state: agent locations (ypt9mrgu), agent resources (qp2t6u4s),
        # free locations (fguw2142), and existing cb6fw2xd already extracted above.
        agent_location = {}  # agent -> location
        agent_resource = {}  # agent -> resource (qp2t6u4s)
        free_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "ypt9mrgu" and len(parts) == 3:
                agent, loc = parts[1], parts[2]
                agent_location[agent] = loc
            elif pred == "qp2t6u4s" and len(parts) == 3:
                agent, res = parts[1], parts[2]
                agent_resource[agent] = res
            elif pred == "fguw2142" and len(parts) == 2:
                free_locations.add(parts[1])

        # If there are no agents, all (non-satisfied) goals are impossible in this
        # approximation: return very large heuristic value.
        if not agent_location:
            return self.INF_COST * len(self.goals)

        total_estimated_cost = 0

        # For each goal not yet satisfied, compute minimal estimated cost
        for (loc, res) in self.goals:
            if (loc, res) in satisfied_goals:
                continue  # already achieved

            # Determine positions stq where an agent must stand: these are the
            # second-arguments in any static adjacency fact (P loc stq).
            required_positions = self.required_agent_positions.get(loc, set())
            if not required_positions:
                # No static relation allows an agent to act on loc => unreachable
                total_estimated_cost += self.INF_COST
                continue

            loc_free_penalty = 0 if loc in free_locations else 1

            best_cost_for_goal = self.INF_COST

            # Evaluate each agent as candidate performer
            for agent, a_loc in agent_location.items():
                # Check resource capability / change cost
                if agent in agent_resource:
                    if agent_resource[agent] == res:
                        resource_change_cost = 0
                    else:
                        # Can only switch to res if res is a known resource in static
                        resource_change_cost = 1 if res in self.available_resources else self.INF_COST
                else:
                    # Agent has no qp2t6u4s mapping and therefore cannot use wvs to obtain one
                    resource_change_cost = self.INF_COST

                if resource_change_cost >= self.INF_COST:
                    # This agent is not usable for this goal under our relaxation
                    continue

                # Distance from agent location to any required position
                move_dist = self._bfs_distance(a_loc, required_positions)
                if move_dist >= self.INF_COST:
                    # Agent cannot reach required position
                    continue

                # 1 action to perform the goal-creating operator, plus move and resource-change costs
                cost = move_dist + resource_change_cost + 1 + loc_free_penalty

                if cost < best_cost_for_goal:
                    best_cost_for_goal = cost

            if best_cost_for_goal >= self.INF_COST:
                # No agent can achieve this goal under the relaxation
                total_estimated_cost += self.INF_COST
            else:
                total_estimated_cost += best_cost_for_goal

        # Return the estimated number of actions to satisfy all remaining goals.
        return total_estimated_cost