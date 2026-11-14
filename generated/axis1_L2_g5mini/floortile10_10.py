from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the list of tokens in a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `secret_domain` (internal name hi4l5zmb).
    The domain's goals require facts of the form (cb6fw2xd <location> <object>).
    Agents (k9b7lknr) carry objects (qp2t6u4s agent object) and can deposit them at
    a goal location using specialized drop operators. Movement between locations is
    possible along several typed (static) relations; depositing requires the agent
    to stand at a specific neighbor of the goal location (the static relations that
    allow depositing are `ga4l9do0` and `s4jrpk2b` in this domain).

    The heuristic estimates the number of actions to satisfy all remaining cb6fw2xd
    goal facts. For each unsatisfied goal (destination, obj) it computes the minimal
    cost over agents that could plausibly perform the deposit:
      - movement cost: shortest-path length in the static movement graph from the
        agent's current location to any location from which the agent can drop
        the object for that destination;
      - swap/load cost: if the agent currently carries the required object, cost 0;
        if the agent carries a different object, cost 1 (use wvswfzmn to switch
        carried object to any fezojtj3-marked object);
      - drop cost: 1 (the action that creates the cb6fw2xd fact);
      - occupancy penalty: small optimism penalty (0 or 1) if the destination is
        currently not free (fguw2142 missing) — we add 1 in that case to reflect
        the need to free the spot (optimistic approximation).
    The heuristic sums the per-goal minimal costs. It is not required to be admissible;
    it is a relatively informed, efficiently computed estimate that captures the key
    structure of how goals are achieved in the domain (agent movement + having the
    correct object + executing a deposit action).

    Assumptions
    -----------
    - Movement between locations can be modeled as a directed graph built from the
      static relation facts of types vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0.
      A static fact (REL dest src) gives a directed edge src -> dest (matching
      movement operator preconditions).
    - Deposits (cb6fw2xd) at a destination `d` can be performed only when an agent
      stands at one of the source locations `s` for which a static fact
      (ga4l9do0 d s) or (s4jrpk2b d s) exists (these correspond to the two
      deposit-producing operators in the domain).
    - Agents cannot pick up objects from the world with a dedicated operator in the
      domain; they can only transform the object they carry via wvswfzmn. Therefore,
      the heuristic looks for agents that already carry some object (qp2t6u4s).
      If no agent carries anything, we return a finite but pessimistic estimate.
    - We ignore dynamic constraints such as temporarily occupied intermediate nodes
      when computing shortest paths (optimistic assumption).

    Heuristic Initialization
    ------------------------
    The constructor extracts the following from task.static:
    - movement adjacency graph (adjacency: source -> set(dest))
      built from static facts of predicates: vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0.
    - deposit_predecessors: for each destination `d`, the set of agent-locations `s`
      such that (ga4l9do0 d s) or (s4jrpk2b d s). These are the standing positions
      from which the deposit action can be executed.
    - fezo_objects: set of objects that satisfy (fezojtj3 obj) (these can be the
      target objects an agent can switch to using wvswfzmn).
    The overall goal set is stored (task.goals) to quickly detect goal states.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node:
    1. If all goal facts are already true in the node's state, return 0.
    2. Parse the node's facts and extract:
       - agent locations: mapping agent -> location from ypt9mrgu facts;
       - agent-carried objects: mapping agent -> object from qp2t6u4s facts;
       - set of free locations from fguw2142 facts;
       - set of already satisfied cb6fw2xd facts.
    3. For each unsatisfied goal (cb6fw2xd dest obj) compute a minimal cost:
       a. If no deposit-predecessors exist for `dest` (i.e., static has no valid
          standing positions), treat goal cost as a moderate large constant.
       b. Consider every agent that carries some object (qp2t6u4s exists for it):
          i. For every standing position `s` in deposit_predecessors[dest], compute
             the shortest-path distance (number of move actions) from the agent's
             current location to `s` using the movement adjacency graph. If no
             path exists, this standing position is ignored for that agent.
         ii. Calculate swap_cost: 0 if the agent already carries `obj`, else 1 (one
             wvswfzmn action to switch carried object to a fezojtj3 object).
        iii. occupancy_penalty: 0 if dest in free_locations, else 1.
         iv. deposit action cost is 1.
          v. The candidate total for this agent-and-standing-position is:
             distance + swap_cost + deposit_cost(=1) + occupancy_penalty.
       c. The goal's estimated cost is the minimum candidate across all agents and
          standing positions. If no agent/position is usable, the heuristic uses
          a fallback cost (a moderately large finite number).
    4. Return the sum of the estimated costs for all unsatisfied goals. The heuristic
       is intentionally optimistic in some respects (ignores some freeing/mutual
       exclusion complexities) but captures the dominant structure of the required
       actions: reach the correct standing position, have the right object, and
       execute the deposit.
    """

    def __init__(self, task):
        # Goals (set of strings)
        self.goals = task.goals
        static_facts = task.static

        # Build directed movement graph from source -> dest using all movement relation types.
        self._movement_preds = {"vjcqgdgn", "s4jrpk2b", "vom4pfig", "ga4l9do0"}
        self.adjacency = {}  # source -> set(dest)
        # deposit-related predecessors: dest -> set(standing_location)
        self.deposit_predecessors = {}
        self.fezo_objects = set()  # objects with (fezojtj3 obj)

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in self._movement_preds and len(parts) == 3:
                dest, src = parts[1], parts[2]
                self.adjacency.setdefault(src, set()).add(dest)
                # if this predicate is ga4l9do0 or s4jrpk2b, also record that standing
                # at `src` allows depositing at `dest`.
                if pred in {"ga4l9do0", "s4jrpk2b"}:
                    self.deposit_predecessors.setdefault(dest, set()).add(src)
            elif pred == "fezojtj3" and len(parts) == 2:
                self.fezo_objects.add(parts[1])

        # Heuristic fallback costs
        self._infeasible_goal_cost = 10  # finite but discouraging cost for goals lacking reachable agents
        self._no_carrier_penalty = 5  # penalty when no agent carries anything (finite)

    def _bfs_distance_to_targets(self, start, targets):
        """
        BFS in the static adjacency graph to compute the shortest number of moves
        from `start` to any node in `targets`. If start is already in targets
        returns 0. If no path exists, returns None.
        """
        if start in targets:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for neigh in self.adjacency.get(node, ()):
                if neigh in visited:
                    continue
                if neigh in targets:
                    return dist + 1
                visited.add(neigh)
                queue.append((neigh, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # 1) If goal already reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse relevant facts from the state.
        agent_location = {}  # agent -> location (from ypt9mrgu)
        agent_carried = {}  # agent -> object (from qp2t6u4s)
        free_locations = set()  # locations with fguw2142
        achieved_cb = set()  # cb6fw2xd facts present already

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "ypt9mrgu" and len(parts) == 3:
                agent, loc = parts[1], parts[2]
                agent_location[agent] = loc
            elif pred == "qp2t6u4s" and len(parts) == 3:
                agent, obj = parts[1], parts[2]
                agent_carried[agent] = obj
            elif pred == "fguw2142" and len(parts) == 2:
                free_locations.add(parts[1])
            elif pred == "cb6fw2xd" and len(parts) == 3:
                achieved_cb.add((parts[1], parts[2]))

        # Collect unsatisfied cb6fw2xd goals:
        unsatisfied_goals = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "cb6fw2xd" and len(parts) == 3:
                dest, obj = parts[1], parts[2]
                if (dest, obj) not in achieved_cb:
                    unsatisfied_goals.append((dest, obj))

        total_estimate = 0
        # If there are no agents carrying anything, we produce a finite but
        # pessimistic estimate.
        if not agent_carried:
            # Heuristic: each unsatisfied goal requires at least _no_carrier_penalty actions.
            return len(unsatisfied_goals) * self._no_carrier_penalty

        # For each unsatisfied goal compute minimal estimated cost.
        for dest, obj in unsatisfied_goals:
            best_cost = None

            # Identify valid standing positions from which a deposit into `dest` is possible.
            standing_positions = self.deposit_predecessors.get(dest, set())
            if not standing_positions:
                # No static deposit positions -> infeasible in normal domain semantics.
                # Return a finite penalty for this goal.
                total_estimate += self._infeasible_goal_cost
                continue

            # occupancy penalty (optimistically assume we may need +1 if not free)
            occupancy_penalty = 0 if dest in free_locations else 1

            # For every agent that carries something, consider them as candidate deliverer.
            for agent, carried_obj in agent_carried.items():
                # agent must have a known location to move
                if agent not in agent_location:
                    continue
                start_loc = agent_location[agent]

                # distance to any standing position
                dist = self._bfs_distance_to_targets(start_loc, standing_positions)
                if dist is None:
                    # this agent cannot reach any standing position (in static graph)
                    continue

                # swap cost: 0 if already carrying the wanted object, 1 if carrying something else
                swap_cost = 0 if carried_obj == obj else 1

                # deposit action cost is 1
                deposit_cost = 1

                candidate = dist + swap_cost + deposit_cost + occupancy_penalty

                if best_cost is None or candidate < best_cost:
                    best_cost = candidate

            # If we found a candidate cost, add it; otherwise add fallback penalty.
            if best_cost is not None:
                total_estimate += best_cost
            else:
                total_estimate += self._infeasible_goal_cost

        return int(total_estimate)