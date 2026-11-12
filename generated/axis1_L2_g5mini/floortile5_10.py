from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return components of a PDDL fact string by stripping surrounding parentheses and splitting."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (hi4l5zmb). The domain's
    goals require facts of the form (cb6fw2xd <node> <target-resource>).
    Producing such a fact requires (briefly):
      - A node <node> that currently holds the marker (fguw2142).
      - An agent <en> that has an association (ypt9mrgu) with a neighboring node
        <stq> such that there is a static relation from <node> to <stq>
        (specifically ga4l9do0 or s4jrpk2b in the domain).
      - The agent must hold the specific qp2t6u4s pairing with the target resource
        (qp2t6u4s <en> <tlfqu>). If not, it can possibly be changed using
        wvswfzmn provided the target resource is marked fezojtj3 somewhere
        in the problem.

    Assumptions
    -----------
    - The heuristic is not admissible (it may overestimate) but aims to be
      informative and fast to compute.
    - Static relations ga4l9do0 and s4jrpk2b define the adjacency used by the
      two operators that add cb6fw2xd; only these adjacency types are used to
      identify feasible neighboring positions for agents.
    - If an agent does not currently have the required qp2t6u4s pairing, we
      assume a single wvswfzmn action will suffice when a fezojtj3 fact for
      the target resource exists (this is a light abstraction).
    - We ignore detailed interdependencies between different goals (sharing of
      agents or tokens) and sum per-goal estimated costs. This is a heuristic
      choice that trades accuracy for speed.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and index:
    - The list of goal (cb6fw2xd ...) facts to satisfy.
    - Static adjacency relations:
        - successors_via_ga4l9do0: nodes reachable via (ga4l9do0 X Y) -> Y is a neighbor of X
        - successors_via_s4jrpk2b: similarly for (s4jrpk2b X Y)
      These are used to find candidate neighbor nodes where an agent must be present.
    - The static set of fezojtj3 resource nodes (targets that can be swapped in via wvswfzmn).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state) the heuristic proceeds as follows for each
    goal (cb6fw2xd node target):
      1. If the goal fact is already true in the state, cost contribution is 0.
      2. Otherwise, estimate minimal actions to produce cb6fw2xd(node, target) as:
           a) 1 for the final operator that creates cb6fw2xd (either kkqx0eg7 or
              wbsgr2yj).
           b) cost_qp: cost to ensure some agent has qp2t6u4s <agent> <target>.
              - 0 if such an agent already exists.
              - 1 if there exists an agent that can change to that tlfqu using
                wvswfzmn (i.e., some agent already has a qp2t6u4s pairing and
                fezojtj3 <target> is a static fact).
              - large_penalty otherwise (we cannot cheaply reach this target).
           c) cost_fguw: cost to ensure fguw2142 is true at the goal node.
              - 0 if (fguw2142 node) is present.
              - 1 if any fguw2142 exists somewhere (we assume one move sequence
                can propagate the token to the desired node).
              - large_penalty if no fguw2142 exists at all in the state.
           d) cost_agent_pos: cost to ensure an agent that will perform the
              final action is located at an appropriate neighbor <stq> of
              the goal node (a neighbor is any Y in successors_via_ga4l9do0[X]
              or successors_via_s4jrpk2b[X]).
              - 0 if an agent with the needed qp2t6u4s (or one that can be
                changed) is already at such a neighbor.
              - 1 otherwise (we assume repositioning the agent can be done
                with a small number of actions).
      3. Sum the above contributions to get the estimate for that goal.
      4. The heuristic value is the sum of estimated costs for all unsatisfied
         goals. A moderate large_penalty is used for goals that appear
         unreachable under the current abstraction to keep the heuristic finite.

    This heuristic is cheap: it uses only current-state lookups and simple
    graph membership tests over static adjacency to form an estimate. It is
    deliberately approximate (domain-aware but abstract) to remain fast and
    effective for greedy best-first search guidance.
    """

    def __init__(self, task):
        # store goals (set of strings)
        self.goals = set(task.goals)

        # static facts (frozenset of strings)
        static_facts = task.static

        # build adjacency from static relations ga4l9do0 and s4jrpk2b:
        # for (ga4l9do0 X Y) and (s4jrpk2b X Y) we record Y as a successor (neighbor) of X.
        self.successors = {}  # node -> set(neighbors)
        def add_succ(a, b):
            self.successors.setdefault(a, set()).add(b)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "ga4l9do0" and len(parts) == 3:
                add_succ(parts[1], parts[2])
            elif parts[0] == "s4jrpk2b" and len(parts) == 3:
                add_succ(parts[1], parts[2])

        # set of resources that can be the target of wvswfzmn (fezojtj3 ...)
        self.fezo_targets = {get_parts(f)[1] for f in static_facts if get_parts(f)[0] == "fezojtj3"}

        # A moderate finite penalty for "hard/impossible" under this abstraction.
        self.large_penalty = 1000

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # Fast 0 for goal states (check per checklist)
        if self.goals <= state:
            return 0

        # Index useful current-state facts:
        # - fguw2142 nodes
        fguw_nodes = {get_parts(f)[1] for f in state if get_parts(f)[0] == "fguw2142"}

        # - qp2t6u4s mapping agent -> resource
        qp_map = {}
        for f in state:
            parts = get_parts(f)
            if parts[0] == "qp2t6u4s" and len(parts) == 3:
                qp_map[parts[1]] = parts[2]

        # - set of (agent, location) for ypt9mrgu
        agent_loc = {}
        for f in state:
            parts = get_parts(f)
            if parts[0] == "ypt9mrgu" and len(parts) == 3:
                agent_loc[parts[1]] = parts[2]

        # Precompute agents available (those appearing in qp_map or agent_loc)
        agents = set(agent_loc.keys()) | set(qp_map.keys())

        total_cost = 0

        # Parse goals: extract (cb6fw2xd node target) from self.goals
        goal_list = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "cb6fw2xd" and len(parts) == 3:
                goal_list.append((parts[1], parts[2]))
            else:
                # if some other goal form exists, treat naively: cost 1 if not present
                if g not in state:
                    total_cost += 1

        for node_x, target_t in goal_list:
            goal_fact = f"(cb6fw2xd {node_x} {target_t})"
            if goal_fact in state:
                continue  # already satisfied

            # Base cost: 1 for the action that creates cb6fw2xd
            cost = 1

            # (b) cost to ensure an agent has qp2t6u4s <agent> <target>
            agent_with_qp = [a for a, t in qp_map.items() if t == target_t]
            if agent_with_qp:
                cost_qp = 0
            else:
                # can we change some agent's qp to target_t using wvswfzmn?
                # wvswfzmn requires some agent that already has qp2t6u4s and fezojtj3 target_t static.
                if self.fezo_targets and target_t in self.fezo_targets and qp_map:
                    cost_qp = 1
                else:
                    # no way in this abstraction to obtain the required qp pairing
                    cost_qp = self.large_penalty
            cost += cost_qp

            # (c) cost to ensure fguw2142 at node_x
            if node_x in fguw_nodes:
                cost_fguw = 0
            else:
                # if there exists any fguw token somewhere we assume it can be brought here with small cost
                if fguw_nodes:
                    cost_fguw = 1
                else:
                    # nothing to move to this node in this abstraction
                    cost_fguw = self.large_penalty
            cost += cost_fguw

            # (d) cost to position an agent at one of the neighbors st such that
            # the final operator preconditions are satisfied. The domain's final
            # operators require a neighbor st where (ga4l9do0 node_x st) or (s4jrpk2b node_x st).
            neighbors = self.successors.get(node_x, set())
            # If no neighbor, it's impossible under real domain to use those operators (but keep finite)
            if not neighbors:
                # we charge a penalty (but finite)
                cost_agent_pos = 2
            else:
                # If there is an agent with required qp already at a neighbor -> 0
                found_zero = False
                for a in agents:
                    loc = agent_loc.get(a)
                    qp = qp_map.get(a)
                    # if qp is already correct or can be changed cheaply:
                    qp_ok = qp == target_t or (qp is not None and target_t in self.fezo_targets)
                    if qp_ok and loc in neighbors:
                        found_zero = True
                        break
                if found_zero:
                    cost_agent_pos = 0
                else:
                    # otherwise we assume a small cost to reposition an agent (abstract)
                    cost_agent_pos = 1
            cost += cost_agent_pos

            total_cost += cost

        # keep heuristic finite
        if total_cost >= self.large_penalty:
            return self.large_penalty
        return total_cost