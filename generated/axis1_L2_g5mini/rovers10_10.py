from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses
    and splitting the inner contents.

    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic implements an additive relaxed-plan style estimate (h_add)
    tailored to the given domain encoding. It performs a relaxed-graph propagation
    from the current state's known facts (including static facts) and computes
    the minimum summed cost to achieve the goal facts assuming action effects
    only add facts (i.e., ignoring delete effects). This provides a fast,
    domain-aware estimate that captures positive precondition-effect causal
    chains present in the operators.

    Assumptions
    -----------
    - Operators provided in task.operators are represented with their
      preconditions, add_effects and del_effects as strings identical to the
      runtime state's fact representation (i.e., "(pred arg1 arg2)").
    - Static facts (task.static) are true in every state; they are treated as
      available at zero cost.
    - Every operator has uniform action cost of 1. The heuristic therefore
      measures the minimum number of actions (in the relaxed sense) required
      to make goal facts true.
    - If a goal fact cannot be derived in the relaxed graph, the heuristic
      returns a large finite value (1e6) to indicate practical impossibility.

    Heuristic Initialization
    ------------------------
    During construction we:
    - Store the set of goal facts (task.goals).
    - Keep references to the operator set (task.operators) and static facts
      (task.static) so they can be used in the per-state relaxed propagation.
    - Note: no heavy preprocessing is required because operators already use
      concrete fact strings; this keeps initialization cheap.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node (with node.state being a frozenset of fact strings):
    1. Build an initial cost map:
       - Facts true in node.state or in task.static are assigned cost 0.
       - All other facts are initialized with infinite cost.
    2. Repeatedly consider every operator and try to derive new facts:
       - For an operator, if every precondition has a finite cost, then the
         operator can (in the relaxed sense) be applied. The cost to "achieve"
         any of its add_effects via this operator is:
             cost(op) = 1 + sum(cost(pre) for pre in preconditions)
         (i.e., additive combination of preconditions plus the action itself).
       - If cost(op) is smaller than the current known cost for an add_effect,
         update the add_effect's cost.
    3. Iterate this relaxation until no cost improves (a fixed point),
       or until all goal facts have finite costs.
    4. The heuristic value is the sum of the individual minimal costs of each
       goal fact (h_add). This encourages achieving multiple goals by
       recognizing shared preconditions (their costs are included only once
       per precondition in the operator-cost computation but the final sum
       still adds per-goal costs as in standard h_add).
    5. If one or more goal facts remain unreachable (cost = infinity), return
       a large finite number (1e6) rather than infinity to maintain finite
       heuristic values for the search algorithm.
    """

    def __init__(self, task):
        # Goals are expected to be an iterable/set of fact strings.
        self.goals = set(task.goals)
        # Operators are Operator objects with preconditions/add_effects as fact strings.
        self.operators = list(task.operators)
        # Static facts are facts true in every state; treat them as cost 0.
        self.static = set(task.static)

    def __call__(self, node):
        """
        Compute the additive relaxed-plan heuristic (h_add) for the node.

        Returns an integer estimating the number of actions to reach the goal.
        """
        state = set(node.state)  # current true facts
        # Initial facts are those in the state plus static facts.
        known_facts = state | self.static

        INF = float("inf")

        # Initialize cost map: 0 for known facts, INF otherwise.
        cost = {}
        # It's more efficient to initialize only facts that appear in operators
        # or goals, but for simplicity we lazily add facts as we encounter them.
        for f in known_facts:
            cost[f] = 0

        # Ensure all goal facts are present in the cost map (possibly INF).
        for g in self.goals:
            if g not in cost:
                cost[g] = INF

        # Also add every operator's add effects and preconditions to cost map.
        for op in self.operators:
            for f in op.preconditions | op.add_effects:
                if f not in cost:
                    cost[f] = INF

        # Relaxation loop: repeatedly try to improve costs using operators.
        improved = True
        while improved:
            improved = False
            for op in self.operators:
                # Sum the costs of preconditions; if any precondition unreachable,
                # skip this operator for now.
                pre_sum = 0
                skip = False
                for p in op.preconditions:
                    c = cost.get(p, INF)
                    if c == INF:
                        skip = True
                        break
                    pre_sum += c
                if skip:
                    continue
                op_cost = pre_sum + 1  # 1 for applying this operator

                # Try to improve costs for add effects.
                for eff in op.add_effects:
                    if op_cost < cost.get(eff, INF):
                        cost[eff] = op_cost
                        improved = True

            # Early exit: if all goals have finite cost, we could stop relaxing.
            all_finite = True
            for g in self.goals:
                if cost.get(g, INF) == INF:
                    all_finite = False
                    break
            if all_finite:
                break

        # Sum the costs of the goals to produce h_add.
        total = 0.0
        unreachable = False
        for g in self.goals:
            cg = cost.get(g, INF)
            if cg == INF:
                unreachable = True
            else:
                total += cg

        if unreachable:
            # Return a large finite value to indicate goal(s) unreachable in relaxed graph.
            return int(1e6)

        # Return integer heuristic (rounded up for safety).
        return int(math.ceil(total))