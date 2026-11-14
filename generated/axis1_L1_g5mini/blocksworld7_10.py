from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing the surrounding parentheses
    and splitting the inner string.

    Example:
      "(s0899d5u b1 b2)" -> ["s0899d5u", "b1", "b2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A delete-relaxed additive heuristic (h_add) tailored to the secret_domain
    (domain name z1spq3ra). The heuristic estimates the number of actions
    required to satisfy all goal facts by computing minimal costs to achieve
    individual facts in the delete-relaxed problem (i.e., ignoring delete
    effects). The final heuristic value is the sum of the minimal costs for
    each goal fact (additive combination). This tends to be informative for
    this domain because goals are a conjunction of ground facts (unary and
    binary) and most operators produce a small number of add effects that
    correspond directly to those goals.

    Assumptions
    -----------
    - Operators available in `task.operators` are ground (no parameters) and
      expressed as strings, consistent with the planner's internal
      representation (Operator.preconditions/add_effects/del_effects are
      frozensets of fact strings).
    - The delete-relaxed problem (ignoring delete effects) provides a useful
      heuristic signal in this domain because many actions primarily add the
      predicates required by the goals (e.g., s0899d5u links).
    - All action costs are uniform (1 per action). The additive heuristic
      (h_add) sums precondition costs plus one for the action itself.
    - The planner represents facts as strings of the form "(pred arg1 ...)".
      The helper `get_parts` removes the surrounding parentheses before parsing.

    Heuristic Initialization
    ------------------------
    During initialization the heuristic extracts and stores:
    - self.goals: the set of goal facts (task.goals).
    - self.static: the set of static facts (task.static). Although this domain's
      static set may be empty, we store it for completeness and potential use.
    - self.operators: the set/list of operator objects for fast iteration.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the current state already satisfies all goals, return 0.
    2. Initialize a dictionary `fact_cost` mapping every fact string to its
       current best-known cost to achieve in the delete-relaxed model. Facts
       present in the current state are assigned cost 0; all others start at
       +inf.
    3. Iteratively relax operators:
       - For each operator, check whether all its preconditions currently have
         finite cost. If so, compute the operator cost as
           op_cost = 1 + sum(cost(pre) for pre in op.preconditions)
         (this is the standard additive cost combination).
       - For each fact added by the operator (fact in op.add_effects), if
         op_cost is smaller than the current recorded cost for that fact,
         update the fact's cost to op_cost.
       - Repeat until a full pass over all operators produces no updates.
    4. Compute the heuristic value as the sum of the costs of each goal fact
       (additive combination). If any goal fact remains at +inf, the relaxed
       model considers it unreachable and we return a large finite value
       (instead of inf) to keep the heuristic finite; however, for reachable
       problems in the delete-relaxed model this should not occur.
    5. Return the integer heuristic value (rounded up from floating values if
       necessary, but here all costs are integers so rounding is not required).

    Notes on design choices
    -----------------------
    - The additive delete-relaxed heuristic balances informativeness and
      computational efficiency. It captures some interactions between goals
      through shared preconditions without attempting expensive exact
      reasoning. For this domain, where connecting/creating s0899d5u links and
      toggling unary predicates are the main effects, the h_add signal is
      typically effective for greedy best-first search guidance.
    - To ensure the heuristic remains finite for solvable states, unreachable
      goals in the relaxed model are mapped to a large finite penalty.
    """

    def __init__(self, task):
        # Store goals and static facts.
        self.goals = set(task.goals)
        self.static = set(task.static)
        # Operators are assumed to be ground Operator instances.
        # Convert to a list for repeated iteration.
        self.operators = list(task.operators)

        # A safe large finite penalty used when a goal is unreachable in the
        # delete-relaxed model (keeps heuristic finite as required).
        self._unreachable_penalty = 10**6

    def __call__(self, node):
        state = node.state

        # If all goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Initialize fact costs: facts true in the current state cost 0,
        # others start at +inf.
        fact_cost = {}

        # Collect universe of facts we might consider:
        # - All facts appearing in operators' add effects or preconditions.
        # This limits dictionary size to relevant facts.
        universe = set()
        for op in self.operators:
            universe.update(op.preconditions)
            universe.update(op.add_effects)
        # Also include goals to ensure they are present in the universe.
        universe.update(self.goals)
        # Initialize costs
        for f in universe:
            if f in state:
                fact_cost[f] = 0.0
            else:
                fact_cost[f] = inf

        # Iteratively relax operators using additive combination until fixpoint.
        changed = True
        while changed:
            changed = False
            for op in self.operators:
                # Sum costs of preconditions; if any precondition unreachable, skip.
                pre_cost_sum = 0.0
                preconditions = op.preconditions
                skip = False
                for pre in preconditions:
                    # If a precondition is not in our universe it may still be in the
                    # initial state; check state as fallback (but we included
                    # operator preconditions in universe above, so this is safe).
                    c = fact_cost.get(pre, 0.0 if pre in state else inf)
                    if c == inf:
                        skip = True
                        break
                    pre_cost_sum += c
                if skip:
                    continue
                op_cost = 1.0 + pre_cost_sum
                # Update costs for each add effect if we found a cheaper way.
                for add in op.add_effects:
                    old = fact_cost.get(add, inf)
                    if op_cost < old:
                        fact_cost[add] = op_cost
                        changed = True

        # Sum costs for goal facts. If any goal is unreachable (inf) map to large penalty.
        total = 0.0
        for g in self.goals:
            c = fact_cost.get(g, inf)
            if c == inf:
                # Return a large finite value rather than infinity to obey the
                # checklist requirement that heuristic is finite for solvable
                # states. If the relaxed model truly deems the goal unreachable,
                # this large value will strongly deprioritize the node.
                return self._unreachable_penalty
            total += c

        # Costs are integral in this domain model; cast to int for consistency.
        return int(total)