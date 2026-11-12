from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact string by removing surrounding parentheses
    and splitting on whitespace.

    Example:
      "(on-table item1 item6)" -> ["on-table", "item1", "item6"]

    Note: This deliberately strips only the first and last character, which are
    expected to be '(' and ')', as required by the checklist.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the (secret) blocksworld-like domain provided.
    The heuristic computes a delete-relaxed (i.e., ignoring delete effects) forward
    reachability estimate: starting from the union of the current state's facts and
    the static facts, it repeatedly applies applicable operators (those whose
    preconditions are satisfied) and accumulates their positive effects, counting
    one cost unit per operator application until all goal facts are achieved.
    To make the estimate tighter while remaining efficient, at each step the heuristic
    greedily chooses an applicable operator that yields the largest number of new
    facts (largest marginal gain). The returned value is the number of operator
    applications required in this relaxed model.

    Assumptions
    -----------
    - Operator instances provided in task.operators are ground (all parameters
      instantiated) and their preconditions/add-effects are strings matching the
      planner's internal fact representation (e.g., "(predicate arg1 arg2)").
    - Static facts provided in task.static hold in all states and should be
      considered initially true for the relaxed reachability.
    - Each operator application costs 1 action; the relaxed model ignores delete
      effects (so add effects persist) which makes the approach admissible only
      in the deletion-free sense, but the heuristic is intentionally not required
      to be admissible for greedy best-first search.
    - The heuristic returns 0 iff all goal facts hold in the given state.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - self.goals: the set of goal facts that must be achieved.
    - self.static: the set of static facts that are true in every state.
    - self.operators: the set of ground Operator instances available in the task.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Combine the dynamic facts of the current state with the static facts to
       form the starting fact set for the relaxed reachability computation.
       (This uses get_parts internally for fact structure when needed; facts are
       otherwise treated as opaque strings.)
    2. If all goal facts are already present in the starting fact set, return 0.
    3. Otherwise, perform an iterative forward expansion (delete-relaxed):
       a. Identify all operators whose preconditions are a subset of the currently
          known facts (i.e., they are applicable in the relaxed model).
       b. From those applicable operators, select one that introduces the largest
          number of new facts (maximizes len(add_effects - current_facts)).
          - If multiple operators tie, any can be chosen; choosing the largest
            marginal gain improves informativeness.
       c. Apply that operator in the relaxed model by adding its add_effects to
          the current fact set and incrementing the cost by 1.
       d. Repeat until either all goal facts are present or no applicable operator
          can add any new facts.
    4. If the goals are reached in the relaxed model, return the accumulated cost
       (an integer count of operator applications). If the goals cannot be reached
       (no operator can extend the fact set further), return a large finite
       penalty (9999) to indicate that the relaxed model could not find a path;
       this keeps the heuristic finite for solvable states and provides a strong
       (high) estimate for dead-ends in the relaxed model.
    """

    def __init__(self, task):
        """
        Initialize the heuristic with task-level information.

        Parameters:
        - task: an instance of Task (see code-file-task) which exposes:
            - task.goals: set of goal fact strings
            - task.static: set of static fact strings
            - task.operators: set of Operator instances (ground operators)
        """
        self.goals = set(task.goals)
        self.static = set(task.static)
        # Operators are expected to be Operator instances with .preconditions
        # and .add_effects as sets/frozensets of fact strings.
        self.operators = set(task.operators)

    def __call__(self, node):
        """
        Compute the heuristic value for a given search node.

        Returns:
        - an integer estimate of the number of actions to reach the goal from
          node.state according to the delete-relaxed greedy forward expansion.
        - returns 0 if the goal is already satisfied in node.state.
        - returns a large integer (9999) if the relaxed expansion cannot reach all goals.
        """
        state = set(node.state)  # node.state is a frozenset of fact strings
        current_facts = set(state) | set(self.static)

        # Quick goal check (checklist item 2)
        if self.goals <= current_facts:
            return 0

        # Prepare a list of operators that can ever help (copy to avoid mutating task data)
        operators = list(self.operators)

        cost = 0
        # Repeatedly apply the applicable operator that gives the largest marginal gain
        while not (self.goals <= current_facts):
            best_op = None
            best_gain = 0

            for op in operators:
                # An operator is considered applicable in the relaxed model if all preconditions are present.
                if op.preconditions <= current_facts:
                    gain = len(op.add_effects - current_facts)
                    if gain > best_gain:
                        best_gain = gain
                        best_op = op

            if best_op is None or best_gain == 0:
                # No operator can further extend the fact set -> goals unreachable in relaxed model.
                # Return a large finite penalty. This keeps the heuristic finite for solvable states
                # (they should have been reachable here) and penalizes dead-ends.
                return 9999

            # Apply the best operator in the relaxed model (add its positive effects).
            current_facts |= best_op.add_effects
            cost += 1

            # Small optimization: if cost grows beyond a reasonable bound, return large value.
            # This avoids pathological loops in degenerate operator sets.
            if cost > 10000:
                return 9999

        return cost