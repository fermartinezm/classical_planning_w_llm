from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact string by removing the surrounding
    parentheses and splitting on whitespace.

    Example:
      "(arm-empty item2 item17)" -> ["arm-empty", "item2", "item17"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal
    for the secret_domain (a Blocks-like domain where goals are typically
    `arm-empty` facts). The returned value is the count of goal facts that are
    not true in the current state. This provides a fast, informative estimate
    for greedy best-first search: states closer to satisfying more goal facts
    receive lower heuristic values.

    Assumptions
    -----------
    - Goals are expressed as ground facts (strings) in task.goals.
    - Operators tend to add or remove one relevant goal fact at a time
      (e.g., `arm-empty` facts are created/removed by few actions), so the
      number of unsatisfied goal facts is a reasonable indicator of distance.
    - static facts (task.static) contain reusable relation patterns that may
      help to recognize feasible paths; we store them for possible future use.
    - Heuristic need not be admissible; it should be fast and reduce node
      expansions in greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts as given by task.goals.
    - Parse static facts into a mapping from predicate name to a set of
      argument-tuples for quick lookup (this code does not rely heavily on
      static facts for the core estimate but keeps them available).
    - The constructor performs inexpensive preprocessing only.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. The heuristic first checks which goal facts are already true in the
       given state. The planner represents states and facts as strings like
       "(predicate arg1 arg2)".
    2. We compute the set difference: missing_goals = goals - state.
       - If this is empty, the state satisfies all goals and the heuristic
         returns 0 (this guarantees the heuristic is 0 exactly for goal
         states).
    3. We return the number of missing goal facts as the heuristic value.
       Rationale:
       - In this domain many operators add or remove single, clearly
         identifiable facts (notably `arm-empty`), so the count of missing
         facts correlates well with the number of operator applications
         required to reach the goal.
       - Counting missing goal facts is extremely cheap to compute and yields
         finite values for solvable states.
    4. Optional future refinements (not required here) could:
       - weight missing `arm-empty` facts differently depending on nearby
         static structure,
       - detect groups of goals that can be achieved together and reduce
         the estimate accordingly.
    """

    def __init__(self, task):
        """
        Extract and store goal facts and a simple index of static facts.

        Parameters
        ----------
        - task: the planner Task object providing .goals and .static
        """
        # Store the set of goal fact strings (ground facts).
        self.goals = frozenset(task.goals)

        # Keep static facts grouped by predicate for potential use.
        self.static = frozenset(task.static)
        self.static_by_pred = {}
        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            args = tuple(parts[1:])
            self.static_by_pred.setdefault(pred, set()).add(args)

    def __call__(self, node):
        """
        Compute and return the heuristic value for the given search node.

        Returns an integer equal to the number of goal facts that are not
        present in the node's state. This value is 0 iff the state satisfies
        all goal facts.
        """
        state = node.state  # a frozenset of fact strings

        # Quickly compute which goal facts are still missing.
        missing_goals = self.goals - state

        # The heuristic is the count of missing goal facts.
        return len(missing_goals)