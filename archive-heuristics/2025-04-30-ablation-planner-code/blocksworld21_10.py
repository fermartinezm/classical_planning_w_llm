from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of goal predicates that are not satisfied in the
    current state. Each unsatisfied goal predicate contributes 1 to the
    heuristic value.

    # Assumptions
    - The heuristic counts unsatisfied goal predicates of the form (on ?x ?y),
      (on-table ?x), and (clear ?x).
    - The goal state is defined by the conjunction of predicates provided in
      the PDDL problem file's :goal section.
    - This heuristic is non-admissible but efficiently computable. It does not
      consider the dependencies between predicates or the cost of achieving them.

    # Heuristic Initialization
    - The heuristic stores the set of goal predicates from the task definition
      in a set for efficient lookup during heuristic computation.
    - Static facts (from task.static) are not used as the Blocksworld domain
      typically has no static predicates relevant to the goal structure or
      this heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value, `heuristic_value`, to 0.
    2. Retrieve the set of goal predicates, `self.goal_facts`, stored during
       the heuristic's initialization.
    3. Access the current state, `node.state`, which is represented as a
       frozenset of PDDL fact strings.
    4. Iterate through each predicate string in `self.goal_facts`.
    5. For the current goal predicate string, check if it exists within the
       `node.state` frozenset.
    6. If the goal predicate string is NOT found in the `node.state` frozenset,
       it means this goal condition is not satisfied in the current state.
       Increment `heuristic_value` by 1.
    7. After iterating through all goal predicates, the final `heuristic_value`
       represents the total count of unsatisfied goal conditions.
    8. Return `heuristic_value`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal predicates.

        Args:
            task: The planning task object containing goal and static information.
                  task.goals is expected to be an iterable of predicate strings.
                  task.static is expected to be an iterable of static predicate strings.
        """
        # Store the set of goal predicates for quick lookup.
        # Convert the iterable task.goals into a set.
        self.goal_facts = set(task.goals)

        # Blocksworld domain typically has no static facts relevant to the goal structure.
        # We don't need to process task.static for this heuristic.
        # static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        The heuristic value is the count of goal predicates from the task's
        goal set that are not present in the current state.

        Args:
            node: The search node containing the current state.
                  node.state is a frozenset of PDDL fact strings.

        Returns:
            An integer representing the estimated cost (number of unsatisfied
            goal predicates) to reach the goal state.
        """
        state = node.state  # The current state as a frozenset of facts (strings).

        heuristic_value = 0

        # Count how many goal predicates are not in the current state.
        # Checking membership in a frozenset is efficient (average O(1)).
        for goal_fact in self.goal_facts:
            if goal_fact not in state:
                heuristic_value += 1

        return heuristic_value
