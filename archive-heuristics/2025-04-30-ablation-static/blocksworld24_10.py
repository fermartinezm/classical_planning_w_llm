from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting:
    1. Goal 'on' predicates that are not satisfied in the current state.
    2. Goal 'on-table' predicates that are not satisfied in the current state.
    3. Current 'on' predicates that are *not* goal 'on' predicates (representing incorrect stacks that need to be dismantled).

    # Assumptions
    - Standard Blocksworld domain rules apply (single arm, blocks on table or other blocks).
    - The heuristic focuses on achieving the correct 'on' and 'on-table' relationships. 'clear' and 'arm-empty' goals are implicitly handled as preconditions/effects of actions needed to satisfy 'on'/'on-table' goals.

    # Heuristic Initialization
    - The heuristic extracts all goal predicates of the form `(on ?x ?y)` and `(on-table ?x)` from the task definition and stores them for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the heuristic value `h` to 0.
    2. Identify all `(on ?x ?y)` and `(on-table ?x)` facts that are true in the current state.
    3. Iterate through the goal `(on ?x ?y)` predicates stored during initialization:
       - If a goal `(on x y)` is *not* true in the current state, increment `h` by 1. This counts a required stacking relationship that is missing.
    4. Iterate through the goal `(on-table ?x)` predicates stored during initialization:
       - If a goal `(on-table z)` is *not* true in the current state, increment `h` by 1. This counts a required table placement that is missing.
    5. Iterate through the current `(on ?x ?y)` facts identified in step 2:
       - If a current `(on x y)` fact is *not* one of the goal `(on x y)` predicates, increment `h` by 1. This counts an existing stacking relationship that is incorrect and must be undone (unstacked).
    6. The total value of `h` is the heuristic estimate for the state.

    This heuristic is non-admissible as it sums unsatisfied goal conditions and incorrect state conditions, potentially overestimating the minimum number of actions. It is 0 if and only if the state is a goal state (all required 'on'/'on-table' facts are true, and no incorrect 'on' facts exist).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.

        @param task: The planning task object.
        """
        self.goals = task.goals

        # Extract goal 'on' and 'on-table' predicates
        self.goal_on = set()
        self.goal_on_table = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                self.goal_on.add((parts[1], parts[2]))
            elif parts[0] == 'on-table':
                self.goal_on_table.add(parts[1])
            # Ignore 'clear' goals for this heuristic calculation

        # Static facts are not used in this heuristic
        # static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: The estimated cost to reach the goal.
        """
        state = node.state

        # Extract current 'on' and 'on-table' predicates from the state
        current_on = set()
        current_on_table = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                current_on.add((parts[1], parts[2]))
            elif parts[0] == 'on-table':
                current_on_table.add(parts[1])
            # Ignore other predicates like 'clear', 'arm-empty', 'holding'

        h = 0

        # Part 1: Count unsatisfied goal 'on' predicates
        for goal_pair in self.goal_on:
            if goal_pair not in current_on:
                h += 1

        # Part 2: Count unsatisfied goal 'on-table' predicates
        for goal_block in self.goal_on_table:
            if goal_block not in current_on_table:
                h += 1

        # Part 3: Count current 'on' predicates that are not goal 'on' predicates
        # These represent incorrect stacks that need to be undone.
        for current_pair in current_on:
            if current_pair not in self.goal_on:
                h += 1

        return h

