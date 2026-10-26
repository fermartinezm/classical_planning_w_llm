from heuristics.heuristic_base import Heuristic
# No need for fnmatch if we parse facts manually

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact string defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
         return [] # Return empty list for invalid fact strings
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting the number
    of unsatisfied goal conditions related to the position of blocks (i.e., `on`
    and `on-table` predicates) and multiplying this count by 2.

    # Assumptions
    - The primary goal is to achieve a specific configuration of blocks in stacks
      or on the table, defined by `on` and `on-table` predicates.
    - `clear` predicates in the goal are assumed to be satisfied as a side effect
      of achieving the correct stack configuration (i.e., the block is the top
      of its goal stack).
    - Each block that is not in its final goal position needs to be moved at least
      once. Moving a block typically involves an unstack/pickup action and a
      stack/putdown action, hence the multiplication by 2.
    - This heuristic is non-admissible as it doesn't precisely account for the
      cost of clearing blocks that are in the way or dependencies between blocks.

    # Heuristic Initialization
    - Extracts all `on` and `on-table` predicates from the goal state. These
      represent the desired final positions of the blocks.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of goal facts that specify the final position of blocks
       (i.e., `(on ?x ?y)` and `(on-table ?x)`). This is done once during initialization
       by iterating through the task's goal conditions.
    2. For a given state (represented as a frozenset of fact strings), check which
       of these stored goal position facts are currently present in the state.
    3. Count the number of goal position facts that are *not* present in the current state.
       Each such fact represents a block that is not in its desired location relative
       to its support (another block or the table).
    4. The heuristic value is twice this count. This estimates that each block
       involved in an unsatisfied positional goal needs to be moved, costing
       approximately 2 actions (one to pick it up/unstack, one to put it down/stack).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the set of target positional facts
        from the goal state.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goal_position_facts = set()
        for goal in task.goals:
            parts = get_parts(goal)
            # Check if the fact is a positional predicate ('on' or 'on-table')
            if parts and (parts[0] == 'on' or parts[0] == 'on-table'):
                self.goal_position_facts.add(goal)

        # Static facts are not used in this heuristic as block positions are dynamic.
        # static_facts = task.static

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions based on
        unsatisfied positional goals.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated number of actions to reach the goal.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Count the number of goal position facts that are not true in the current state.
        unsatisfied_count = 0
        for goal_fact in self.goal_position_facts:
            if goal_fact not in state:
                unsatisfied_count += 1

        # Estimate the cost: Each unsatisfied positional goal might require moving
        # at least one block, which takes roughly 2 actions (pickup/unstack + stack/putdown).
        # This is a simple, non-admissible estimate designed for greedy search.
        heuristic_value = 2 * unsatisfied_count

        return heuristic_value

