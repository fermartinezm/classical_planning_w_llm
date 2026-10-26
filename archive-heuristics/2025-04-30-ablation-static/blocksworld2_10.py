# from fnmatch import fnmatch # Not used in this version
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    configuration by counting the number of unsatisfied 'on' and 'on-table'
    goal predicates. It assigns a cost of 2 actions (pickup/unstack + stack/putdown)
    for each unsatisfied goal predicate where the target block is not held,
    and a cost of 1 action (stack/putdown) if the target block is already held.

    # Assumptions:
    - The goal state is defined primarily by a set of (on ?x ?y) and (on-table ?x) predicates.
    - (clear ?x) and (arm-empty) goals are typically consequences of achieving the
      main structural goals and are not explicitly counted towards the heuristic value.
    - Moving a block from its current position to its goal position relative to its base
      typically takes two actions: one to pick it up (pickup or unstack) and one to
      place it (stack or putdown).
    - If the block that needs to be moved is already held by the arm, only the
      placement action (stack or putdown) is needed, costing one action.
    - This heuristic is non-admissible as it does not account for the cost of
      clearing blocks that are on top of a block that needs to be moved or on
      top of a target location, nor does it account for the cost of freeing the
      arm if it holds a different block.

    # Heuristic Initialization
    - The heuristic stores the goal predicates from the task. Static facts are
      not used in this specific heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Determine if the arm is currently holding any block, and identify which block it is.
       This requires iterating through the current state facts to find the `(holding ?x)` predicate.
    3. Iterate through each goal predicate specified in the task's goals.
    4. For each goal predicate:
       - Check if the predicate is of the form `(on A B)` or `(on-table A)`. These are the
         predicates defining the desired block structure.
       - If the goal predicate is of the relevant form (`on` or `on-table`):
         - Check if this exact predicate string is present in the current state.
         - If the goal predicate is NOT present in the current state (meaning this part
           of the goal structure is not yet achieved):
           - Identify the block `A` involved in the goal (the block that needs to be
             on `B` or on the table).
           - Compare this block `A` with the block currently held by the arm (if any).
           - If block `A` is the block currently held by the arm:
             - Add 1 to `h`. This estimates that only the final placement action
               (stack or putdown) is needed for this specific goal predicate.
           - If block `A` is NOT the block currently held by the arm (or the arm is empty):
             - Add 2 to `h`. This estimates that both a pickup/unstack action and a
               stack/putdown action will be needed to achieve this specific goal predicate.
       - Ignore any other types of goal predicates (e.g., `(clear ?x)`, `(arm-empty)`),
         as they are treated as derived conditions or less critical for the structural
         heuristic estimate.
    5. After iterating through all goal predicates, the total accumulated value `h`
       is the heuristic estimate for the current state. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal conditions."""
        self.goals = task.goals
        # Static facts are not needed for this heuristic.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Check if any block is currently held by the arm
        held_block = None
        # Iterate through state facts to find the 'holding' predicate
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'holding':
                held_block = parts[1]
                break # Only one block can be held at a time

        # Iterate through goal predicates
        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            # Ensure the fact has parts before checking the predicate
            if not parts:
                continue

            predicate = parts[0]

            # We only care about 'on' and 'on-table' goals for the structural heuristic
            if predicate == 'on':
                # Check if the goal predicate is NOT in the current state
                if goal_fact_str not in state:
                    # Ensure the 'on' predicate has the expected number of arguments
                    if len(parts) == 3:
                        block_A = parts[1] # The block that needs to be on top
                        # Estimate cost based on whether block_A is held
                        if held_block == block_A:
                            h += 1 # Needs stack action
                        else:
                            h += 2 # Needs pickup/unstack + stack actions
            elif predicate == 'on-table':
                 # Check if the goal predicate is NOT in the current state
                 if goal_fact_str not in state:
                    # Ensure the 'on-table' predicate has the expected number of arguments
                    if len(parts) == 2:
                        block_A = parts[1] # The block that needs to be on the table
                        # Estimate cost based on whether block_A is held
                        if held_block == block_A:
                            h += 1 # Needs putdown action
                        else:
                            h += 2 # Needs pickup/unstack + putdown actions
            # Ignore other goal types like 'clear' or 'arm-empty' as they are often consequences

        return h
