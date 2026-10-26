from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
# Assuming heuristic_base.py is available in the 'heuristics' directory
# and defines a base class 'Heuristic' with __init__ and __call__ methods.

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle potential empty fact strings
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to move blocks
    that are not in their goal positions. It counts 2 actions for each block
    that is not currently held and is not in its goal location, and 1 action
     for a block that is held but not in its goal location.

    # Heuristic Initialization
    - Extracts the goal positions for each block from the task goals.
      Only 'on' and 'on-table' goal predicates are considered for goal positions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Goal Positions:** Parse the task goals to create a mapping
        from each block to its desired final location (either another block
        or 'table'). Blocks not explicitly mentioned in 'on' or 'on-table'
        goal predicates are considered to have no specific goal location
        relevant to this heuristic.
    2.  **Identify Current State:** Parse the current state facts to determine
        the current location of each block (on another block, on the table,
        or held by the arm). Also, identify which block, if any, is currently held.
        Collect all blocks present in the state.
    3.  **Calculate Heuristic:** Initialize the heuristic value `h` to 0.
        Iterate through all blocks found in the state:
        -   Get the block's current position and its goal position (if specified).
        -   If the block has a specified goal position and its current position
            does not match the goal position:
            -   If the block is currently held by the arm, it needs at least one
                action (stack or putdown) to reach its goal location. Add 1 to `h`.
            -   If the block is not held, it needs at least two actions (pickup/unstack
                + stack/putdown) to reach its goal location. Add 2 to `h`.
    4.  Return the total calculated value `h`.

    This heuristic is non-admissible because it does not guarantee a lower bound
    on the true cost. It simplifies the problem by ignoring dependencies like
    blocks being in the way of placing another block, except implicitly by
    counting the misplaced blocks themselves. However, it provides a relatively
    efficient and informative estimate for greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for each block.

        Args:
            task: The planning task object containing goals and other information.
        """
        self.goals = task.goals

        # Map goal positions for blocks: block -> target (block or 'table')
        # Only consider 'on' and 'on-table' goal predicates
        self.goal_pos = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                # Goal is (on block1 block2)
                self.goal_pos[parts[1]] = parts[2]
            elif predicate == 'on-table' and len(parts) == 2:
                # Goal is (on-table block)
                self.goal_pos[parts[1]] = 'table'
            # Ignore other goal predicates like (clear X) or (arm-empty) for this heuristic

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions for a given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining cost to reach a goal state.
        """
        state = node.state

        # Identify current positions and held block
        current_pos = {}
        held_block = None
        all_blocks = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                current_pos[parts[1]] = parts[2]
                all_blocks.add(parts[1])
                all_blocks.add(parts[2])
            elif predicate == 'on-table' and len(parts) == 2:
                current_pos[parts[1]] = 'table'
                all_blocks.add(parts[1])
            elif predicate == 'holding' and len(parts) == 2:
                held_block = parts[1]
                current_pos[parts[1]] = 'holding' # Represent being held as a position
                all_blocks.add(parts[1])
            elif predicate == 'clear' and len(parts) == 2:
                 all_blocks.add(parts[1]) # Ensure all blocks are known even if clear on table/another block

        h = 0

        # Iterate through all blocks found in the state
        for block in all_blocks:
            target = self.goal_pos.get(block)
            current = current_pos.get(block)

            # Only consider blocks that have a specified goal position
            if target is not None:
                # If the block is not in its goal position
                if current != target:
                    # Estimate cost to move this block
                    if block == held_block:
                        # If it's the held block, it needs 1 action (putdown or stack)
                        h += 1
                    else:
                        # If it's not held, it needs at least 2 actions (pickup/unstack + putdown/stack)
                        h += 2
            # Note: Blocks present in the state but not in goal_pos are ignored.
            # Their position doesn't matter for goal satisfaction unless they block
            # a goal block, which this heuristic doesn't explicitly penalize.

        return h

# Example usage (assuming a Task object 'task' and a Node object 'node' exist):
# heuristic = blocksworldHeuristic(task)
# h_value = heuristic(node)
# print(f"Heuristic value: {h_value}")
