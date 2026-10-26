from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Basic check for fact format
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
         return [] # Return empty list for malformed facts
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by counting the number of blocks
    that are not currently in their correct position within the goal stack structure.
    A block is considered correctly placed if it is on the table and the goal requires it to be,
    or if it is on the block it is supposed to be on according to the goal, AND the block
    it is on is itself correctly placed.

    # Assumptions
    - The goal defines a set of desired 'on' and 'on-table' relationships forming stacks.
    - Blocks not explicitly mentioned in 'on' or 'on-table' goal facts are not considered for the heuristic count.
    - The goal stacks are acyclic.

    # Heuristic Initialization
    - Parses the goal conditions to create a mapping from each block to its desired support
      (either another block or the table). This mapping defines the target stack structure.

    # Step-By-Step Thinking for Computing Heuristic
    1. In `__init__`, iterate through the goal facts (`task.goals`).
    2. For each goal fact `(on X Y)`, record that block `X` should be on block `Y`. Store this in `self.goal_pos[X] = Y`.
    3. For each goal fact `(on-table X)`, record that block `X` should be on the table. Store this in `self.goal_pos[X] = 'table'`.
    4. In `__call__`, initialize a counter `misplaced_count` to 0.
    5. Initialize a memoization dictionary `memo` to store results of the `is_correctly_placed` check to avoid redundant computations in the recursive calls.
    6. Define a recursive helper function `is_correctly_placed(block, state, goal_positions, memo_dict)`:
       - If `block` is already in `memo_dict`, return the stored result `memo_dict[block]`.
       - If the `block` is not a key in `goal_positions`, it means its final position is not explicitly specified in the goal. For this heuristic, we consider such blocks "correctly placed" relative to the goal structure we are tracking. Store `True` in `memo_dict[block]` and return `True`.
       - Get the target position `target = goal_positions[block]`.
       - Initialize a boolean variable `is_correct` to `False`.
       - If `target` is the string `'table'`: Check if the string representation of the fact `(on-table block)` is present in the current `state`. Set `is_correct` accordingly.
       - If `target` is another block name: Check if the string representation of the fact `(on block target)` is present in the current `state` AND recursively call `is_correctly_placed(target, state, goal_positions, memo_dict)` to check if the block it's supposed to be on is itself correctly placed. Set `is_correct` accordingly.
       - Store the computed `is_correct` value in `memo_dict[block]` and return it.
    7. Iterate through each `block` that is a key in `self.goal_pos` (i.e., each block whose goal position is specified).
    8. Call the `is_correctly_placed` helper function for the current `block`, passing the state, goal positions map, and memoization dictionary.
    9. If the result of `is_correctly_placed` is `False`, increment `misplaced_count`.
    10. After checking all blocks in `self.goal_pos`, return the final `misplaced_count`.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for blocks.
        """
        self.goals = task.goals
        # Map block -> target_block or 'table' based on goal facts
        self.goal_pos = {}

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip malformed facts if any
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, target = parts[1], parts[2]
                self.goal_pos[block] = target
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_pos[block] = 'table'
            # Ignore other goal predicates like (clear) or (arm-empty)

        # Static facts are empty in blocksworld, no need to process task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Memoization dictionary for the recursive function
        memo = {}

        def is_correctly_placed(block, current_state, goal_positions, memo_dict):
            """
            Recursively check if a block is in its correct goal position relative
            to the stack structure defined by goal_positions.
            """
            if block in memo_dict:
                return memo_dict[block]

            # If the block's goal position is not specified, consider it correctly placed
            # for the purpose of this heuristic count.
            if block not in goal_positions:
                 memo_dict[block] = True
                 return True

            target = goal_positions[block]
            is_correct = False

            if target == 'table':
                # Goal: (on-table block)
                is_correct = f"(on-table {block})" in current_state
            else:
                # Goal: (on block target)
                # Check if (on block target) is true AND target is correctly placed
                is_correct = f"(on {block} {target})" in current_state and \
                             is_correctly_placed(target, current_state, goal_positions, memo_dict)

            memo_dict[block] = is_correct
            return is_correct

        misplaced_count = 0
        # Iterate only over blocks whose goal position is explicitly defined
        for block in self.goal_pos:
            if not is_correctly_placed(block, state, self.goal_pos, memo):
                misplaced_count += 1

        return misplaced_count
