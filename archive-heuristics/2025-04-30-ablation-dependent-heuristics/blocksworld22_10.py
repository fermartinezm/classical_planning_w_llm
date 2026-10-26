class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the distance to the goal by counting the number
        of blocks that are part of a goal stack but are not currently in their
        correct position relative to the base of that goal stack. It uses a
        recursive definition of "correctly placed" to capture dependencies:
        a block is correctly placed only if the block it should be on is also
        correctly placed (or is the table). Additionally, it adds a penalty
        if the goal requires the arm to be empty but it is currently holding
        a block.

    Assumptions:
        - The input task is a valid Blocksworld task.
        - Goal states define complete stacks down to the table using (on X Y)
          and (on-table B) facts for the stack bases.
        - The state representation is a frozenset of strings as described.

    Heuristic Initialization:
        The constructor processes the goal facts from the task. It builds a
        dictionary `self.goal_pos` mapping each block that should be on
        another block or the table to its desired target (block name or 'table').
        It also checks if `(arm-empty)` is a goal fact.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state. If yes, the heuristic
           value is 0.
        2. Parse the current state facts to determine the current position of
           each block (on another block, on the table, or being held) and
           whether the arm is empty. Store this in a `current_pos` dictionary
           mapping blocks to what they are on ('table' or another block name).
           Also store the block being held in `current_holding`.
        3. Initialize a memoization dictionary `memo` to store the results of
           the `_is_correctly_placed` helper function to avoid redundant computations
           in the recursive calls.
        4. Define a recursive helper function `_is_correctly_placed(block, current_pos, goal_pos, current_holding, memo)`:
           - If the result for `block` is already in `memo`, return the stored value.
           - If `block` is not a key in `goal_pos`, it means this block's position
             is not constrained by the goal stacks. It is considered correctly placed
             relative to a goal stack base (as it's not part of one). Store True in memo and return True.
           - Get the `desired_target` for `block` from `goal_pos`.
           - Get the `current_target` for `block`. If `current_holding` is `block`,
             the `current_target` is 'holding'. Otherwise, look up the block in
             `current_pos` (it will be 'table' or another block name, or None if
             the block is not on anything and not held, which indicates an invalid state).
           - If `current_target` does not match `desired_target`, the block is not
             in its immediate goal position. Store False in memo and return False.
           - If `desired_target` is 'table', the block is correctly placed on the table.
             Store True in memo and return True.
           - If `desired_target` is another block, recursively call
             `_is_correctly_placed` for the `desired_target`. The result for the
             current `block` is the same as the result for its `desired_target`.
             Store the result in memo and return it.
        5. Initialize the heuristic `count` to 0.
        6. Iterate through all blocks that are keys in `self.goal_pos` (i.e., blocks
           that are part of a goal stack and have a defined goal position).
        7. For each such block, call `_is_correctly_placed` with the current state
           information and the memoization dictionary.
        8. If `_is_correctly_placed` returns False for a block, increment the `count`.
        9. After checking all blocks in goal stacks, check if `(arm-empty)` is a goal
           and the arm is currently holding a block (`current_holding` is not None).
           If both are true, increment the `count` by 1.
        10. Return the final `count` as the heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing the goal state.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.goal_pos = {}
        self.goal_arm_empty = False

        # Parse goal facts to build goal_pos and check for arm-empty goal
        for fact in task.goals:
            if fact == '(arm-empty)':
                self.goal_arm_empty = True
            elif fact.startswith('(on '):
                parts = fact[len('(on '):-1].split()
                if len(parts) == 2:
                    block, target = parts[0], parts[1]
                    self.goal_pos[block] = target
            elif fact.startswith('(on-table '):
                block = fact[len('(on-table '):-1]
                self.goal_pos[block] = 'table'

        # Static facts are ignored as per domain definition (empty)
        # self.static_info = task.static # Not used in this heuristic

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The heuristic value (integer).
        """
        # If the goal is reached, the heuristic is 0
        if self.task.goal_reached(state):
            return 0

        # Parse current state facts
        current_pos = {}
        current_holding = None
        # current_arm_empty = False # Not needed for heuristic calculation directly

        for fact in state:
            # if fact == '(arm-empty)':
            #     current_arm_empty = True # Not needed
            if fact.startswith('(on '):
                parts = fact[len('(on '):-1].split()
                if len(parts) == 2:
                    block, target = parts[0], parts[1]
                    current_pos[block] = target
            elif fact.startswith('(on-table '):
                block = fact[len('(on-table '):-1]
                current_pos[block] = 'table'
            elif fact.startswith('(holding '):
                block = fact[len('(holding '):-1]
                current_holding = block

        # Memoization dictionary for _is_correctly_placed
        memo = {}

        # Helper function to check if a block is correctly placed relative to its goal stack base
        def _is_correctly_placed(block, current_pos, goal_pos, current_holding, memo):
            if block in memo:
                return memo[block]

            # If block is not in goal_pos, its position is not constrained by goal stacks
            # It's "correct" in the sense that it doesn't violate a goal position.
            if block not in goal_pos:
                memo[block] = True
                return True

            desired_target = goal_pos[block]

            # Find current target
            if current_holding == block:
                current_target = 'holding'
            else:
                current_target = current_pos.get(block) # None if block is not on anything or table

            # Check if immediate position matches desired position
            if current_target != desired_target:
                memo[block] = False
                return False

            # If immediate position matches, check the target recursively (unless target is table)
            if desired_target == 'table':
                memo[block] = True
                return True
            else:
                # Desired target is another block. Check if that block is correctly placed.
                # The desired_target must also be in goal_pos or it's a base implicitly on table.
                # The recursive call handles the case where desired_target is not in goal_pos.
                is_target_correct = _is_correctly_placed(desired_target, current_pos, goal_pos, current_holding, memo)
                memo[block] = is_target_correct
                return is_target_correct

        # Calculate heuristic: count blocks in goal stacks that are not correctly placed
        misplaced_count = 0
        for block in self.goal_pos:
            if not _is_correctly_placed(block, current_pos, self.goal_pos, current_holding, memo):
                misplaced_count += 1

        # Add penalty if arm-empty is a goal and arm is not empty
        arm_penalty = 0
        if self.goal_arm_empty and current_holding is not None:
             arm_penalty = 1

        return misplaced_count + arm_penalty
