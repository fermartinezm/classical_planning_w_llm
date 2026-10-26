import fnmatch # Not strictly needed for this heuristic, but often useful for PDDL parsing
from heuristics.heuristic_base import Heuristic
# Assuming the Task class definition is available in the execution environment
# from task import Task # Example import path

def get_parts(fact):
    """
    Helper function to parse PDDL fact strings.
    Removes the surrounding parentheses and splits the string by spaces.
    Example: "(on b1 b2)" -> ["on", "b1", "b2"]

    Args:
        fact: A string representing a PDDL fact.

    Returns:
        A list of strings representing the parts of the fact, or an empty list if input is invalid.
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return []

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It works by counting the number of blocks that are not in their final goal
    configuration (position and supporting block/table). Each such "misplaced"
    block is assumed to require two actions (one to pick it up/unstack it, and
    one to put it down/stack it correctly). An adjustment is made if the arm
    is currently holding a misplaced block, reducing the estimate by one.

    # Assumptions
    - The goal state specifies the final position (`on(A,B)` or `on-table(A)`) for every block involved in the goal configuration.
    - Blocks mentioned in the problem but not in a goal `on`/`on-table` fact are handled; if they obstruct goal blocks, they contribute to the heuristic count as misplaced.
    - Goal states typically require the arm to be empty. If a goal requires holding a block, this heuristic might not be perfectly accurate for the final step.
    - All actions (pickup, putdown, stack, unstack) have a cost of 1.
    - The heuristic does not need to be admissible (it can overestimate) and aims for informativeness to guide Greedy Best-First Search.

    # Heuristic Initialization
    - The constructor parses the goal conditions (`task.goals`) to build a representation
      of the target tower configuration(s). This is stored in `self.goal_support`,
      a dictionary mapping each block to the block below it or the special value 'table'.
    - It also identifies all unique block objects involved in the problem by examining
      the initial state and goal facts using relevant predicates (on, on-table, clear, holding).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Determine the current position of each block (on table, on another block, or held by the arm) and whether the arm is holding a block. Store the support structure in a `current_support` dictionary and note the `holding` block.
    2.  **Define Correctness:** A block `b` is considered "correct" if:
        a.  It has a defined goal position in `self.goal_support`.
        b.  Its current position (e.g., `on(b, c)` or `on-table(b)`) matches its goal position.
        c.  AND the block supporting it (`c` or the table) is also "correct". This check is done recursively down to the table (which is always considered correct support).
        d.  A block currently held by the arm cannot be in its final correct position within a tower.
    3.  **Identify Misplaced Blocks:** Iterate through all known block objects. Any block that is not "correct" according to the recursive definition (or is currently held, or has no defined current position, or has no defined goal position) is marked as "misplaced". Use memoization (caching) to avoid redundant correctness checks for blocks lower in a stack.
    4.  **Calculate Base Cost:** The initial heuristic estimate `h` is calculated as `2 * number_of_misplaced_blocks`. This represents the estimated cost of picking up/unstacking and then putting down/stacking each misplaced block to reach its goal state.
    5.  **Adjust for Arm State:** If the arm is currently `holding` a block `x`, and that block `x` has been identified as "misplaced" (which it always will be if held and part of the goal config), reduce the heuristic estimate `h` by 1. This accounts for the fact that the pickup/unstack action for this block has effectively already been performed (or is not needed immediately).
    6.  **Final Value:** The final heuristic value is `max(0, h)`. It returns 0 if all blocks are in their correct goal positions and the arm is empty (consistent with a typical goal state).
    """

    def __init__(self, task):
        """
        Initializes the heuristic.
        - Parses goal conditions to build the target configuration map.
        - Identifies all block objects relevant to the task.

        Args:
            task: The planning task object containing initial state, goals, operators, etc.
        """
        self.goals = task.goals
        self.static = task.static # Usually empty for blocksworld

        # Precompute goal configuration: map block -> support (other block or 'table')
        self.goal_support = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            pred = parts[0]
            if pred == "on" and len(parts) == 3:
                # Goal is (on block_A block_B)
                self.goal_support[parts[1]] = parts[2]
            elif pred == "on-table" and len(parts) == 2:
                # Goal is (on-table block_A)
                self.goal_support[parts[1]] = 'table'
            # We ignore 'clear' predicates in the goal, assuming they are implied
            # by the 'on'/'on-table' structure and necessary for actions.

        # Identify all unique block objects from init and goal states
        self.all_objects = self._get_all_objects(task)


    def _get_all_objects(self, task):
        """
        Extracts all unique object names mentioned in relevant predicates
        from the initial state and goals.

        Args:
            task: The planning task object.

        Returns:
            A set of strings, where each string is the name of a block object.
        """
        objects = set()
        # Predicates involving blocks that define configuration or state
        relevant_preds = {"on", "on-table", "clear", "holding"}
        # Combine initial state and goals to find all mentioned objects
        facts_to_scan = task.initial_state.union(task.goals)

        for fact in facts_to_scan:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            pred = parts[0]
            if pred in relevant_preds:
                # Add all arguments of the predicate as potential objects
                objects.update(parts[1:])
        return objects

    def _is_correct(self, block, current_support, memo):
        """
        Recursively checks if a block is in its correct final position
        with correct support beneath it. Uses memoization.

        Args:
            block: The block object (string name) to check.
            current_support: Dict mapping block -> support ('table' or other block) in the current state.
            memo: Dict used for memoization to store computed correctness status (block -> bool).

        Returns:
            True if the block is correctly positioned with correct support, False otherwise.
        """
        # Base case: The table is always considered correct support.
        if block == 'table':
            return True
        # Return cached result if available
        if block in memo:
            return memo[block]

        # Get the goal position for this block (what should be under it)
        goal_support_for_block = self.goal_support.get(block)
        # Get the current position for this block (what is actually under it)
        current_support_for_block = current_support.get(block)

        # If the block has no defined goal position in self.goal_support,
        # it cannot be considered "correct" in the sense of achieving a goal stack.
        # It might need to be moved out of the way if it obstructs others. Treat as incorrect.
        if goal_support_for_block is None:
             memo[block] = False
             return False

        # If the block's current support doesn't match its goal support, it's incorrect.
        # Also handles the case where the block isn't currently placed (e.g., held, invalid state)
        # because current_support_for_block would be None or different.
        if current_support_for_block != goal_support_for_block:
            memo[block] = False
            return False

        # Current position matches goal position. Now, recursively check the support block/table.
        # The required support is defined by goal_support_for_block.
        support_correct = self._is_correct(goal_support_for_block, current_support, memo)

        # Cache and return the result: correct only if position matches AND support is correct
        memo[block] = support_correct
        return support_correct

    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.

        Args:
            node: The search node containing the state (node.state) to evaluate.

        Returns:
            An integer estimate of the cost (number of actions) to reach the goal
            from the node's state.
        """
        state = node.state

        # 1. Parse current state configuration
        current_support = {}  # block -> block/'table'
        holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            pred = parts[0]
            if pred == "on" and len(parts) == 3:
                current_support[parts[1]] = parts[2]
            elif pred == "on-table" and len(parts) == 2:
                current_support[parts[1]] = 'table'
            elif pred == "holding" and len(parts) == 2:
                holding = parts[1]
            # We don't need to explicitly track 'clear' or 'arm-empty' for this heuristic logic

        # 2. Identify misplaced blocks using the recursive check
        misplaced_count = 0
        memo = {}  # Memoization cache for _is_correct checks {block: bool}

        for block in self.all_objects:
            is_block_correct = False # Assume incorrect initially
            if block == holding:
                # A held block is never in its final tower position.
                is_block_correct = False
                memo[block] = False # Cache result for held block
            elif block not in current_support:
                # Block exists but is not on table, not on another block, and not held.
                # This indicates an unexpected state or an unhandled block. Treat as incorrect.
                # This block needs to be placed somewhere eventually.
                is_block_correct = False
                memo[block] = False
            else:
                # Block is on the table or on another block. Check its correctness recursively.
                is_block_correct = self._is_correct(block, current_support, memo)

            # Increment count if the block is not correct
            if not is_block_correct:
                misplaced_count += 1

        # 3. Calculate base heuristic value
        # Each misplaced block estimated to cost 2 actions (pickup/unstack + putdown/stack)
        h_value = misplaced_count * 2

        # 4. Adjust heuristic if arm is holding a misplaced block
        if holding is not None:
            # If the held block 'holding' is considered misplaced (which it always is
            # according to our definition if it's held and has a goal position),
            # subtract 1 because the pickup/unstack action is effectively already done.
            # We check if it was processed and marked incorrect in the memo.
            if holding in memo and not memo[holding]:
                 h_value -= 1
            # If 'holding' block wasn't in all_objects or goal_support, it still contributes
            # to misplaced_count, and this subtraction correctly accounts for it being held.

        # Ensure heuristic is non-negative
        h_value = max(0, h_value)

        # The heuristic should be 0 for a goal state because misplaced_count will be 0,
        # and goal states typically require arm-empty (so holding is None, no negative adjustment).
        return h_value
