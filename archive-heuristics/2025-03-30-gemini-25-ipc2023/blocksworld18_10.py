import re
from fnmatch import fnmatch
# Assuming Heuristic base class is available at this path
# Make sure this import path is correct for the target environment.
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string like '(on a b)'.

    Args:
        fact: The PDDL fact string.

    Returns:
        A list of strings [predicate, arg1, arg2, ...], or an empty list if parsing fails.

    Raises:
        ValueError: If the fact format is unexpected (e.g., missing parentheses).
    """
    fact = fact.strip()
    if not fact.startswith("(") or not fact.endswith(")"):
        # Raise error for unexpected format, assuming standard PDDL input.
         raise ValueError(f"Fact '{fact}' is not in the expected format '(predicate ...)'")
    # Split the content within parentheses by whitespace
    parts = fact[1:-1].split()
    # Filter out empty strings that might result from multiple spaces
    return [part for part in parts if part]


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts the number of blocks that are not in their final goal position
    (either misplaced relative to the block below them, or blocking other blocks that need moving)
    and assigns a cost of 2 actions per such block (one move to clear/pickup, one move to place correctly).
    It adjusts the cost based on whether a block is currently held by the arm. This heuristic
    is designed for Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - The goal configuration is primarily defined by `(on blockA blockB)` and `(on-table blockC)` predicates.
    - `(clear block)` goals are implicitly satisfied if the blocks are stacked correctly according to the `on` goals.
    - `(arm-empty)` goal predicate is considered for goal completion but not directly in cost calculation,
      except through the adjustment for a held block.
    - Each action (pickup, putdown, stack, unstack) has a uniform cost of 1.
    - Block names do not contain spaces.

    # Heuristic Initialization
    - Parses the goal conditions (`task.goals`) provided during initialization.
    - Builds a `goal_on` dictionary mapping each block to the object it should be resting on
      in the goal state (either another block name or the special string 'TABLE').
    - Stores the complete set of goal facts (`task.goals`) to accurately check if a state is a goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal Achievement:** If the current state (`node.state`) contains all facts
        present in the goal set (`self.goals`), the state is a goal state, and the heuristic value is 0.
    2.  **Parse Current State:** Extract the current world configuration from the set of facts in `node.state`:
        - `current_on`: Dictionary mapping block -> object below (block name or 'TABLE').
        - `current_block_above`: Dictionary mapping block -> block directly on top (or None if clear).
        - `held_block`: The name of the block currently held by the arm, or None if the arm is empty.
    3.  **Identify Misplaced and Interfering Blocks:**
        a.  **Directly Misplaced:** Identify all blocks `b` whose current support object (`current_on.get(b)`)
            is different from their target support object (`self.goal_on.get(b)`). This includes blocks
            that are on the wrong block, on the table when they should be stacked, stacked when they
            should be on the table, or held when they should be placed. Add these blocks to a set
            called `misplaced_and_interfering`.
        b.  **Interfering Blocks:** Identify blocks that are currently stacked above any block already in the
            `misplaced_and_interfering` set. These blocks must be moved to access the misplaced blocks below them.
            Iteratively find all such blocks (directly or indirectly above misplaced ones) and add them
            to the `misplaced_and_interfering` set.
    4.  **Calculate Base Cost:** The initial heuristic estimate is `cost = 2 * len(misplaced_and_interfering)`.
        This estimates two actions per block: one action to pick up or unstack the block, and one action
        to put it down or stack it in its correct final position (or temporarily on the table if it's an interfering block).
    5.  **Adjust for Held Block:**
        - If the arm is currently holding a block (`held_block` is not None):
            - If `held_block` is present in the `misplaced_and_interfering` set: Subtract 1 from the `cost`.
              This adjustment reflects that the first action (pickup/unstack) for this block has effectively
              already been performed.
            - If `held_block` is *not* in the `misplaced_and_interfering` set: Add 1 to the `cost`.
              This accounts for the fact that the held block, although potentially in its final tower already,
              is occupying the arm. The arm needs to be freed (e.g., via `putdown` or `stack` if applicable)
              to manipulate other blocks, thus requiring at least one extra action.
    6.  **Return Cost:** The final calculated value is the heuristic estimate. Ensure the cost is non-negative (it should be by construction, but a `max(0, cost)` safeguard can be added if needed).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goal conditions from the task.

        Args:
            task: The planning task object, containing task.goals, task.static, etc.
        """
        self.goals = task.goals
        self.goal_on = {} # Stores goal position: block -> object_below ('TABLE' or another block)
        # Blocksworld typically doesn't use static facts for this type of heuristic.
        # self.static = task.static

        # Parse goal state to build the target configuration map
        for fact in self.goals:
            try:
                parts = get_parts(fact)
                if not parts: continue # Skip if parsing failed or fact is empty

                predicate = parts[0]
                if predicate == "on" and len(parts) == 3:
                    # Goal is (on blockA blockB)
                    block, below = parts[1], parts[2]
                    self.goal_on[block] = below
                elif predicate == "on-table" and len(parts) == 2:
                    # Goal is (on-table blockC)
                    block = parts[1]
                    self.goal_on[block] = 'TABLE'
                # Other goal predicates like 'clear' or 'arm-empty' are captured in self.goals
                # for the goal check, but not needed for the goal_on map.
            except ValueError as e:
                # Log a warning if a goal fact cannot be parsed.
                # This might indicate an issue with the PDDL file or the parser.
                print(f"Warning: Could not parse goal fact '{fact}'. Error: {e}")
                continue


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.

        Args:
            node: The search node containing the state (node.state) to evaluate.

        Returns:
            An integer estimate of the cost (number of actions) to reach the goal.
        """
        state = node.state

        # Check if the current state satisfies all goal conditions.
        # The '<=' operator checks if self.goals is a subset of state.
        if self.goals <= state:
            return 0

        # Parse the current state to understand the block configuration
        current_on = {} # block -> object_below ('TABLE' or another block)
        current_block_above = {} # block -> block_on_top (None if clear)
        held_block = None

        for fact in state:
            try:
                parts = get_parts(fact)
                if not parts: continue # Skip if parsing failed

                predicate = parts[0]
                if predicate == "on" and len(parts) == 3:
                    # State fact is (on blockA blockB)
                    block, below = parts[1], parts[2]
                    current_on[block] = below
                    current_block_above[below] = block # Record what's on top of 'below'
                elif predicate == "on-table" and len(parts) == 2:
                    # State fact is (on-table blockC)
                    block = parts[1]
                    current_on[block] = 'TABLE'
                elif predicate == "holding" and len(parts) == 2:
                    # State fact is (holding blockD)
                    held_block = parts[1]
                # 'clear' and 'arm-empty' predicates are derived from the above and not explicitly stored here.
            except ValueError as e:
                 # Log a warning if a state fact cannot be parsed.
                print(f"Warning: Could not parse state fact '{fact}'. Error: {e}")
                continue

        # Identify all unique block names involved to iterate over them
        all_blocks = set(self.goal_on.keys()) | set(self.goal_on.values()) | \
                     set(current_on.keys()) | set(current_on.values())
        if held_block:
            all_blocks.add(held_block)
        if 'TABLE' in all_blocks:
            all_blocks.remove('TABLE') # 'TABLE' is a location identifier, not a block object

        # Set to store blocks that are misplaced or interfering with misplaced blocks
        misplaced_and_interfering = set()

        # Phase 1: Identify blocks whose current support does not match the goal support
        for block in all_blocks:
            goal_support = self.goal_on.get(block)
            current_support = current_on.get(block) # This will be None if the block is held

            # A block is considered misplaced if its goal position is defined
            # AND its current position (what it's on, or being held) is different.
            if goal_support is not None and current_support != goal_support:
                 misplaced_and_interfering.add(block)

        # Phase 2: Identify blocks stacked above any misplaced block (these are interfering)
        # Use a list as a queue for breadth-first traversal up the stacks
        processing_queue = list(misplaced_and_interfering)
        # Keep track of blocks already added to the set to avoid redundant checks and infinite loops (though cycles shouldn't occur in valid states)
        processed_for_interference = set(misplaced_and_interfering)

        idx = 0
        while idx < len(processing_queue): # Iterate through the queue
            block_below = processing_queue[idx]
            idx += 1 # Move to the next item index

            # Find the block directly on top of block_below, if any
            block_above = current_block_above.get(block_below)

            # If there is a block above, and it hasn't been marked as misplaced/interfering yet
            if block_above is not None and block_above not in processed_for_interference:
                 misplaced_and_interfering.add(block_above) # Mark it as needing to move
                 processing_queue.append(block_above) # Add it to the queue to check blocks above it
                 processed_for_interference.add(block_above) # Mark it as processed

        # Phase 3: Calculate the base heuristic cost
        # Estimate 2 actions for each block identified: 1 to pick/unstack, 1 to place/putdown
        cost = 2 * len(misplaced_and_interfering)

        # Phase 4: Adjust the cost based on the block currently held by the arm
        if held_block is not None:
            if held_block in misplaced_and_interfering:
                # If the held block is one that needs to be moved anyway,
                # subtract 1 from the cost because the pickup/unstack action is already done.
                cost -= 1
            else:
                # If the held block does NOT need to be moved for its own goal position,
                # it means the arm is occupied when it might need to be free.
                # This block needs to be put down somewhere (e.g., table) to allow other actions. Add 1 cost for this required putdown/stack action.
                cost += 1

        # Ensure the heuristic value is non-negative.
        cost = max(0, cost)

        return cost
