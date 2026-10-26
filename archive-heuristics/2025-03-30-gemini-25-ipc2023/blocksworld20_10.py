import os
import sys
from typing import Set, Dict, Optional

# Assuming the planner infrastructure provides Heuristic in this path
# and the Task representation as shown in the examples.
# If the path is different, adjust the import accordingly.
try:
    from heuristics.heuristic_base import Heuristic
    from planning.strips.representation import Task # Assuming Task class path
except ImportError:
    # Fallback for environments where the exact path might differ
    # This assumes the script is run in a context where these modules are accessible
    # You might need to adjust sys.path or the import statement based on your setup
    # For example, if representation is in the same directory:
    # from representation import Task
    # If heuristic_base is in a parent directory's 'heuristics' folder:
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    # from heuristics.heuristic_base import Heuristic
    # For this example, we'll assume the original import works.
    from heuristics.heuristic_base import Heuristic
    # If Task is needed and not found, define a placeholder or ensure it's available
    # For this heuristic, we only need task.goals from the Task object in __init__
    # and node.state from the Node object in __call__.
    pass


def get_parts(fact: str) -> list[str]:
    """
    Extract the components of a PDDL fact string by removing parentheses
    and splitting by space. Handles potential extra whitespace.

    Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    Example: " ( on   b1  b2 ) " -> ["on", "b1", "b2"]
    """
    # Remove leading/trailing whitespace and the parentheses
    content = fact.strip()[1:-1].strip()
    # Split by whitespace and filter out empty strings
    return [part for part in content.split() if part]


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It primarily counts the number of blocks that are not in their final goal position
    relative to the object (block or table) directly beneath them. It assumes that
    each such misplaced block requires at least two actions (one to pick/unstack, one
    to place/stack). An additional action is counted if the robot arm is currently
    holding a block, as that block needs to be placed.

    # Assumptions
    - The goal state fully specifies the desired final configuration using 'on'
      and 'on-table' predicates. Blocks mentioned in the state but not in the
      goal's 'on'/'on-table' predicates are assumed to need moving out of the way,
      likely ending up on the table if not held.
    - Each action (pickup, putdown, stack, unstack) has a cost of 1.
    - The heuristic does not need to be admissible (can overestimate) but aims for
      informativeness to guide Greedy Best-First Search effectively by providing
      a reasonable estimate of the remaining work.

    # Heuristic Initialization
    - The constructor parses the goal specification (`task.goals`) to build
      data structures representing the target configuration:
        - `goal_on`: A dictionary mapping each block to the block it should be on
                     in the goal state (e.g., `{'b1': 'b2'}` for `(on b1 b2)`).
        - `goal_on_table`: A set of blocks that should be on the table in the
                           goal state (e.g., `{'b2'}` for `(on-table b2)`).
        - `goal_blocks`: A set of all blocks mentioned in the goal configuration's
                         `on` or `on-table` predicates.
    - Static facts (`task.static`) are not used in this heuristic as Blocksworld
      typically doesn't have relevant static information beyond object declarations,
      which are implicitly handled by the task structure and state representation.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check for Goal State:** If the current state already satisfies all goal
        predicates, the heuristic value is 0.
    2.  **Parse Current State:** Extract the current configuration from the `node.state`:
        - `current_on`: Dictionary mapping blocks to the block they are currently on.
        - `current_on_table`: Set of blocks currently on the table.
        - `current_holding`: The block currently held by the arm (or `None`).
        - `current_blocks_in_config`: Set of blocks physically placed (on another
                                      block or on the table).
        - `current_blocks`: Set of all blocks present in the current state (either
                            in the configuration or held by the arm).
    3.  **Identify All Relevant Blocks:** Combine blocks from the current state and
        the pre-parsed goal state to get `all_blocks`. This ensures we consider
        blocks that need moving into place and blocks that might be obstructing.
    4.  **Count Misplaced (Not Held) Blocks:** Iterate through `all_blocks`:
        - If a block `b` is currently held by the arm, skip it in this step (it's
          handled separately).
        - Determine the `current_support` of `b` using the parsed current state
          (the block below it, 'table', or 'unknown' if it's not placed).
        - Determine the `goal_support` of `b` using the pre-parsed goal structures
          (the block that should be below it, 'table', or 'unknown' if not
          specified in goal `on`/`on-table`).
        - If the block `b` is currently placed (`current_support` is not 'unknown')
          AND its current support differs from its goal support
          (`current_support != goal_support`), increment a counter
          `num_blocks_off_goal_support`. This signifies that block `b` needs to be
          moved from its current position.
            - Note: If `goal_support` is 'unknown' (block exists now but not in goal
              structure), it's considered different from any known `current_support`,
              correctly identifying it as needing to be moved.
    5.  **Calculate Base Cost:** The base heuristic estimate `h` is calculated as
        `2 * num_blocks_off_goal_support`. This estimates two actions (one pick/unstack
        and one put/stack) for each misplaced block that is not currently held.
    6.  **Add Cost for Held Block:** If `current_holding` is not `None` (the arm is
        holding a block), increment `h` by 1. This accounts for the single action
        (putdown or stack) needed to place the held block.
    7.  **Return Value:** The final value of `h` (guaranteed non-negative) is the
        heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by parsing the goal state from the task definition.

        Args:
            task: The planning task object, containing goals, initial state, etc.
                  Expected to have a `task.goals` attribute (set or list of strings).
        """
        self.goals: frozenset[str] = task.goals
        # Static facts are not used by this heuristic for Blocksworld
        # self.static: frozenset[str] = task.static

        self.goal_on: Dict[str, str] = {} # block -> block_below
        self.goal_on_table: Set[str] = set() # blocks on table

        for fact in self.goals:
            parts = get_parts(fact)
            if not parts: continue # Skip if parsing results in empty list

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                # Goal fact is (on block_a block_b)
                block_above = parts[1]
                block_below = parts[2]
                self.goal_on[block_above] = block_below
            elif predicate == 'on-table' and len(parts) == 2:
                # Goal fact is (on-table block_a)
                block = parts[1]
                self.goal_on_table.add(block)
            # 'clear' goals are implicitly handled by ensuring blocks are moved
            # off others if needed to achieve the 'on'/'on-table' structure.

        # Store all blocks mentioned in the goal configuration's structure
        self.goal_blocks: Set[str] = set(self.goal_on.keys()) | set(self.goal_on.values()) | self.goal_on_table


    def _get_support(self, block: str, on_map: Dict[str, str], on_table_set: Set[str]) -> str:
        """
        Helper function to find what supports a block in a given configuration.
        Returns the block name below it, the string 'table', or 'unknown' if
        the block is not found in the provided configuration maps.

        Args:
            block: The name of the block to check.
            on_map: Dictionary mapping blocks to the block below them.
            on_table_set: Set of blocks on the table.

        Returns:
            A string representing the support ('block_name', 'table', or 'unknown').
        """
        if block in on_map:
            return on_map[block]
        if block in on_table_set:
            return 'table'
        return 'unknown'

    def __call__(self, node) -> int:
        """
        Compute the heuristic value (estimated cost to goal) for the given state node.

        Args:
            node: The search node containing the state to evaluate.
                  Expected to have a `node.state` attribute (frozenset of strings).

        Returns:
            An integer heuristic value (>= 0).
        """
        state: frozenset[str] = node.state

        # Optimization: Check if goal is already reached
        if self.goals <= state:
            return 0

        h_value: int = 0

        # 1. Parse Current State
        current_on: Dict[str, str] = {}
        current_on_table: Set[str] = set()
        current_holding: Optional[str] = None
        current_blocks_in_config: Set[str] = set() # Blocks physically placed (on/on-table)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_above = parts[1]
                block_below = parts[2]
                current_on[block_above] = block_below
                current_blocks_in_config.add(block_above)
                current_blocks_in_config.add(block_below)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_on_table.add(block)
                current_blocks_in_config.add(block)
            elif predicate == 'holding' and len(parts) == 2:
                current_holding = parts[1]
            # We don't need 'clear' or 'arm-empty' for this heuristic calculation

        # Set of all blocks currently existing (placed or held)
        current_blocks: Set[str] = set(current_blocks_in_config)
        if current_holding:
            current_blocks.add(current_holding)

        # 2. Identify All Relevant Blocks
        # Consider blocks present now and blocks needed for the goal
        all_blocks: Set[str] = current_blocks | self.goal_blocks

        # 3. Count Misplaced (Not Held) Blocks
        num_blocks_off_goal_support: int = 0
        for block in all_blocks:
            if block == current_holding:
                continue # Handle the held block cost separately

            # Find current support (where the block is physically located)
            current_support_val = self._get_support(block, current_on, current_on_table)

            # Find goal support (where the block should be)
            goal_support_val = self._get_support(block, self.goal_on, self.goal_on_table)

            # If the block is currently placed somewhere...
            if current_support_val != 'unknown':
                # ...and its current support is not its goal support...
                if current_support_val != goal_support_val:
                    # ...then it's considered misplaced and needs moving.
                    # This correctly handles blocks currently placed but not in the
                    # goal structure (goal_support_val == 'unknown'), as they
                    # also need moving.
                    num_blocks_off_goal_support += 1

        # 4. Calculate Base Cost (2 actions per misplaced, non-held block)
        h_value = 2 * num_blocks_off_goal_support

        # 5. Add Cost for Held Block (1 action to place it)
        if current_holding is not None:
            h_value += 1

        # 6. Goal State Check (already done at the start)

        # 7. Return Value (ensure non-negative, though logic should guarantee it)
        return max(0, h_value)

