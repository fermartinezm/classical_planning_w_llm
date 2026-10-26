from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def current_stack_height_above(block, current_above_map):
    """
    Recursively calculate the number of blocks stacked directly on top of the given block
    in the current state using the current_above_map.
    """
    # Find the block directly on top of 'block'
    block_on_top = current_above_map.get(block)

    if block_on_top is None:
        # Nothing is directly on this block, stack height above is 0
        return 0
    else:
        # There is a block on top, add 1 for it and recurse up the stack
        return 1 + current_stack_height_above(block_on_top, current_above_map)


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts blocks that are not on their correct goal parent (or table) and adds
    the estimated cost to clear the stacks above these misplaced blocks and their
    goal parents. It also adds a cost if the arm needs to be empty.

    # Assumptions
    - The goal specifies the desired parent for each block (either another block or the table)
      using `(on B A)` or `(on-table B)` predicates. All blocks in the problem are assumed
      to appear as the first argument of an `on` or `on-table` goal predicate.
    - The goal may also specify `(clear X)` for some blocks and `(arm-empty)`.
    - The cost of moving a block out of the way (unstack/pickup + putdown/stack) is estimated as 2 actions.
    - The base cost for moving a misplaced block to its correct immediate position is estimated as 2 actions.
    - The cost of putting down a held block (to achieve arm-empty) is 1 action.

    # Heuristic Initialization
    - Extracts the goal configuration to build a map from each block to its desired parent
      (block or 'table').
    - Checks if `(arm-empty)` is a goal predicate.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Parse Goal:**
        - Create `goal_parent_map`: Iterate through `task.goals`. For each `(on B A)` goal, store `goal_parent_map[B] = A`. For each `(on-table B)` goal, store `goal_parent_map[B] = 'table'`.
        - Check if `(arm-empty)` is present in `task.goals`.

    2.  **Parse State:**
        - Create `current_below_map`: Maps each block B to the block A it is currently `on` (`(on B A)`), or 'table' if `(on-table B)`.
        - Create `current_above_map`: Maps each block A to the block B currently `on` it (`(on B A)`). This allows finding the block directly on top.
        - Identify if the arm is holding a block (`is_holding`).

    3.  **Compute Heuristic Value:** Initialize `heuristic_value = 0`.

    4.  **Add Cost for Misplaced Blocks and Clearing:**
        - Iterate through each block B that is a key in `goal_parent_map` (i.e., blocks that have a specified goal position).
        - Get B's goal parent (`goal_parent = goal_parent_map[B]`).
        - Determine B's current parent: This is either the block/table from `current_below_map` or 'arm' if held.
        - Initialize `current_below = None`.
        - If `is_holding == block`:
             current_below = 'arm'
        else:
             current_below = current_below_map.get(block)


        - If current_below != goal_parent:
            # Block is misplaced relative to its immediate goal parent
            heuristic_value += 2 # Base cost (pickup + putdown)

            # Add cost to clear the block itself
            # If held, height above is 0. If on table/block, use map.
            if is_holding == block:
                height_b = 0
            else:
                height_b = current_stack_height_above(block, current_above_map)
            heuristic_value += 2 * height_b

            # Add cost to clear the goal parent (if it's a block)
            if goal_parent != 'table':
                height_gp = current_stack_height_above(goal_parent, current_above_map)
                heuristic_value += 2 * height_gp

    5.  **Add Cost for Arm State:**
        - If `self.goal_arm_empty` is True and `is_holding` is not None:
             heuristic_value += 1 # Cost to put down the held block

    6.  **Return Total Heuristic:** Return the final `heuristic_value`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration."""
        self.goals = task.goals
        self.goal_parent_map = {}
        self.goal_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, parent = parts[1], parts[2]
                self.goal_parent_map[block] = parent
            elif parts[0] == "on-table":
                block = parts[1]
                self.goal_parent_map[block] = 'table'
            elif parts[0] == "arm-empty":
                self.goal_arm_empty = True
            # Ignore (clear X) goals for this heuristic's structure

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_below_map = {}
        current_above_map = {}
        is_holding = None

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, parent = parts[1], parts[2]
                current_below_map[block] = parent
                current_above_map[parent] = block # Assumes only one block can be on top
            elif parts[0] == "on-table":
                block = parts[1]
                current_below_map[block] = 'table'
            elif parts[0] == "holding":
                is_holding = parts[1]
            # Ignore (clear X) and (arm-empty) facts here, derive clear from above_map

        heuristic_value = 0

        # Add cost for misplaced blocks and clearing stacks
        for block, goal_parent in self.goal_parent_map.items():
            # Determine current parent: block below, table, or arm if held
            current_below = None
            if is_holding == block:
                 current_below = 'arm'
            else:
                 current_below = current_below_map.get(block)


            if current_below != goal_parent:
                # Block is misplaced relative to its immediate goal parent
                heuristic_value += 2 # Base cost (pickup + putdown)

                # Add cost to clear the block itself
                # If held, height above is 0. If on table/block, use map.
                if is_holding == block:
                    height_b = 0
                else:
                    height_b = current_stack_height_above(block, current_above_map)
                heuristic_value += 2 * height_b

                # Add cost to clear the goal parent (if it's a block)
                if goal_parent != 'table':
                    height_gp = current_stack_height_above(goal_parent, current_above_map)
                    heuristic_value += 2 * height_gp

        # Add cost for arm state if arm-empty is a goal and arm is not empty
        if self.goal_arm_empty and is_holding is not None:
             heuristic_value += 1 # Cost to put down the held block

        return heuristic_value
