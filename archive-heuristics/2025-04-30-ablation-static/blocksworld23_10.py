from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Ensure fact is treated as a string and strip potential whitespace
    fact_str = str(fact).strip()
    # Check if it's a parenthesized fact like '(predicate arg1 ...)'
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    # Based on the provided state example, facts are always '(...)'.
    # If the format is unexpected, splitting might fail or return incorrect parts.
    # For robustness, could add more checks or error handling, but sticking to expected format.
    return fact_str[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the number of blocks that are not in their correct position relative
    to the block below them in the goal stack, the number of blocks that are
    currently on top of any other block, and a penalty if the arm is holding a block.
    This heuristic is designed to be non-admissible and guide a greedy best-first search.

    # Assumptions
    - The goal state is primarily defined by a set of `(on X Y)` and `(on-table Z)`
      predicates, forming one or more stacks on the table.
    - `(clear)` and `(arm-empty)` goals are implicitly handled by the cost components
      related to clearing stacks and arm usage.
    - Each block that is not in its correct goal position relative to the block
      below it adds a base cost of 2 actions (approximating the necessary pickup/unstack
      and stack/putdown operations).
    - Each block currently on top of another block adds a cost of 1 action
      (representing the need to move it out of the way).
    - Having the arm occupied adds a cost of 1 action (representing the need
      to place the held block).

    # Heuristic Initialization
    - Parses the goal conditions to determine the desired block-on-block or
      block-on-table relationships, storing them in `self.goal_on`.
    - Identifies all blocks mentioned in the goal predicates (`self.goal_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine:
       - The block currently below each block (`current_on`), or 'table'.
       - The block currently directly on top of each block (`is_on_top_of`).
       - The block currently held by the arm (`holding`).
       - Collect all unique block names present in the state and goals (`all_blocks`).
    2. Identify blocks that are misplaced relative to their goal position:
       - A block `B` is considered misplaced if it has a specified goal position
         (`B` is a key in `self.goal_on`) AND it is currently held OR its current
         position relative to below (`current_on.get(B)`) is different from its
         goal position (`self.goal_on[B]`).
       - Store these misplaced blocks in a set `misplaced_relative`.
    3. Initialize the heuristic value `h` to 0.
    4. Add cost for misplaced blocks:
       - Iterate through each block `B` that has a specified goal position
         (i.e., `B` is a key in `self.goal_on`).
       - If `B` is in the `misplaced_relative` set: Add 2 to `h`. This base cost
         represents the effort to pick up/unstack `B` and place it in its correct
         relative position.
    5. Add cost for blocks that are on top of other blocks:
       - Iterate through all blocks `B` that are present in the state or goals (`all_blocks`).
       - If there is a block currently stacked directly on top of `B`
         (`is_on_top_of.get(B)` is not None): Add 1 to `h`. This represents the
         action needed to move the block on top out of the way. This counts
         each block that is *not* clear.
    6. Add cost for the arm being occupied:
       - If the arm is currently holding a block (`holding` is not None):
         - Add 1 to `h`. This penalizes having the arm busy, as the held block
           will eventually need to be placed (stack or putdown).
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals
        # Static facts are not needed for this heuristic.
        # task.static

        # Map block to the block it should be on in the goal, or 'table'.
        self.goal_on = {}
        # Keep track of all blocks mentioned in the goal (as either object or subject of on/on-table)
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, below = parts[1], parts[2]
                self.goal_on[block] = below
                self.goal_blocks.add(block)
                self.goal_blocks.add(below)
            elif predicate == "on-table":
                block = parts[1]
                self.goal_on[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore (clear ?) and (arm-empty) goals for the core heuristic structure,
            # their cost is implicitly captured by misplaced/clearing/arm costs.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Parse the current state
        current_on = {} # block -> block_below or 'table'
        is_on_top_of = {} # block_below -> block_on_top (or None)
        holding = None
        all_blocks = set(self.goal_blocks) # Start with blocks from goals

        # First pass state: find holding, add all mentioned objects to all_blocks
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "holding":
                holding = parts[1]
                all_blocks.add(holding)
            elif len(parts) > 1: # Add other objects mentioned in predicates like (on), (on-table), (clear)
                 for part in parts[1:]:
                     all_blocks.add(part)

        # Initialize is_on_top_of for all blocks to None (assume clear initially)
        for block in all_blocks:
            is_on_top_of[block] = None

        # Second pass state: populate current_on, is_on_top_of
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, below = parts[1], parts[2]
                current_on[block] = below
                is_on_top_of[below] = block # This block is on top of 'below'
            elif pred == "on-table":
                block = parts[1]
                current_on[block] = 'table'
            # holding is already found

        # 2. Identify blocks that are misplaced relative to their goal position
        misplaced_relative = set()
        for block in self.goal_on: # Iterate through blocks that are keys in goal_on
            is_held = (holding == block)
            current_pos_of_block = current_on.get(block) # Use .get for held blocks

            # Check if the block is in the wrong place relative to what's below it
            # A held block is considered not to be on its goal_on location.
            if is_held or current_pos_of_block != self.goal_on[block]:
                misplaced_relative.add(block)

        # 3. Initialize heuristic
        h = 0

        # 4. Add cost for misplaced blocks
        for block in self.goal_on: # Iterate through blocks that are keys in goal_on
            if block in misplaced_relative:
                h += 2 # Base cost for moving this block

        # 5. Add cost for blocks that are on top of other blocks
        # Iterate through all blocks that can potentially have something on top
        for block in all_blocks:
             # Check if there is a block directly on top of 'block'
             if is_on_top_of.get(block) is not None:
                 h += 1 # The block on top needs to be moved

        # 6. Add cost for the arm being occupied
        if holding is not None:
             h += 1 # Penalty for having the arm busy

        # 7. Return the total heuristic value
        return h
