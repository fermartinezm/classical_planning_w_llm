from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """Parses a fact string like '(predicate arg1 arg2)' into a list ['predicate', 'arg1', 'arg2']."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    The heuristic estimates the cost by counting the number of blocks that are
    not in their correct goal position relative to the block below them (or the table),
    and adds the number of blocks currently stacked on top of these misplaced blocks.
    It also adds a penalty if the arm is holding a block. This captures the idea
    that a misplaced block needs to be moved, any blocks on top of it must be
    moved first, and the arm must be free to perform actions.

    Assumptions:
    - The goal state consists of one or more disjoint stacks of blocks on the table.
    - The goal predicates primarily include (on ?x ?y) and (on-table ?x) to define
      the desired stack structure. (clear ?x) goals for stack tops are implicitly
      handled if the stack structure is correct. (arm-empty) is implicitly handled
      by the arm penalty.
    - All objects mentioned in the goal are present in the initial state.
    - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes the desired position for each block
    by parsing the goal predicates. It creates a dictionary `goal_below` where
    `goal_below[block]` is the block that `block` should be directly on top of
    in the goal state, or the string 'table' if the block should be on the table.
    It also identifies all blocks that are the first argument of an 'on' or 'on-table'
    goal predicate, storing them in `goal_blocks`. These are the blocks whose
    goal position relative to the block below is explicitly defined.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to determine the immediate support for each block
       and which block is immediately on top of another. This creates
       `current_below` (mapping block to block below or 'table' or 'arm') and
       `current_above` (mapping block below to block above). It also identifies
       if the arm is holding a block (`arm_holding`).
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through all blocks in `self.goal_blocks` (blocks whose goal position
       relative to the block below is explicitly defined).
    4. For each such block `b`:
       a. Get its current immediate support (`current_pos = current_below.get(b)`).
       b. Get its goal immediate support (`goal_pos = self.goal_below.get(b)`).
       c. If `current_pos` exists and is not equal to `goal_pos`:
          i. Increment `h` by 1 (representing the cost to move block `b` itself).
          ii. Traverse upwards from block `b` in the current stack using `current_above`. For each block `x` found directly on top of `b` (then on top of `x`, and so on), increment `h` by 1 (representing the cost to move `x` out of the way).
    5. After iterating through all goal blocks, check if the arm is holding any block (`arm_holding is not None`). If so, increment `h` by 1 (representing the cost associated with the arm being busy).
    6. The final value of `h` is the heuristic estimate.
    """
    def __init__(self, task):
        self.goals = task.goals

        # Precompute goal configuration: what should be immediately below each block?
        self.goal_below = {}
        # Collect all blocks whose goal position relative to below is specified
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, below_block = parts[1], parts[2]
                self.goal_below[block] = below_block
                self.goal_blocks.add(block)
            elif predicate == "on-table":
                block = parts[1]
                self.goal_below[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore 'clear' goals for this heuristic calculation

    def __call__(self, node):
        state = node.state

        # Compute current configuration: what is immediately below/above each block?
        current_below = {}
        current_above = {}
        arm_holding = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, below_block = parts[1], parts[2]
                current_below[block] = below_block
                current_above[below_block] = block
            elif predicate == "on-table":
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == "holding":
                block = parts[1]
                current_below[block] = 'arm'
                arm_holding = block
            # Ignore 'clear' and 'arm-empty'

        h = 0

        # Iterate through blocks that have a defined goal position relative to below
        for block in self.goal_blocks:
            current_pos = current_below.get(block)
            goal_pos = self.goal_below.get(block)

            # Check if the block's current immediate support is not its goal immediate support
            # We assume blocks in goal_blocks are always present in the state and thus in current_below
            if current_pos != goal_pos:
                h += 1 # Count the block itself as misplaced

                # Count blocks currently stacked on top of this misplaced block
                block_on_top = current_above.get(block)
                while block_on_top is not None:
                    h += 1 # Count the block on top
                    block_on_top = current_above.get(block_on_top)

        # Add penalty if the arm is holding a block
        if arm_holding is not None:
             h += 1

        return h
