from heuristics.heuristic_base import Heuristic

# Define get_parts helper function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the difficulty of reaching the goal state by counting:
    1. Blocks that are not in their correct position relative to their support as specified in the goal.
    2. Blocks that are currently on top of a goal-relevant block but are not the block that should be there according to the goal (i.e., they are obstructing).
    3. Whether the robot's arm is holding a block.

    # Assumptions
    - The goal specifies the desired position (on another block or on the table) for a subset of blocks using `on` and `on-table` predicates.
    - Blocks not explicitly given a position relative to a support in the goal are assumed to not obstruct goal stacks unless they are on top of a goal-relevant block.
    - The cost of each action is implicitly assumed to be uniform (e.g., 1).

    # Heuristic Initialization
    - Parses the goal facts (`task.goals`) to determine the desired support for each block mentioned as being 'on' something or 'on-table'. This forms `self.goal_config`.
    - Identifies all blocks mentioned in any goal fact (`on`, `on-table`, `clear`) as 'goal-relevant' (`self.all_goal_relevant_blocks`).
    - Creates an inverse mapping (`self.goal_block_on_top`) from a support block to the block that should be directly on top of it according to the goal.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state (`node.state`) to determine the current support for each block (`current_config`) and if the arm is holding a block (`current_holding`).
    2. Initialize heuristic value `h` to 0.
    3. **Component 1 (Misplaced Blocks):** Iterate through each block `b` that is explicitly given a support in the goal (`self.goal_blocks_in_config`). If `b` is not currently in the state's configuration (e.g., it's being held) or its current support (`current_config.get(b)`) is different from its goal support (`self.goal_config[b]`), increment `h` by 1. Using `.get(b)` handles the case where the block is held and not in `current_config`.
    4. **Component 2 (Obstructing Blocks):** Iterate through each block `b` and its current support `y` (`(on b y)` is true in the state, captured in `current_config`). If `y` is a block relevant to the goal (`y` is in `self.all_goal_relevant_blocks`) and `b` is *not* the block that should be directly on top of `y` according to the goal (`self.goal_block_on_top.get(y)`), increment `h` by 1. This counts blocks that are in the way of forming goal stacks or clearing blocks needed for goal stacks.
    5. **Component 3 (Busy Arm):** If the robot's arm is currently holding a block (`current_holding` is not None), increment `h` by 1. This accounts for the action needed to place the held block before other operations can begin.
    6. Return the total heuristic value `h`. The heuristic is designed such that `h` is 0 if and only if the state is the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and relevant blocks.
        """
        # self.task = task # Store task for potential future use, although not strictly needed for this heuristic

        # Map block -> goal_support ('table' or block) for blocks explicitly placed in goal
        self.goal_config = {}
        # Set of blocks that are the 'x' in (on x y) or (on-table x) goal facts
        self.goal_blocks_in_config = set()
        # Set of all blocks mentioned in any goal fact (on x y, on-table x, clear x)
        self.all_goal_relevant_blocks = set()

        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    self.goal_config[block] = support
                    self.goal_blocks_in_config.add(block)
                    self.all_goal_relevant_blocks.add(block)
                    self.all_goal_relevant_blocks.add(support)
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_config[block] = 'table'
                    self.goal_blocks_in_config.add(block)
                    self.all_goal_relevant_blocks.add(block)
            elif predicate == 'clear':
                 if len(parts) == 2:
                    block = parts[1]
                    self.all_goal_relevant_blocks.add(block)
            # Ignore arm-empty goal if present, it's usually a consequence

        # Pre-calculate inverse goal config for component 2 in __call__
        # Maps support -> block that should be on it (if any)
        self.goal_block_on_top = {v: k for k, v in self.goal_config.items() if v != 'table'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        current_config = {} # Maps block -> current_support ('table' or block)
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    current_config[block] = support
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    current_config[block] = 'table'
            elif predicate == 'holding':
                 if len(parts) == 2:
                    current_holding = parts[1]
            # Ignore clear and arm-empty in state parsing for this heuristic

        h = 0

        # Component 1: Blocks not in their goal position relative to their support
        # Count blocks whose desired support in the goal is different from their current support.
        for block in self.goal_blocks_in_config:
            # Check if the block exists in the current state configuration (it might be held)
            # Use .get() to safely check current_config. If block is held, it's not in current_config, get() returns None.
            # If goal_config[block] is 'table' or another block, None != goal_config[block], so it counts as misplaced.
            if current_config.get(block) != self.goal_config[block]:
                 h += 1

        # Component 2: Blocks obstructing a goal-relevant position
        # Count blocks that are currently on top of some block Y, where Y is goal-relevant,
        # and the block currently on top (B) is *not* the block that should be on Y according to the goal.
        for block, support in current_config.items():
            if support != 'table' and support in self.all_goal_relevant_blocks:
                correct_block = self.goal_block_on_top.get(support) # None if support should be clear in goal
                if block != correct_block:
                    # `block` is on `support`, and either `support` should be clear, or `correct_block` should be there. `block` is obstructing.
                    h += 1

        # Component 3: Arm is holding a block
        # If the arm is holding a block, it needs to be put down or stacked, costing at least 1 action.
        if current_holding is not None:
            h += 1

        return h
