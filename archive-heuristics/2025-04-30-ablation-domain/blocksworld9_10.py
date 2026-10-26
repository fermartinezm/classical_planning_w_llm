# from fnmatch import fnmatch # Not strictly needed for this heuristic logic
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by summing costs for:
    1. Blocks not on their correct goal base (estimated 2 actions each).
    2. Blocks currently on top of other blocks that should be clear in the goal
       but are not (estimated 2 actions per block to clear).
    3. The arm holding a block (estimated 1 action).

    # Assumptions:
    - The goal specifies the desired base for a subset of blocks (either on another
      block or on the table) and which blocks should be clear.
    - Each action (pickup, putdown, stack, unstack) has a cost of 1.
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - Parse the goal facts to determine the desired base for each block involved
      in the goal configuration (`self.goal_config`) and the set of blocks that
      should be clear (`self.goal_clear`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state to determine:
       - The current base for each block (`current_config`).
       - Which block is directly on top of another block (`current_block_above`).
       - Which blocks are currently clear (`current_clear`).
       - Whether the arm is holding a block (`current_holding`).
    2. Initialize the heuristic value `h` to 0.
    3. **Cost for Misplaced Base:** Identify blocks that are part of the goal
       configuration (`B` in `self.goal_config`) but are currently not on their
       desired base (`current_config.get(B) != self.goal_config[B]`). Add 2 to `h`
       for each such block. This estimates the pickup/unstack and stack/putdown
       actions needed for the block itself.
    4. **Cost for Clearing Unclear Goal Blocks:** Identify blocks `B` that should
       be clear in the goal (`B in self.goal_clear`) but are currently not clear
       (`B not in current_clear`). For each such block `B`, find the stack of
       blocks currently on top of it. Add 2 to `h` for *each* block in these stacks.
       This estimates the unstack/pickup and putdown actions needed to clear `B`.
    5. **Cost for Holding:** If the arm is currently holding a block (`current_holding is not None`),
       add 1 to `h`. This estimates the action needed to free the arm.
    6. Explicitly check if the current state is the goal state. If it is, return 0.
       Otherwise, return the calculated `h`. This ensures the heuristic is 0 only
       at the goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and clear conditions.

        @param task: The planning task object.
        """
        self.goals = task.goals

        # Map blocks to their desired base (block or 'table') based on goal facts.
        self.goal_config = {}
        # Set of blocks that should be clear in the goal.
        self.goal_clear = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                self.goal_config[block] = base
            elif predicate == "on-table":
                block = parts[1]
                self.goal_config[block] = 'table'
            elif predicate == "clear":
                block = parts[1]
                self.goal_clear.add(block)
            # Ignore (arm-empty) goals for config/clear sets

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        @param node: The search node containing the current state.
        @return: The estimated heuristic cost.
        """
        state = node.state

        # Explicitly return 0 if it's a goal state.
        if self.is_goal(state):
             return 0

        # Determine the current state configuration
        current_config = {} # block -> base
        current_block_above = {} # base -> block_on_top (only direct top)
        current_clear = set()
        current_holding = None
        # arm_is_empty = False # Not directly used in calculation

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                current_config[block] = base
                current_block_above[base] = block # Assumes only one block can be directly on top
            elif predicate == "on-table":
                block = parts[1]
                current_config[block] = 'table'
                # current_block_above['table'] is not useful as multiple can be on table
            elif predicate == "holding":
                block = parts[1]
                current_holding = block
            elif predicate == "clear":
                block = parts[1]
                current_clear.add(block)
            # elif predicate == "arm-empty":
            #     arm_is_empty = True # Not directly used

        # Helper to find the stack of blocks above a given base
        def get_stack_above(base, current_block_above_map):
            stack = []
            current = current_block_above_map.get(base)
            while current:
                stack.append(current)
                current = current_block_above_map.get(current)
            return stack

        # Heuristic calculation
        h = 0

        # 1. Cost for blocks not on their goal base (estimated 2 actions each)
        # These blocks need to be moved from their current base to their goal base.
        misplaced_base_count = 0
        for block, goal_base in self.goal_config.items():
            current_base = current_config.get(block)
            if current_base != goal_base:
                misplaced_base_count += 1
        h += 2 * misplaced_base_count

        # 2. Cost for clearing blocks that are on top of blocks that should be clear but aren't
        # Each block in the stack above a block that needs clearing requires unstack/pickup + putdown (2 actions).
        blocks_to_clear_set = set()
        for block_to_be_clear in self.goal_clear:
            # Check if this block is currently NOT clear
            if block_to_be_clear not in current_clear:
                # Find the stack of blocks on top of it
                stack_above = get_stack_above(block_to_be_clear, current_block_above)
                # Add all blocks in the stack to the set of blocks that need clearing
                blocks_to_clear_set.update(stack_above)

        h += 2 * len(blocks_to_clear_set)

        # 3. Cost for holding a block (estimated 1 action)
        # If the arm is holding a block, it needs one action (putdown or stack) to become free.
        if current_holding is not None:
             h += 1

        return h

    def is_goal(self, state):
        """Check if the current state satisfies all goal conditions."""
        # The state is a frozenset of fact strings.
        # The goals are a frozenset of fact strings.
        # Goal is reached if all goal facts are in the state.
        return self.goals <= state
