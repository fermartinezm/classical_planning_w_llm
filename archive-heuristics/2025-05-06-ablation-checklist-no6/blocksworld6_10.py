from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like '(predicate arg1 arg2)'
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of blocks that are not in their correct goal position
    relative to the block below them, plus a penalty for blocks that should
    be clear but are not, and a penalty if the arm is not empty when it
    should be.

    This heuristic is non-admissible and designed to guide a greedy best-first
    search by prioritizing states where more blocks are in their desired
    relative positions and clear status.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals

        # Extract goal stacking relationships and clear requirements
        self.goal_below = {}
        self.goal_clear_blocks = set()
        self.goal_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                # Goal: (on block support)
                block, support = parts[1], parts[2]
                self.goal_below[block] = support
            elif predicate == 'on-table':
                # Goal: (on-table block)
                block = parts[1]
                self.goal_below[block] = 'table' # Use a special string for the table
            elif predicate == 'clear':
                # Goal: (clear block)
                block = parts[1]
                self.goal_clear_blocks.add(block)
            elif predicate == 'arm-empty':
                # Goal: (arm-empty)
                self.goal_arm_empty = True

        # Static facts are empty in Blocksworld, no need to process task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state
        h = 0

        # Check arm-empty goal
        arm_is_empty_in_state = '(arm-empty)' in state
        if self.goal_arm_empty and not arm_is_empty_in_state:
            h += 1

        # Build current state information about block positions and clear status
        current_below = {}
        current_clear_blocks = set()
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_below[block] = support
            elif predicate == 'on-table':
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == 'clear':
                block = parts[1]
                current_clear_blocks.add(block)
            elif predicate == 'holding':
                current_holding = parts[1] # Assuming only one block can be held at a time

        # Check position and clear status for blocks involved in goal stacks
        # Iterate over blocks that *should* be in a specific position according to the goal
        for block, goal_support in self.goal_below.items():
            is_held = (current_holding == block)
            # Find what is currently below this block. If the block is held, it's not
            # on anything, but the 'is_held' check handles this case. If the block
            # is not in the state at all (shouldn't happen in valid states), get() returns None.
            current_support = current_below.get(block, None)

            needs_to_be_clear_in_goal = (block in self.goal_clear_blocks)
            is_clear_in_state = (block in current_clear_blocks)

            # A block is considered "correctly positioned" for the heuristic if:
            # 1. It is not currently held.
            # 2. It is on the correct block or the table as specified by the goal.
            # 3. If the goal requires this block to be clear, it is currently clear.
            is_correctly_positioned = (
                not is_held and
                current_support == goal_support and
                (not needs_to_be_clear_in_goal or is_clear_in_state)
            )

            # If the block is not correctly positioned, it contributes to the heuristic cost.
            if not is_correctly_positioned:
                h += 1

        return h
