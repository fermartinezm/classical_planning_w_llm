# from fnmatch import fnmatch # Not needed with get_parts
from heuristics.heuristic_base import Heuristic # Assuming this exists

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()

# Helper function to check if a block is in its goal position
def is_in_goal_position(block, state, goal_below):
    """
    Checks if a block is currently in its desired goal position relative to its base.
    Assumes block is a goal block (present in goal_below).
    """
    goal_base = goal_below.get(block) # Should exist for goal blocks

    if goal_base == 'table':
        # Check if '(on-table block)' is in state
        return f'(on-table {block})' in state
    else:
        # Check if '(on block goal_base)' is in state
        return f'(on {block} {goal_base})' in state

# Helper function to get the block on top of another block
def get_block_on_top(block, state):
    """
    Finds the block directly on top of the given block in the current state.
    Returns the block name or 'clear' if nothing is on top.
    """
    for fact in state:
        parts = get_parts(fact)
        # Check for '(on ?x block)' fact
        if parts[0] == 'on' and len(parts) == 3 and parts[2] == block:
            return parts[1] # Found block ?x on top of 'block'
    # If no 'on' fact has this block as the base, it must be clear
    # (Assuming state is consistent, i.e., if (clear B) is true, no (on X B) is true)
    return 'clear'


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the estimated costs to move blocks that are not in their correct
    goal position and the estimated costs to move correctly-placed blocks that are
    blocking the movement of misplaced blocks below them.

    # Assumptions
    - The goal specifies the desired stack configuration using `on` and `on-table` predicates.
    - Actions cost 1.
    - Moving a block from its current base to its goal base costs approximately 2 actions (pickup/unstack + stack/putdown), assuming destinations are clearable.
    - Moving a correctly-placed block out of the way costs approximately 2 actions (unstack + putdown).
    - The arm being empty is implicitly handled as a prerequisite for actions, but its state doesn't directly add to the heuristic unless it's holding a block that needs to be moved.

    # Heuristic Initialization
    - Parses the goal facts to determine the desired base (block or table) for each block mentioned in the goal. This mapping is stored in `self.goal_below`.

    # Step-By-Step Thinking for Computing Heuristic
    1. For the current state, identify the blocks currently on top of other blocks using state facts.
    2. Initialize the heuristic value `h` to 0.
    3. Check if the current state is the goal state. If yes, return 0.
    4. Iterate through each block `B` that is mentioned in the goal predicates (`self.goal_below`).
    5. Check if block `B` is currently in its goal position relative to its base (`is_in_goal_position`).
    6. If `B` is NOT in its goal position:
       - Add 2 to `h` (estimated cost to move `B`: pickup/unstack + stack/putdown).
       - Find the block `C` currently on top of `B` (`get_block_on_top`).
       - If there is a block `C` on top of `B` (i.e., `C` is not 'clear'):
         - Check if `C` is mentioned in the goal predicates (`C` in `self.goal_below`).
         - If `C` is a goal block AND `C` IS in its goal position relative to its base (`is_in_goal_position(C, state, self.goal_below)`):
           - Block `C` is a correctly placed goal block sitting on a misplaced block `B`.
           - `C` must be moved out of the way. Add 2 to `h` (estimated cost for unstack + putdown of `C`).
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal base positions for each block.
        """
        self.goals = task.goals

        # Map block to its goal base (block or 'table')
        self.goal_below = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                obj, underob = parts[1], parts[2]
                self.goal_below[obj] = underob
            elif parts[0] == 'on-table':
                obj = parts[1]
                self.goal_below[obj] = 'table'
            # Ignore 'clear' or 'arm-empty' goals for this heuristic as they don't define block positions

    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # If the goal is already reached, the heuristic is 0.
        # This is crucial for correctness and ensures h=0 only at goal.
        if self.goals <= state: # Using the set subset check from Task class
             return 0

        h = 0

        # Iterate through blocks that are part of the goal configuration
        for block in self.goal_below:
            # Check if block is in its goal position relative to its base
            if not is_in_goal_position(block, state, self.goal_below):
                # Block is not in its goal position. It needs to be moved.
                # Estimated cost for pickup/unstack (1) + stack/putdown (1) = 2 actions.
                h += 2

                # Check for correctly placed blockers on top of this block
                blocker_C = get_block_on_top(block, state)

                # If there is a block C on top of 'block' (i.e., blocker_C is not 'clear')
                if blocker_C != 'clear':
                    # Check if C is a goal block AND C is in its goal position relative to its base
                    # We only care about correctly placed *goal* blocks that are blocking.
                    if blocker_C in self.goal_below and is_in_goal_position(blocker_C, state, self.goal_below):
                         # C is a correctly placed goal block sitting on a misplaced block 'block'.
                         # C must be moved out of the way. Estimated cost: unstack C (1) + putdown C (1) = 2 actions.
                         h += 2

        return h
