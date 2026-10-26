from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy base class for standalone testing
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state.
    It counts 2 actions for each block that is not on its correct immediate support
    according to the goal state and is not currently held (representing pickup/unstack
    followed by putdown/stack). It counts 1 action if the block is held but not
    at its goal position (representing putdown/stack).
    Additionally, it adds a penalty of 1 if the arm should be empty in the goal state
    but is not, only if all block positions are otherwise correct.

    # Assumptions
    - The goal state specifies the desired position (on another block or on the table)
      for every block relevant to the goal using 'on' or 'on-table' predicates.
    - The goal state may optionally require the arm to be empty using the 'arm-empty' predicate.
    - Standard Blocksworld goals are used (primarily focusing on block positions).
    - The goal position for a block is never 'held'.

    # Heuristic Initialization
    - Iterates through the goal facts to identify:
        - The desired block immediately below each block ('goal_pos').
        - Whether the 'arm-empty' state is required in the goal ('goal_arm_empty').

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the block position heuristic component (`block_pos_heuristic`) to 0.
    2. Determine the current state of the arm (empty or holding a block) and the
       current position (on another block or on the table) for every block.
    3. For each block X that has a defined goal position (`goal_pos[X]` exists):
       a. Check if block X is currently held.
       b. If block X is held:
          Increment `block_pos_heuristic` by 1 (estimated cost to place it).
       c. If block X is not held:
          i. Find its current position (`current_pos[X]`).
          ii. If block X is found in the current state (either on something or on the table)
              and its `current_pos[X]` is different from its `goal_pos[X]`:
              Increment `block_pos_heuristic` by 2 (estimated cost to unstack/pickup and stack/putdown).
          iii. If block X is in `goal_pos` but is not found in the current state
               (neither held nor in `current_pos`), this indicates an inconsistent state;
               treat as misplaced needing 2 actions.
    4. Initialize the total heuristic value with `block_pos_heuristic`.
    5. Check if the 'arm-empty' state is required in the goal (`self.goal_arm_empty`).
    6. If 'arm-empty' is a goal fact, and the arm is not empty in the current state,
       and the `block_pos_heuristic` is 0 (meaning all block positions are correct),
       increment the total heuristic value by 1.
    7. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and arm-empty goal.
        """
        self.goals = task.goals

        # Store goal positions for each block mentioned in the goal
        # Maps block -> block_below_in_goal or 'table'
        self.goal_pos = {}
        # Check if arm-empty is a goal
        self.goal_arm_empty = False

        for goal in self.goals:
            # Check for arm-empty goal first as it's a simple fact string
            if goal == '(arm-empty)':
                 self.goal_arm_empty = True
                 continue # Move to next goal fact

            parts = get_parts(goal)
            if parts[0] == 'on':
                block, below = parts[1], parts[2]
                self.goal_pos[block] = below
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_pos[block] = 'table'
            # Ignore 'clear' or 'holding' goals for this heuristic version

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current positions and held block
        current_pos = {} # Maps block -> block_below_in_state or 'table'
        current_holding = None # The block being held, or None
        arm_is_empty = '(arm-empty)' in state

        # Populate current_pos and current_holding from state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, below = parts[1], parts[2]
                current_pos[block] = below
            elif parts[0] == 'on-table':
                block = parts[1]
                current_pos[block] = 'table'
            elif parts[0] == 'holding':
                current_holding = parts[1]
            # Ignore 'clear', 'arm-empty'

        block_pos_heuristic = 0

        # Heuristic Component 1: Blocks not on their goal support
        # Iterate through objects that have a defined goal position
        for block, goal_below in self.goal_pos.items():
            if block == current_holding:
                 # Block is held. It's not on its goal support (assuming goal_pos is never 'held').
                 # It needs 1 action (stack/putdown) to be placed.
                 block_pos_heuristic += 1
            elif block in current_pos:
                # Block is on something or on the table
                if current_pos[block] != goal_below:
                    # Block is in the wrong position. Needs pickup/unstack + stack/putdown.
                    block_pos_heuristic += 2
            else:
                 # This case implies the block is not held, not on anything, and not on the table.
                 # This shouldn't happen in a valid Blocksworld state for an object that exists.
                 # If it occurs, the block is certainly not on its goal support.
                 # Treat as misplaced on something/table, needing 2 actions.
                 # This might indicate a malformed state, but we handle it defensively.
                 block_pos_heuristic += 2


        heuristic_value = block_pos_heuristic

        # Heuristic Component 2: Arm-empty goal penalty
        # Add penalty only if block positions are correct AND arm should be empty but isn't
        # This penalty ensures h=0 only at goal states (assuming standard goals).
        if block_pos_heuristic == 0 and self.goal_arm_empty and not arm_is_empty:
             heuristic_value += 1

        return heuristic_value
