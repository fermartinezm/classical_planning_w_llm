# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic: # Assuming Heuristic base class is available and used elsewhere
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting the number of blocks that are not in their correct goal position
    relative to their immediate base (either another block or the table),
    plus a penalty if the arm needs to be empty but is not.

    # Assumptions
    - The goal state specifies the desired position (on another block or on the table)
      for a subset of blocks.
    - The arm should be empty in the goal state if '(arm-empty)' is a goal fact.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack).

    # Heuristic Initialization
    - Parses the goal facts to determine the desired immediate base for each block
      that is part of the goal configuration.
    - Determines if '(arm-empty)' is a goal fact.
    - Stores this goal information for use during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Determine the current state for each block:
       - If `(on B U)` is true, block B is currently on U.
       - If `(on-table B)` is true, block B is currently on the table.
       - If `(holding B)` is true, block B is currently held by the arm.
    3. Check the state of the robot's arm.
    4. For each block `B` that has a defined goal position (either on another block or on the table):
       - Get the block's current state (on U, on table, or held).
       - Compare the block's current state with its goal position.
       - If the current state does not match the goal position, increment `h`.
    5. If the goal requires the arm to be empty, and the arm is not empty in the current state,
       increment `h` by 1.
    6. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for blocks
        and checking for the arm-empty goal.
        """
        self.goal_pos = {}
        self.goal_arm_empty = False

        # Extract goal positions and arm-empty goal from task.goals
        for goal in task.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, base = parts[1], parts[2]
                self.goal_pos[block] = base
            elif predicate == 'on-table':
                block = parts[1]
                # If a block has both an 'on' goal and an 'on-table' goal,
                # the 'on' goal takes precedence as it defines its base.
                if block not in self.goal_pos:
                    self.goal_pos[block] = 'table'
            elif predicate == 'arm-empty':
                self.goal_arm_empty = True
            # 'clear' goals are typically satisfied if the stack below is correct
            # and the block is the top of its goal stack. We don't explicitly
            # count unsatisfied 'clear' goals in this heuristic.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        current_state_map = {} # Map block -> its current state ('on U', 'on-table', 'holding')
        current_arm_empty = False

        # Determine current state for each block and arm state
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, base = parts[1], parts[2]
                current_state_map[block] = base # Store the base block
            elif predicate == 'on-table':
                block = parts[1]
                current_state_map[block] = 'table' # Store 'table'
            elif predicate == 'holding':
                block = parts[1]
                current_state_map[block] = 'holding' # Store 'holding'
            elif predicate == 'arm-empty':
                current_arm_empty = True

        h = 0
        # Count blocks whose current state does not match their goal position
        for block, goal_base in self.goal_pos.items():
            current_base_or_state = current_state_map.get(block, None) # Get current base or 'holding'

            if current_base_or_state != goal_base:
                 h += 1

        # Add penalty if arm needs to be empty but isn't
        if self.goal_arm_empty and not current_arm_empty:
             h += 1

        return h
