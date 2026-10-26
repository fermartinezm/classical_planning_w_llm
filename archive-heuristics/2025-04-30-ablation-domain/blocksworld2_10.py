from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting the number of blocks that are not on their correct base (the block
    or table they should be directly on in the goal state), plus the number of
    blocks that are currently on top of a block that is itself not on its correct
    goal base. This captures both misplaced blocks and blocks that are blocking
    access to or movement of misplaced blocks below them. It also adds a cost
    if the arm is holding a block and the goal requires the arm to be empty.

    # Assumptions
    - The goal state specifies the desired position (on-table or on another block)
      for all blocks involved in the final configuration.
    - All actions have a cost of 1.
    - The heuristic does not explicitly count actions for achieving `clear` goals
      for blocks that are the top of a goal stack, but the structure of the count
      is intended to correlate with the operations needed.

    # Heuristic Initialization
    - Parses the goal conditions to build `self.goal_base_map`, which maps each
      block to its required base (another block or 'table') in the goal state.
    - Identifies `self.goal_blocks`, the set of all blocks whose position is
      specified in the goal.
    - Checks if `(arm-empty)` is a goal condition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Analyze the current state to determine the immediate base for each block
       (the block it's on, or 'table') and identify if any block is being held.
       Store this in `current_base_map` and `holding`. Also, create `current_on_top_map`
       to quickly find which block is on top of another.
    2. Initialize the heuristic value `h_value` to 0.
    3. Identify blocks that are not on their correct goal base:
       - Iterate through each block `b` in `self.goal_blocks`.
       - Determine `b`'s goal base from `self.goal_base_map`.
       - Determine `b`'s current base from `current_base_map` or check if it's `holding`.
       - If `b` is being held, or its current base is different from its goal base,
         increment `h_value` by 1 and add `b` to a set `misplaced_bases`.
    4. Identify blocks that are blocking misplaced blocks:
       - Iterate through the `current_on_top_map`. For each pair `(base, block_on_top)`:
       - If `base` is in the `misplaced_bases` set (meaning the block below is not
         on its correct goal base), then `block_on_top` needs to be moved to allow
         the `base` block to be repositioned. Increment `h_value` by 1.
    5. Check for `(arm-empty)` goal: If `self.arm_empty_goal` is True and `holding` is not None,
       increment `h_value` by 1. This counts the necessary put-down action.

    This heuristic is non-admissible. It provides a rough estimate by counting
    structural mismatches and necessary clearing operations. It should be 0
    if and only if the state matches the goal configuration captured by the
    heuristic's terms.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals

        # Map block -> block_it_should_be_on (or 'table')
        self.goal_base_map = {}
        # Set of all blocks mentioned in goal positions
        self.goal_blocks = set()
        # Check if (arm-empty) is a goal
        self.arm_empty_goal = False

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty or malformed facts
            if parts[0] == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_base_map[block] = base
                self.goal_blocks.add(block)
                self.goal_blocks.add(base)
            elif parts[0] == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_base_map[block] = 'table'
                self.goal_blocks.add(block)
            elif parts[0] == "arm-empty" and len(parts) == 1:
                 self.arm_empty_goal = True
            # Ignore (clear ?) goals for goal_base_map

        # Remove 'table' from goal_blocks if it was added (it's not a block)
        self.goal_blocks.discard('table')

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        # task = node.task # Access task from node if needed, though __init__ already has it

        # Check if goal is reached (optional but good practice for h=0 iff goal)
        # The search algorithm should handle this before calling the heuristic,
        # but the heuristic value *must* be 0 iff the state is a goal state.
        # The current calculation should satisfy this for standard Blocksworld problems.
        # if self.task.goal_reached(state):
        #     return 0


        # 1. Parse current state
        current_base_map = {} # block -> block_it_is_on (or 'table')
        current_on_top_map = {} # block_below -> block_on_top (assuming only one)
        holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            if parts[0] == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                current_base_map[block] = base
                current_on_top_map[base] = block # Assuming only one block on top
            elif parts[0] == "on-table" and len(parts) == 2:
                block = parts[1]
                current_base_map[block] = 'table'
            elif parts[0] == "holding" and len(parts) == 2:
                holding = parts[1]
            # Ignore (clear ?) and (arm-empty)

        # 2. Initialize heuristic
        h_value = 0
        misplaced_bases = set() # Keep track of blocks whose base is wrong

        # 3. Count blocks not on their goal base (Term 1)
        for block in self.goal_blocks:
            goal_base = self.goal_base_map.get(block)
            # goal_base should exist for all blocks in goal_blocks

            current_base = current_base_map.get(block) # None if held or not found
            is_held = (holding == block)

            if is_held:
                 # If held, it's not on its goal base (unless goal is to hold it, which is not standard)
                 h_value += 1
                 misplaced_bases.add(block)
            elif current_base is not None:
                 if current_base != goal_base:
                     h_value += 1
                     misplaced_bases.add(block)
            else:
                 # Block is in goal_blocks but not in current_base_map and not held. Inconsistent state?
                 # Treat as misplaced.
                 h_value += 1
                 misplaced_bases.add(block)


        # 4. Count blocks blocking misplaced blocks (Term 2)
        # Iterate through blocks that are currently bases (i.e., have something on them)
        for base, block_on_top in current_on_top_map.items():
             # If the block below (the base) is misplaced relative to its goal base
             if base in misplaced_bases:
                 # The block on top needs to be moved to clear the base
                 h_value += 1

        # 5. Check for (arm-empty) goal (Term 3)
        if self.arm_empty_goal and holding is not None:
             h_value += 1 # Add cost for putting down the held block

        # The heuristic value is guaranteed to be 0 only if:
        # 1. No block in goal_blocks is misplaced relative to its base (Term 1 = 0).
        # 2. No block is on top of a block that is misplaced relative to its base (Term 2 = 0).
        # 3. Arm is empty if (arm-empty) is a goal (Term 3 = 0).
        # This structure implies the goal configuration is achieved for standard problems.

        return h_value
