from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_stack_height_above(block, current_block_on_top_map):
    """
    Calculates the number of blocks currently stacked directly or indirectly on the given block.
    """
    height = 0
    current = current_block_on_top_map.get(block)
    while current is not None:
        height += 1
        current = current_block_on_top_map.get(current)
    return height

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the costs associated with:
    1. Blocks that are not in their goal position (requiring pickup/unstack and putdown/stack).
    2. Clearing stacks of blocks that are currently on top of blocks needed for goal placements
       or needed for picking up/unstacking blocks.
    3. Clearing the arm if the goal requires it to be empty.

    # Assumptions
    - Standard Blocksworld rules apply.
    - Each action (pickup, putdown, stack, unstack) has a cost of 1.
    - The cost to clear a stack of height H above a block is approximately 2*H actions
      (unstacking each block and putting it down somewhere).

    # Heuristic Initialization
    - Parses the goal state to identify the desired support for each block
      (either another block or the table) and whether the arm must be empty.
    - Identifies all blocks mentioned in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Parse Current State:**
        - Identify all blocks present in the current state and the goal state.
        - Determine the current support for each block (on another block or on the table).
        - Build a map `current_block_on_top` indicating which block is directly on top of another.
        - Determine which block (if any) is being held.
        - Determine which blocks are currently clear based on explicit `(clear X)` facts and the `(holding X)` fact.

    2.  **Parse Goal State:** (Done in `__init__`)
        - `goal_support`: Map from block to its desired support (block or 'table').
        - `goal_arm_empty`: Boolean indicating if `(arm-empty)` is a goal fact.

    3.  **Calculate Base Cost from Misplaced Blocks:**
        - Initialize `cost = 0`.
        - Initialize sets `blocks_needing_pickup` and `supports_needing_clear`.
        - Iterate through each block that has a specified goal location in `goal_support`.
        - If a block's current location is different from its goal location:
            - Add 1 to `cost` (representing the final stack/putdown action).
            - If the block is not currently held, add it to `blocks_needing_pickup`
              (representing the need for a pickup/unstack action).
            - If the block's goal location is another block (not 'table'), add the goal
              support block to `supports_needing_clear` (as it must be clear to stack upon).

    4.  **Add Cost for Pickup Actions and Clearing Blocks to be Picked Up:**
        - For each block in `blocks_needing_pickup`:
            - Add 1 to `cost` (representing the pickup/unstack action).
            - To pickup/unstack, the block must be clear. If it's not clear, calculate the height
              of the stack above it using `current_block_on_top` and add `2 * height` to `cost`
              (cost to clear the stack).

    5.  **Add Cost for Clearing Support Blocks:**
        - For each block in `supports_needing_clear`:
            - The support block must be clear to stack on it. If it's not clear, calculate the height
              of the stack above it using `current_block_on_top` and add `2 * height` to `cost`
              (cost to clear the stack).

    6.  **Add Cost for Clearing the Arm:**
        - If the goal requires the arm to be empty (`goal_arm_empty` is True) and the
          arm is currently holding a block (`is_holding` is not None), add 1 to `cost`
          (representing the putdown action).

    7.  **Return Total Cost:** The accumulated `cost` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Parse goal facts to build goal_support map and check for arm-empty goal
        self.goal_support = {}
        self.goal_arm_empty = False
        self.all_blocks_in_goals = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]
            if pred == 'on' and len(parts) == 3:
                on_block, under_block = parts[1:]
                self.goal_support[on_block] = under_block
                self.all_blocks_in_goals.add(on_block)
                self.all_blocks_in_goals.add(under_block)
            elif pred == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
                self.all_blocks_in_goals.add(block)
            elif pred == 'clear' and len(parts) == 2:
                 self.all_blocks_in_goals.add(parts[1])
            elif pred == 'holding' and len(parts) == 2:
                 self.all_blocks_in_goals.add(parts[1])
            elif pred == 'arm-empty' and len(parts) == 1:
                self.goal_arm_empty = True


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Step 1: Parse Current State ---
        current_support = {}
        current_block_on_top = {} # Map: block -> block_on_top_of_it
        is_holding = None
        is_clear = set() # Blocks explicitly stated as clear in the state
        all_blocks = set(self.all_blocks_in_goals) # Start with blocks from goals

        # First pass: Identify all blocks in the current state
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 all_blocks.update(parts[1:])

        # Second pass: Process state facts to build maps and is_clear set
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]
            if pred == 'on' and len(parts) == 3:
                on_block, under_block = parts[1:]
                current_support[on_block] = under_block
                current_block_on_top[under_block] = on_block
            elif pred == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif pred == 'holding' and len(parts) == 2:
                held_block = parts[1]
                is_holding = held_block
            elif pred == 'clear' and len(parts) == 2:
                block = parts[1]
                is_clear.add(block) # Add blocks explicitly stated as clear

        # A held block is never clear, even if explicitly stated as clear in state
        if is_holding is not None and is_holding in is_clear:
             is_clear.remove(is_holding)


        # --- Step 3: Calculate Base Cost from Misplaced Blocks ---
        cost = 0
        blocks_needing_pickup = set()
        supports_needing_clear = set()

        # Consider only blocks that have a defined goal location
        for block_in_goal, goal_loc in self.goal_support.items():
            current_loc = current_support.get(block_in_goal)

            if current_loc != goal_loc:
                # Block is not in its goal location
                cost += 1 # Cost for final placement (stack/putdown)

                # It needs to be picked up/unstacked
                if is_holding != block_in_goal:
                     blocks_needing_pickup.add(block_in_goal)

                # If it needs to be stacked on another block, that block must be clear
                if goal_loc != 'table':
                    supports_needing_clear.add(goal_loc)

        # --- Step 4: Add Cost for Pickup Actions and Clearing Blocks to be Picked Up ---
        for block_to_get in blocks_needing_pickup:
            cost += 1 # Cost for pickup/unstack action

            # To pickup/unstack, the block must be clear. If not, clear the stack above it.
            if block_to_get not in is_clear:
                 height_above = get_stack_height_above(block_to_get, current_block_on_top)
                 cost += 2 * height_above # Cost to clear the stack above it

        # --- Step 5: Add Cost for Clearing Support Blocks ---
        for support_block in supports_needing_clear:
            # The support block must be clear to stack on it. If not, clear the stack above it.
            if support_block not in is_clear:
                 height_above = get_stack_height_above(support_block, current_block_on_top)
                 cost += 2 * height_above # Cost to clear the stack above it

        # --- Step 6: Add Cost for Clearing the Arm ---
        if self.goal_arm_empty and is_holding is not None:
            cost += 1 # Cost for putdown action to empty the arm

        # --- Step 7: Return Total Cost ---
        return cost
