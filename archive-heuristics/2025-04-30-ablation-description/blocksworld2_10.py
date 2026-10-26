# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


# class blocksworldHeuristic(Heuristic): # Uncomment and inherit if Heuristic base class is provided
class blocksworldHeuristic:
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost based on how many blocks are not in their
    correct position within the goal stacks, considering blocks that need to be
    removed from on top, and whether the arm needs to be empty.

    # Assumptions:
    - Actions have unit cost.
    - The goal specifies the desired stack configuration for all relevant blocks using `on` and `on-table` predicates.
    - The goal may include `clear` predicates for blocks that need to be clear in the final state.
    - The goal may include `arm-empty`.

    # Heuristic Initialization
    - Extracts the desired base block or 'table' for each block from the goal state's `on` and `on-table` facts, storing this in `self.goal_base`.
    - Identifies the set of blocks that need to be clear in the goal state from the goal's `clear` facts, storing this in `self.goal_clear`.
    - Determines if `(arm-empty)` is a goal fact, storing this in `self.arm_empty_goal`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the desired base (`goal_base`) for each block from the goal facts (`(on X Y)` or `(on-table X)`). This is done during initialization.
    2. Identify the set of blocks that must be clear in the goal (`goal_clear`). This is done during initialization.
    3. Determine if `(arm-empty)` is a goal fact. This is done during initialization.
    4. Determine for each block whether it is currently in its "goal stack position". A block X is in its goal stack position if:
       - Its goal base is 'table' AND it is on the table in the current state.
       - OR its goal base is block Y AND it is on block Y in the current state AND block Y is in its goal stack position.
       This check is done recursively with memoization, considering only blocks that have a defined goal base (`block` in `self.goal_base`).
    5. Initialize the heuristic cost to 0.
    6. For each block X that has a defined goal base (`block` in `self.goal_base`):
       - If X is NOT in its goal stack position (based on step 4):
         - Add 1 to the cost (representing the need to move X).
         - Find the block Y that X is currently on (if any).
         - If X is currently on a block Y, count the number of blocks stacked directly on top of X in the current state and add this count to the cost. This accounts for the need to clear X.
    7. For each block X that must be clear in the goal (`block` in `self.goal_clear`):
       - If `(clear X)` is NOT true in the current state:
         # This block needs to be clear but isn't. Add cost for clearing it.
         # We only add this cost if it wasn't already accounted for in Step 6.
         # A block X is accounted for in Step 6 if X is a key in goal_base AND not correctly stacked.
         is_accounted_in_step6 = (block in self.goal_base) and (not is_correctly_stacked(block))

         if not is_accounted_in_step6:
             # Add cost for clearing this block. This is the number of blocks on top.
             cost += count_blocks_on_top(block)

    8. Check if `(arm-empty)` is a goal fact (`self.arm_empty_goal` is True) AND `(arm-empty)` is NOT true in the current state. If so, add 1 to the cost. This accounts for needing to put down a held block.
    9. The total cost is the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        self.goal_base = {}
        self.goal_clear = set()
        self.arm_empty_goal = False

        # Extract goal_base, goal_clear, and arm_empty_goal from goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_base[block] = base
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_base[block] = 'table'
            elif predicate == 'clear' and len(parts) == 2:
                 block = parts[1]
                 self.goal_clear.add(block)
            elif predicate == 'arm-empty' and len(parts) == 1:
                 self.arm_empty_goal = True
            # Ignore other predicates like 'holding' if they appear in goals (unlikely for static goals)


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        cost = 0

        # --- Helper functions (defined inside __call__ to access state) ---

        # Helper to find current location/holder
        def get_current_loc(block):
            for fact in state:
                parts = get_parts(fact)
                if not parts: continue
                if parts[0] == 'on' and len(parts) == 3 and parts[1] == block:
                    return parts[2] # block is on parts[2]
                elif parts[0] == 'on-table' and len(parts) == 2 and parts[1] == block:
                    return 'table'
                elif parts[0] == 'holding' and len(parts) == 2 and parts[1] == block:
                    return 'arm'
            return None # Block not found on table, on another block, or held. Should not happen for blocks in the problem.

        # Helper to count blocks on top
        def count_blocks_on_top(base_block):
            count = 0
            current = base_block
            while True:
                block_above = None
                for fact in state:
                    parts = get_parts(fact)
                    if not parts: continue
                    if parts[0] == 'on' and len(parts) == 3 and parts[2] == current:
                        block_above = parts[1]
                        break
                if block_above:
                    count += 1
                    current = block_above
                else:
                    break
            return count

        # Helper recursive function with memoization for goal stack position
        memo_correctly_stacked = {}
        def is_correctly_stacked(block):
            """Checks if a block is in its correct goal stack position relative to the base."""
            if block in memo_correctly_stacked:
                return memo_correctly_stacked[block]

            # Only evaluate for blocks that have a defined goal base
            if block not in self.goal_base:
                 # This block is not part of the goal stack structure defined by on/on-table goals.
                 # It cannot be "correctly stacked" within this structure.
                 # We only calculate the "misplaced" cost (Step 6) for blocks *in* self.goal_base.keys().
                 # So returning False here is fine for the recursive check logic.
                 memo_correctly_stacked[block] = False
                 return False

            base = self.goal_base[block]

            if base == 'table':
                result = '(on-table ' + block + ')' in state
            else: # base is a block
                # The base block must also have a defined goal base or be 'table'
                # for the goal structure to be well-defined.
                # If base is a block and not in self.goal_base, something is wrong with the goal.
                # Assuming valid goals where all bases are defined:
                result = ('(on ' + block + ' ' + base + ')' in state) and \
                         is_correctly_stacked(base) # Recursive call

            memo_correctly_stacked[block] = result
            return result


        # --- Step 6: Cost for blocks not on their correct goal base relative to the stack ---
        # Iterate over blocks that are supposed to be on a specific base or table.
        for block in self.goal_base.keys():
             if not is_correctly_stacked(block):
                 cost += 1 # Cost for moving the block itself
                 current_loc = get_current_loc(block)
                 # Only count blocks on top if the block is actually placed on something
                 if current_loc is not None and current_loc != 'table' and current_loc != 'arm':
                     cost += count_blocks_on_top(block)


        # --- Step 7: Cost for blocks that need to be clear but aren't, and not already accounted for ---
        # Iterate over blocks that must be clear in the goal.
        for block in self.goal_clear:
             if '(clear ' + block + ')' not in state:
                 # This block needs to be clear but isn't. Add cost for clearing it.
                 # We only add this cost if it wasn't already accounted for in Step 6.
                 # A block X is accounted for in Step 6 if X is a key in goal_base AND not correctly stacked.
                 is_accounted_in_step6 = (block in self.goal_base) and (not is_correctly_stacked(block))

                 if not is_accounted_in_step6:
                     # Add cost for clearing this block. This is the number of blocks on top.
                     cost += count_blocks_on_top(block)


        # --- Step 8: Cost for arm not being empty if needed ---
        if self.arm_empty_goal and '(arm-empty)' not in state:
             cost += 1 # Need to put down the block

        # Check if goal is reached. If so, cost must be 0.
        # The logic should ensure this.

        return cost
