from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Remove parentheses and split by whitespace
    # Assumes standard PDDL fact format like "(predicate arg1 arg2)"
    # Handle potential empty fact strings
    fact = fact.strip()
    if not fact or fact == '()':
        return []
    # Handle facts without parens if necessary, though example shows parens
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    else:
        # Assuming facts like 'arm-empty' might appear without parens sometimes?
        # Based on example state, this path might not be needed, but defensive.
        return fact.split()


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

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing several components:
    1. The number of blocks that are not on their correct goal base (block or table).
    2. The number of blocks that are currently on top of another block which is itself
       not on its correct goal base.
    3. The number of blocks that are required to be clear in the goal but are not clear
       in the current state.
    4. A penalty if the arm is holding a block.

    This heuristic is non-admissible but aims to guide a greedy search effectively
    by penalizing common forms of "disorder" relative to the goal configuration
    and necessary intermediate conditions (like clearing blocks).

    # Assumptions
    - The goal state specifies the desired configuration of blocks using `on` and
      `on-table` predicates, and potentially `clear` and `arm-empty`.
    - The heuristic focuses on achieving the correct relative positions of blocks
      as defined by the goal `on` and `on-table` predicates, and ensuring blocks
      are clear if required by the goal.
    - Each misplaced block, each block blocking a misplaced block, each block
      that needs to be clear but isn't, and the arm holding a block, represent
      work that needs to be done.

    # Heuristic Initialization
    - Parses the goal conditions to identify the desired base for each block
      (`goal_base` mapping) and the set of blocks that must be clear (`goal_clear`).
    - Static facts are not relevant for this heuristic in Blocksworld.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine:
       - The block directly below each block (`current_on` mapping: block -> block_below).
       - The set of blocks currently on the table (`current_on_table`).
       - The set of blocks that are currently clear (`current_clear`).
       - Whether the arm is holding a block, and if so, which one (`arm_holding`).
    2. Initialize the heuristic value `h = 0`.
    3. Count blocks not on their correct goal base: Iterate through each block `X`
       that is a key in the `goal_base` mapping. Determine its goal base (`goal_base[X]`).
       Determine its current base using `current_on`, `current_on_table`, and `arm_holding`.
       If the current base is known and is different from the goal base, increment `h`.
    4. Count blocks blocking wrongly placed blocks: Iterate through each block `Y`
       that is currently on top of another block `X` (i.e., `current_on[Y] == X`).
       Check if block `X` is in the `goal_base` mapping. If it is, determine its
       current base. If `X` is in `goal_base` and its current base is different
       from its goal base, increment `h` (because `Y` is blocking `X` which is
       misplaced).
    5. Count blocks that need to be clear but aren't: Iterate through each block `X`
       that is in the `goal_clear` set. If `X` is not in the `current_clear` set,
       increment `h`.
    6. Add penalty for holding a block: If the `arm_holding` variable is not None,
       increment `h` by 1.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Extract goal base for blocks in on/on-table goals
        self.goal_base = {}
        # Extract blocks that need to be clear in the goal
        self.goal_clear = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                # (on block1 block2) means block1 should be on block2
                self.goal_base[parts[1]] = parts[2]
            elif predicate == 'on-table' and len(parts) == 2:
                # (on-table block1) means block1 should be on the table
                self.goal_base[parts[1]] = 'table'
            elif predicate == 'clear' and len(parts) == 2:
                # (clear block1) means block1 should be clear
                self.goal_clear.add(parts[1])
            # We ignore (arm-empty) goal for base calculation, handle it separately

        # Static facts are not needed for this heuristic in blocksworld.
        # static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Parse the current state
        current_on = {} # block -> block_below
        current_on_table = set() # blocks on table
        current_clear = set() # clear blocks
        arm_holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                current_on[parts[1]] = parts[2]
            elif predicate == 'on-table' and len(parts) == 2:
                current_on_table.add(parts[1])
            elif predicate == 'clear' and len(parts) == 2:
                current_clear.add(parts[1])
            elif predicate == 'holding' and len(parts) == 2:
                arm_holding = parts[1]
            # arm-empty is implicitly handled if arm_holding is None

        h = 0

        # Helper function to get current base considering all possibilities
        def get_current_base_for_block(block):
             if block in current_on:
                 return current_on[block]
             elif block in current_on_table:
                 return 'table'
             elif arm_holding == block:
                 return 'arm'
             else:
                 # If a block is not in current_on, current_on_table, or held,
                 # it's not described by these facts in the state.
                 # For blocks in goal_base, this means they are missing or location unknown.
                 # We treat this as a mismatch.
                 return None # Location unknown from state facts

        # 3. Count blocks not on their correct goal base
        # We only care about blocks that are part of the goal configuration (in goal_base)
        for block_in_goal, goal_b in self.goal_base.items():
            current_b = get_current_base_for_block(block_in_goal)

            # If the block's location is known and is different from the goal base
            # or if the block's location is unknown (e.g., not in state facts)
            if current_b is None or current_b != goal_b:
                 h += 1

        # 4. Count blocks blocking wrongly placed blocks
        # Iterate through blocks that are currently on top of another block
        for block_on_top, block_below in current_on.items():
            # Check if the block below (block_below) is part of the goal_base
            # and is currently in the wrong position relative to its goal base.
            block_below_is_wrongly_placed = False
            if block_below in self.goal_base:
                goal_b_below = self.goal_base[block_below]
                current_b_below = get_current_base_for_block(block_below)

                if current_b_below is None or current_b_below != goal_b_below:
                    block_below_is_wrongly_placed = True

            if block_below_is_wrongly_placed:
                h += 1 # block_on_top is blocking block_below which is wrongly placed

        # 5. Count blocks that need to be clear but aren't
        for block_needs_clear in self.goal_clear:
            if block_needs_clear not in current_clear:
                h += 1

        # 6. Add penalty for holding a block
        if arm_holding is not None:
            h += 1

        return h
