from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Handle simple facts like 'arm-empty'
         return [fact_str]

    # Remove parentheses and split by whitespace
    return fact_str[1:-1].split()


# The match function is not strictly used in the heuristic logic itself,
# but included as it was in the example code structure.
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
    This heuristic estimates the number of actions required to reach the goal
    by counting blocks that are either not in their correct goal position
    relative to the block below them (or the table) or are in the correct
    relative position but are blocked by a wrong block on top. Each such
    "problem" block contributes 1 to the heuristic value.

    # Assumptions
    - The goal state defines a specific configuration of blocks, forming stacks
      or being on the table.
    - The heuristic counts blocks that are "misplaced" or "blocking" others
      from achieving the goal configuration.
    - Each misplaced or blocking block is assumed to require some effort (at
      least one unstack/pickup and one stack/putdown sequence) to fix,
      contributing to the heuristic value.
    - Goal definitions are well-formed, meaning every block mentioned in a goal
      predicate has a defined position relative to the block below it (or the
      table) via 'on' or 'on-table' facts, or is the top of a stack whose base
      is defined.

    # Heuristic Initialization
    - Parses the goal conditions to determine the desired position for each
      block (what it should be directly on, or if it should be on the table).
    - Creates mappings:
        - `goal_under[block]`: The block that `block` should be directly on
          in the goal, or 'table'.
        - `goal_above[block]`: The block that should be directly on `block`
          in the goal, or None if `block` should be clear.
    - Identifies all blocks involved in the goal configuration.
    - Precomputes the goal base for blocks that are the top of a stack in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h = 0`.
    2. Check if the current state is the goal state. If yes, return 0.
    3. Parse the current state to determine the current position of each block:
       - `current_under[block]`: The block `block` is currently directly on,
         or 'table', or 'arm' if held.
       - `current_above[block]`: The block currently directly on `block`, or None.
       - `held_block`: The block currently held by the arm, or None.
    4. Iterate through each block relevant to the goal (identified during initialization):
       - Determine the block's goal base (`goal_base`) using the precomputed mappings.
       - Determine the block's current base (`current_base`) from the parsed state.
       - Determine the block that should be on top in the goal (`goal_top`).
       - Determine the block currently on top (`current_top`).
       - If the block is currently held (`current_base == 'arm'`):
         - It is definitely not in its goal position. Increment `h`.
       - If the block is not held:
         - Check if its current base matches its goal base (`current_base == goal_base`).
         - If `current_base != goal_base`:
           - The block is in the wrong place relative to what's directly below it. Increment `h`.
         - If `current_base == goal_base`:
           - The block is in the correct place relative to what's directly below it.
           - Now check if there is a *wrong* block on top of it that prevents
             building the stack above it.
           - If `current_top` is not None AND (`goal_top` is None
             OR `current_top` != `goal_top`):
             - The block is in place but blocked by a wrong block on top.
             - This wrong block needs to be moved. Increment `h`.
    5. Return the total count `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals
        # static facts are empty in blocksworld, so no need to process task.static

        # Collect all blocks mentioned in goal facts
        self.all_blocks = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Add all arguments except the predicate name
            self.all_blocks.update(parts[1:])
        # Remove 'table' if it was added as an argument (e.g., in (on-table b)).
        self.all_blocks.discard('table')

        # Extract goal configuration: what should each block be on?
        self.goal_under = {} # block -> block_below or 'table'
        self.goal_above = {} # block -> block_on_top or None (if should be clear)

        blocks_explicitly_cleared_in_goal = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block_on_top, block_below = parts[1], parts[2]
                self.goal_under[block_on_top] = block_below
                self.goal_above[block_below] = block_on_top
            elif predicate == "on-table":
                block = parts[1]
                self.goal_under[block] = 'table'
            elif predicate == "clear":
                 block = parts[1]
                 blocks_explicitly_cleared_in_goal.add(block)
            # Ignore arm-empty goal

        # For blocks explicitly marked clear in the goal, ensure goal_above is None
        for block in blocks_explicitly_cleared_in_goal:
             self.goal_above[block] = None

        # Precompute the goal base for blocks that are the top of a stack in the goal.
        # These are blocks in self.all_blocks that are not keys in self.goal_under.
        # Their goal base is the block below them, found via self.goal_above.
        self.goal_base_for_top_block = {}
        for block in self.all_blocks:
            if block not in self.goal_under: # This block is a top block in the goal
                 # Find the block below it in the goal stack
                 # Iterate through goal_above to find which block has this block on top
                 base_found = False
                 for block_below, block_on_top in self.goal_above.items():
                     if block_on_top == block:
                         self.goal_base_for_top_block[block] = block_below
                         base_found = True
                         break
                 # If base_found is still False, it means the block is in all_blocks
                 # but isn't in goal_under and isn't on top of anything in goal_above.
                 # This implies it's an isolated block in the goal. Assume it should be on the table.
                 if not base_found:
                      self.goal_base_for_top_block[block] = 'table' # Fallback assumption


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is already reached
        if self.goals <= state:
             return 0

        # Parse current state to find locations and stacking
        current_under = {} # block -> block_below or 'table' or 'arm'
        current_above = {} # block_below -> block_on_top or None (if clear)
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, under_block = parts[1], parts[2]
                current_under[block] = under_block
                current_above[under_block] = block
            elif predicate == "on-table":
                block = parts[1]
                current_under[block] = 'table'
            elif predicate == "holding":
                block = parts[1]
                held_block = block
                current_under[block] = 'arm' # Indicate it's held
            # 'clear' and 'arm-empty' are state properties, not block locations/stacking directly

        h = 0 # Initialize heuristic value

        # Iterate through all blocks relevant to the goal
        for block in self.all_blocks:
            # Skip 'table' if it somehow got into all_blocks (shouldn't happen with current parsing)
            if block == 'table':
                continue

            current_base = current_under.get(block) # 'table', block_name, 'arm', or None (if block not in state facts)
            goal_top = self.goal_above.get(block) # block_name or None (if should be clear)
            current_top = current_above.get(block) # block_name or None (if clear)

            # Determine the block's goal base
            goal_base = self.goal_under.get(block) # 'table', block_name, or None (if block is a top block in goal_under)

            # If the block is a top block in the goal (not in goal_under), its goal base is the block below it.
            if goal_base is None:
                 goal_base = self.goal_base_for_top_block.get(block)
                 # If goal_base is still None here, it means the block is in all_blocks
                 # but isn't in goal_under and isn't on top of anything in goal_above.
                 # This implies it's an isolated block in the goal.
                 # The fallback 'table' assumption is handled during __init__ precomputation.
                 # So goal_base should be defined here for all blocks in self.all_blocks.
                 # assert goal_base is not None, f"Block {block} in all_blocks has no defined goal base."
                 # The fallback handles this, no assert needed unless debugging goal parsing.
                 pass # goal_base is already set by get() or the fallback lookup


            # Problem 1: Block is not on its correct base (or is held)
            # Check if current_base matches the determined goal_base
            if current_base == 'arm' or current_base != goal_base:
                h += 1
            # Problem 2: Block is on its correct base, but has a wrong block on top
            # This check only applies if Problem 1 is NOT present.
            elif current_top is not None and current_top != goal_top:
                h += 1

        return h
