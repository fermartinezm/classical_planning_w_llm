from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Utility functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class definition
class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting blocks
    that are not in their correct goal position relative to their base (table or another block),
    and adding the number of blocks that are currently stacked on top of these
    misplaced blocks. It also adds a penalty if the arm is holding a block,
    as the arm usually needs to be empty to make progress.

    # Assumptions:
    - The goal specifies the desired base (table or another block) for all blocks
      that are part of the goal configuration using `(on X Y)` or `(on-table X)` facts.
    - `(clear X)` goals are implicitly handled; if a block is correctly stacked
      according to the `on`/`on-table` goals and nothing is on it that shouldn't be,
      it will likely be clear.
    - Standard Blocksworld problems typically require the arm to be empty in the goal state.
      The heuristic includes a penalty for a non-empty arm based on this common pattern.

    # Heuristic Initialization
    - Extracts the desired base (block or 'table') for each block mentioned in the
      '(on X Y)' or '(on-table X)' goal facts. This creates the `goal_base_map`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the actual base (block or 'table')
       for every block that is currently on the table or on another block.
       Store this in `current_base_map`. Also identify which blocks are currently
       on top of other blocks by storing `(on X Y)` facts in `on_facts`.
       Identify if the arm is holding a block.
    2. Identify the set of "misplaced blocks": These are blocks `X` for which a desired
       base is specified in the `goal_base_map`. If the block `X` is currently held,
       or if its current base in the state (`current_base_map.get(X)`) is different
       from the desired base (`goal_base_map[X]`), it is considered misplaced.
       Count the number of such blocks. This is the first component of the heuristic.
    3. Identify the set of "blocker blocks": These are blocks `Y` that are currently
       stacked directly on top of any block `Z` that is in the set of "misplaced blocks".
       Count the number of such blocker blocks. This is the second component of the heuristic.
    4. Check if the robot's arm is holding any block. If it is, add 1 to the heuristic
       value. This accounts for the action needed to free the arm (putdown or stack).
    5. The total heuristic value is the sum of the counts from step 2, step 3, and step 4.
       This estimates the number of blocks that need to be moved (misplaced blocks),
       plus the number of blocks that must be moved *first* because they are on top
       of a misplaced block, plus the cost to free the arm if it's busy.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal base positions.
        """
        self.goals = task.goals

        # Map block -> desired_base ('table' or block_name)
        self.goal_base_map = {}

        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            if not parts: # Handle empty facts if any
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_base_map[block] = base
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_base_map[block] = 'table'
            # Ignore (clear X) goals as they are usually consequences of the stack structure.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Step 1: Parse the current state
        current_base_map = {} # Map block -> current_base ('table' or block_name)
        on_facts = {} # Map block_on -> block_under

        held_block = None
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Handle empty facts
                continue
            predicate = parts[0]
            if predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_base_map[block] = 'table'
            elif predicate == "on" and len(parts) == 3:
                block_on, block_under = parts[1], parts[2]
                current_base_map[block_on] = block_under
                on_facts[block_on] = block_under
            elif predicate == "holding" and len(parts) == 2:
                held_block = parts[1]
                # Held block is not on anything, so it's not added to current_base_map here.

        # Step 2: Identify misplaced blocks relative to their goal base
        misplaced_blocks = set()
        for block, desired_base in self.goal_base_map.items():
            # Check if the block is currently held
            is_held = (held_block == block)

            if is_held:
                 current_base = 'arm' # Explicitly set current base if held
            else:
                 current_base = current_base_map.get(block)

            # If a block is in the goal_base_map but not found in the state's
            # on/on-table/holding facts, it's definitely not in the right place.
            # This indicates an invalid state or a block from the goal not in the initial state.
            # Treat as misplaced.
            if current_base is None and not is_held:
                 misplaced_blocks.add(block)
                 continue # Cannot compare bases if current base is unknown

            if current_base != desired_base:
                misplaced_blocks.add(block)

        # Step 3: Identify blocker blocks (blocks on top of misplaced blocks)
        blocker_blocks = set()
        for block_on, block_under in on_facts.items():
            if block_under in misplaced_blocks:
                blocker_blocks.add(block_on)

        # Step 4: Add penalty for held block
        arm_penalty = 1 if held_block is not None else 0

        # Step 5: Calculate total heuristic value
        total_cost = len(misplaced_blocks) + len(blocker_blocks) + arm_penalty

        # The heuristic is 0 iff the state is the goal state (assuming standard goals).
        # If state is goal:
        # - All (on/on-table) goal facts are true -> misplaced_blocks is empty.
        # - If misplaced_blocks is empty, blocker_blocks is empty.
        # - Standard goals include (arm-empty) -> held_block is None -> arm_penalty is 0.
        # - Total cost = 0.
        # If state is not goal:
        # - Either some (on/on-table) goal fact is false -> misplaced_blocks > 0 -> h > 0.
        # - Or all (on/on-table) goal facts are true, but (arm-empty) is a goal and arm is not empty -> held_block is not None -> arm_penalty > 0 -> h > 0.

        return total_cost
