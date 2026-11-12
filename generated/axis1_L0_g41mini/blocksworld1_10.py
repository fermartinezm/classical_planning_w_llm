from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Summary:
    --------
    This heuristic estimates the number of actions required to reach the goal state 
    from the current state by counting misplaced blocks and penalizing blocking 
    conditions in the stacks. It focuses especially on blocks that are not on their 
    goal positions and considers the order of blocks to approximate the effort needed 
    for rearrangement with pick, unstack, stack, and putdown actions.

    Assumptions:
    ------------
    - The arm can hold exactly one block at a time.
    - Blocks are moved by picking up or unstacking and placing them on the table or 
      on other blocks.
    - The goal specifies which blocks should be on which blocks or on the table, and 
      being clear.
    - The heuristic is not required to be admissible, but should guide the search 
      efficiently.
    - Any held block not in its goal position contributes to the heuristic effort.
    - The planner represents states as frozensets of fact strings exactly as in PDDL.

    Heuristic Initialization:
    -------------------------
    - Extract all blocks mentioned in the task from goal facts by collecting all 
      objects appearing in (on ?x ?y), (on-table ?x), and (clear ?x).
    - Parse goal relations to create mappings:
        - goal_on[block] = what the block should be on (another block or 'table')
        - goal_clear[block] = True if block must be clear in goal
    - Store empty arm requirement (used for checks if needed).
    - Static facts are not used here because the domain is simple and static-free.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Extract the current configuration of blocks:
       - For each block, determine what it is currently on (another block or table).
       - For each block, determine if it is currently clear.
       - Determine if the arm is empty or holding a block, and which block if held.
    2. For each block, compare current position to goal position:
       - If block is not in its goal position, estimate actions needed to move it.
       - Blocks that are "out of place" must be unstacked or picked and stacked appropriately.
    3. Additionally, consider blocking situations:
       - Count every block that stands (directly or indirectly) on a block that is not 
         its target support as additional moves needed to "clear the way."
    4. Estimate handling of the held block if any:
       - If holding a block, one or more actions may be needed to place it correctly.
    5. The cost estimate is a sum of:
       - Number of blocks not on the correct support (each at least 1 unstack/pick + 1 stack/putdown).
       - Additional costs for blocks that block others from reaching their goals.
       - Adjustments for held block situation.
    6. The heuristic returns 0 only if the current state matches the goal exactly.

    The heuristic provides a finite, efficiently computable estimate that encourages 
    reducing misplaced blocks and clearing blocks blocking goal placement, thereby 
    guiding greedy search effectively.
    """

    def __init__(self, task):
        """Initialize the heuristic by parsing the goal and extracting block goals."""
        self.goals = task.goals
        self.static = task.static  # Not used in this heuristic.

        # Identify all blocks by extracting objects from goal predicates of interest.
        blocks = set()
        # goal_on: block -> what it should be on; 'table' if on-table
        self.goal_on = {}
        # goal_clear: block -> whether block must be clear at goal
        self.goal_clear = {}

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on" and len(parts) == 3:
                block, below = parts[1], parts[2]
                blocks.add(block)
                blocks.add(below)
                self.goal_on[block] = below
            elif pred == "on-table" and len(parts) == 2:
                block = parts[1]
                blocks.add(block)
                self.goal_on[block] = "table"
            elif pred == "clear" and len(parts) == 2:
                block = parts[1]
                blocks.add(block)
                self.goal_clear[block] = True
            # other predicates (like holding or arm-empty) are irrelevant here for block positions

        # For blocks with no on goal specified (should not happen in well-formed tasks), assume table.
        for block in blocks:
            if block not in self.goal_on:
                self.goal_on[block] = "table"
            if block not in self.goal_clear:
                self.goal_clear[block] = False

        self.blocks = blocks

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from this state."""

        state = node.state

        # Quick check: if goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Current position maps: block -> what it is on ('table' or another block)
        current_on = {}
        # Clear state of blocks: block -> True/False
        current_clear = {}
        # Set of blocks on table
        blocks_on_table = set()
        # Arm state: empty or holding a specific block
        arm_empty = False
        holding_block = None

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on" and len(parts) == 3:
                block, below = parts[1], parts[2]
                current_on[block] = below
            elif pred == "on-table" and len(parts) == 2:
                block = parts[1]
                current_on[block] = "table"
                blocks_on_table.add(block)
            elif pred == "clear" and len(parts) == 2:
                block = parts[1]
                current_clear[block] = True
            elif pred == "arm-empty":
                arm_empty = True
            elif pred == "holding" and len(parts) == 2:
                holding_block = parts[1]
                arm_empty = False

        # Ensure each block has clear state (default False)
        for b in self.blocks:
            if b not in current_clear:
                current_clear[b] = False

        # Ensure each block has current_on; if missing assume table (to be safe)
        for b in self.blocks:
            if b not in current_on:
                current_on[b] = "table"

        # To estimate moves, we consider:
        #  - Blocks not on their goal target support
        #  - Blocks blocking other blocks (blocks on top of blocks that should be clear)
        #  - Held block disposition

        # Build reverse mapping: support -> set of blocks on it
        # This is needed to assess stacking and blocking relationships
        supports = {}
        for b in self.blocks:
            support = current_on[b]
            if support not in supports:
                supports[support] = set()
            supports[support].add(b)

        # We'll identify misplaced blocks: those not on their goal support
        misplaced_blocks = set(
            b for b in self.blocks if current_on[b] != self.goal_on[b]
        )

        # Count blocks that block goal placement by being on blocks that should be clear at goal
        blocking_count = 0

        # For this, find blocks that stand (directly or indirectly) on a block that must be clear at goal
        # but that block is currently occupied.

        # Approach:
        # For each block that must be clear at goal:
        #   If it has any block on top now, those blocks must be removed, penalize accordingly.

        # We'll count how many blocks are on top of goal-clear blocks
        for b in self.blocks:
            if self.goal_clear.get(b, False):
                # Get blocks currently on this block
                on_top = supports.get(b, set())
                # Blocks on top of this must be moved away before this block can be clear at goal
                blocking_count += len(on_top)

        # Adding penalties for blocks misplaced beneath others:
        # A block can only be placed if the block below it is correctly placed.
        # To approximate, count how many blocks are stacked on blocks that are misplaced themselves.
        # This will estimate the need to rearrange stacks.

        blocking_below_count = 0
        for b in self.blocks:
            below = current_on[b]
            if below != "table" and below in misplaced_blocks:
                blocking_below_count += 1

        # Estimate cost for held block:
        # If holding a block, and that block is misplaced or is held but arm not empty,
        # assume at least one put down or stack action needed.
        hold_cost = 0
        if holding_block is not None:
            if holding_block in misplaced_blocks:
                hold_cost += 1
        else:
            # Arm empty is good: no immediate cost
            pass

        # Estimate cost per misplaced block:
        # Each misplaced block needs at least 2 actions:
        #  - One to pick/unstack it from its current place
        #  - One to putdown or stack it to its goal place

        misplaced_cost = 2 * len(misplaced_blocks)

        # Sum heuristic components:
        # misplaced_cost: base actions per misplaced block
        # blocking_count: moves to clear blocks that block goal-clear blocks
        # blocking_below_count: rearranging stacks with misplaced supports
        # hold_cost: extra cost for held block if misplaced

        heuristic_value = (
            misplaced_cost + blocking_count + blocking_below_count + hold_cost
        )

        # The heuristic should never return zero unless the goal is reached (already handled above)

        return max(1, heuristic_value)
