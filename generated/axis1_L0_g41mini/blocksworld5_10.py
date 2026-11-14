from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing surrounding parentheses and splitting the string."""
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
    Domain-dependent heuristic for the BlocksWorld domain.

    Summary:
    This heuristic estimates the number of block manipulation actions needed to reach the goal state from the current state.
    It focuses on counting blocks that are incorrectly placed relative to the goal stacking configuration, accounting for
    blocks held by the arm, blocks on the table but misplaced, and misplaced sub-towers. It also considers that the arm can hold
    at most one block, and that each action (pickup, putdown, stack, unstack) counts as one step.

    Assumptions:
    - Blocks can be on the table or stacked on other blocks.
    - Only one block can be held at a time.
    - A block must be clear to be picked up or stacked onto.
    - The goal can specify blocks on blocks, on table, and clear blocks.
    - The arm may already be holding one block.
    - The heuristic does not have to be admissible but aims to guide search efficiently.

    Heuristic Initialization:
    - Extract goal conditions partitioned into:
      - Blocks that must be on table.
      - Blocks that must be on another block.
      - Blocks that must be clear.
    - Extract the full goal "on" relationships in a dictionary for quick lookup:
      goal_on[block] = block it should be on (or None if on table).
    - Extract all blocks mentioned in goals and initial facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. If current state is goal (all goals hold), heuristic = 0.
    2. Identify which block (if any) is held by the arm.
    3. For each block, check if it is placed correctly:
       a) If the block should be on another block in the goal,
          check if currently that condition holds; if not, count at least one move to fix it.
       b) If the block should be on table in the goal,
          check if it's currently on table; if not, count a move to fix it.
    4. Count each misplaced block independently since placing a block requires unstacking and stacking actions.
       We approximate by counting each incorrect placement as 1-2 actions.
    5. Add 1 action if holding a wrong block (may need to put it down).
    6. If a block that should be clear at the goal is not clear, count the required moves to clear it.
       This means we count the number of blocks stacked on top of blocks that should be clear.
    7. Sum all these to get the heuristic estimate.

    This heuristic provides a rough lower bound of moves needed, counting misplaced blocks and the need to clear blocks,
    while being efficient to compute from state and goals.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static  # Not used in this domain but saved for completeness.

        # Parse goal predicates into dictionaries and sets for quick access.
        # goal_on maps each block to the block it should be on or None if on table.
        self.goal_on = {}
        self.goal_clear = set()

        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]
            if pred == "on":
                block, under = parts[1], parts[2]
                self.goal_on[block] = under
            elif pred == "on-table":
                block = parts[1]
                self.goal_on[block] = None  # On table indicated by None
            elif pred == "clear":
                block = parts[1]
                self.goal_clear.add(block)

    def __call__(self, node):
        state = node.state

        # Quick check: If goal already satisfied, heuristic = 0
        if self.goals <= state:
            return 0

        # Maps to quickly get where each block currently is:
        # current_on[block] = block under it OR None if on table
        current_on = {}
        # Tracks which blocks are clear currently
        current_clear = set()
        # Blocks being held by the arm (0 or 1, but must check)
        held_block = None
        # Arm empty flag
        arm_empty = False

        # Extract current relations
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "holding":
                held_block = parts[1]
            elif pred == "arm-empty":
                arm_empty = True
            elif pred == "on":
                block, under = parts[1], parts[2]
                current_on[block] = under
            elif pred == "on-table":
                block = parts[1]
                current_on[block] = None
            elif pred == "clear":
                block = parts[1]
                current_clear.add(block)

        # Collect all blocks known from goal and current_on keys to ensure coverage
        blocks = set(self.goal_on.keys()) | set(current_on.keys())
        if held_block:
            blocks.add(held_block)
        # Clear blocks may or may not be listed in goal_on (e.g. empty blocks)
        blocks |= set(self.goal_clear)
        blocks |= current_clear

        # Heuristic components
        misplaced_count = 0   # Number of blocks incorrectly placed
        blocks_to_clear = 0   # Count needed to clear blocks that should be clear
        holding_penalty = 0   # Penalty if holding a wrongly held block

        # Set for fast membership tests of goal positions
        # For each block in goal_on, get desired support (block or None)
        # We'll check current_on for correctness
        for block in blocks:
            goal_support = self.goal_on.get(block, None)  # Where block should be
            current_support = current_on.get(block, None)

            # If block is held currently:
            if block == held_block:
                # If block is correctly held only if goal requires holding?
                # The domain does not have goals like "holding", so assume holding is never a goal.
                # Hence, holding any block is "wrong" (unless arm-empty goal, which does not constrain holding)
                holding_penalty = 1
                # We still check placement for block when it's put down, so count it as misplaced if not at goal
                # but since the block is held, current_on is None here by definition (not on anything).
                # So count the block as misplaced since it must be put somewhere.
                if goal_support is not None:  # Should be on another block
                    misplaced_count += 1
                elif goal_support is None:  # Should be on table
                    misplaced_count += 1
                continue

            # Block is not held, check placement correctness:
            if current_support != goal_support:
                # Block is misplaced.
                misplaced_count += 1

        # Count how many blocks need to be cleared because they should be clear at goal
        # but currently have blocks on top.
        # To do this, invert current_on to find which blocks have something on top:
        blocks_on_top_of = {under: [] for under in blocks}
        for b, under in current_on.items():
            if under is not None:
                blocks_on_top_of.setdefault(under, []).append(b)

        for block in self.goal_clear:
            # If block is not clear currently, count how many blocks prevent it from being clear
            if block not in current_clear:
                # The blocks on top of this block contribute to penalty:
                # Each block stacked above will eventually have to be moved.
                # But for efficiency, only count 1 for each block that is stacked on block or atop it.
                # Here just count direct blocks on top conservatively.
                obstructing_blocks = blocks_on_top_of.get(block, [])
                blocks_to_clear += len(obstructing_blocks)
                # If no direct info (empty list), blocks_to_clear += 0

        # The heuristic sums:
        # - misplaced_count: to fix wrong placements (each at least 1 action)
        # - blocks_to_clear: moves needed to clear blocks that should be clear (likely multiple actions)
        # - holding_penalty: extra action to put down or stack held block if wrongly held
        #
        # We multiply misplaced_count by 2 to roughly account for unstack + stack actions,
        # add blocks_to_clear as is, and add holding_penalty as 1 (putdown or stack).
        #
        # This is a heuristic, so approximate to encourage progress and pruning.

        heuristic_value = 2 * misplaced_count + blocks_to_clear + holding_penalty

        # At minimum return 1 to avoid zero heuristic for non-goal states without misplaced blocks (unlikely)
        return max(heuristic_value, 1)
