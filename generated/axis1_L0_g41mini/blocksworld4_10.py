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

    Summary
    -------
    This heuristic estimates the number of actions needed to transform the current state
    into the goal state by counting the discrepancies in the positions of blocks compared
    to the goal configuration. It focuses on the blocks that are out of place and
    accounts for actions required to unstack misplaced blocks, pick them up, and stack 
    them correctly on the goal-specified supports or put them down on the table. 

    Assumptions
    -----------
    - The planner has one arm and can hold at most one block at a time.
    - Blocks are moved only by sequences of unstack, pickup, putdown, and stack actions.
    - The heuristic does not need to be admissible but should guide the search efficiently.
    - Goal facts specify desired positions of blocks on other blocks or on the table, and their clear-ness.
    - Holding or arm-empty facts do not appear in the goal (typical in blocksworld).

    Heuristic Initialization
    ------------------------
    The constructor extracts the goal relations into two dictionaries:
    - `goal_on`: maps each block to which block it should be on (or None if on table).
    - `goal_clear`: set of blocks that must be clear in the goal.
    
    This representation allows fast lookups of goal relations during heuristic computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state satisfies the goals, return 0.
    
    2. Extract from the current state the following relations:
       - `current_on`: maps each block to the block it is currently on, or None if on table.
       - `current_clear`: set of blocks that are currently clear.
       - `holding_block`: which block (if any) is currently held by the arm.
       - `arm_empty`: whether the arm is empty.

    3. The heuristic will count the minimal approximate number of block moves needed.
    
       The main idea:
       - For each block, if its current support differs from its goal support, 
         consider that it needs to be moved.
       - Moving a block typically costs at least one pickup (or unstack) and one putdown (or stack).
       - Blocks stacked incorrectly above a block (blocking it) will cause additional moves.

    4. Calculation details:
       - Identify blocks misplaced relative to the goal `goal_on` relation.
       - For each block out of place, count:
         a) One action to pick it up (unstack or pickup),
         b) One action to put it down in the correct place (stack or putdown).
       - Blocks currently held count as already picked up (no pick cost).
       - If the arm is not empty and holding a wrong block, extra putdown or stack may be needed.
       - Misplaced clear-ness facts are implicitly accounted by block moves.
       
    5. Additionally, stacking order matters:
       - To move a block to the goal support, the support must be clear.
       - If the support is currently not clear, the blocking blocks need to be moved first.
       - The heuristic approximates this by adding moves for blocks stacked above their goal positions.

    6. The heuristic sums these costs for all blocks to estimate minimal actions required.

    Note: The heuristic is not admissible but designed to provide informative guidance for greedy best-first search.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal relations for quick comparisons."""
        self.goals = task.goals
        static_facts = task.static  # Not used here as blocksworld domain has no static facts.

        # Parse goal 'on' relations and 'on-table' relations:
        # Map block -> support block or None (if on table)
        self.goal_on = {}
        self.goal_clear = set()

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, support = parts[1], parts[2]
                self.goal_on[block] = support
            elif pred == "on-table":
                block = parts[1]
                self.goal_on[block] = None
            elif pred == "clear":
                block = parts[1]
                self.goal_clear.add(block)
        # Collect all blocks mentioned in the goal for convenience
        self.goal_blocks = set(self.goal_on.keys())

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract current 'on' relations: block -> support block or None (if on table)
        current_on = {}
        current_clear = set()
        holding_block = None
        arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, support = parts[1], parts[2]
                current_on[block] = support
            elif pred == "on-table":
                block = parts[1]
                current_on[block] = None
            elif pred == "clear":
                current_clear.add(parts[1])
            elif pred == "holding":
                holding_block = parts[1]
            elif pred == "arm-empty":
                arm_empty = True

        # Blocks appearing in current_on only:
        blocks = set(current_on.keys())

        # Heuristic calculation:
        # We estimate the number of moves required to get all blocks onto their goal support.

        # Count misplaced blocks: those that are not on their goal support.
        misplaced_blocks = set()
        for block in blocks:
            goal_support = self.goal_on.get(block)
            current_support = current_on.get(block)
            if goal_support != current_support:
                misplaced_blocks.add(block)

        # If a block is holding_block, it is already picked up
        # so the pickup cost for that block is zero.

        # Check if arm is holding a block that is not supposed to be held in goal (typically it is not).
        # If holding a block unnecessarily, it requires at least one putdown action.

        # Count pickup/unstack and putdown/stack actions:
        # Estimate:
        # - For each misplaced block:
        #     - If block is currently held, no pickup action needed.
        #     - Else, count 1 pickup/unstack.
        # - For each misplaced block:
        #     - Count 1 putdown/stack action.

        # Additionally, blocks stacked on a wrong block may require moves:
        # We also add penalty for blocks that are supporting misplaced blocks incorrectly.
        # For simplicity, count how many blocks are above their goal location wrongly.

        # To approximate some of the ordering constraints, we compute a "stack depth penalty":
        # Count the number of blocks above misplaced blocks that must be moved to free them.

        # Build inverse current 'on' mapping: support -> set of blocks on it
        support_to_blocks = {}
        for b, s in current_on.items():
            support_to_blocks.setdefault(s, set()).add(b)

        # Recursively find blocks above each block. We count for misplaced blocks how many blocks above must be moved.
        def count_blocks_above(block):
            above_blocks = support_to_blocks.get(block, set())
            count = len(above_blocks)
            for ab in above_blocks:
                count += count_blocks_above(ab)
            return count

        # Count blocks above misplaced blocks to approximate blocking moves
        blocking_moves = 0
        counted = set()
        for b in misplaced_blocks:
            # Count blocks above b only once per block
            blocks_above = count_blocks_above(b)
            # It may overestimate because some blocks could be misplaced themselves,
            # but provides a guiding estimate.
            blocking_moves += blocks_above

        # Compute number of pickup actions (or unstack):
        # If currently holding a misplaced block, its pickup is free (already held)
        # But if holding block is not misplaced, it costs at least one putdown.
        pickups = 0
        putdowns = 0

        # Putdowns needed for all misplaced blocks (each misplaced block must be put down or stacked correct)
        putdowns = len(misplaced_blocks)

        if holding_block is None:
            # Arm empty, so all misplaced blocks require pickup
            pickups = len(misplaced_blocks)
        else:
            # Holding some block
            if holding_block in misplaced_blocks:
                # Already picked up, no pickup cost for it
                pickups = len(misplaced_blocks) - 1
            else:
                # Holding block that should not be held -> add one putdown for it
                # plus pickup for all misplaced blocks
                putdowns += 1
                pickups = len(misplaced_blocks)

        # Total estimated cost: pickups + putdowns + blocking moves
        # This sums actions: pickups/unstack + putdowns/stack + extra moves to free blocks blocking misplaced blocks
        heuristic_value = pickups + putdowns + blocking_moves

        return heuristic_value
