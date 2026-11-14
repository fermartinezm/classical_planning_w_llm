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
    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state in the Blocksworld domain.
    It measures how many blocks are currently misplaced or need rearrangement relative to the goal configuration,
    and accounts for the arm state (holding or empty) to approximate the remaining effort to stack blocks correctly.

    Assumptions
    -----------
    - We have a single manipulator arm which can hold only one block at a time.
    - Moving a block involves picking it up from either the table or another block and then placing it either on another block or on the table.
    - Blocks that are already on the correct block in the goal pose and have the correct blocks beneath them do not add to the heuristic cost.
    - Clearing blocks that are on top of the desired block is necessary, and each wrongly stacked block on top will roughly account for multiple actions.
    - The heuristic does not need to be admissible but should be efficient and reduce expanded nodes.
    - The goal facts are conjuncts of predicates `(on ?x ?y)` (and possibly `(clear ?z)` and `(on-table ?z)`) defining the goal stacking.

    Heuristic Initialization
    ------------------------
    - Extract the goal on-relations into a dictionary mapping each block to the block/table it should be on.
    - Extract the set of blocks appearing in the goal, to consider all relevant blocks.
    - No static facts are used since Blocksworld is fully dynamic except for object identities.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    1. Parse the current state to:
       - Identify what block (or table) each block is currently on.
       - Identify which blocks are clear.
       - Identify if the arm is holding a block (and which one) or empty.

    2. For each block that appears in the goal:
       - If the block is currently held by the arm:
         - If the arm should be empty in the goal (usually it should be empty):
           - Count the cost for putting down this block on the correct place.
       - Else, if the block is on something else than the goal requires:
         - The block and all blocks above it block progress and need rearrangement.
         - Recursively count the number of misplaced blocks stacked above this block.

    3. Define a helper function to follow a tower of blocks from the current position up to clear block,
       counting how many are out of place compared to the goal stack below them.

    4. Add a cost of roughly 2 actions per misplaced block (unstack + stack), adjusted for blocks already held or clear.

    5. Add 1 if the arm holds a wrong block (to put it down first).

    6. The heuristic is zero only if the current state exactly matches the goal (already checked by planner),
       otherwise a positive integer.

    7. Return the total heuristic cost.

    This heuristic captures both the number of misplaced blocks and the effort necessary to move them,
    guiding the search towards assembling the blocks into the correct final stacks efficiently.
    """

    def __init__(self, task):
        # Extract goal on-relations: dict block -> block/table it should be on
        self.goal_on = {}
        self.blocks_in_goal = set()
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            if pred == "on":
                block, below = parts[1], parts[2]
                self.goal_on[block] = below
                self.blocks_in_goal.add(block)
                self.blocks_in_goal.add(below)
            elif pred == "on-table":
                block = parts[1]
                self.goal_on[block] = "table"
                self.blocks_in_goal.add(block)
            elif pred == "clear":
                # clear goals are less relevant as they hold implicitly by stacking,
                # but we keep track of blocks anyway
                block = parts[1]
                self.blocks_in_goal.add(block)
        # Sometimes the table is not explicitly named in on-table but we use "table" as a marker.

    def __call__(self, node):
        state = node.state

        # Parse current positions:
        # Map block -> what it is on (block or "table")
        current_on = {}
        # Set of clear blocks
        clear_blocks = set()
        # If arm holds a block, store it, else None
        holding_block = None

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, below = parts[1], parts[2]
                current_on[block] = below
            elif pred == "on-table":
                block = parts[1]
                current_on[block] = "table"
            elif pred == "clear":
                block = parts[1]
                clear_blocks.add(block)
            elif pred == "holding":
                holding_block = parts[1]
            elif pred == "arm-empty":
                # arm empty implies holding_block is None
                if holding_block is None:
                    holding_block = None

        # Now we analyze misplaced blocks:
        # We want to measure how many blocks are out of place compared to goal stacks.
        # We will identify for each block if it is correctly placed on the block below it,
        # recursively counting misplaced towers.

        # Helper: count how many blocks are misplaced in the stack starting from block
        def misplaced_stack_height(block):
            """
            Returns 0 if block is correctly placed and the whole tower above it is correct,
            otherwise counts how many blocks from this block upwards are misplaced.

            Misplaced means either:
            - block is not on its goal block,
            - the block above is misplaced recursively.

            If the block is not in the goal_on dict, we consider it misplaced to
            avoid ignoring out-of-goal blocks.

            Stops when block is clear (no block on top).
            """
            if block not in self.blocks_in_goal:
                # Block not in goal, count as 1 misplaced block
                return 1

            goal_below = self.goal_on.get(block, None)
            current_below = current_on.get(block, None)
            if goal_below is None:
                # No info about where this block should be - consider misplaced
                return 1
            if current_below != goal_below:
                # Block is not on the correct support
                return 1

            # We check the block above this block, if any:
            # To find block(s) above this one, find blocks where current_on[other] == block
            blocks_above = [b for b, below in current_on.items() if below == block]
            if len(blocks_above) == 0:
                return 0  # no blocks above, correct position and so tower above correct
            assert len(blocks_above) == 1, "Blocksworld multiple blocks above one block violation"
            return misplaced_stack_height(blocks_above[0])

        total_cost = 0

        # If arm holds a block:
        if holding_block is not None:
            # If holding block is correct to hold in goal? Usually arm should be empty at goal.
            # So consider cost to put it down correctly to contribute to heuristic.
            # Putting held block down and stacking it correctly costs at least one or two actions.
            # We add 1 action as minimal cost for putting down or stacking.
            total_cost += 1

        # For each block in goal, compute misplaced stack height and sum costs:
        # Each misplaced block roughly costs 2 actions: unstack + stack.
        # For simplicity, multiply by 2.
        # To avoid double counting sub-towers, only count blocks whose current support is not correct,
        # or blocks directly above misplaced blocks.

        # Keep track of blocks we have counted to avoid double counting.
        counted_blocks = set()

        for block in self.blocks_in_goal:
            if block in counted_blocks:
                continue

            misplaced_height = misplaced_stack_height(block)
            if misplaced_height > 0:
                # Count action cost: 2 actions per misplaced block
                actions_for_subtower = 2 * misplaced_height

                total_cost += actions_for_subtower

                # Mark all blocks in this misplaced tower as counted to avoid double count
                # We define helper to find all blocks in that tower:

                def mark_tower(block_, count_set):
                    count_set.add(block_)
                    above_blocks = [b for b, below in current_on.items() if below == block_]
                    if above_blocks:
                        assert len(above_blocks) == 1
                        mark_tower(above_blocks[0], count_set)

                mark_tower(block, counted_blocks)

        # The heuristic should never be zero unless the state matches the goal exactly.
        # The goal test is done by planner, so here just return total_cost.
        # If total_cost is zero, it's safe to return 0 meaning goal reached.

        return total_cost
