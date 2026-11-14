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
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state by 
    measuring how many blocks are "out of place" with respect to their desired position 
    in the goal configuration, plus costs to unstack misplaced blocks below correct ones 
    and to account for the robot arm's holding status.

    Assumptions:
    - The arm can hold at most one block at a time.
    - Actions allowed are pickup, putdown, stack, unstack according to the domain.
    - The goal specifies a conjunction of "on" and "clear" and/or "on-table" facts for blocks.
    - The state is a set of facts representing block positions and arm status.
    - The heuristic does not have to be admissible but should be efficiently computable.
    
    Heuristic Initialization:
    - Parse and store the goal "on" relations as a mapping: block -> what it should be on.
    - Parse and store which blocks should be clear in the goal.
    - Parse and store which blocks should be on the table in the goal.
    - We do not use static facts as none are defined for this domain.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the arm status: if the arm is holding a block, this likely means some ongoing action.
       Add at least 1 action cost for putdown or stack.

    2. For each block:
       - Check if the block is on what it should be on in the goal.
         If not, count it as misplaced.
       - Check if the block's "supporting" block is misplaced (unstacking below correct blocks is costly).
    
    3. For blocks that should be on the table in the goal, check if they are on the table now.
       If not, count them as misplaced.

    4. The heuristic estimates the number of "moves" as:
       - One action per misplaced block to move it to the correct position.
       - Additional actions to free blocks stacked above misplaced blocks.
       - One extra action if the arm is holding a block.

    5. This is a coarse but informative approximation, guiding the search towards 
       configurations with fewer misplaced blocks and fewer obstructing blocks.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions:
        - A dictionary for goal "on" relations: goal_on[block] = what block should be under it or "table"
        - Sets for goal clear blocks and blocks on table.
        """
        self.goals = task.goals
        # Parse goal relations:
        # goal_on: block -> block or "table"
        self.goal_on = {}
        self.goal_clear = set()
        self.goal_on_table = set()

        for fact in self.goals:
            if match(fact, "on", "*", "*"):
                parts = get_parts(fact)
                _, block, below = parts
                self.goal_on[block] = below
            elif match(fact, "on-table", "*"):
                parts = get_parts(fact)
                _, block = parts
                self.goal_on[block] = "table"
                self.goal_on_table.add(block)
            elif match(fact, "clear", "*"):
                parts = get_parts(fact)
                _, block = parts
                self.goal_clear.add(block)
        # No static facts needed in this domain.

    def __call__(self, node):
        """
        Compute heuristic value estimating the number of actions needed to reach the goal.
        """
        state = node.state

        # Extract current arm status:
        holding_block = None
        if any(fact.startswith("(holding ") for fact in state):
            for fact in state:
                if match(fact, "holding", "*"):
                    holding_block = get_parts(fact)[1]
                    break

        # Map: block -> what it is currently on ("table" or another block)
        current_on = {}
        # Set of blocks currently clear
        current_clear = set()
        # Set of blocks currently on the table
        current_on_table = set()

        for fact in state:
            if match(fact, "on", "*", "*"):
                parts = get_parts(fact)
                _, block, below = parts
                current_on[block] = below
            elif match(fact, "on-table", "*"):
                parts = get_parts(fact)
                _, block = parts
                current_on[block] = "table"
                current_on_table.add(block)
            elif match(fact, "clear", "*"):
                parts = get_parts(fact)
                _, block = parts
                current_clear.add(block)

        # For efficient lookup: build a reverse map: below_block -> set(blocks on top)
        above_map = {}
        for block, below in current_on.items():
            above_map.setdefault(below, set()).add(block)

        # Calculate misplaced blocks:
        # A block is misplaced if it is not sitting on the correct block/table in the goal.
        # Also consider blocks that are blocking the correct stacking.
        misplaced_blocks = set()
        # Blocks obstructing correct blocks need to be counted as well.

        # Recursive helper to find all blocks that need to be moved above a target block
        def blocking_blocks(block):
            # Returns all blocks stacked above this block (directly or indirectly)
            result = set()
            to_visit = list(above_map.get(block, []))
            while to_visit:
                b = to_visit.pop()
                result.add(b)
                to_visit.extend(above_map.get(b, []))
            return result

        # Identify all blocks misplaced by checking their current support vs goal support
        for block in self.goal_on.keys():
            goal_below = self.goal_on[block]  # what block it should be on or "table"
            current_below = current_on.get(block, None)
            if current_below != goal_below:
                # Block is misplaced and must be moved
                misplaced_blocks.add(block)
                # Also, blocks stacked above it (obstructing) must be moved first
                misplaced_blocks.update(blocking_blocks(block))

        # Also handle blocks that are in current_on but not in goal_on (should not happen in well-formed problems)
        # or blocks that are in goal but not in current_on (should not happen, but treat as misplaced)
        for block in current_on.keys():
            if block not in self.goal_on:
                # The heuristic counts this block as misplaced as well (no goal position)
                misplaced_blocks.add(block)
                misplaced_blocks.update(blocking_blocks(block))

        # Heuristic cost starts at zero
        cost = 0

        # Each misplaced block costs at least one action (to pick or put or stack/unstack)
        cost += len(misplaced_blocks)

        # Add an extra cost if the arm is holding a block (at least one putdown or stack action required)
        if holding_block is not None:
            cost += 1

        # Additional heuristic refinements:
        # If a block is misplaced but already clear, one action suffices to pick it.
        # If not clear, we anticipate moves to clear it.
        # Add one extra action for each misplaced block that is not clear,
        # since the blocks above it must be unstacked first.
        for b in misplaced_blocks:
            if b not in current_clear:
                cost += 1

        # Ensure heuristic zero only if goals are reached:
        if self.goals <= state:
            cost = 0

        return cost
