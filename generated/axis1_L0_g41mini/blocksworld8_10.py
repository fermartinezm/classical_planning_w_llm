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
    A domain-dependent heuristic for the blocksworld domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve the goal stacking configuration
    from the current state in the classical blocksworld problem. It guides a greedy best-first search
    by focusing on mismatches between the current and goal arrangements of blocks.

    Assumptions
    -----------
    - The arm can hold at most one block.
    - Blocks can be on the table or stacked on other blocks.
    - To achieve the goal configuration, blocks out of place must be unstacked and moved.
    - The heuristic does not have to be admissible but aims to be informative and efficiently computable.
    - The given goal state only specifies a partial stack configuration (some 'on' relations and 'clear' facts).
      The heuristic handles these relations and implicit table conditions.
    - The state representation uses facts like (on ?x ?y), (on-table ?x), (clear ?x), (holding ?x), and (arm-empty).

    Heuristic Initialization
    ------------------------
    - Extract all goal 'on' relations as a dictionary mapping each block to its goal support (block or table).
    - Extract the set of goal clear blocks.
    - Extract goal 'on-table' blocks for reference.
    - Initialize internal data structures for goal lookup efficient checks.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state satisfies all goal conditions, the heuristic is 0.
    2. Otherwise, for each block mentioned in the goal 'on' relations:
       - Check whether the block currently rests on the correct block or table as specified by the goal.
       - Each block that does not satisfy its goal support condition is considered misplaced.
    3. For blocks that must be on the table in the goal but are currently stacked on another block, count them as misplaced.
    4. Count blocks currently held by the arm as blocks requiring at least one putdown action.
    5. Each misplaced block usually requires at least two moves:
       - Unstacking or picking it up from the wrong position.
       - Putting it down or stacking it onto the correct support.
    6. Blocks currently held require at least one putdown or stack action, thus add moves accordingly.
    7. Count the number of blocks that must be moved, weighing those on wrong support twice (unstack + stack).
    8. The heuristic is the sum of these estimates.
    9. This heuristic deliberately ignores precise ordering or intermediate block removals for efficiency.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal stacking relations."""
        self.goals = task.goals

        # Parse goal 'on' relations: block -> target support (block or 'table')
        self.goal_on = {}  # block -> support (block or 'table')
        self.goal_clear = set()
        self.goal_on_table = set()

        for fact in self.goals:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "on":
                block, support = parts[1], parts[2]
                self.goal_on[block] = support
            elif pred == "clear":
                self.goal_clear.add(parts[1])
            elif pred == "on-table":
                self.goal_on_table.add(parts[1])

    def __call__(self, node):
        state = node.state

        if self.goals <= state:
            # Goal reached exactly
            return 0

        # Extract current 'on' relations: block -> support (block or 'table')
        current_on = {}
        holding_blocks = set()
        arm_empty = False
        clear_blocks = set()
        blocks = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]

            if pred == "on":
                block, support = parts[1], parts[2]
                current_on[block] = support
                blocks.add(block)
                blocks.add(support)
            elif pred == "on-table":
                block = parts[1]
                current_on[block] = "table"
                blocks.add(block)
            elif pred == "holding":
                holding_blocks.add(parts[1])
                blocks.add(parts[1])
            elif pred == "arm-empty":
                arm_empty = True
            elif pred == "clear":
                clear_blocks.add(parts[1])
                blocks.add(parts[1])

        # All blocks known from state and goal
        all_blocks = set(current_on.keys()) | set(self.goal_on.keys()) | holding_blocks

        misplaced_count = 0  # Counts blocks not on their correct goal support.
        need_put_down = 0    # Blocks currently held => need putdown or stack.

        # For each block that has a goal support:
        for block in self.goal_on:
            goal_support = self.goal_on[block]

            # Determine current support for block; treat absent as unknown -> considered misplaced
            current_support = current_on.get(block)
            if current_support is None:
                # Block missing in state facts? Conservative: count as misplaced
                misplaced_count += 1
                continue

            # Check if block is held currently: if held, its current support is "holding"
            if block in holding_blocks:
                # Must at least putdown or stack
                need_put_down += 1
                # Also must be placed on correct support eventually, so count as misplaced
                misplaced_count += 1
                continue

            # If the goal is 'on-table', the support must be 'table' currently
            # Otherwise, it must be goal_support exactly
            if goal_support == "table":
                # Current support must be 'table'
                if current_support != "table":
                    misplaced_count += 1
            else:
                # Goal support is a block; current support must match exactly
                if current_support != goal_support:
                    misplaced_count += 1

        # Blocks that are held but not in goal_on (e.g., blocks with no specified goal support, but held):
        # They must be put down somewhere; count their putdown cost minimally.
        for block in holding_blocks:
            if block not in self.goal_on:
                need_put_down += 1

        # Each misplaced block requires at least 2 moves: unstack/pickup + stack/putdown
        # Each held block requires at least 1 move to put down or stack
        # So estimate as: 2 * misplaced + 1 * number of held blocks not double counted

        # Because held blocks were added to misplaced_count (above), we do not add them twice.
        heuristic_value = 2 * misplaced_count + need_put_down

        # At minimum, heuristic should be at least 1 if not solved (no zero for unsolved states)
        return max(heuristic_value, 1)
