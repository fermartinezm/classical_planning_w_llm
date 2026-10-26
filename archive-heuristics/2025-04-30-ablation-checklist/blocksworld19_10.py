from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL states are structured.
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    # Split by spaces, ignoring spaces within parameters if they were quoted (not typical in Blocksworld)
    # Simple split is sufficient for Blocksworld predicates like (on b1 b2)
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern using fnmatch.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting:
    1. The number of blocks that are not on their correct support (block or table) according to the goal.
    2. The number of blocks currently stacked on top of blocks identified in step 1.

    # Assumptions
    - The goal specifies the desired support for each block (either another block or the table).
    - Blocks not mentioned as being 'on' something or 'on-table' in the goal are ignored by the heuristic.
    - The cost of moving a block includes clearing anything on top of it and placing it in the correct location.

    # Heuristic Initialization
    - Extract the desired support for each block from the goal conditions.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Parse the goal facts to determine the target support (another block or 'table') for each block that needs to be in a specific location.
    2. Parse the current state facts to determine the current support for each block (another block, 'table', or 'arm' if held). Also, build a mapping of which block is directly on top of which other block.
    3. Initialize the heuristic value to 0.
    4. For each block that has a specified target support in the goal:
       a. Check if its current support matches its target support.
       b. If the supports do *not* match:
          i. Increment the heuristic value by 1 (representing the cost to move this block).
          ii. Recursively count all blocks currently stacked directly or indirectly on top of this block in the current state. Add this count to the heuristic value (representing the cost to clear the block).
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support information.

        @param task: The planning task object containing goals and initial state.
        """
        self.goals = task.goals
        # Map block -> its desired support (another block or 'table')
        self.goal_supports = {}
        # Set of blocks that are explicitly positioned in the goal
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_supports[block] = support
                self.goal_blocks.add(block)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_supports[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore other goal predicates like (clear ?) or (arm-empty) for this heuristic

    def _count_blocks_on_top(self, block, blocks_on_top_map):
        """
        Recursively counts the number of blocks stacked directly or indirectly
        on top of the given block in the current state.

        @param block: The block to start counting from.
        @param blocks_on_top_map: A dictionary mapping a block to the block directly on top of it.
        @return: The total count of blocks above the given block.
        """
        # Find the block directly on top of 'block'
        block_above = blocks_on_top_map.get(block)

        if block_above is None:
            # Nothing is on top
            return 0
        else:
            # One block on top, plus whatever is on top of that block
            return 1 + self._count_blocks_on_top(block_above, blocks_on_top_map)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: The estimated cost (heuristic value) to reach the goal.
        """
        state = node.state
        heuristic_value = 0

        # Build current support map and blocks-on-top map from the state
        current_supports = {}
        # We need the inverse mapping: block -> block_directly_on_top
        blocks_on_top = {} # Maps block_below -> block_above

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_supports[block] = support
                blocks_on_top[support] = block # Inverse mapping
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_supports[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_supports[block] = 'arm'
            # Ignore clear, arm-empty, etc. for support mapping

        # Compute heuristic based on goal blocks
        for block in self.goal_blocks:
            goal_support = self.goal_supports[block]
            current_support = current_supports.get(block) # Use .get to handle blocks not explicitly in state facts (e.g., held)

            # If the block is not on its correct support
            if current_support != goal_support:
                # Add 1 for the cost of moving the block itself
                heuristic_value += 1
                # Add the cost of clearing the block (moving everything on top)
                # Note: A block held by the arm has nothing on top in the state representation
                if current_support != 'arm':
                     heuristic_value += self._count_blocks_on_top(block, blocks_on_top)

        return heuristic_value

