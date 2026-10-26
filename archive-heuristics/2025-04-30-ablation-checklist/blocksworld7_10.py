from fnmatch import fnmatch
# Assume Heuristic base class is available in heuristics.heuristic_base
# For standalone testing, you might need a mock base class like:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal
    by counting structural differences between the current state and the goal state.
    It counts blocks that are not on their correct support (table or another block)
    and blocks that have the wrong block on top of them. It also adds a cost
    if the arm is holding a block (assuming arm-empty is desired in the goal).

    # Heuristic Initialization
    - Parses the goal predicates to determine the desired support and the desired
      block on top for each block involved in the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the desired position for each block in the goal: which block
       it should be on, or if it should be on the table. Store this in `goal_below`.
    2. Identify which block should be directly on top of each block in the goal,
       or if it should be clear. Store this in `goal_above`.
    3. For a given state, parse the current positions and blocks on top. Store
       this in `current_below` and `current_above`. Also, check if the arm is
       holding a block.
    4. Initialize heuristic value `h = 0`.
    5. Iterate through each block that is part of the goal configuration (i.e.,
       appears in `goal_below` or `goal_above`).
    6. For each block `B`:
       - Determine `B`'s current support (the block it's on, or 'table', or 'arm' if held).
       - Compare `B`'s current support with its goal support (`goal_below[B]`). If they don't match, increment `h` by 1. This counts blocks that are in the wrong stack or on the wrong level.
       - Determine the block currently on top of `B` (or None if clear).
       - Compare the block currently on top of `B` with the block that should be on top of `B` in the goal (`goal_above[B]`). If they don't match, increment `h` by 1. This counts blocks that are obstructing or are themselves misplaced on top of another block.
    7. If the arm is currently holding a block, increment `h` by 1. This accounts
       for the action needed to free the arm (putdown or stack), assuming the goal
       implicitly requires the arm to be empty.
    8. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.

        @param task: The planning task object.
        """
        self.goals = task.goals

        # Dictionaries to store the desired configuration for each block in the goal
        # goal_below[block] = block_it_should_be_on or 'table'
        # goal_above[block] = block_that_should_be_on_it or None (if should be clear)
        self.goal_below = {}
        self.goal_above = {}
        self.goal_blocks = set() # Collect all blocks mentioned in goal predicates

        # Parse goal predicates to build the desired configuration
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                obj, underob = parts[1], parts[2]
                self.goal_below[obj] = underob
                self.goal_above[underob] = obj
                self.goal_blocks.add(obj)
                self.goal_blocks.add(underob)
            elif predicate == "on-table" and len(parts) == 2:
                obj = parts[1]
                self.goal_below[obj] = 'table'
                # If a block is on the table in the goal, it should implicitly be clear
                # unless another goal predicate explicitly puts something on it.
                # We handle this by initializing goal_above entries only when an 'on' goal exists.
                self.goal_blocks.add(obj)
            # We don't explicitly process 'clear' or 'arm-empty' goals here,
            # as they are implicitly handled by the structural counts and the holding check.

        # Add any blocks that appear as supports in 'on' goals but not as the
        # object being placed ('on X Y' where X is not in goal_below yet).
        # These blocks are bases of goal stacks and should be on the table.
        for block in list(self.goal_above.keys()): # Iterate over a copy
             if block not in self.goal_below:
                 self.goal_below[block] = 'table'
                 self.goal_blocks.add(block)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The current state node.
        @return: The estimated number of actions to reach the goal.
        """
        state = node.state

        # Parse the current state to determine block positions and arm state
        current_below = {}
        current_above = {}
        holding_block = None
        # arm_empty = False # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                obj, underob = parts[1], parts[2]
                current_below[obj] = underob
                current_above[underob] = obj
            elif predicate == "on-table" and len(parts) == 2:
                obj = parts[1]
                current_below[obj] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                holding_block = parts[1]
            # elif predicate == "arm-empty":
            #     arm_empty = True # Not strictly needed

        h = 0

        # Count discrepancies for blocks involved in the goal
        for block in self.goal_blocks:
            # Check current support vs goal support
            # If block is held, its current location is effectively the arm, not on/on-table
            current_loc = current_below.get(block, 'arm' if holding_block == block else None)

            # If a block is in the goal_blocks set but doesn't have a goal_below entry,
            # it means it only appeared as a support (underob in an 'on' goal)
            # and should have been added to goal_below as 'table' in __init__.
            # So goal_loc should always exist here.
            goal_loc = self.goal_below.get(block)

            # If current_loc is None, the block is not on anything, not on table, and not held.
            # This shouldn't happen in a valid blocksworld state for blocks that exist.
            # We'll treat None as a mismatch if the goal_loc is defined.
            if current_loc != goal_loc:
                h += 1

            # Check current block on top vs goal block on top
            current_top = current_above.get(block, None) # None if nothing is on it
            goal_top = self.goal_above.get(block, None) # None if it should be clear in goal

            if current_top != goal_top:
                h += 1

        # Add cost if the arm is holding a block (assuming arm-empty is desired in goal)
        # This covers the action needed to put the block down or stack it.
        if holding_block is not None:
             h += 1

        return h

