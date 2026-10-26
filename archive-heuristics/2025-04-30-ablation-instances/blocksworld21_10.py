from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return [] # Or raise an error, depending on expected input robustness
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the remaining effort by counting the number of
    blocks that are not yet in their correct position relative to the goal
    stack structure, built from the bottom up. A block is considered
    "correctly stacked" if it is on the table and the goal requires it to be
    on the table, OR if it is on block A and the goal requires it to be on
    block A, AND block A is itself correctly stacked. The heuristic value is
    the total number of blocks minus the number of correctly stacked blocks.

    # Assumptions
    - The goal state defines a specific configuration of blocks on the table
      or stacked on top of each other using `(on ?x ?y)` and `(on-table ?x)`
      predicates.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack) are used.
    - The heuristic does not need to be admissible.

    # Heuristic Initialization
    - The constructor extracts the goal configuration for each block (what it
      should be directly on top of, or if it should be on the table) from the
      task's goal conditions.
    - It also identifies all blocks involved in the goal.
    - Static facts are not used as Blocksworld typically has none relevant
      to the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the immediate support for each block
       (what it is currently on, or if it's on the table, or if it's being held).
    2. Initialize a set `correct_set` to store blocks that are currently in
       their correct position relative to the goal stack, built from the bottom up.
    3. Identify blocks that should be on the table according to the goal. For
       each such block, if it is currently on the table in the state, add it
       to `correct_set`. These are the base cases for the correctly built stacks.
    4. Iteratively expand the `correct_set`: In each iteration, check all blocks
       that are *not* yet in `correct_set`. If a block `B` should be on block `A`
       according to the goal (`(on B A)` is a goal fact), and block `A` is already
       in `correct_set`, AND block `B` is currently on block `A` in the state
       (`(on B A)` is a state fact), then add block `B` to a temporary set of
       `newly_correct` blocks. Add `newly_correct` blocks to `correct_set`.
       Repeat this step until no new blocks are added to `correct_set` in an iteration.
    5. The heuristic value is the total number of blocks involved in the goal
       minus the number of blocks in the final `correct_set`. This counts the
       blocks that are not part of the correctly formed goal stacks. Each such
       block (and potentially blocks above it) will require actions to move it
       into or towards its correct position.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and blocks.
        """
        self.goals = task.goals  # Goal conditions.

        # Map block to what it should be directly on top of in the goal state.
        # Value is a block name or the string 'table'.
        self.goal_under_map = {}

        # Set of all blocks involved in the goal.
        self.all_blocks = set()

        # Parse goal facts to build the goal_under_map and all_blocks set.
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip empty or invalid facts
                continue
            predicate = parts[0]
            if predicate == "on":
                # Goal is (on block1 block2)
                block1, block2 = parts[1], parts[2]
                self.goal_under_map[block1] = block2
                self.all_blocks.add(block1)
                self.all_blocks.add(block2)
            elif predicate == "on-table":
                # Goal is (on-table block)
                block = parts[1]
                self.goal_under_map[block] = 'table'
                self.all_blocks.add(block)
            # Ignore other goal predicates like (clear) or (arm-empty) for this heuristic

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state  # Current world state as a frozenset of strings.

        # Map block to what it is currently directly on top of.
        # Value is a block name or the string 'table'.
        # Blocks being held are not included here initially.
        current_under_map = {}
        # Track the block being held, if any.
        holding_block = None

        # Parse state facts to build current_under_map and find holding_block.
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty or invalid facts
                continue
            predicate = parts[0]
            if predicate == "on":
                # State has (on block1 block2)
                block1, block2 = parts[1], parts[2]
                current_under_map[block1] = block2
            elif predicate == "on-table":
                # State has (on-table block)
                block = parts[1]
                current_under_map[block] = 'table'
            elif predicate == "holding":
                # State has (holding block)
                holding_block = parts[1]
            # Ignore other state predicates like (clear) or (arm-empty)

        # Set to store blocks that are correctly stacked from the bottom up.
        correct_set = set()

        # Step 3: Add blocks that are correctly on the table according to the goal.
        for block in self.all_blocks:
            # Check if the block should be on the table in the goal
            if self.goal_under_map.get(block) == 'table':
                # Check if the block is currently on the table in the state
                if current_under_map.get(block) == 'table':
                    correct_set.add(block)

        # Step 4: Iteratively add blocks that are correctly stacked on top of
        # already correctly stacked blocks.
        changed = True
        while changed:
            changed = False
            newly_correct = set()
            # Iterate through all blocks that are not yet marked as correct
            for block in self.all_blocks - correct_set:
                goal_under = self.goal_under_map.get(block)

                # Check if the block should be on another block (not table) in the goal
                if goal_under is not None and goal_under != 'table':
                    # Check if the block it should be on is already correctly stacked
                    if goal_under in correct_set:
                        # Check if the block is currently on the correct block in the state
                        current_under = current_under_map.get(block)
                        if current_under == goal_under:
                            # This block is correctly stacked on a correctly stacked block
                            newly_correct.add(block)
                            changed = True # We found new correctly stacked blocks

            # Add the newly found correctly stacked blocks to the main set
            correct_set.update(newly_correct)

        # Step 5: The heuristic is the number of blocks not correctly stacked.
        # If the arm is holding a block, that block is definitely not in its
        # final place, so we should count it if it wasn't already counted.
        # However, the current_under_map approach naturally excludes the held block,
        # and the logic correctly identifies it as not being on its goal_under.
        # So, simply counting blocks not in correct_set is sufficient.

        # The total number of blocks considered is len(self.all_blocks).
        # The number of blocks correctly placed in the goal structure is len(correct_set).
        # The number of blocks not correctly placed is the difference.
        heuristic_value = len(self.all_blocks) - len(correct_set)

        # Ensure heuristic is 0 only at the goal state.
        # The goal state implies all goal facts are true. If all goal facts
        # (on and on-table) are true, then all blocks involved in the goal
        # will be correctly stacked by the logic above, resulting in 0.
        # If not all goal facts are true, at least one block is not on its
        # goal_under, or its goal_under is not correctly stacked, resulting in > 0.
        # This check is implicitly handled by the logic.

        return heuristic_value
