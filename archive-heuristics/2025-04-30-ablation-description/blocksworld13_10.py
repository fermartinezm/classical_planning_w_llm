from fnmatch import fnmatch
# Assuming Heuristic base class is available in a module named heuristics
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, you might need a mock Heuristic class:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return [] # Return empty list for malformed facts
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing three components:
    1. The number of blocks that are not part of a correctly built stack segment
       from the table up, according to the goal configuration.
    2. The number of blocks that are required to be clear in the goal but are
       not clear in the current state.
    3. A penalty of 1 if the arm is required to be empty in the goal but is
       not empty in the current state.

    # Assumptions:
    - The goal state consists of one or more stacks of blocks, defined by
      `(on ?x ?y)` and `(on-table ?z)` predicates, potentially including
      `(clear ?x)` and `(arm-empty)`.
    - All blocks mentioned in the goal predicates are considered relevant
      for the heuristic calculation.
    - The heuristic assumes a block is correctly placed in a stack segment
      only if it is on its correct goal support (another block or the table)
      AND that support is itself correctly placed in its goal stack segment
      down to the table.

    # Heuristic Initialization
    - The heuristic parses the goal conditions (`task.goals`) to extract:
      - The target support for each block (`goal_target`: block -> block_below or 'table').
      - The set of all blocks involved in the goal structure (`goal_blocks`).
      - The set of blocks that must be clear in the goal (`goal_clear`).
      - Whether the arm must be empty in the goal (`goal_arm_empty`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state facts (`node.state`) to determine:
       - The current support for each block (`current_on`: block -> block_below).
       - The set of blocks currently on the table (`current_on_table`).
       - The set of blocks currently clear (`current_clear`).
       - Whether the arm is currently empty (`current_arm_empty`).
    2. Calculate the "Stack Misplacement Count":
       - Initialize a set `correctly_placed_blocks`.
       - Add blocks whose goal target is 'table' AND are currently on the table to `correctly_placed_blocks`.
       - Iteratively add blocks `B` to `correctly_placed_blocks` if their goal target is `UnderB`, they are currently on `UnderB`, AND `UnderB` is already in `correctly_placed_blocks`. Repeat until no new blocks are added.
       - The count is the total number of blocks in the goal structure (`len(self.goal_blocks)`) minus the number of blocks in `correctly_placed_blocks`.
    3. Calculate the "Unsatisfied Clear Count":
       - Count how many blocks in `self.goal_clear` are NOT in `current_clear`.
    4. Calculate the "Unsatisfied Arm-Empty Count":
       - If `self.goal_arm_empty` is True AND `current_arm_empty` is False, the count is 1, otherwise 0.
    5. The total heuristic value is the sum of the Stack Misplacement Count, the Unsatisfied Clear Count, and the Unsatisfied Arm-Empty Count.
    6. This sum is 0 if and only if all goal conditions (positional, clear, arm-empty) are met, which corresponds exactly to the goal state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        # Parse goal facts to build the target stack structure and identify other goals
        self.goal_target = {} # Maps block -> block_below or 'table'
        self.goal_blocks = set() # All blocks mentioned in goal positions or clear goals
        self.goal_clear = set() # Blocks that must be clear in the goal
        self.goal_arm_empty = False # True if arm must be empty in the goal

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                # (on ?x ?y) means ?x should be on ?y
                block, under_block = parts[1], parts[2]
                self.goal_target[block] = under_block
                self.goal_blocks.add(block)
                self.goal_blocks.add(under_block)
            elif predicate == "on-table":
                # (on-table ?x) means ?x should be on the table
                block = parts[1]
                self.goal_target[block] = 'table'
                self.goal_blocks.add(block)
            elif predicate == "clear":
                # (clear ?x) means ?x must be clear
                block = parts[1]
                self.goal_clear.add(block)
                self.goal_blocks.add(block) # A block that must be clear is also goal-relevant
            elif predicate == "arm-empty":
                # (arm-empty) must be true
                self.goal_arm_empty = True

        # 'table' is not a block object, remove if added to goal_blocks
        self.goal_blocks.discard('table')

        # Static facts are not used in this heuristic for blocksworld
        # static_facts = task.static


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Build current state representation
        current_on = {} # Maps block -> block_below
        current_on_table = set() # Set of blocks on the table
        current_clear = set() # Set of clear blocks
        current_arm_empty = False # Is arm empty?

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                block, under_block = parts[1], parts[2]
                current_on[block] = under_block
            elif predicate == "on-table":
                block = parts[1]
                current_on_table.add(block)
            elif predicate == "clear":
                block = parts[1]
                current_clear.add(block)
            elif predicate == "arm-empty":
                current_arm_empty = True
            # Ignore 'holding' predicate for this heuristic calculation

        # --- Calculate Stack Misplacement Count ---
        # Identify blocks that are correctly placed relative to a correctly placed support (bottom-up)
        correctly_placed_blocks = set()

        # Start with blocks whose goal target is the table and are currently on the table
        initial_correct = {
            block for block in self.goal_blocks
            if self.goal_target.get(block) == 'table' and block in current_on_table
        }
        correctly_placed_blocks.update(initial_correct)

        # Iteratively add blocks correctly placed on top of already correct blocks
        changed = True
        while changed:
            changed = False
            newly_correct = set()
            # Iterate through all blocks that are part of the goal structure
            for block in self.goal_blocks:
                # Skip if already determined to be correctly placed
                if block in correctly_placed_blocks:
                    continue

                # Get the block this block should be on according to the goal
                target = self.goal_target.get(block)

                # If the target exists and is not 'table'
                if target and target != 'table':
                    # Check if the block is currently on its target AND the target is correctly placed
                    if current_on.get(block) == target and target in correctly_placed_blocks:
                        newly_correct.add(block)
                        changed = True

            correctly_placed_blocks.update(newly_correct)

        # The stack misplacement count is the number of goal blocks not found to be correctly placed
        stack_misplacement_count = len(self.goal_blocks) - len(correctly_placed_blocks)


        # --- Calculate Unsatisfied Clear Count ---
        # Count how many blocks required to be clear in the goal are not clear in the state
        unsatisfied_clear_count = sum(1 for block in self.goal_clear if block not in current_clear)


        # --- Calculate Unsatisfied Arm-Empty Count ---
        # Penalty if arm-empty is a goal but is false in the state
        unsatisfied_arm_empty_count = 1 if self.goal_arm_empty and not current_arm_empty else 0


        # --- Total Heuristic Value ---
        # The total heuristic is the sum of the three components
        total_cost = stack_misplacement_count + unsatisfied_clear_count + unsatisfied_arm_empty_count

        return total_cost
