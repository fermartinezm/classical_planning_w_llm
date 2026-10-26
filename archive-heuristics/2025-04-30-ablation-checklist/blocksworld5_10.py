from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function (optional, but good practice)
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
    This heuristic estimates the number of actions required to achieve the goal
    configuration of blocks by counting blocks that are not in their correct
    position within the goal stacks and multiplying by an estimated cost per move.
    It considers a block correctly placed only if it is on its goal support
    AND its goal support is also correctly placed (recursively).

    # Heuristic Initialization
    - Parses the goal state to determine the desired support (block or table)
      for each block mentioned in the goal (on or on-table predicates).
    - Identifies all blocks that are part of the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the desired support for each block based on the goal facts
       ((on ?x ?y) and (on-table ?x)). Store this mapping (block -> support).
    2. Identify all blocks that are part of the goal configuration (i.e.,
       mentioned in the goal (on) or (on-table) facts).
    3. Determine the current support for each block in the current state
       ((on ?x ?y) or (on-table ?x)).
    4. Determine which blocks are "correctly placed" relative to the goal
       stack structure. A block B is correctly placed if:
       - It is supposed to be on the table AND is currently on the table.
       - OR, it is supposed to be on block A AND is currently on block A,
         AND block A is itself correctly placed.
       This is computed iteratively, propagating correctness up the goal stacks
       starting from blocks correctly placed on the table.
    5. Count the number of blocks that are *not* correctly placed according
       to step 4. Let this be N_incorrect. These are the blocks that need
       to be moved or are on top of blocks that need to be moved to achieve
       the goal structure.
    6. The base heuristic estimate is 2 * N_incorrect. This assumes each
       incorrectly placed block needs at least two actions (pickup/unstack +
       putdown/stack) to be moved towards its correct position. This is a
       simplification that works well for greedy search.
    7. Check if the arm is currently holding a block. If it is, and that block
       is one of the incorrectly placed goal blocks, subtract 1 from the
       heuristic. This accounts for the pickup/unstack action already being
       completed for that block.
    8. Return the calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships
        and identifying all blocks involved in the goal.
        """
        self.goals = task.goals
        # static_facts = task.static # Not needed for this heuristic

        # Map block to its desired support ('table' or another block)
        self.goal_support = {}
        # Set of all blocks that are part of the goal configuration
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    self.goal_support[block] = support
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(support) # Support block is also part of the goal structure
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_support[block] = 'table'
                    self.goal_blocks.add(block)
            # Ignore (clear ?x) goals for the support structure calculation,
            # as they are typically satisfied if the stack below is correct.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Map block to its current support ('table' or another block)
        current_support = {}
        # Track the block being held, if any
        held_block = None

        # Build current support map and find held block
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on":
                 if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    current_support[block] = support
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    current_support[block] = 'table'
            elif predicate == "holding":
                 if len(parts) == 2:
                    held_block = parts[1]

        # Determine which goal blocks are correctly placed relative to the goal structure
        # Initialize all goal blocks as not correctly placed
        is_correct = {block: False for block in self.goal_blocks}
        changed = True

        # Propagate correctness up the goal stacks iteratively
        while changed:
            changed = False
            for block in self.goal_blocks:
                if is_correct[block]:
                    continue # Already marked correct

                goal_sup = self.goal_support.get(block)
                current_sup = current_support.get(block)

                # If the block is not mentioned in goal support, it's not part of a specific stack goal.
                # We only consider blocks that have a defined goal support in the goal facts.
                if goal_sup is None:
                     # This block isn't part of a specific stack goal (e.g., it might only have a (clear) goal).
                     # For this heuristic, we focus on the stack structure defined by (on) and (on-table).
                     # Blocks not in goal_support are effectively ignored in the count of 'incorrect' blocks.
                     # This is a simplification. A more complex heuristic might count blocks blocking (clear) goals.
                     continue

                # Check if the block is correctly placed on the table
                if goal_sup == 'table' and current_sup == 'table':
                    is_correct[block] = True
                    changed = True
                # Check if the block is correctly placed on another block
                elif goal_sup != 'table' and current_sup == goal_sup:
                    # Check if the block it's supposed to be on is also correct
                    # Use .get(goal_sup, False) to handle cases where goal_sup might not be in goal_blocks
                    # (e.g., if goal is (on b1 b2) but b2 has no further goal support defined - though goal_blocks includes b2)
                    # The check `goal_sup in self.goal_blocks` is implicitly handled by `is_correct.get(goal_sup, False)`
                    # if goal_sup is not in is_correct keys.
                    if is_correct.get(goal_sup, False):
                        is_correct[block] = True
                        changed = True

        # Count the number of goal blocks that are not correctly placed
        n_incorrect = sum(1 for block in self.goal_blocks if not is_correct[block])

        # Base heuristic: Estimate 2 actions per incorrectly placed block
        # (one to pick it up/unstack, one to put it down/stack).
        h = 2 * n_incorrect

        # Adjust if the arm is currently holding an incorrectly placed goal block.
        # If the held block is a goal block and is not correctly placed, the first
        # action (pickup/unstack) is already done for this block.
        if held_block and held_block in self.goal_blocks and not is_correct[held_block]:
             h = max(0, h - 1) # Subtract 1, ensuring the heuristic doesn't go below zero

        return h

