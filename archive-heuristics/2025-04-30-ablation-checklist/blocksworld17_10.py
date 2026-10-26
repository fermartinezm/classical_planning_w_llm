from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()

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

    Estimates the cost based on the number of blocks not in their
    correct goal position relative to their support (recursively),
    plus a penalty if the arm is holding a block.

    Heuristic Value:
    2 * (Number of blocks whose goal position relative to their support
         is not achieved, considering the support must also be settled)
    + 1 (if the arm is holding any block)

    A block B is "settled" if:
    1. Its goal is (on-table B) AND (on-table B) is true in the state.
    2. Its goal is (on B UnderB) AND (on B UnderB) is true in the state AND UnderB is "settled".
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships.
        """
        self.goals = task.goals
        self.goal_support = {}

        # Build goal_support map for all blocks whose position is specified in the goal
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_support[block] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for the structural part of the heuristic

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current_support map and check if arm is holding something
        current_support = {}
        arm_is_holding = False

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                current_support[block] = support
            elif parts[0] == 'on-table':
                block = parts[1]
                current_support[block] = 'table'
            elif parts[0] == 'holding':
                arm_is_holding = True
                # We could store which block is held, but the penalty is fixed at +1
                # if any block is held.
                # held_block = parts[1]

        # Compute the set of settled blocks
        # A block is settled if it's in its goal position AND its support is settled.
        # This is computed iteratively starting from blocks whose goal support is the table.
        settled = set()
        
        # Blocks whose goal support is the table (base case for settled)
        # Only consider blocks that are actually in the goal_support map
        for block in self.goal_support:
             if self.goal_support[block] == 'table':
                 if current_support.get(block) == 'table':
                     settled.add(block)

        # Iteratively add blocks whose support is settled and are in the correct place
        # Only consider blocks whose position is specified in the goal and are not yet settled
        potentially_settled_candidates = set(self.goal_support.keys()) - settled

        while True:
            newly_settled = set()
            current_settled_count = len(settled)

            for block in potentially_settled_candidates:
                 # block is guaranteed to be in self.goal_support.keys()
                 goal_supp = self.goal_support[block]
                 if goal_supp != 'table': # Ensure block is not a table goal (handled in base case)
                     under_block = goal_supp
                     # Check if the block below is settled AND the current support is correct
                     # Use .get() for current_support in case the block is held or not in state facts
                     if under_block in settled and current_support.get(block) == under_block:
                          newly_settled.add(block)

            if not newly_settled:
                break # No new blocks were settled in this iteration

            settled.update(newly_settled)
            potentially_settled_candidates -= newly_settled # Remove newly settled from candidates

            # Safety break - should not be needed with correct logic but good practice
            if len(settled) == current_settled_count:
                 break


        # Calculate the number of unsettled blocks whose position is specified in the goal
        # We only care about blocks whose goal position is explicitly defined in goal_support
        n_unsettled = len(self.goal_support) - len(settled)

        # Heuristic value: 2 actions per unsettled block + penalty for held block
        # Each unsettled block needs at least a pickup/unstack and a stack/putdown (2 actions).
        heuristic_value = 2 * n_unsettled

        # Add penalty if arm is holding something. This block needs to be put down
        # before the arm can be used for other operations.
        if arm_is_holding:
             heuristic_value += 1

        return heuristic_value
