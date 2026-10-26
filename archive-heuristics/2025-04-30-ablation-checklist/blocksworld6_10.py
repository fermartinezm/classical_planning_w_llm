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
    # Ensure we have at least as many parts as args, unless args contains wildcards
    if len(parts) < len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost to reach the goal state by considering
    blocks that are not in their correct goal stack position and blocks that
    are obstructing the movement of misplaced blocks.

    Heuristic value = (Number of blocks not correctly stacked) +
                      2 * (Number of blocks on top of a not correctly stacked block)

    A block is considered "correctly stacked" if:
    1. It is supposed to be on the table in the goal AND it is currently on the table.
    2. OR it is supposed to be on block A in the goal AND it is currently on block A AND block A is correctly stacked.
    Blocks not explicitly part of a goal stack (i.e., not mentioned in goal 'on' or 'on-table' predicates)
    are considered correctly placed if they are currently on the table.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying all blocks.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Extract goal support relationships: block -> block_below or 'table'
        # This map stores the desired support for each block that is part of a goal stack.
        self._goal_support_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self._goal_support_map[block] = support
            elif parts[0] == 'on-table':
                block = parts[1]
                self._goal_support_map[block] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for the stack structure heuristic

        # Collect all blocks mentioned in goals and initial state.
        # This set represents all objects that are blocks in the problem instance.
        self.all_blocks = set()
        # Collect from initial state
        for fact in self.initial_state:
             parts = get_parts(fact)
             for part in parts[1:]: # Skip predicate name
                 # Exclude keywords/non-objects like 'table', 'arm-empty'
                 if part not in ['table', 'arm-empty']:
                      self.all_blocks.add(part)
        # Collect from goals
        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts[1:]: # Skip predicate name
                 if part not in ['table', 'arm-empty']:
                      self.all_blocks.add(part)

        # Ensure any block mentioned as a support in the goal map is also considered a block
        self.all_blocks.update(self._goal_support_map.keys())
        self.all_blocks.update(self._goal_support_map.values())
        self.all_blocks.discard('table') # Make sure 'table' is not in the set of blocks


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Extract current support relationships: block -> block_below or 'table' or 'arm'
        state_support = {}
        # We don't strictly need blocks_on_top map here, just iterate state facts later
        # held_block = None # Not needed for this version of heuristic

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                state_support[block] = support
            elif parts[0] == 'on-table':
                block = parts[1]
                state_support[block] = 'table'
            elif parts[0] == 'holding':
                 block = parts[1]
                 state_support[block] = 'arm' # Represent held block position


        # If the goal is empty, the heuristic is 0.
        # In Blocksworld, goals typically define stack configurations.
        # If _goal_support_map is empty, there are no goal stacks defined.
        # The heuristic focuses on achieving goal stacks.
        if not self._goal_support_map:
             # If there are no goal stacks, the goal might be just arm-empty or clear specific blocks.
             # A simple heuristic could be 0, or count blocks not on table?
             # Let's return 0 if no stack goals are defined.
             return 0


        # --- Calculate H1: Number of blocks not correctly stacked ---

        # Initialize correctness status for all blocks. Assume False initially.
        is_correct = {block: False for block in self.all_blocks}
        changed = True

        # Iteratively determine which blocks are correctly stacked.
        # A block is correctly stacked if its goal support is met AND its goal support is correct.
        # Base cases: Blocks whose goal is 'on-table' and are currently 'on-table'.
        # Also, blocks not in goal_support_map are correct if on the table.
        while changed:
            changed = False
            for block in self.all_blocks:
                if not is_correct[block]: # Only try to update if not already correct
                    goal_sup = self._goal_support_map.get(block) # What should this block be on?
                    current_sup = state_support.get(block) # What is this block on?

                    if goal_sup is not None: # Block is part of a goal stack structure
                        if goal_sup == 'table':
                            # Goal: block should be on the table
                            if current_sup == 'table':
                                is_correct[block] = True
                                changed = True
                        else:
                            # Goal: block should be on goal_sup (which must be another block)
                            # Check if current support matches goal support AND the support block is correct
                            if current_sup == goal_sup and is_correct.get(goal_sup, False):
                                # is_correct.get(goal_sup, False) handles cases where goal_sup might not be in all_blocks
                                # (though it should be if collected properly), defaulting to False if unknown.
                                if goal_sup in self.all_blocks and is_correct[goal_sup]:
                                    is_correct[block] = True
                                    changed = True
                    else:
                        # Block is not explicitly in _goal_support_map.
                        # Its implicit goal is to be on the table and clear.
                        # Consider it "correctly placed" if it's on the table.
                        if current_sup == 'table':
                             is_correct[block] = True
                             changed = True
                        # If current_sup is 'arm' or another block, it remains False.


        # H1: Count blocks that are not correctly stacked
        h1 = sum(1 for block in self.all_blocks if not is_correct[block])


        # --- Calculate H2: Number of blocks on top of a not correctly stacked block ---
        # These are blocks that are blocking access to a block that needs to be moved.

        h2 = 0
        # Iterate through state facts to find (on C B)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block_c, block_b = parts[1], parts[2]
                # Check if block_b is in our set of blocks and is not correctly stacked
                # If block_b is not correctly stacked, then block_c sitting on it is a blocker.
                if block_b in self.all_blocks and not is_correct[block_b]:
                    h2 += 1
                # If block_b is not in all_blocks, it means it wasn't in init or goal.
                # This shouldn't happen in valid PDDL instances for blocks.
                # We assume all relevant blocks are in all_blocks collected from init/goal.


        # Heuristic value is the sum of H1 and H2 * 2.
        # H1 counts blocks that need to be moved to their correct place.
        # H2 counts blocks that need to be moved *out of the way* of a block that needs moving.
        # Moving a blocker typically takes 2 actions (unstack/pickup + putdown).
        heuristic_value = h1 + h2 * 2

        # Optional: Add 1 if the arm is holding a block. This block needs a final action (stack/putdown).
        # This held block is already counted in H1 if it's not correctly stacked.
        # Adding 1 here might slightly overestimate but penalizes having the arm busy.
        # Let's include it for potentially better performance in greedy search.
        arm_empty = "(arm-empty)" in state
        if not arm_empty:
             heuristic_value += 1


        return heuristic_value

