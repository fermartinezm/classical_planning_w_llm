from fnmatch import fnmatch
# Assuming Heuristic base class is available in the environment as heuristics.heuristic_base.Heuristic
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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check: just zip and match. fnmatch handles differing lengths gracefully
    # if the pattern is shorter than the fact parts.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)) and len(parts) == len(args)


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not in their correct goal stack position,
    plus penalties for blocks that need to be cleared and for the arm not being empty
    when required by the goal.

    # Assumptions
    - Standard Blocksworld domain with predicates (on, on-table, clear, holding, arm-empty)
      and actions (pickup, putdown, stack, unstack).
    - Goal states specify the desired configuration of blocks in stacks (on/on-table)
      and which blocks should be clear, and potentially if the arm should be empty.

    # Heuristic Initialization
    - Parses the goal conditions to determine the desired support (block or 'table')
      for each block and which blocks should be clear in the goal state.
    - Identifies all relevant blocks by examining initial state and goal facts.
    - Stores this goal configuration information for use in the heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a given state is computed as follows:

    1.  **Parse Current State:** Extract the current configuration of blocks (which block is on which, or on the table), identify which blocks are clear, and check if the arm is holding a block.
    2.  **Identify Correctly Stacked Blocks:** Define a recursive helper function `is_correctly_stacked(block)` that determines if a block is in its correct goal position relative to its goal support, AND if its goal support is also correctly stacked. The base case is a block whose goal is to be on the table, and it is currently on the table. Memoization is used to efficiently compute this for all blocks within the current state evaluation.
    3.  **Count Misplaced Blocks:** Initialize the heuristic value `h` to 0. Iterate through all relevant blocks (those appearing in the initial state or goals). If a block is determined *not* to be correctly stacked according to the goal configuration, increment `h`. This counts blocks that are fundamentally in the wrong place within the desired stack structure.
    4.  **Count Unsatisfied Clear Goals:** Iterate through the goal conditions. For each goal predicate `(clear B)`, if block `B` is *not* clear in the current state, increment `h`. This adds a penalty for blocks that are blocked by others and need to be cleared before they or blocks below them can be moved correctly.
    5.  **Penalty for Busy Arm:** If the goal includes `(arm-empty)` and the arm is currently holding a block, increment `h`. This adds a penalty if the arm is in a state that needs to be resolved to meet a common goal condition.
    6.  **Return Heuristic Value:** The final value of `h` is the estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying all relevant blocks.
        """
        self.goals = task.goals

        # Map block -> goal_support (block or 'table')
        self.goal_config = {}
        # Set of blocks that should be clear in the goal
        self.goal_clear = set()
        # Check if arm-empty is a goal
        self.goal_arm_empty = '(arm-empty)' in self.goals

        # Extract goal configuration from goal facts
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                self.goal_config[block] = support
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_config[block] = 'table'
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear.add(block)
            # arm-empty goal is handled by self.goal_arm_empty

        # Identify all blocks present in the problem (from initial state and goals)
        self.all_blocks = set()
        def add_objects_from_fact(fact_string):
             parts = get_parts(fact_string)
             predicate = parts[0]
             # Consider arguments of predicates that involve objects
             if predicate in ['on', 'on-table', 'clear', 'holding']:
                 for arg in parts[1:]:
                     # Basic check to exclude non-block terms like 'table' or 'arm'
                     if arg != 'table' and not arg.startswith('arm'):
                         self.all_blocks.add(arg)

        for fact in task.initial_state:
             add_objects_from_fact(fact)
        for fact in task.goals:
             add_objects_from_fact(fact)

        # Memoization dictionary for the recursive correctly_stacked check, reset per state
        self._correctly_stacked_memo = {}


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Parse Current State
        current_pos = {} # Map block -> current_support (block or 'table')
        is_clear_set = set() # Set of blocks that are clear
        is_holding_block = None # The block being held, or None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_pos[block] = support
            elif predicate == 'on-table':
                block = parts[1]
                current_pos[block] = 'table'
            elif predicate == 'clear':
                block = parts[1]
                is_clear_set.add(block)
            elif predicate == 'holding':
                is_holding_block = parts[1]
            # arm-empty is checked directly

        # Reset memoization for this state
        self._correctly_stacked_memo = {}

        # 2. Define and use recursive helper to identify correctly stacked blocks
        def is_correctly_stacked_recursive(block):
            """
            Recursive helper to check if a block is correctly stacked from the bottom up.
            Uses memoization stored in self._correctly_stacked_memo.
            """
            if block in self._correctly_stacked_memo:
                return self._correctly_stacked_memo[block]

            # If block is currently held, it's not correctly stacked relative to a fixed position
            if is_holding_block == block:
                 self._correctly_stacked_memo[block] = False
                 return False

            # Find goal support for this block
            goal_support = self.goal_config.get(block)

            # If a block has no specified goal position, it's considered "correctly stacked"
            # for the purpose of this part of the heuristic.
            if goal_support is None:
                 self._correctly_stacked_memo[block] = True
                 return True

            # Find current support for this block
            current_support = current_pos.get(block)

            # If block is not on anything or table (and not held), something is wrong, not correctly stacked
            if current_support is None:
                 self._correctly_stacked_memo[block] = False
                 return False

            # Check if current support matches goal support
            if current_support != goal_support:
                self._correctly_stacked_memo[block] = False
                return False

            # If current support matches goal support, check the support recursively
            if goal_support == 'table':
                # Base case: If the goal is on the table and it is on the table, it's correctly stacked
                self._correctly_stacked_memo[block] = True
                return True
            else:
                # Recursive step: Check if the support block itself is correctly stacked
                # Ensure the support block is a relevant block in the problem
                if goal_support not in self.all_blocks:
                     # If the block is on something that isn't a known block, it's wrong
                     self._correctly_stacked_memo[block] = False
                     return False
                else:
                     is_support_correct = is_correctly_stacked_recursive(goal_support)
                     self._correctly_stacked_memo[block] = is_support_correct
                     return is_support_correct


        # 3. Count Misplaced Blocks (not correctly stacked)
        h = 0
        for block in self.all_blocks:
            if not is_correctly_stacked_recursive(block):
                h += 1

        # 4. Count Unsatisfied Clear Goals
        for block_to_be_clear in self.goal_clear:
            if block_to_be_clear not in is_clear_set:
                h += 1 # Needs clearing

        # 5. Penalty for Busy Arm if arm-empty is a goal
        if self.goal_arm_empty and is_holding_block is not None:
             h += 1 # Needs putdown/stack

        # 6. Return Heuristic Value
        return h
