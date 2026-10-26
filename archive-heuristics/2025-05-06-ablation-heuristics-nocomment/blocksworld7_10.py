# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Parses a PDDL fact string into a list of parts."""
    # Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
    # Assumes fact is a string starting with '(' and ending with ')'
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        three components:
        1. The number of blocks that are part of a goal stack but are not
           currently in their correct position relative to the base of the
           stack, considering the goal configuration.
        2. The number of blocks that are required to be clear in the goal
           state (either explicitly by a (clear ?) goal or implicitly by
           having another block stacked on them in the goal) but are not
           clear in the current state.
        3. A penalty if the arm is not empty.

    Assumptions:
        - The goal state defines a set of stacks and blocks on the table.
        - All blocks mentioned in (on ?x ?y) or (on-table ?x) goal facts
          form valid stacks ultimately resting on the table.
        - The heuristic is non-admissible and designed for greedy best-first
          search to minimize node expansions.
        - The input PDDL goal is well-formed for Blocksworld (no cycles in
          'on' goals, no block simultaneously on table and on another block
          in the goal).

    Heuristic Initialization:
        The constructor processes the goal facts from the task.
        - `self.goal_pos`: A dictionary mapping each block (that is part of
          a goal stack or goal on-table fact) to the block it should be
          directly on top of in the goal state, or None if it should be
          on the table.
        - `self.goal_blocks`: A set containing all blocks that are keys or
          values in `self.goal_pos`. These are the blocks whose final
          position is specified by the goal.
        - `self.goal_needs_clear`: A set containing all blocks that must
          be clear in the goal state, either because there is a (clear ?)
          goal fact for them, or because another block is supposed to be
          stacked directly on top of them according to an (on ?x ?) goal fact.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state, the heuristic value is computed as h1 + h2 + h3:

        1. Calculate h1 (Misplaced blocks in goal stacks):
           - Identify the current position of every block: on another block,
             on the table, or held by the arm. Store this in `current_pos`.
           - Define a recursive helper function `is_correct(block)` that checks
             if a block is in its correct goal position AND the block below it
             (if any, according to the goal) is also correctly positioned.
             Memoization is used to avoid redundant calculations.
             - A block `B` is `is_correct` if:
               - It is not in `self.goal_pos` (its position is not specified by the goal).
               - OR its goal position is on the table (`self.goal_pos[B]` is None)
                 AND it is currently on the table (`current_pos.get(B)` is None).
               - OR its goal position is on block `UnderB` (`self.goal_pos[B]` is `UnderB`)
                 AND it is currently on `UnderB` (`current_pos.get(B)` is `UnderB`)
                 AND `is_correct(UnderB)` is true.
           - Initialize `h1 = 0`.
           - Iterate through every block in `self.goal_blocks`. If `is_correct(block)`
             returns False, increment `h1`.

        2. Calculate h2 (Blocks that need to be clear but aren't):
           - Identify which blocks are currently clear in the state. A block is
             clear if no other block is currently on top of it and it's not held.
             Store this in `current_clear_set`.
           - Initialize `h2 = 0`.
           - Iterate through every block in `self.goal_needs_clear`. If the block
             is not in `current_clear_set`, increment `h2`.

        3. Calculate h3 (Arm penalty):
           - Check if the arm is empty in the current state.
           - Initialize `h3 = 1` if the arm is not empty, otherwise `h3 = 0`.

        - The total heuristic value is `h1 + h2 + h3`.
        - An explicit check is performed at the beginning of `__call__` to return
          0 immediately if the state is the goal state, ensuring the heuristic
          is zero only for goal states.
    """
    def __init__(self, task):
        self.goals = task.goals
        # static_facts = task.static # Blocksworld has no static facts

        # 1. Parse goal facts to build goal structure
        self.goal_pos = {} # block -> block_below or None (for on-table)
        self.goal_blocks = set() # All blocks mentioned in goal_pos
        self.goal_needs_clear = set() # Blocks that need to be clear in goal

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, under_block = parts[1], parts[2]
                self.goal_pos[block] = under_block
                self.goal_blocks.add(block)
                self.goal_blocks.add(under_block)
                self.goal_needs_clear.add(under_block) # Block below needs to be clear
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_pos[block] = None # None signifies on the table
                self.goal_blocks.add(block)
            elif predicate == 'clear':
                block = parts[1]
                self.goal_needs_clear.add(block)
            # Ignore other predicates like arm-empty or holding if they appear in goals

    def __call__(self, node):
        state = node.state

        # Heuristic is 0 only for goal states
        if self.goals.issubset(state):
             return 0

        # 2. Parse current state
        current_pos = {} # block -> block_below or None (on-table) or 'holding'
        current_clear_set = set()
        is_arm_empty = False
        blocks_on_top = set() # Helper to find clear blocks

        # Collect all blocks present in the state facts
        all_blocks_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, under_block = parts[1], parts[2]
                current_pos[block] = under_block
                blocks_on_top.add(block)
                all_blocks_in_state.add(block)
                all_blocks_in_state.add(under_block)
            elif predicate == 'on-table':
                block = parts[1]
                current_pos[block] = None
                all_blocks_in_state.add(block)
            elif predicate == 'holding':
                block = parts[1]
                current_pos[block] = 'holding' # Special value for held block
                blocks_on_top.add(block) # Held block is not clear underneath (conceptually)
                all_blocks_in_state.add(block)
            elif predicate == 'arm-empty':
                is_arm_empty = True
            # Ignore 'clear' facts in state for parsing positions, we calculate clear status

        # Determine which blocks are clear in the current state
        # Iterate over all blocks found in the state
        for block in all_blocks_in_state:
             # A block is clear if nothing is on top of it (i.e., it's not in the blocks_on_top set)
             if block not in blocks_on_top:
                 current_clear_set.add(block)

        # Calculate h1 (Misplaced blocks in goal stacks)
        memo = {} # Memoization dictionary for is_correct

        def is_correct(block):
            """
            Recursive helper to check if a block is in its correct goal position
            relative to the base, considering the goal stack structure.
            """
            if block in memo:
                return memo[block]

            # If block is not part of the goal structure, it doesn't contribute
            # to the 'misplaced in goal stack' count.
            if block not in self.goal_pos:
                memo[block] = True
                return True

            goal_below = self.goal_pos[block]
            current_below = current_pos.get(block) # Get current position, None if not found

            result = False
            if goal_below is None: # Goal is on table
                result = (current_below is None) # Is it currently on table?
            else: # Goal is on another block
                # Is it currently on the correct block AND is the block below correct?
                result = (current_below == goal_below and is_correct(goal_below))

            memo[block] = result
            return result

        h1 = 0
        # Only consider blocks that are part of the goal structure for h1
        for block in self.goal_blocks:
            if not is_correct(block):
                h1 += 1

        # Calculate h2 (Blocks that need to be clear but aren't)
        h2 = 0
        for block in self.goal_needs_clear:
            if block not in current_clear_set:
                h2 += 1

        # Calculate h3 (Arm penalty)
        h3 = 0 if is_arm_empty else 1

        return h1 + h2 + h3
