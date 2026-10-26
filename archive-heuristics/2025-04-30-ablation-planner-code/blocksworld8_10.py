import functools

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if fact == '(arm-empty)':
        return ['arm-empty']
    return fact[1:-1].split()


class blocksworldHeuristic:
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not in their correct final position within the
    goal stacks and blocks that are currently on top of such misplaced blocks.
    Each such block is estimated to require 2 actions (e.g., unstack/pickup + stack/putdown)
    to be moved out of the way or into its correct place.

    # Assumptions:
    - The goal defines specific stacks of blocks. Blocks not mentioned in goal 'on' or 'on-table'
      predicates are assumed to belong on the table in any clear location.
    - Moving a block requires clearing it first, picking it up, and then placing it.
    - A simplified cost of 2 actions is assigned for moving a block that is
      either in the wrong final position or obstructing a block that is.

    # Heuristic Initialization
    - Parses the goal conditions to determine the desired support (the block or table
      it should be on) for each block that is part of a goal stack.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current support for each block
       (what it is immediately on) and which blocks are currently on top of others.
       Also, identify all blocks present in the state or goal. Track the block, if any, held by the arm.
    2. For each block, determine if it is "correctly stacked". A block B is correctly
       stacked if:
       - It is NOT currently held by the arm.
       - If it is supposed to be on the table in the goal (or not part of any goal stack), it must be currently on the table.
       - If it is supposed to be on block U in the goal, it must be currently on U, AND U must be correctly stacked.
       This check is performed recursively and memoized to handle stack dependencies efficiently.
    3. Identify the set of "incorrect blocks" - those for which the "correctly stacked" check is false.
    4. Initialize the heuristic cost to 0.
    5. Add 2 to the cost for each block in the set of incorrect blocks. This accounts for the estimated cost of moving the block itself into a better position.
    6. Identify the set of "obstructing blocks" - those that are currently on top of any block in the set of incorrect blocks.
    7. Add 2 to the cost for each block in the set of obstructing blocks. This accounts for the estimated cost of moving these blocks out of the way so the incorrect blocks can be moved.
    8. The total cost is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals
        # Blocksworld has no static facts, task.static is frozenset()
        # self.static_facts = task.static

        # Build the goal support map: block -> desired_support ('table' or another block)
        self.goal_support = {}
        # Collect all blocks mentioned in the goal
        self.goal_blocks = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
                self.goal_blocks.add(block)
                self.goal_blocks.add(support)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_support[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore 'clear' goals for building support structure

    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # 1. Parse the current state
        current_support = {}
        blocks_on_top = {'table': set()} # Initialize table support
        all_blocks = set(self.goal_blocks) # Start with blocks from goal
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                current_support[block] = support
                blocks_on_top.setdefault(support, set()).add(block)
                all_blocks.add(block)
                all_blocks.add(support)
            elif parts[0] == 'on-table':
                block = parts[1]
                current_support[block] = 'table'
                blocks_on_top['table'].add(block)
                all_blocks.add(block)
            elif parts[0] == 'holding':
                 held_block = parts[1]
                 current_support[held_block] = 'arm' # Represent being held
                 all_blocks.add(held_block)
            # Ignore 'clear', 'arm-empty' for this calculation

        # Ensure all blocks seen are in blocks_on_top dict, even if nothing is on them
        for block in all_blocks:
             blocks_on_top.setdefault(block, set())

        # 2. Compute is_correctly_stacked status for each block
        # Use memoization for efficiency
        @functools.lru_cache(None)
        def is_correctly_stacked(block):
            # A held block is never correctly stacked
            if current_support.get(block) == 'arm':
                return False

            current_sup = current_support.get(block)

            # If block is not in goal_support, it should be on the table
            if block not in self.goal_support:
                return current_sup == 'table'

            # If block is in goal_support, check its desired position
            desired_support = self.goal_support[block]

            # If block should be on table
            if desired_support == 'table':
                return current_sup == 'table'

            # If block should be on another block U
            # Check if it's currently on U AND U is correctly stacked
            if current_sup == desired_support:
                 # Check if the desired_support block exists in the state/goal blocks
                 # It might not exist if the goal is impossible or malformed,
                 # but assuming valid PDDL, it should exist.
                 if desired_support not in all_blocks:
                     # This case should ideally not happen in valid problems
                     return False
                 return is_correctly_stacked(desired_support)
            else:
                return False

        # 3. Identify incorrect blocks
        incorrect_blocks = set()
        for block in all_blocks:
            if not is_correctly_stacked(block):
                incorrect_blocks.add(block)

        # 4. Initialize cost
        cost = 0

        # 5. Add cost for incorrect blocks
        cost += len(incorrect_blocks) * 2

        # 6. Identify and add cost for obstructing blocks
        obstructing_blocks = set()
        for block in incorrect_blocks:
            # Blocks on top of an incorrect block are obstructing its movement
            # Held blocks don't have blocks on top of them in the state representation
            if current_support.get(block) != 'arm':
                obstructing_blocks.update(blocks_on_top.get(block, set()))

        cost += len(obstructing_blocks) * 2

        # 8. Return total cost
        return cost
