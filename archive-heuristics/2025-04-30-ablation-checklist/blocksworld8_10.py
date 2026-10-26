# Assuming heuristic_base.py exists and defines a Heuristic base class
# For submission, this base class might need to be included or assumed available.
# If running standalone without the planning framework, you might need a dummy
# Heuristic class and a dummy Task class for testing.

# Example dummy classes if needed:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError
# class Task:
#      def __init__(self, name, facts, initial_state, goals, operators, static):
#          self.name = name
#          self.facts = facts
#          self.initial_state = initial_state
#          self.goals = goals
#          self.operators = operators
#          self.static = static
# class Node:
#      def __init__(self, state, parent=None, action=None, cost=0):
#          self.state = state
#          self.parent = parent
#          self.action = action
#          self.cost = cost


from fnmatch import fnmatch
# from heuristics.heuristic_base import Heuristic # Uncomment if running within the framework

# Assume Heuristic base class is available as per problem description
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal
    by counting:
    1. Blocks that are not on their correct goal base (either another block or the table).
    2. Blocks that are currently stacked directly on top of a block that is
       not on its correct goal base.
    3. An additional cost if the arm is holding a block but the goal requires
       the arm to be empty.

    This heuristic is non-admissible but aims to be informative for greedy
    best-first search by identifying key discrepancies related to block positions
    and necessary unstacking operations.

    # Heuristic Initialization
    - Parses the goal state to determine the desired base for each block
      and which block should be on top of which (or if it should be clear).
    - Collects all block names from the initial and goal states.
    - Assumes blocks not explicitly placed in the goal should be on the table.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal position (block or 'table') for every block.
       Store this in `self.goal_below`. Blocks not in 'on' or 'on-table' goals
       are assumed to have a goal of being 'on-table'.
    2. Identify the block that should be directly on top of each block in the goal,
       or if it should be clear. Store this in `self.goal_above`.
    3. In the current state, determine the base for every block ('on', 'on-table', or 'holding').
       Store this in `current_below`.
    4. In the current state, determine which block is directly on top of each block.
       Store this in `current_above`.
    5. Identify blocks whose current base is different from their goal base.
       These blocks are considered "misplaced" relative to their base.
    6. Initialize heuristic value `h = 0`.
    7. Count the number of blocks identified in step 5 (misplaced base blocks).
       Add this count to `h`. Each such block needs to be moved.
    8. For each block identified in step 5 (misplaced base block), check if there is a block
       currently stacked directly on top of it in the current state. If so, increment `h`.
       This block on top needs to be unstacked to allow the misplaced block below to move.
    9. Check if the arm is holding a block and the goal requires the arm to be empty.
       If both are true, increment `h`. The held block needs to be put down.
    10. Return the total value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and blocks.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        self.all_blocks = set()

        # Build goal configuration maps
        # Maps a block to the block or 'table' it should be directly on in the goal.
        self.goal_below = {}
        # Maps a block to the block that should be directly on top of it in the goal, or None if it should be clear.
        self.goal_above = {}
        self.goal_arm_empty = False

        # Collect all blocks from initial state and goal state
        # This ensures we consider all relevant objects in the domain
        for fact in task.initial_state:
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate in ["on", "on-table", "holding"]:
                 if len(parts) > 1: self.all_blocks.add(parts[1])
                 if predicate == "on" and len(parts) > 2:
                     self.all_blocks.add(parts[2])

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on" and len(parts) > 2:
                block, base = parts[1], parts[2]
                self.goal_below[block] = base
                self.goal_above[base] = block # Assumes only one block on top in goal
                self.all_blocks.add(block)
                self.all_blocks.add(base)
            elif predicate == "on-table" and len(parts) > 1:
                block = parts[1]
                self.goal_below[block] = 'table'
                self.all_blocks.add(block)
            elif predicate == "clear" and len(parts) > 1:
                block = parts[1]
                self.goal_above[block] = None # Explicitly mark as needing to be clear
                self.all_blocks.add(block)
            elif predicate == "arm-empty":
                self.goal_arm_empty = True

        # Ensure 'table' is not treated as a block object
        if 'table' in self.all_blocks:
             self.all_blocks.remove('table')

        # For blocks not mentioned as being 'on' or 'on-table' in the goal,
        # assume the goal is for them to be on the table.
        for block in self.all_blocks:
            if block not in self.goal_below:
                self.goal_below[block] = 'table'


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current configuration maps
        current_below = {} # block -> block or 'table' or 'holding'
        current_above = {} # block -> block or None
        holding_block = None

        # Initialize current_above to None for all blocks
        for block in self.all_blocks:
            current_above[block] = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on" and len(parts) > 2:
                block, base = parts[1], parts[2]
                current_below[block] = base
                current_above[base] = block
            elif predicate == "on-table" and len(parts) > 1:
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == "holding" and len(parts) > 1:
                block = parts[1]
                current_below[block] = 'holding'
                holding_block = block
            # 'clear' facts are implicitly captured by the absence in current_above
            # 'arm-empty' is checked separately

        h = 0
        misplaced_base_blocks = set()

        # Count blocks with misplaced bases
        for block in self.all_blocks:
            current_b = current_below.get(block)
            goal_b = self.goal_below.get(block) # Default handled in init

            if current_b != goal_b:
                h += 1 # Block needs to move
                misplaced_base_blocks.add(block)

        # Count blocks on top of misplaced blocks
        # These blocks need to be unstacked to free up the misplaced block below.
        for block in misplaced_base_blocks:
            block_on_top = current_above.get(block)
            if block_on_top is not None:
                h += 1 # Block on top needs to be unstacked

        # Add cost for holding a block if arm-empty is a goal
        if self.goal_arm_empty and holding_block is not None:
             h += 1 # Held block needs to be put down

        return h

