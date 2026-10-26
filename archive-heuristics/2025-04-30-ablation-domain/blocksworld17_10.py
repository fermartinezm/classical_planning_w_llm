from heuristics.heuristic_base import Heuristic

def get_parts(fact_string):
    """Parses a PDDL fact string into a list of parts."""
    # Remove parentheses and split by spaces
    return fact_string[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are in the wrong position relative to their base
    (or the table) and blocks that are obstructing positions needed for goal stacks.
    Each such "misplaced" or "blocking" block is estimated to require approximately
    2 actions (e.g., unstack/pickup + stack/putdown) to fix its state or move it
    out of the way. An additional cost of 1 is added if the arm is holding a block.

    # Assumptions
    - Actions have unit cost.
    - The primary goal is to achieve the specified `on` and `on-table` predicates,
      forming specific stacks. `clear` goals are implicitly handled as the top
      of a correctly built stack segment.
    - Blocks not mentioned in the goal can be placed anywhere as long as they
      don't obstruct goal requirements.

    # Heuristic Initialization
    - Extracts the desired base for each block from the goal `on` and `on-table`
      predicates (`self.goal_below`, `self.goal_on_table`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the base of each block (`current_below`,
       `current_on_table`), which blocks are clear (`current_clear`), and if the
       arm is holding a block (`current_holding`).
    2. Identify "misplaced blocks": These are blocks that are part of the goal
       configuration (appear as keys in `self.goal_below` or in `self.goal_on_table`)
       but are not currently on their correct base or the table as specified by the goal.
       Add 2 to the heuristic cost for each misplaced block. This estimates the
       cost to move the block to its correct base.
    3. Identify blocks that "need to be clear": This set includes all blocks that
       serve as a base in a goal `on` predicate, plus all the "misplaced blocks"
       identified in step 2 (since a misplaced block needs to be moved, its current
       location might need to be clear for something else, or it needs to be picked
       up, requiring it to be clear). 'table' is excluded as it doesn't need clearing.
    4. Identify "blocking blocks": These are blocks that are currently stacked
       directly on top of any block identified in step 3 (a block that needs to be clear).
       Add 2 to the heuristic cost for each blocking block. This estimates the
       cost to move the blocking block out of the way.
    5. If the arm is currently holding a block, add 1 to the heuristic cost.
       This accounts for the immediate action needed to put the block down or stack it.
    6. The total heuristic value is the sum of costs from steps 2, 4, and 5.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals

        # Build goal configuration dictionaries
        self.goal_below = {}
        self.goal_on_table = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == 'on':
                block, base = parts[1:]
                self.goal_below[block] = base
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_on_table.add(block)
            # Ignore 'clear' goals for heuristic structure, they are implicitly handled
            # by ensuring blocks below are correctly placed and clear.

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Parse the current state
        current_below = {}
        current_on_table = set()
        current_clear = set()
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, base = parts[1:]
                current_below[block] = base
            elif predicate == 'on-table':
                block = parts[1]
                current_on_table.add(block)
            elif predicate == 'clear':
                block = parts[1]
                current_clear.add(block)
            elif predicate == 'holding':
                current_holding = parts[1]
            # Ignore 'arm-empty'

        total_cost = 0

        # 2. Identify misplaced blocks
        misplaced_blocks = set()
        # Check blocks that should be on a specific base
        for block, goal_base in self.goal_below.items():
            if current_below.get(block) != goal_base:
                misplaced_blocks.add(block)
        # Check blocks that should be on the table
        for block in self.goal_on_table:
            if block not in current_on_table:
                 misplaced_blocks.add(block)

        total_cost += 2 * len(misplaced_blocks)

        # 3. Identify blocks that need to be clear
        # These are blocks that are goal bases OR are misplaced (and thus need to be moved)
        blocks_that_are_goal_bases = {base for base in self.goal_below.values() if base != 'table'}
        blocks_that_need_to_be_clear = blocks_that_are_goal_bases.union(misplaced_blocks)
        # 'table' does not need to be cleared, so remove it if present
        blocks_that_need_to_be_clear.discard('table')


        # 4. Identify blocking blocks
        # These are blocks currently on top of blocks that need to be clear
        blocking_blocks = set()
        for block_A, base_B in current_below.items():
             if base_B in blocks_that_need_to_be_clear:
                 blocking_blocks.add(block_A)

        total_cost += 2 * len(blocking_blocks)

        # 5. Penalty for holding a block
        if current_holding is not None:
            total_cost += 1

        return total_cost
