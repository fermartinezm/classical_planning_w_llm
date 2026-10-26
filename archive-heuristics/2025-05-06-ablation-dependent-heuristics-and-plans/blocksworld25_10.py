from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and multiple spaces between parts
    return fact.strip()[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are currently in the wrong
    position (either on the wrong block/table or held) plus the number of blocks
    that are blocking non-goal stack segments. It also adds a cost if the arm
    is not empty.

    # Assumptions
    - The goal state defines specific stacks of blocks on the table.
    - Blocks not mentioned in goal 'on' or 'on-table' predicates are not considered
      for their final position, only if they are blocking goal structures.
    - The arm should typically be empty to perform most rearrangement actions.

    # Heuristic Initialization
    - Parses the goal conditions to identify the desired 'on' relationships
      (goal_below map), blocks that should be on the table (goal_on_table set),
      and the set of all goal 'on' facts (goal_on_facts).
    - Identifies all blocks involved in goal stacks (goal_blocks).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine:
       - Which block is currently on which other block or the table (current_below map).
       - Which block is currently holding which block (held_block).
       - Which blocks are clear (clear_blocks set).
       - Which 'on' facts are currently true (current_on_facts set).
    2. Initialize the heuristic cost to 0.
    3. Count Misplaced Blocks: Iterate through each block that is part of a goal stack (goal_blocks).
       - Determine its goal position (block below it or 'table').
       - Determine its current position (block below it, 'table', or 'held').
       - If the block is currently held, add 1 to the cost.
       - If the block is currently on a block/table, but it's the wrong one according to the goal, add 1 to the cost.
    4. Count Blocking Blocks: Iterate through each 'on' relationship `(on X B)` in the current state (current_on_facts).
       - If this specific `(on X B)` relationship is *not* one of the desired 'on' relationships in the goal (i.e., not in goal_on_facts).
       - AND block X (the one on top) is currently clear (i.e., `(clear X)` is in the state).
       - Then, block X is the topmost block of a non-goal stack segment and needs to be moved. Add 1 to the cost.
    5. Arm Empty Cost: If the arm is not empty in the current state, add 1 to the cost.
    6. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        self.goal_below = {}
        self.goal_on_table = set()
        self.goal_on_facts = set()
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                block_on_top, block_below = args
                self.goal_below[block_on_top] = block_below
                self.goal_on_facts.add(goal)
                self.goal_blocks.add(block_on_top)
                self.goal_blocks.add(block_below)
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                self.goal_on_table.add(block)
                self.goal_blocks.add(block)
            elif predicate == 'clear' and len(args) == 1:
                 # Add block to goal_blocks if it's only mentioned in a clear goal.
                 # This ensures we consider all blocks mentioned in the goal.
                 self.goal_blocks.add(args[0])
            # Ignore arm-empty goal explicitly, handle it based on state

        # Refine goal_on_table: A block is a goal_on_table base only if it's not supposed to be on another block.
        self.goal_on_table = {b for b in self.goal_on_table if b not in self.goal_below}

        # Ensure blocks mentioned only as bases (values in goal_below) are also in goal_blocks
        for block_below in self.goal_below.values():
             if block_below != 'table':
                 self.goal_blocks.add(block_below)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_below = {}
        held_block = None
        clear_blocks = set()
        current_on_facts = set() # Keep track of current 'on' facts

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                block_on_top, block_below = args
                current_below[block_on_top] = block_below
                current_on_facts.add(fact)
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                current_below[block] = 'table'
            elif predicate == 'holding' and len(args) == 1:
                held_block = args[0]
            elif predicate == 'clear' and len(args) == 1:
                clear_blocks.add(args[0])
            # Ignore arm-empty for state parsing here, handle it separately

        cost = 0

        # 1. Count blocks that are on the wrong block or table, or held.
        #    We only consider blocks that are part of the goal stacks.
        for block in self.goal_blocks:
            goal_pos_B = self.goal_below.get(block, 'table' if block in self.goal_on_table else None)

            # If the block is not part of the goal structure (e.g., only mentioned in a clear goal),
            # we don't count it as misplaced in this step.
            if goal_pos_B is None:
                 continue

            current_pos_B = current_below.get(block, 'held' if (held_block == block) else 'unknown')

            if current_pos_B == 'held':
                cost += 1 # Held blocks are not in their final place
            elif current_pos_B != 'unknown' and current_pos_B != goal_pos_B:
                cost += 1 # Block is on the wrong block/table

        # 2. Count Blocking Blocks: Iterate through each 'on' relationship `(on X B)` in the current state (current_on_facts).
        #    If this specific `(on X B)` relationship is *not* one of the desired 'on' relationships in the goal (i.e., not in goal_on_facts).
        #    AND block X (the one on top) is currently clear (i.e., `(clear X)` is in the state).
        #    Then, block X is the topmost block of a non-goal stack segment and needs to be moved. Add 1 to the cost.
        for fact in current_on_facts:
            # fact is like '(on b1 b2)'
            parts = get_parts(fact)
            block_on_top = parts[1]
            # If this (on X B) relationship is not a goal relationship
            if fact not in self.goal_on_facts:
                # And the block on top is clear
                if block_on_top in clear_blocks:
                    cost += 1 # This block needs to be moved.

        # 3. Arm Empty Cost: If the arm is not empty in the current state, add 1 to the cost.
        if held_block is not None:
             cost += 1

        return cost
