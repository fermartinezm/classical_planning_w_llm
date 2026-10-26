from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
        Estimates the cost by counting blocks that are not in their correct
        position within the goal stacks, blocks that are currently
        sitting on top of such misplaced blocks, and blocks that need to be
        clear but are not.

    Assumptions:
        - The input state and goal are valid for the blocksworld domain.
        - All blocks mentioned in the goal are present in the initial state
          and subsequent states.
        - The goal consists of 'on', 'on-table', and 'clear' predicates.
        - The state representation includes 'on', 'on-table', 'clear',
          'holding', and 'arm-empty' facts.

    Heuristic Initialization:
        - Parses the goal state to build the desired stack structure.
        - Creates a mapping `goal_succ` where `goal_succ[B]` is the block
          that block B should be directly on top of in the goal, or 'table'
          if B should be on the table.
        - Identifies the set of all blocks involved in the goal stacks (`goal_blocks`).
        - Identifies the set of blocks that should be clear in the goal (`goal_clear`).

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state to determine the current position of each block.
           Create a mapping `current_succ` where `current_succ[B]` is the block
           that block B is currently directly on top of, or 'table' if B is on
           the table. Identify the `held_block`.
           Also, collect all 'on' facts and 'clear' facts in the current state.
        2. Determine the set of blocks that are "in-stack-correct". A block B is
           in-stack-correct if its goal successor is 'table' and its current
           successor is 'table', OR if its goal successor Y is the same as its
           current successor, and Y is itself in-stack_correct. This is computed
           iteratively starting from blocks correctly placed on the table according
           to the goal.
        3. Identify `wrong_blocks`: the set of blocks in `goal_blocks` that are
           *not* in `in_stack_correct`. These are the blocks whose position
           relative to the goal stack base is incorrect.
        4. Calculate `count1`: The number of blocks in `wrong_blocks`. Each of
           these blocks needs to be moved at least once to get into its correct
           goal stack. Moving a block typically requires two actions (pickup/unstack
           + putdown/stack). This contributes `2 * count1` to the heuristic.
        5. Calculate `count2`: The number of blocks A such that `(on A B)` is true
           in the current state and B is in `wrong_blocks`. These blocks A are
           sitting on top of a block B that is in a wrong position relative to
           the goal stack base. Block A must be moved off B to allow B to be moved.
           This requires at least one action (unstack). This contributes `count2`
           to the heuristic.
        6. Calculate `count3`: The number of blocks B such that `(clear B)` is a
           goal fact and `(clear B)` is not true in the current state. These blocks
           need to be cleared. This contributes `count3` to the heuristic.
        7. Check if the arm is holding a block. If so, add 1 to the heuristic value,
           as this block must be placed before the arm is free.
        8. The final heuristic value is `2 * count1 + count2 + count3 + (1 if arm is holding)`.
           This heuristic is non-admissible but aims to guide the search towards
           building the correct goal stacks by penalizing misplaced blocks,
           blocks blocking them, and blocks that need to be clear.
    """
    def __init__(self, task):
        self.goals = task.goals

        # Precompute goal structure
        self.goal_succ = {}
        self.goal_blocks = set()
        self.goal_clear = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'on':
                block, under_block = parts[1], parts[2]
                self.goal_succ[block] = under_block
                self.goal_blocks.add(block)
                self.goal_blocks.add(under_block)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_succ[block] = 'table'
                self.goal_blocks.add(block)
            elif parts[0] == 'clear':
                self.goal_clear.add(parts[1])

    def __call__(self, node):
        state = node.state

        # 1. Parse current state
        current_succ = {}
        held_block = None
        current_on_facts = set()
        current_clear_facts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, under_block = parts[1], parts[2]
                current_succ[block] = under_block
                current_on_facts.add(fact)
            elif parts[0] == 'on-table':
                block = parts[1]
                current_succ[block] = 'table'
            elif parts[0] == 'holding':
                held_block = parts[1]
            elif parts[0] == 'clear':
                 current_clear_facts.add(fact)

        # 2. Determine in-stack-correct blocks
        in_stack_correct = set()
        # Initialize with blocks correctly on the table according to the goal
        for block in self.goal_blocks:
             if self.goal_succ.get(block) == 'table' and current_succ.get(block) == 'table':
                 in_stack_correct.add(block)

        # Iteratively add blocks correctly stacked on in-stack-correct blocks
        changed = True
        while changed:
            changed = False
            # Iterate over blocks whose goal successor is known and is a block (not 'table')
            # We only care about goal blocks for in_stack_correct calculation
            for block in list(self.goal_succ.keys()): # Iterate over a copy as set might change
                if block not in in_stack_correct:
                    goal_under = self.goal_succ.get(block)
                    if goal_under is not None and goal_under != 'table':
                        # Check if block is currently on its goal successor
                        current_under = current_succ.get(block)
                        if current_under == goal_under:
                            # Check if the block below it (goal_under) is in-stack-correct
                            if goal_under in in_stack_correct:
                                in_stack_correct.add(block)
                                changed = True

        # 3. Identify wrong_blocks
        wrong_blocks = self.goal_blocks - in_stack_correct

        # 4. Calculate count1
        count1 = len(wrong_blocks)

        # 5. Calculate count2
        # Count blocks A currently on B, where B is in wrong_blocks
        count2 = 0
        for fact in current_on_facts:
            parts = get_parts(fact)
            # block_a = parts[1] # The block on top
            block_b = parts[2] # The block below
            if block_b in wrong_blocks:
                 count2 += 1

        # 6. Calculate count3 (unsatisfied clear goals)
        count3 = len(self.goal_clear - current_clear_facts)

        # 7. Check if arm is holding a block
        held_cost = 1 if held_block else 0

        # 8. Calculate final heuristic value
        heuristic_value = 2 * count1 + count2 + count3 + held_cost

        return heuristic_value
