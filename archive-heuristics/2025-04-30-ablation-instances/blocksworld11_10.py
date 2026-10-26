from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a valid PDDL fact string like '(predicate arg1 arg2)'
    # Handles potential empty fact string or invalid format gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
         return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not currently in their
    correct goal position relative to a correctly positioned base (either the table
    or another block that is itself correctly positioned according to the goal).
    It counts the number of blocks that are part of the goal configuration but
    are not yet part of a correctly built stack segment originating from the table.

    # Assumptions
    - The goal state is defined primarily by `(on ?x ?y)` and `(on-table ?x)` predicates,
      forming specific stack configurations.
    - All blocks relevant to achieving the goal configuration are mentioned in
      these `on` or `on-table` goal predicates.
    - The heuristic focuses only on satisfying the `on` and `on-table` goal predicates.
      Other goal predicates like `(clear ?x)` or `(arm-empty)` are not directly
      counted, as they are typically side effects of achieving the main stack goals.

    # Heuristic Initialization
    - The heuristic stores the set of all goal facts (`self.goals`).
    - It identifies all blocks that are mentioned in the `on` or `on-table` goal
      predicates and stores them in `self.goal_blocks`. These are the blocks
      whose final position matters for the heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is computed by identifying which blocks are "correctly supported"
    according to the goal configuration and the current state, starting from the table.

    1. Initialize a set `correctly_supported` with the literal 'table', as the table
       is the ultimate base for all stacks.
    2. Identify blocks whose goal is `(on-table X)` and which are currently `(on-table X)`
       in the state. These blocks are correctly supported from the base layer. Add these
       blocks to `correctly_supported` and also to a set `newly_supported` for the
       first iteration.
    3. Enter an iterative loop that continues as long as `newly_supported` is not empty:
       a. Create a copy of `newly_supported` called `current_newly_supported`. This set
          represents the blocks that were just identified as correctly supported in the
          previous iteration.
       b. Clear `newly_supported` for the current iteration's findings.
       c. Iterate through all goal facts `(on X Y)`:
          - If block `X` is a goal block and its required base `Y` is in `current_newly_supported`
            (meaning `Y` was just found to be correctly supported), check if the fact
            `(on X Y)` is true in the current state.
          - If `(on X Y)` is true and `X` is not already in `correctly_supported`, then `X`
            is now correctly supported. Add `X` to both `correctly_supported` and `newly_supported`.
    4. The loop terminates when no new blocks are found to be correctly supported in an iteration.
    5. The set `correctly_supported` now contains 'table' and all goal blocks that are
       part of a correctly built stack segment from the table up.
    6. The heuristic value is the total number of blocks in `self.goal_blocks` minus the
       number of goal blocks found in the final `correctly_supported` set. This counts
       the goal blocks that are *not* correctly supported.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying
        all blocks involved in the goal configuration.
        """
        self.goals = task.goals
        self.goal_blocks = set()

        # Identify all blocks mentioned in 'on' or 'on-table' goal predicates
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_blocks.add(block)
                self.goal_blocks.add(base)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_blocks.add(block)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions based on
        the number of goal blocks not correctly supported.
        """
        state = node.state

        # 'table' is always a correctly supported base
        correctly_supported = {'table'}

        # Identify blocks whose goal is (on-table X) and are currently (on-table X)
        # These are the initial set of newly supported blocks (excluding 'table')
        initial_on_table_goals_met = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             if parts[0] == "on-table" and len(parts) == 2:
                 block = parts[1]
                 # Check if this block is a goal block and is currently on the table
                 if block in self.goal_blocks and f"(on-table {block})" in state:
                     initial_on_table_goals_met.add(block)

        correctly_supported.update(initial_on_table_goals_met)
        newly_supported = initial_on_table_goals_met.copy()

        # Iteratively find correctly supported blocks based on 'on' goals
        # A block X is correctly supported if (on X Y) is a goal, (on X Y) is true,
        # AND Y was *just* added to correctly_supported in the previous step.
        # This ensures we build from the bottom up layer by layer.
        while newly_supported:
            current_newly_supported = newly_supported.copy()
            newly_supported = set()

            # Check goal facts (on X Y) where Y is now correctly supported (was newly supported in the last step)
            for goal in self.goals:
                parts = get_parts(goal)
                if not parts: continue
                if parts[0] == "on" and len(parts) == 3:
                    block, base = parts[1], parts[2]
                    # Check if this block is a goal block and its base was newly supported in the *last* iteration
                    if block in self.goal_blocks and base in current_newly_supported:
                         # Check if the 'on' fact is true in the current state
                        if f"(on {block} {base})" in state:
                            # If block is not already marked correctly supported, add it
                            if block not in correctly_supported:
                                correctly_supported.add(block)
                                newly_supported.add(block)

        # The heuristic is the number of goal blocks that are NOT correctly supported
        # correctly_supported includes 'table', which is not a block.
        # We only care about blocks from self.goal_blocks.
        correctly_supported_goal_blocks = correctly_supported.intersection(self.goal_blocks)
        heuristic_value = len(self.goal_blocks) - len(correctly_supported_goal_blocks)

        return heuristic_value
