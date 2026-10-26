from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Split by spaces, ignoring spaces within quoted strings if any (not typical in basic BW)
    # For basic BW, simple split is fine.
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing costs associated with:
    1. Blocks that are not in their correct goal position (on the table or on the correct block),
       or are currently held when they should be placed.
    2. Blocks that are currently on top of other blocks that are part of the goal structure,
       as these must be moved to allow goal stacks to be built or cleared.
    3. The state of the arm if it needs to be empty but isn't.

    # Assumptions
    - Standard Blocksworld actions (pickup, putdown, stack, unstack).
    - Each necessary move for a block from a wrong location (pickup/unstack + stack/putdown) costs approximately 2 actions.
    - Placing a block that is currently held costs approximately 1 action (stack/putdown).
    - Clearing a block costs approximately 2 actions (unstack + putdown).
    - Putting down a held block to free the arm costs 1 action.
    - The heuristic is non-admissible and designed to guide a greedy best-first search.

    # Heuristic Initialization
    The heuristic pre-processes the goal state and initial state to identify:
    - `goal_support`: A mapping from each block to the block it should be on, or 'table', based on goal `on` and `on-table` facts.
    - `goal_blocks`: The set of all blocks explicitly mentioned in goal `on` or `on-table` predicates. These are the blocks whose final position matters.
    - `goal_arm_empty`: Whether the arm must be empty in the goal.
    - `all_blocks`: The set of all blocks present in the initial state or goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine:
       - `current_support`: A mapping from each block to the block it is on, 'table', or 'arm' (if held).
       - `current_on`: A set of `(block_on_top, support_block)` tuples for all `(on ...)` facts.
       - `current_holding`: The block currently held, or None.
       - `current_arm_empty`: Whether the arm is empty.
    2. Initialize the heuristic cost `h` to 0.
    3. Add cost for blocks not in their goal position:
       For each block `B` in `self.goal_blocks`:
       - Get its goal support (`goal_sup`) from `self.goal_support`.
       - Get its current support (`curr_sup`) from the parsed state.
       - If `curr_sup` is 'arm': Add 1 to `h`. This block is held and needs to be placed (stack/putdown).
       - Else if `goal_sup` is defined (i.e., B has a specific goal position) and `curr_sup` is different from `goal_sup`: Add 2 to `h`. This block is in the wrong place and needs a full move cycle (unstack/pickup + stack/putdown).
    4. Add cost for blocks that are blocking goal blocks:
       Identify the set of `blockers`: blocks `A` such that `(on A B)` is true in the current state for some block `B` where `B` is in `self.goal_blocks`. These `blockers` are preventing `B` from being cleared or moved.
       Add 2 to `h` for each unique blocker (cost to unstack the blocker from `B` and put it down elsewhere).
    5. Add cost for the arm state if `(arm-empty)` is a goal and the arm is not empty:
       If `self.goal_arm_empty` is True and `current_arm_empty` is False: Add 1 to `h` (cost to put down the held block to achieve arm-empty).
    6. Return the total cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and initial state facts.
        """
        self.goals = task.goals

        self.goal_support = {}  # block -> support_block or 'table'
        self.goal_blocks = set()  # blocks that appear in goal on/on-table
        self.goal_arm_empty = False

        # Parse goal facts to build goal structure and identify goal blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                b_on_top, b_support = parts[1:]
                self.goal_support[b_on_top] = b_support
                self.goal_blocks.add(b_on_top)
                self.goal_blocks.add(b_support)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
                self.goal_blocks.add(block)
            elif predicate == 'arm-empty' and len(parts) == 1:
                self.goal_arm_empty = True
            # 'clear' goals are implicitly handled by the 'blockers' logic,
            # as a block needing to be clear means nothing should be on it.

        # Collect all blocks from initial state as well, as some might not be in goal
        self.all_blocks = set(self.goal_blocks) # Start with blocks from goal
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             # Add objects from relevant predicates
             if predicate in ['on', 'on-table', 'clear', 'holding']:
                 for arg in parts[1:]:
                     # Simple check if it looks like an object name (starts with a letter)
                     # More robust parsing would check object types from domain file
                     if isinstance(arg, str) and arg and arg[0].isalpha():
                         self.all_blocks.add(arg)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Parse current state
        current_support = {}  # block -> support_block or 'table' or 'arm'
        current_on = set()    # (block_on_top, support_block) tuples
        current_holding = None
        current_arm_empty = False

        # Populate current_support and current_on from state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                b_on_top, b_support = parts[1:]
                current_on.add((b_on_top, b_support))
                current_support[b_on_top] = b_support
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'arm'
                current_holding = block
            elif predicate == 'arm-empty' and len(parts) == 1:
                current_arm_empty = True

        cost = 0

        # 1. Cost for blocks not in goal position (relative to support/arm)
        # Iterate only through blocks that are part of the goal structure
        for block in self.goal_blocks:
            goal_sup = self.goal_support.get(block)
            curr_sup = current_support.get(block) # Will be None if block is not in current_support map (e.g., not on/on-table/held)

            if curr_sup == 'arm':
                # Holding the block. Needs to be placed (stack/putdown). Cost 1.
                cost += 1
            elif goal_sup is not None and curr_sup != goal_sup:
                # Block is in the wrong place (on wrong block or table). Needs moving.
                # Cost 2 (unstack/pickup + stack/putdown).
                cost += 2
            # If goal_sup is None, this block is in goal_blocks but not in goal_support.
            # This implies it's a goal_top_block but not a support or on-table block in goal,
            # which is inconsistent with standard BW goals parsed from on/on-table.
            # We only consider blocks with explicit goal support/on-table positions.


        # 2. Cost for blocks that are blocking a goal block:
        # Identify blocks that are currently on top of any block that is part of the goal structure.
        blockers = set()
        for (block_on_top, support_block) in current_on:
            if support_block in self.goal_blocks:
                 blockers.add(block_on_top)

        # Each unique blocker needs to be moved (unstack + putdown), costing 2 actions.
        cost += 2 * len(blockers)

        # 3. Cost for arm-empty goal:
        # If arm-empty is required in the goal and the arm is not empty,
        # the held block must be put down (cost 1).
        if self.goal_arm_empty and not current_arm_empty:
            cost += 1

        # The heuristic should be 0 iff the state is a goal state.
        # The current logic should achieve this:
        # If state is goal:
        # - All goal_blocks are in goal_support position (cost 0 from step 1).
        # - No blocks are on top of goal_blocks (cost 0 from step 2).
        # - Arm state matches goal_arm_empty (cost 0 from step 3).
        # If state is not goal: At least one goal predicate is false.
        # - If an (on B C) or (on-table B) goal is false, either B is in wrong place (cost >= 1/2 from step 1)
        #   or C (or B if on-table) is blocked (cost >= 2 from step 2).
        # - If a (clear B) goal is false, B must have something on it, and B must be a goal block
        #   (or its support is a goal block), leading to cost >= 2 from step 2.
        # - If (arm-empty) goal is false, cost >= 1 from step 3.
        # So, cost should be > 0 for non-goal states.

        return cost
