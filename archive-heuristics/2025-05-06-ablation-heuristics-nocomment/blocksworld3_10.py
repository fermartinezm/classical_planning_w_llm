from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to parse a PDDL fact string into its components."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost to reach the goal state by considering
    blocks that are not in their goal location and blocks that are stacked
    on top of them, as these blocks are "in the way". It is designed for
    greedy best-first search and is not admissible.

    Summary:
    The heuristic counts the number of blocks that are not in their desired
    goal position (either on a specific block or on the table). Additionally,
    it counts blocks that are currently stacked on top of any block that
    is itself misplaced. If the calculated count is zero but the state is
    not the goal state, it returns 1 to ensure the heuristic is zero only
    at the goal.

    Assumptions:
    - The input is a valid Blocksworld state represented as a frozenset of
      PDDL fact strings.
    - The task object provides access to initial_state, goals, and static
      facts, and a goal_reached method.
    - The domain uses the standard Blocksworld predicates: on, on-table,
      clear, holding, arm-empty.
    - Blocks are the only objects relevant to the heuristic calculation.

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes the goal state to
    determine the desired location (parent block or 'table') for each block
    mentioned in the goal 'on' or 'on-table' facts. It also identifies which
    blocks must be clear in the goal state (though goal_clear is not directly
    used in the final heuristic formula, it's extracted as per instructions).
    All blocks present in the initial state or the goal state are collected
    from the initial state and goal facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is the goal state using the task's
       goal_reached method. If it is, the heuristic value is 0.
    2. If not the goal state, initialize a raw heuristic count to 0.
    3. Parse the current state to determine the current location (on a block,
       on the table, or in the arm) and the block directly on top for every
       block in the problem.
    4. Identify "misplaced" blocks: Iterate through all blocks collected
       during initialization. If a block is mentioned in the goal's 'on' or
       'on-table' facts, and its current location is different from its goal
       location, increment the raw count by 1 and mark the block as misplaced.
    5. Identify "blocking" blocks: Iterate through all blocks. For each block,
       traverse the stack of blocks currently on top of it using the parsed
       'current_top' information. If any block in this stack is sitting on a
       block that was marked as misplaced in the previous step, increment the
       raw count by 1 for the block that is directly on top of the misplaced
       block. This counts blocks that need to be moved out of the way.
    6. After calculating the raw count (sum of misplaced blocks and blocking blocks),
       if the raw count is 0, return 1 (since we already know it's not the goal state).
       Otherwise, return the raw count.
    """
    def __init__(self, task):
        self.task = task
        self.goal_pos = {} # block -> parent_block or 'table'
        self.goal_clear = set() # set of blocks that must be clear (extracted but not used in final h)
        self.all_blocks = set() # set of all blocks in the problem

        # Collect all blocks from initial state facts
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] in ['on', 'on-table', 'holding', 'clear']:
                 # Add all arguments except the predicate name
                for arg in parts[1:]:
                    self.all_blocks.add(arg)

        # Process goal facts to get goal_pos and goal_clear, and collect blocks
        for goal in task.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                b1, b2 = parts[1:]
                self.goal_pos[b1] = b2
                self.all_blocks.add(b1)
                self.all_blocks.add(b2)
            elif predicate == 'on-table':
                b1 = parts[1]
                self.goal_pos[b1] = 'table'
                self.all_blocks.add(b1)
            elif predicate == 'clear':
                b1 = parts[1]
                self.goal_clear.add(b1)
                self.all_blocks.add(b1)
            # Ignore arm-empty in goal for heuristic calculation

        # 'table' is not a block object
        self.all_blocks.discard('table')

    def __call__(self, node):
        state = node.state

        # 1. Check if goal reached
        if self.task.goal_reached(state):
            return 0

        # 2. Initialize raw heuristic count
        raw_cost = 0

        # 3. Parse state to get current_pos and current_top
        current_pos = {} # block -> parent_block or 'table' or 'arm'
        current_top = {b: None for b in self.all_blocks} # block -> block_on_top or None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                b1, b2 = parts[1:]
                current_pos[b1] = b2
                current_top[b2] = b1
            elif predicate == 'on-table':
                b1 = parts[1]
                current_pos[b1] = 'table'
            elif predicate == 'holding':
                b1 = parts[1]
                current_pos[b1] = 'arm'
            # 'clear' and 'arm-empty' are not needed for pos/top mapping

        # Ensure current_pos contains all blocks from all_blocks.
        # Blocks not found in on/on-table/holding facts would indicate an invalid state.
        # We assume valid states where every block is in one of these configurations.
        # The loop below will handle blocks correctly as long as they are in current_pos.

        # 4. Identify misplaced blocks and calculate base cost
        misplaced_blocks = set()
        for b in self.all_blocks:
            goal_loc = self.goal_pos.get(b)
            current_loc = current_pos.get(b) # Use .get() for safety, though should be present

            # A block is misplaced if its goal location is defined and different from its current location
            if goal_loc is not None and current_loc != goal_loc:
                raw_cost += 1 # Cost for the block itself needing to be moved
                misplaced_blocks.add(b)

        # 5. Identify blocking blocks and calculate penalty
        # A block is blocking if it is on top of a misplaced block
        for b in self.all_blocks: # Iterate through potential bottom blocks
            current = b
            while True:
                block_on_top = current_top.get(current)
                if block_on_top is None:
                    break # Reached the top of the current stack

                # block_on_top is currently on 'current'.
                # If 'current' is misplaced, block_on_top is in the way.
                if current in misplaced_blocks:
                    raw_cost += 1 # Penalty for block_on_top being in the way
                
                current = block_on_top # Move up the current stack

        # 6. Apply final adjustment
        # If the raw cost is 0, it means all blocks mentioned in goal_pos are in their goal_pos,
        # and nothing is on top of any block that is in goal_pos but not in its goal_pos.
        # This doesn't guarantee the goal is reached (e.g., clear facts might be missing).
        # Since we already checked that it's not the goal state, a raw_cost of 0
        # indicates an underestimate. Return 1 instead of 0.
        if raw_cost == 0:
            return 1
        else:
            return raw_cost
