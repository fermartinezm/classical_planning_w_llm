from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Removes leading/trailing parentheses and splits by space."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
        Estimates the cost by counting the number of blocks that are not
        part of a "correct prefix" of a goal stack. A block is part of a
        correct prefix if it is in its goal position relative to the block
        below it, and the block below it is also part of a correct prefix
        (recursively down to the table).

    Assumptions:
        - The goal state primarily consists of 'on' and 'on-table' predicates
          defining the desired block configuration. 'clear' and 'arm-empty'
          goals are typically consequences of achieving the stack configuration.
        - The heuristic is non-admissible and designed for greedy best-first search.
        - The heuristic value is 0 if and only if the state is a goal state
          (with respect to 'on' and 'on-table' predicates).

    Heuristic Initialization:
        The constructor pre-processes the goal facts to build a map
        `self.goal_below` which stores the desired block immediately below
        each block in the goal state ('table' for blocks on the table).
        It also identifies all blocks involved in the goal state in
        `self.goal_blocks`.
        A map `self.goal_on_map` is created to quickly find which blocks
        should be on top of a given block according to the goal.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state to determine the immediate block below
           each block (`current_below` map).
        2. Initialize a boolean map `is_correct_prefix` for all blocks
           involved in the goal, setting all values to False initially.
        3. Initialize a queue (`deque`) with blocks `X` that are supposed
           to be on the table in the goal (`self.goal_below.get(X) == 'table'`)
           AND are currently on the table in the state (`current_below.get(X) == 'table'`).
           Mark these blocks as `is_correct_prefix = True` and add them to the queue.
           These blocks form the base of correct goal stacks.
        4. Process the queue:
           - Dequeue a block `current_block`.
           - Find all blocks `next_block` that are supposed to be directly
             on top of `current_block` in the goal state (using `self.goal_on_map`).
           - For each `next_block`, check if it is not already marked as
             `is_correct_prefix = True`.
           - If it's not already marked, check if `next_block` is currently
             directly on `current_block` in the state (`current_below.get(next_block) == current_block`).
           - If both conditions are met, mark `is_correct_prefix[next_block]`
              as True and enqueue `next_block`. This propagates the "correctly
              stacked" property upwards.
        5. After the queue is empty, iterate through all blocks in
           `self.goal_blocks`. Count how many blocks `b` have
           `is_correct_prefix[b]` set to False.
        6. The heuristic value is this count. This represents the number
           of blocks that are not in their correct position relative to
           the goal stack structure they belong to. Each such block
           needs to be moved at least once to be placed correctly.
    """
    def __init__(self, task):
        self.goals = task.goals
        # Blocksworld has no static facts relevant to the heuristic
        # static_facts = task.static

        self.goal_below = {}
        self.goal_blocks = set()
        self.goal_on_map = {} # base -> list of blocks that should be on base

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                self.goal_below[block] = base
                self.goal_blocks.add(block)
                self.goal_blocks.add(base)
                self.goal_on_map.setdefault(base, []).append(block)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_below[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore 'clear' and 'arm-empty' goals for this heuristic

    def __call__(self, node):
        state = node.state

        # 1. Parse current state structure
        current_below = {}
        # current_clear = set() # Not strictly needed for this heuristic logic
        # current_holding = None # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, base = parts[1], parts[2]
                current_below[block] = base
            elif parts[0] == 'on-table':
                block = parts[1]
                current_below[block] = 'table'
            # elif parts[0] == 'clear':
            #     current_clear.add(parts[1])
            # elif parts[0] == 'holding':
            #     current_holding = parts[1]

        # 2. Identify correctly stacked blocks (correct prefix)
        # Initialize all goal blocks as not part of a correct prefix
        is_correct_prefix = {b: False for b in self.goal_blocks}
        q = deque()

        # Initialize queue with blocks that are correctly on the table according to the goal
        for block in self.goal_blocks:
            if self.goal_below.get(block) == 'table':
                if current_below.get(block) == 'table':
                    q.append(block)
                    is_correct_prefix[block] = True # Mark as correct when adding to queue

        while q:
            current_block = q.popleft()

            # Find blocks that should be on top of current_block in the goal
            blocks_on_top_in_goal = self.goal_on_map.get(current_block, [])
            for next_block in blocks_on_top_in_goal:
                # If next_block is not already marked as correct prefix
                if not is_correct_prefix[next_block]:
                    # Check if next_block is currently on current_block in the state
                    if current_below.get(next_block) == current_block:
                        # It's in the correct position relative to this correct base.
                        # Mark it as correct and add to queue to propagate upwards.
                        is_correct_prefix[next_block] = True
                        q.append(next_block)


        # 3. Calculate heuristic
        # Count blocks in the goal that are NOT part of a correct prefix
        misplaced_count = sum(1 for block in self.goal_blocks if not is_correct_prefix[block])

        return misplaced_count
