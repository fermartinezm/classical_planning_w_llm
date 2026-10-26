# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
         return [] # Return empty for malformed facts
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions required by counting blocks not in their
    correct goal stack position and multiplying by 2 (representing pickup/unstack + stack/putdown).

    A block is in its correct goal stack position if:
    1. It should be on the table in the goal AND it is currently on the table.
    2. It should be on block Y in the goal AND it is currently on block Y AND Y is in its correct goal stack position.

    This is computed iteratively from the base (table) upwards.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal stack configurations and all blocks.
        """
        self.goal_below_map = {}
        self.goal_above_map = {} # Map Y to list of B where (on B Y) is a goal
        self.all_blocks = set()

        # Parse goal predicates to build goal stack structure and collect blocks
        for goal in task.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    b, y = parts[1], parts[2]
                    self.goal_below_map[b] = y
                    self.goal_above_map.setdefault(y, []).append(b)
                    self.all_blocks.add(b)
                    self.all_blocks.add(y)
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    b = parts[1]
                    self.goal_below_map[b] = 'table'
                    self.all_blocks.add(b)
            # Ignore 'clear' goals

        # Collect all blocks from the initial state as well
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             # Assuming block names are the arguments in predicates like (on X Y), (on-table X), (clear X), (holding X)
             if len(parts) > 1:
                 for arg in parts[1:]:
                     # Simple check to avoid adding non-block terms like 'table', 'arm-empty', gripper names etc.
                     # In blocksworld, arguments are typically blocks.
                     if arg not in ['table', 'arm-empty']:
                         self.all_blocks.add(arg)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current stack configuration
        current_below_map = {}
        # We don't strictly need current_clear or current_holding for this specific heuristic logic,
        # but parsing them doesn't hurt and might be useful for other heuristics.
        # current_clear = set()
        # current_holding = None

        for fact in state:
             parts = get_parts(fact)
             if not parts: continue

             predicate = parts[0]
             if predicate == 'on':
                 if len(parts) == 3:
                     b, y = parts[1], parts[2]
                     current_below_map[b] = y
             elif predicate == 'on-table':
                 if len(parts) == 2:
                     b = parts[1]
                     current_below_map[b] = 'table'
             # elif predicate == 'clear':
             #     if len(parts) == 2:
             #         current_clear.add(parts[1])
             # elif predicate == 'holding':
             #     if len(parts) == 2:
             #         current_holding = parts[1]


        # Compute which blocks are in their correct goal stack position
        # Initialize all blocks as not being in goal stack position
        is_in_goal_stack_pos = {block: False for block in self.all_blocks}
        q = [] # Queue for blocks confirmed to be in goal stack position

        # Base cases: blocks that should be on the table in the goal AND are currently on the table
        for block in self.all_blocks:
            goal_base = self.goal_below_map.get(block)
            if goal_base == 'table':
                 # Check if the block is currently on the table
                 if current_below_map.get(block) == 'table':
                     # If it's on the table and should be, it's in goal stack position
                     # Mark it True and add to Q to propagate upwards
                     if not is_in_goal_stack_pos[block]: # Avoid duplicates in Q and reprocessing
                         is_in_goal_stack_pos[block] = True
                         q.append(block)

        # Propagate correctness upwards through the goal stacks
        # Use Q as a list acting as a queue (BFS)
        q_index = 0
        while q_index < len(q):
            y = q[q_index] # Get the next block from the queue
            q_index += 1

            # Find blocks B that should be directly on Y in the goal according to goal_above_map
            blocks_to_check_above_y = self.goal_above_map.get(y, [])

            for b in blocks_to_check_above_y:
                # If B is currently directly on Y
                if current_below_map.get(b) == y:
                    # If B is on Y currently, and Y is already confirmed in goal stack pos,
                    # then B is also in goal stack pos.
                    # Mark it True and add to Q to continue propagation, if not already marked.
                    if not is_in_goal_stack_pos[b]:
                         is_in_goal_stack_pos[b] = True
                         q.append(b)


        # Count blocks not in goal stack position
        # These are the blocks that are not part of the correctly built bottom-up goal stacks.
        # Each such block needs to be moved at least once.
        n_not_in_goal_stack = sum(1 for block in self.all_blocks if not is_in_goal_stack_pos[block])

        # Heuristic value: 2 * number of blocks not in goal stack position.
        # This is a simple estimate assuming each 'misplaced' block requires
        # roughly 2 actions (pickup/unstack + stack/putdown) to eventually
        # get it into its correct position or out of the way.
        # This heuristic is not admissible but aims to guide greedy search effectively.
        return 2 * n_not_in_goal_stack
