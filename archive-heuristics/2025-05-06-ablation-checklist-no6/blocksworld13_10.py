from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by counting blocks
    that are not in their correct position relative to the goal stack structure,
    plus blocks that are obstructing the movement of other blocks, plus
    unsatisfied (clear X) goals.

    # Heuristic Initialization
    - Parses the goal facts to determine the desired parent (block or table)
      for each block and identifies all blocks involved in the goal.
    - Also identifies blocks that need to be clear in the goal.

    # Step-by-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current parent (block, table, or holding)
       for each block and identify which blocks are clear.
    2. Identify the set of blocks that are "correctly positioned" relative to the goal stack.
       A block B is correctly positioned if its goal is (on-table B) and it is on the table, OR
       its goal is (on B U) and it is on U, AND U is correctly positioned.
       This is computed iteratively starting from blocks correctly placed on the table
       and propagating upwards through the goal stacks. Blocks whose location is not
       explicitly given in the state facts are assumed to be on the table and clear.
    3. The first part of the heuristic (h1) is the number of blocks involved in the goal
       minus the number of correctly positioned blocks. This counts blocks that
       are not part of a correctly built segment of a goal stack.
    4. The second part of the heuristic (h2) counts blocks that are currently on top
       of a block that is NOT correctly positioned. These blocks must be moved
       out of the way.
    5. The third part of the heuristic (h3) counts goal facts of the form (clear X)
       that are not true in the current state.
    6. The total heuristic is the sum of h1 + h2 + h3.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and clear goals.
        """
        super().__init__(task)

        self.goal_parent = {}
        self.goal_on_table = set()
        self.goal_blocks = set()
        self.goal_clear = set()

        # Parse goal facts to build goal structure and clear goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                child, parent = parts[1], parts[2]
                self.goal_parent[child] = parent
                self.goal_blocks.add(child)
                self.goal_blocks.add(parent)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_on_table.add(block)
                self.goal_blocks.add(block)
            elif predicate == 'clear' and len(parts) == 2:
                 block = parts[1]
                 self.goal_clear.add(block)
                 self.goal_blocks.add(block) # A block might only appear in a clear goal


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Parse the current state
        state_parent = {}
        state_is_clear = set()
        held_block = None
        blocks_explicitly_located = set() # Blocks mentioned in on, on-table, holding

        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'on' and len(parts) == 3:
                 child, parent = parts[1], parts[2]
                 state_parent[child] = parent
                 blocks_explicitly_located.add(child)
                 blocks_explicitly_located.add(parent)
             elif predicate == 'on-table' and len(parts) == 2:
                 block = parts[1]
                 state_parent[block] = 'table'
                 blocks_explicitly_located.add(block)
             elif predicate == 'holding' and len(parts) == 2:
                 block = parts[1]
                 state_parent[block] = 'holding'
                 held_block = block
                 blocks_explicitly_located.add(block)
             elif predicate == 'clear' and len(parts) == 2:
                 block = parts[1]
                 state_is_clear.add(block)
                 # Note: A block can be clear even if its location is known (e.g., clear on-table)


        # If the goal is empty, the heuristic is 0
        if not self.goal_blocks:
             return 0

        # 2. Identify correctly positioned blocks relative to goal stack structure
        correctly_positioned_set = set()
        queue = []

        # Add blocks that are correctly on the table in the state and goal
        for block in self.goal_on_table:
            # Check if the block is on the table in the state.
            # A block is on the table if state_parent[block] is 'table' OR
            # if its location is not explicitly given (not in blocks_explicitly_located)
            # AND it's not the held block.
            is_on_table_in_state = False
            if block in state_parent and state_parent[block] == 'table':
                is_on_table_in_state = True
            elif block not in blocks_explicitly_located and held_block != block:
                 # Implicitly on the table
                 is_on_table_in_state = True

            if is_on_table_in_state:
                 # Ensure block is in goal_blocks (should be by definition of goal_on_table)
                 queue.append(block)

        # Use a set to track items added to the queue to avoid duplicates and cycles
        added_to_queue = set(queue)

        # Propagate correctness up the goal stacks
        while queue:
            current_block = queue.pop(0)
            correctly_positioned_set.add(current_block)

            # Find blocks that should be on current_block in the goal
            for block, parent in self.goal_parent.items():
                if parent == current_block:
                    # Check if block is currently on current_block in the state
                    # A block X is on block Y if state_parent[X] == Y
                    if block in self.goal_blocks and block in state_parent and state_parent[block] == current_block:
                        # Add block to queue to check its correctness
                        if block not in added_to_queue:
                            queue.append(block)
                            added_to_queue.add(block)


        # 3. Calculate h1: Misplaced blocks relative to goal stack structure
        # Count blocks in goal_blocks that are NOT in correctly_positioned_set
        h1 = sum(1 for block in self.goal_blocks if block not in correctly_positioned_set)


        # 4. Calculate h2: Blocks obstructing incorrect blocks
        # Count blocks X that are currently on top of a block Y, where Y is NOT correctly positioned
        h2 = 0
        for block, parent in state_parent.items():
            # Check if the parent is a block (not 'table' or 'holding')
            if parent != 'table' and parent != 'holding':
                 # Check if the parent block is NOT correctly positioned
                 if parent not in correctly_positioned_set:
                     h2 += 1

        # 5. Calculate h3: Unsatisfied (clear X) goals
        h3 = 0
        for block_to_clear in self.goal_clear:
            # Check if (clear block_to_clear) is false in the state
            if block_to_clear not in state_is_clear:
                 h3 += 1

        # 6. Total heuristic
        heuristic_value = h1 + h2 + h3

        return heuristic_value
