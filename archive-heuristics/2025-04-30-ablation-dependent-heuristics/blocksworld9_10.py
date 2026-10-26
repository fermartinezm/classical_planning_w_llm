import collections

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
    The heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are currently in the wrong position. It identifies
    blocks that are not part of a correctly built stack prefix from the bottom-up
    according to the goal, and adds a penalty for blocks that are incorrectly
    placed on top of a correctly built stack prefix. The heuristic value is
    calculated as the sum of blocks not part of a correctly stacked goal prefix
    and blocks that are obstructing a correctly stacked goal prefix.

    Assumptions:
    - The input state and goal are valid for the blocksworld domain.
    - The goal state consists primarily of 'on' and 'on-table' facts defining
      desired stacks, and 'clear' facts for the top blocks. 'arm-empty' is
      usually also a goal fact.
    - The heuristic is non-admissible and designed for greedy best-first search.
      It aims to provide a good estimate to minimize expanded nodes, not to
      guarantee finding the optimal plan.

    Heuristic Initialization:
    The constructor processes the goal facts from the Task object to identify
    the desired 'on' relations and 'on-table' relations. It also extracts
    all unique objects involved in the problem from the initial state and goal.
    These precomputed goal structures (goal_on, goal_on_table, goal_above_map,
    goal_base_blocks) and the total number of blocks are stored for efficient
    lookup during heuristic computation. Static facts are ignored as they are
    not relevant to block positions or arm status in this domain.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is the goal state. If yes, the heuristic is 0.
    2. Parse the current state to extract relevant facts, particularly 'on' and
       'on-table' facts, and build lookup structures (state_on_map, state_on_table_set).
    3. Compute the set of 'correctly stacked' blocks (CS). This set includes blocks
       that are part of a goal stack or goal-on-table block, and are in the correct
       position relative to the block directly below them (or the table), and that
       block below is also correctly stacked (recursively). This is computed using
       a breadth-first search starting from goal base blocks that are correctly
       on the table in the current state, and following the goal 'on' relations
       upwards if the corresponding state 'on' relation exists.
    4. Compute the set of 'obstructing' blocks. An obstructing block Y is one
       for which the fact '(on Y X)' is true in the current state, where X is
       a correctly stacked block (i.e., X is in the CS set), but the fact
       '(on Y X)' is NOT a goal fact. These blocks are sitting incorrectly
       on top of a part of a correctly built goal structure and must be moved.
    5. The heuristic value is calculated as the sum of two components:
       a) The number of blocks that are NOT in the set of correctly stacked blocks.
          These blocks (or something below them) are in the wrong place relative
          to the goal structure.
       b) The number of blocks in the set of obstructing blocks. These blocks
          are hindering the completion of the goal structure even though the
          block below them is correctly placed.
       The sum represents an estimate of the number of blocks that are misplaced
       or obstructing, each potentially requiring one or more actions to fix.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with goal information.

        @param task: The planning task object.
        """
        self.goal_facts = task.goals

        # Extract goal 'on' and 'on-table' facts
        self.goal_on = set()
        self.goal_on_table = set()

        # Precompute goal structure for faster lookup
        # goal_on_map: {block_on_top: block_below} derived from goal_on
        self.goal_on_map = {}
        # goal_above_map: {block_below: block_on_top} derived from goal_on
        self.goal_above_map = {}
        # goal_base_blocks: set of blocks that should be on the table according to goal_on_table
        self.goal_base_blocks = set()

        # Extract all objects from initial state and goal facts
        self.all_objects = set()

        # Collect all facts from initial state and goal to find all objects
        all_relevant_facts = set(task.initial_state) | set(task.goals)

        for fact in all_relevant_facts:
            parts = fact[1:-1].split()
            predicate = parts[0]
            # Only consider facts involving objects that can be moved/stacked
            if predicate in ['on', 'on-table', 'clear', 'holding']:
                 # Ignore predicate name (parts[0])
                 for obj in parts[1:]:
                     self.all_objects.add(obj)

        self.total_blocks = len(self.all_objects)

        # Populate goal structure maps and sets
        for fact in self.goal_facts:
            if fact.startswith('(on '):
                self.goal_on.add(fact)
                parts = fact[1:-1].split()
                block_on_top = parts[1]
                block_below = parts[2]
                self.goal_on_map[block_on_top] = block_below
                # Assuming a block has at most one block directly on top in the goal
                self.goal_above_map[block_below] = block_on_top
            elif fact.startswith('(on-table '):
                self.goal_on_table.add(fact)
                parts = fact[1:-1].split()
                block = parts[1]
                self.goal_base_blocks.add(block)

        # Note: goal_clear and goal_arm_empty are not directly used in the h-value calculation
        # but are part of self.goal_facts used for the final goal state check.


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The heuristic value (integer).
        """
        # Check if goal is reached
        if self.goal_facts <= state:
             return 0

        # Parse current state facts and build lookup structures
        # state_on_map: {block_on_top: block_below}
        state_on_map = {}
        # state_on_table_set: set of blocks on the table
        state_on_table_set = set()

        for fact in state:
            if fact.startswith('(on '):
                parts = fact[1:-1].split()
                block_on_top = parts[1]
                block_below = parts[2]
                state_on_map[block_on_top] = block_below
            elif fact.startswith('(on-table '):
                parts = fact[1:-1].split()
                block = parts[1]
                state_on_table_set.add(block)

        # 1. Compute Correctly Stacked (CS) blocks
        correctly_stacked = set()
        # Use deque for efficient queue operations
        q = collections.deque()

        # Add goal base blocks that are correctly on the table in the state
        for block in self.goal_base_blocks:
            if block in state_on_table_set:
                 q.append(block)

        visited_cs = set() # Prevent cycles and redundant processing

        while q:
            block_below = q.popleft()
            if block_below in visited_cs:
                continue
            visited_cs.add(block_below)
            correctly_stacked.add(block_below)

            # Find the block that should be on top of block_below according to the goal
            block_on_top_goal = self.goal_above_map.get(block_below)

            if block_on_top_goal:
                # Check if this block is actually on top of block_below in the state
                if state_on_map.get(block_on_top_goal) == block_below:
                    # This block is correctly placed on top of a CS block
                    q.append(block_on_top_goal)


        # 2. Compute Obstructing blocks
        obstructing = set()
        # Iterate through blocks that are ON other blocks in the current state
        for block_on_top, block_below in state_on_map.items():
            # Check if the block below is correctly stacked
            if block_below in correctly_stacked:
                # Check if this 'on' relation is NOT a goal fact
                current_on_fact = f'(on {block_on_top} {block_below})'
                if current_on_fact not in self.goal_on:
                    obstructing.add(block_on_top)

        # 3. Calculate heuristic value
        # Count blocks not in CS + Count obstructing blocks
        # This sums two types of 'misplaced' blocks.
        # A block can be both not in CS and obstructing.
        # E.g., Goal (on A B), (on B table). State (on C A), (on A B), (on B table).
        # CS = {A, B}. Blocks not in CS = {C}. Obstructing = {C}. h = 1 + 1 = 2.
        # This seems correct. The counts are separate components of "wrongness".

        h_value = (self.total_blocks - len(correctly_stacked)) + len(obstructing)

        return h_value
