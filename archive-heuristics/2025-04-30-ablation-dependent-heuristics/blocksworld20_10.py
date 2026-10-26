from collections import deque

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the number of actions required to reach the goal
        by counting the number of blocks that are not in their correct goal
        position relative to a correctly placed base block or the table.
        A block is considered "correctly placed" if it is on its correct goal
        location (either the table or the correct block below it) AND the block
        below it is also correctly placed (recursively), AND if the block is
        required to be clear in the goal state, it is also clear in the current state.
        The heuristic value is the total count of blocks that are not correctly placed
        according to this definition.

    Assumptions:
        - The input task is a standard Blocksworld planning task.
        - Goal facts primarily consist of `(on X Y)`, `(on-table X)`, and `(clear X)`
          predicates defining the desired stack configurations.
        - The heuristic focuses only on achieving the goal `on`/`on-table`/`clear`
          relations for blocks mentioned in the goal. Blocks not mentioned in
          these goal facts do not contribute to the heuristic value.
        - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
        The heuristic is initialized with the planning task. During initialization,
        it processes the goal facts to build two internal data structures:
        1. `_goal_structure`: A dictionary mapping each block (that should be on
           another block or the table in the goal) to its required location
           (either the name of the block it should be on, or the string 'table').
           This is built from `(on X Y)` and `(on-table X)` goal facts.
        2. `_goal_clear_blocks`: A set containing the names of blocks that are
           required to be clear in the goal state. This is built from `(clear X)`
           goal facts.
        It also pre-computes `_goal_table_blocks` (blocks that should be on the table)
         and `_goal_on_map` (a reverse map from location to blocks that should be on it)
         for efficient lookup during heuristic computation.
        Static facts from the task are accepted but not used by this specific
        heuristic, as they typically don't define the goal configuration structure
        in Blocksworld.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state (a frozenset of facts), the heuristic is computed as follows:
        1. Initialize a dictionary `correctly_placed` to store the status (True/False)
           for each block encountered.
        2. Initialize a queue with all blocks that are required to be on the table
           in the goal state (based on `_goal_structure`). These are the potential
           bases for goal stacks.
        3. Use a set `processed_q` to keep track of blocks added to the queue to prevent
           redundant processing (though cycles are not expected in valid goal structures).
        4. While the queue is not empty:
           a. Dequeue a block `B`.
           b. If `B` has already been processed, continue. Add `B` to `processed_q`.
           c. Get the goal location `goal_loc` for `B` from `_goal_structure`.
           d. Determine the placement status of `B` relative to its goal location:
              - If `goal_loc` is 'table', the status is true if `(on-table B)` is in the state.
              - If `goal_loc` is a block `Y`, the status is true if `(on B Y)` is in the state AND `Y` is marked as correctly placed in `correctly_placed`. If `Y` is not yet in `correctly_placed`, `B` cannot be correctly placed relative to a correct base yet, so its status is false for now.
           e. If the placement status is true AND `B` is in `_goal_clear_blocks`, also check if `(clear B)` is in the state. The final status is true only if both placement and clear checks pass.
           f. Store the final status for `B` in `correctly_placed[B]`.
           g. If `B` is correctly placed, find all blocks `X` that should be directly on top of `B` in the goal state (using `_goal_on_map`). Enqueue these blocks `X` to check their status in the next iterations.
        5. After the queue is empty, iterate through all blocks that appear as keys
           in `_goal_structure`.
        6. For each such block `B`, check its status in `correctly_placed`. If `B`
           is not found in `correctly_placed` or its status is False, it means
           the block is not in its correct goal position relative to a correct base.
           Increment a counter for each such block.
        7. The final count is the heuristic value.
    """
    def __init__(self, task):
        """
        Initializes the heuristic by processing the goal facts.

        Args:
            task: The planning task object (instance of Task class).
        """
        self._goal_structure = {}
        self._goal_clear_blocks = set()
        self._goal_table_blocks = set()
        self._goal_on_map = {} # Reverse map: location -> blocks that should be on it

        # Process goal facts to build goal structure and identify goal clear blocks
        for fact in task.goals:
            fact_parts = fact.strip('()').split()
            if not fact_parts: # Skip empty facts if any
                continue
            predicate = fact_parts[0]

            if predicate == 'on' and len(fact_parts) == 3:
                # Fact is like '(on block1 block2)'
                block = fact_parts[1]
                location = fact_parts[2]
                self._goal_structure[block] = location
                self._goal_on_map.setdefault(location, []).append(block)
            elif predicate == 'on-table' and len(fact_parts) == 2:
                # Fact is like '(on-table block1)'
                block = fact_parts[1]
                self._goal_structure[block] = 'table'
                self._goal_table_blocks.add(block)
            elif predicate == 'clear' and len(fact_parts) == 2:
                # Fact is like '(clear block1)'
                block = fact_parts[1]
                self._goal_clear_blocks.add(block)
            # Ignore other potential goal facts if any (like arm-empty) for this heuristic

        # Static facts are not used in this heuristic for blocksworld
        self._static = task.static # Store it as per requirement, but not used

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            An integer representing the estimated number of actions to reach the goal.
        """
        # If there are no goal facts related to block positions, the heuristic is 0
        if not self._goal_structure:
             return 0

        correctly_placed = {}
        q = deque(list(self._goal_table_blocks)) # Start with blocks that should be on the table

        processed_q = set() # Keep track of blocks added to queue

        while q:
            block = q.popleft()
            if block in processed_q:
                continue
            processed_q.add(block)

            goal_loc = self._goal_structure.get(block)

            # Determine if the block is correctly placed relative to its goal location
            status = False
            if goal_loc == 'table':
                status = ('(on-table {})'.format(block) in state)
            elif goal_loc in correctly_placed and correctly_placed[goal_loc]:
                 # Only check if the block below is correctly placed AND this block is on it
                 status = ('(on {} {})'.format(block, goal_loc) in state)
            # Note: If goal_loc is a block but not yet in correctly_placed,
            # status remains False, meaning this block cannot be correctly placed
            # relative to a correct base yet.

            # If the placement status is true AND the block is required to be clear, check that too
            if status and block in self._goal_clear_blocks:
                 status = status and ('(clear {})'.format(block) in state)

            correctly_placed[block] = status

            # If this block is correctly placed, check blocks that should be on top of it
            if status:
                blocks_on_top_in_goal = self._goal_on_map.get(block, [])
                for block_on_top in blocks_on_top_in_goal:
                    # Add blocks that should be on top to the queue to check their status
                    # They will be marked correctly placed only if they are on this block
                    # and this block is correctly placed.
                    if block_on_top not in processed_q: # Avoid adding duplicates
                         q.append(block_on_top)

        # Count blocks that are in the goal structure but were not marked as correctly placed
        misplaced_count = 0
        for block in self._goal_structure.keys():
            if not correctly_placed.get(block, False):
                misplaced_count += 1

        return misplaced_count
