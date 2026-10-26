# Import necessary modules
# No external modules needed for this implementation beyond standard Python types

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by counting
        the number of blocks that are not in their correct goal position.
        It assigns a cost of 2 actions for each misplaced block that is
        currently on the table or on another block (representing pickup/unstack
        followed by stack/putdown), and a cost of 1 action for each misplaced
        block that is currently held by the arm (representing stack/putdown).
        Blocks that are not mentioned in the goal's (on) or (on-table) facts
        do not contribute to the heuristic value.

    Assumptions:
        - The input state and goal conform to the standard Blocksworld PDDL domain.
        - Object names in facts are simple strings (e.g., 'b1', 'b2') without spaces
          or parentheses, and do not clash with predicate names ('on', 'on-table',
          'clear', 'holding', 'arm-empty') or the keyword 'table'.
        - The goal state represents a valid configuration of blocks (stacks on the table).
        - The goal does not include '(holding ?x)' facts.
        - The heuristic is used for greedy best-first search and does not need to be admissible.
        - States are valid, meaning every block is either on the table, on another block, or held.

    Heuristic Initialization:
        The __init__ method processes the task definition to precompute
        the desired goal position for each block that is explicitly mentioned
        in the goal's (on) or (on-table) facts. This mapping from block
        to its goal location ('table' or the block it should be on) is stored
        in `self.goal_pos`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is already a goal state using `task.goal_reached()`.
           If it is, the heuristic value is 0.
        2. If not a goal state, determine the current location for each block
           that has a defined goal position (`self.goal_pos`). This is done by
           iterating through the facts in the current state and identifying
           '(on ?ob ?underob)', '(on-table ?ob)', and '(holding ?ob)' facts.
           A dictionary `current_pos` is built mapping these blocks to their
           current location ('table', the block below, or 'arm'). Only blocks
           that are present in `self.goal_pos` are included in `current_pos`.
        3. Initialize the heuristic value `h` to 0.
        4. Iterate through each block that has a defined goal position (i.e.,
           each block in `self.goal_pos`).
        5. For each such block, compare its current location (looked up in `current_pos`)
           with its goal location (`self.goal_pos`). Note: If a block with a goal_pos
           is not found in `current_pos`, it implies an invalid state representation
           or the block is not in a standard location. Assuming valid states,
           `current_pos.get(block)` will return a valid location string or 'arm'.
        6. If the block's current location is different from its goal location:
           a. If the block is currently held by the arm (`current_pos == 'arm'`),
              it needs one more action (stack or putdown) to potentially reach
              its goal location. Add 1 to `h`.
           b. Otherwise (the block is on the table or on another block, or its
              location was not found in the state facts), it needs two actions
              (unstack or pickup, followed by stack or putdown) to potentially
              reach its goal location. Add 2 to `h`.
        7. Blocks that are in their correct goal position do not add to the heuristic.
           Blocks that are not mentioned in the goal's (on) or (on-table) facts
           do not contribute to the heuristic calculation.
        8. The final value of `h` is returned. This value is guaranteed to be
           greater than 0 for any non-goal state (given valid blocksworld goals),
           as at least one block must be misplaced if the goal is not reached.
    """
    def __init__(self, task):
        """
        Initializes the heuristic. Precomputes goal positions for blocks.

        Args:
            task: The planning task (an instance of the Task class).
        """
        self.task = task
        self.goal_pos = {} # Maps block -> goal_location ('table' or block name)
        self._precompute_goal_positions()

    def _precompute_goal_positions(self):
        """
        Parses goal facts to determine the desired location for each block.
        Location is either 'table' or the block it should be on.
        Only considers blocks mentioned in goal (on/on-table) facts.
        """
        for goal_fact in self.task.goals:
            parts = goal_fact.strip("()").split()
            predicate = parts[0]
            if predicate == 'on':
                # (on ?ob ?underob)
                if len(parts) == 3: # Basic check for expected format
                    ob, underob = parts[1], parts[2]
                    self.goal_pos[ob] = underob
            elif predicate == 'on-table':
                # (on-table ?ob)
                if len(parts) == 2: # Basic check for expected format
                    ob = parts[1]
                    self.goal_pos[ob] = 'table'
            # Ignore (clear ?) and (arm-empty) goals for goal_pos mapping


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (a frozenset of fact strings).

        Returns:
            The estimated number of actions to reach the goal.
        """
        # Check if goal is reached first for efficiency and correctness (h=0 iff goal)
        if self.task.goal_reached(state):
            return 0

        current_pos = {} # Maps block -> current_location ('table', block name, or 'arm')

        # Determine current position for all blocks that have a goal_pos
        blocks_with_goal = set(self.goal_pos.keys())

        for fact in state:
            parts = fact.strip("()").split()
            predicate = parts[0]
            if predicate == 'on':
                # (on ?ob ?underob)
                if len(parts) == 3:
                    ob, underob = parts[1], parts[2]
                    if ob in blocks_with_goal:
                         current_pos[ob] = underob
            elif predicate == 'on-table':
                # (on-table ?ob)
                if len(parts) == 2:
                    ob = parts[1]
                    if ob in blocks_with_goal:
                        current_pos[ob] = 'table'
            elif predicate == 'holding':
                # (holding ?ob)
                if len(parts) == 2:
                    ob = parts[1]
                    if ob in blocks_with_goal:
                        current_pos[ob] = 'arm'
                # If the held block is not in blocks_with_goal, it doesn't affect h.


        h = 0

        # Iterate through all blocks that have a defined goal position
        for block, goal_location in self.goal_pos.items():
            current_location = current_pos.get(block) # Get current location, None if not found

            # If current_location is None, it means the block is not in any
            # on/on-table/holding fact in the state. This implies it's not
            # in its goal_pos (unless goal_pos was None, which is not the case here).
            # So, if current_location is None, the block is misplaced.
            # We need to decide the cost for a block whose location is unknown/invalid.
            # Let's assume valid states where current_location is always one of 'table', block_name, or 'arm'.
            # If current_location is None, it's definitely not goal_location.
            # The cost logic handles 'arm' specifically, so None would fall into the 'else' branch, adding 2.
            # This seems like a reasonable penalty for a block in an undefined location.

            if current_location != goal_location:
                 # Block is misplaced
                 if current_location == 'arm':
                     h += 1 # Needs 1 more action (stack/putdown)
                 else: # current_location is 'table', a block name, or None (assuming invalid state)
                     h += 2 # Needs 2 actions (unstack/pickup + stack/putdown)

        return h
