from heuristics.heuristic_base import Heuristic
# No other specific modules are needed based on the logic.
# The task object is passed to __init__

class blocksworldHeuristic(Heuristic):
    """
    Summary:
        A domain-dependent heuristic for the Blocksworld domain.
        It estimates the number of actions required to reach the goal state
        by counting the number of goal blocks that are not in their correct
        position within the goal stack structure.

    Assumptions:
        - This heuristic is designed specifically for the Blocksworld domain
          with standard actions (pickup, putdown, stack, unstack).
        - It is a non-admissible heuristic intended for greedy best-first search.
        - Assumes PDDL facts are represented as strings like '(predicate arg1 arg2)'.

    Heuristic Initialization:
        In the constructor, the heuristic pre-processes the goal state to
        understand the desired stack configuration. It builds a mapping
        `goal_below` where `goal_below[B]` is the block `UnderB` if the goal
        is `(on B UnderB)`, or 'table' if the goal is `(on-table B)`.
        It also identifies all blocks involved in the goal stack structure (`goal_blocks`).

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state. If yes, the heuristic is 0.
        2. If not the goal state, parse the current state facts to determine the
           current position of each block (on another block, on the table, or held).
           This creates a `current_below` mapping.
        3. Identify the set of "In-Place" (IP) blocks. An IP block is one that
           is currently in its correct goal position relative to the block below it,
           AND the block below it (if any) is also In-Place. This is computed
           iteratively starting from blocks that should be on the table in the goal
           and are currently on the table. Only blocks mentioned in the goal are considered.
        4. Calculate the base heuristic value as the number of goal blocks
           that are not in the IP set: `h = len(self.goal_blocks) - len(IP)`.
        5. Adjust the heuristic value: If the calculated value `h` is 0, it means
           the stack structure is perfect according to the goal, but the state is
           not the goal state (this happens if the arm is not empty). In this case,
           the heuristic should be 1 (at least one action, putdown, is needed).
           If `h` is greater than 0, it indicates structural problems, and `h` is returned.
           This can be expressed as `max(1, h)` when the state is not the goal.

    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals # Keep track of full goals for h=0 check

        # Pre-process goal facts to build goal stack structure
        self.goal_below = {}
        self.goal_blocks = set()
        for fact_str in task.goals:
            # Parse fact string by removing parentheses and splitting
            parts = fact_str[1:-1].split()
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on':
                if len(args) == 2:
                    block, under_block = args
                    self.goal_below[block] = under_block
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(under_block)
            elif predicate == 'on-table':
                if len(args) == 1:
                    block = args[0]
                    self.goal_below[block] = 'table'
                    self.goal_blocks.add(block)
            # Ignore 'clear' and 'arm-empty' goals for stack structure analysis


    def __call__(self, node):
        state = node.state

        # Step 1: Check if goal is reached (heuristic is 0)
        if self.goals <= state:
            return 0

        # Step 2: Parse current state facts
        current_below = {}
        # We only need current_below for IP calculation

        for fact_str in state:
            # Parse fact string by removing parentheses and splitting
            parts = fact_str[1:-1].split()
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on':
                if len(args) == 2:
                    block, under_block = args
                    current_below[block] = under_block
            elif predicate == 'on-table':
                if len(args) == 1:
                    block = args[0]
                    current_below[block] = 'table'
            # Ignore other predicates like 'clear', 'holding', 'arm-empty' for this part

        # Step 3: Compute IP (In-Place blocks within goal stack structure)
        # Iterative bottom-up approach
        IP = set()
        
        # Step 3a: Add blocks that should be on the table in the goal and are currently on the table
        for block in self.goal_blocks:
            if self.goal_below.get(block) == 'table' and current_below.get(block) == 'table':
                 IP.add(block)

        # Step 3b: Iteratively add blocks that are correctly stacked on IP blocks
        while True:
            newly_added_to_ip = set()
            for block_above in self.goal_blocks:
                if block_above not in IP:
                    block_below_goal = self.goal_below.get(block_above)
                    # Check if the block below it in the goal is in IP
                    if block_below_goal != 'table' and block_below_goal in IP:
                        # Check if the block_above is currently on the block_below_goal
                        if current_below.get(block_above) == block_below_goal:
                            newly_added_to_ip.add(block_above)

            if not newly_added_to_ip:
                break # No new blocks added in this iteration

            IP.update(newly_added_to_ip)

        # Step 4: Calculate base heuristic
        # Number of goal blocks not in IP
        h = len(self.goal_blocks) - len(IP)

        # Step 5: Adjust heuristic
        # If h is 0, it means stack structure is perfect, but not goal state (arm not empty).
        # Needs at least 1 action (putdown). If h > 0, return h.
        # This is equivalent to max(1, h) when not in goal state.
        return max(1, h)

