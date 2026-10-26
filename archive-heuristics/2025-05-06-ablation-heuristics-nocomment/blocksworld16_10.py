from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Parses a PDDL fact string like '(predicate arg1 arg2)' into a list of strings."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the distance to the goal state by counting
        two types of misplaced blocks:
        1. Blocks that are part of the desired goal stacks but are not
           currently in their correct position relative to the blocks below
           them in the goal structure (h1).
        2. Blocks that are currently placed on top of other blocks, where
           this 'on' relationship is not part of the goal state (h2).
        The heuristic value is the sum of these two counts (h1 + h2).

    Assumptions:
        - The heuristic assumes that all blocks present in the problem instance
          that are relevant to the goal are explicitly mentioned in the goal's
          '(on ?x ?y)' or '(on-table ?z)' predicates. Blocks not mentioned
          in these goal predicates are not considered in the h1 calculation.
        - The heuristic ignores '(clear ?x)' and '(arm-empty)' goals, focusing
          only on the structural arrangement of blocks. These goals are typically
          satisfied as a side effect of achieving the correct block placements.

    Heuristic Initialization:
        In the __init__ method, the heuristic parses the goal predicates
        ('(on ?x ?y)' and '(on-table ?z)') from the task definition.
        It constructs:
        - `self.goal_on`: A dictionary mapping a block to the block it should be
          directly on top of in the goal state.
        - `self.goal_on_table`: A set of blocks that should be directly on the
          table in the goal state.
        - `self.goal_on_facts`: A set of (block, under_block) tuples representing
          all desired '(on block under_block)' facts in the goal.
        - `self.goal_blocks`: A set containing all blocks that are mentioned
          in the goal's '(on ?x ?y)' or '(on-table ?z)' predicates.

    Step-By-Step Thinking for Computing Heuristic:
        The __call__ method computes the heuristic value for a given state (node).
        1.  **Goal Check**: First, it checks if the current state is the goal state.
            If it is, the heuristic value is 0.
        2.  **Parse State**: It iterates through the facts in the current state
            to determine the current position of each block (on which block or
            on the table) and identifies all current '(on ?x ?y)' relationships.
            - `current_pos`: A dictionary mapping a block to its current support
              ('table' or another block). Note: 'holding' is not considered a position
              in this context, as the block is not supporting or being supported.
            - `current_on_facts`: A set of (block, under_block) tuples for all
              '(on block under_block)' facts true in the state.
        3.  **Calculate h2 (Obstructions)**: It calculates the number of current
            '(on ?x ?y)' facts that are *not* present in the set of goal
            '(on ?x ?y)' facts (`self.goal_on_facts`). These represent blocks
            that are on top of other blocks where they shouldn't be according
            to the goal. This count is `h2`.
            `h2 = len(current_on_facts - self.goal_on_facts)`
        4.  **Calculate h1 (Misplaced Goal Blocks)**: It determines which blocks
            that are part of the goal structure (`self.goal_blocks`) are
            "correctly stacked". A block B is correctly stacked if it is in its
            goal position (on the table or on block U) AND, if it's on block U,
            U is also correctly stacked. This is computed iteratively:
            a.  Initialize `correctly_stacked` and `newly_correct` sets.
            b.  Add blocks from `self.goal_blocks` that are currently on the table
                AND are supposed to be on the table (`self.goal_on_table`) to
                `newly_correct`. Update `correctly_stacked`.
            c.  Loop while `newly_correct` is not empty:
                i.  Store `newly_correct` as `current_iteration_correct`. Clear `newly_correct`.
                ii. Iterate through blocks B in `self.goal_blocks` that are not yet
                    in `correctly_stacked`.
                iii. If B is supposed to be on block U (`self.goal_on.get(B) == U`) AND
                     U was identified as correctly stacked in the *previous* iteration
                     (`U in current_iteration_correct`) AND B is currently on U
                     (`(B, U) in current_on_facts`), then add B to `newly_correct`.
                iv. Update `correctly_stacked` with blocks from `newly_correct`.
            d.  After the loop, `correctly_stacked` contains all blocks that form
                correct goal stack segments from the bottom up.
            e.  `h1` is the number of blocks in `self.goal_blocks` that are *not*
                in the `correctly_stacked` set.
            `h1 = len(self.goal_blocks) - len(correctly_stacked)`
        5.  **Total Heuristic**: The final heuristic value is `h1 + h2`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the goal state.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals

        # Parse goal predicates to build the target stack structure
        self.goal_on = {} # Maps block X to block Y if goal is (on X Y)
        self.goal_on_table = set() # Set of blocks that should be on the table
        self.goal_on_facts = set() # Set of (X, Y) tuples for goal (on X Y) facts
        self.goal_blocks = set() # Set of all blocks mentioned in goal ON/ON-TABLE

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, under_block = parts[1], parts[2]
                self.goal_on[block] = under_block
                self.goal_on_facts.add((block, under_block))
                self.goal_blocks.add(block)
                self.goal_blocks.add(under_block)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_on_table.add(block)
                self.goal_blocks.add(block)
            # Ignore (clear ?) and (arm-empty) goals for heuristic calculation

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer heuristic value. Returns 0 if the state is a goal state.
        """
        state = node.state

        # Check if goal is reached (heuristic is 0)
        # This check is important for greedy best-first search termination
        if self.goals <= state:
             return 0

        # --- Step-By-Step Thinking for Computing Heuristic ---
        # (Detailed explanation provided in the class docstring)

        # 1. Parse the current state to find block positions and 'on' relationships.
        current_pos = {} # Maps block X to block Y or 'table'
        current_on_facts = set() # Set of (X, Y) tuples for current (on X Y) facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, under_block = parts[1], parts[2]
                current_pos[block] = under_block
                current_on_facts.add((block, under_block))
            elif parts[0] == 'on-table':
                block = parts[1]
                current_pos[block] = 'table'
            # Ignore (clear ?) and (arm-empty) for position tracking

        # 2. Calculate h2 (Obstructions): Count 'on' facts in the state that are not goal 'on' facts.
        h2 = len(current_on_facts - self.goal_on_facts)

        # 3. Calculate h1 (Misplaced Goal Blocks): Count blocks in the goal structure that are not correctly stacked.
        correctly_stacked = set()
        newly_correct = set()

        # Identify base blocks that are correctly on the table
        for block in self.goal_blocks:
            if block in self.goal_on_table and current_pos.get(block) == 'table':
                newly_correct.add(block)

        correctly_stacked.update(newly_correct)

        # Iteratively identify blocks correctly stacked on top of newly_correct blocks
        while newly_correct:
            current_iteration_correct = newly_correct
            newly_correct = set()
            # Iterate through all goal blocks to find those newly correctly stacked
            for block in self.goal_blocks:
                if block not in correctly_stacked: # Only check blocks not yet marked
                    goal_under_block = self.goal_on.get(block) # Get the block it should be on
                    if goal_under_block is not None: # Check if block has an 'on' goal
                        # Check if the block it should be on was correctly stacked in the previous iteration
                        if goal_under_block in current_iteration_correct:
                             # Check if the block is currently on the correct block
                             if (block, goal_under_block) in current_on_facts:
                                newly_correct.add(block)

            correctly_stacked.update(newly_correct)

        # h1 is the number of goal blocks that are NOT correctly stacked
        h1 = len(self.goal_blocks) - len(correctly_stacked)

        # Total heuristic is the sum of h1 and h2
        return h1 + h2
