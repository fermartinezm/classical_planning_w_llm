from heuristics.heuristic_base import Heuristic
# No other specific imports are necessary for this heuristic.


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by considering
        blocks that are not in their correct position within the goal structure,
        blocks that are blocking correctly placed blocks, and unsatisfied
        arm-empty and clear goals. It is designed for greedy best-first search
        and does not need to be admissible.

    Assumptions:
        - The domain uses standard Blocksworld predicates: (on ?x ?y),
          (on-table ?x), (clear ?x), (holding ?x), (arm-empty).
        - The goal state is defined using a conjunction of these predicates.
        - The goal structure implied by (on ?x ?y) and (on-table ?x) goals
          forms a set of stacks (a Directed Acyclic Graph).

    Heuristic Initialization:
        In the constructor, the heuristic pre-processes the goal facts to build
        the target configuration structure. This includes:
        - `goal_parent_map`: A dictionary mapping a block to the block it should
          be directly on top of in the goal state.
        - `goal_child_map`: A dictionary mapping a block to the block that should
          be directly on top of it in the goal state.
        - `goal_on_table_set`: A set of blocks that should be on the table in
          the goal state.
        - `goal_blocks`: A set of all blocks mentioned in the (on ...) or
          (on-table ...) goal facts.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state, the heuristic value is computed as the sum of four terms:

        1.  **Misplaced Blocks in Goal Structure:** Count the number of blocks `b`
            that are part of the goal configuration (`b` is in `goal_blocks`)
            but are *not* correctly placed. A block `b` is considered correctly
            placed if its current position matches its goal position (on the
            table or on a specific parent block) AND the block it is on (if any)
            is also correctly placed. This is computed recursively with memoization.

        2.  **Blocks Blocking Correctly Placed Blocks:** Count the number of blocks `c`
            such that `c` is currently on top of block `b` (`(on c b)` is true),
            where `b` is a block from the goal configuration (`b` is in `goal_blocks`),
            `b` *is* correctly placed, but `c` is *not* the block that should be
            directly on top of `b` according to the goal structure. This term
            penalizes states where a correctly built part of a goal stack is
            obstructed by the wrong block.

        3.  **Unsatisfied Arm-Empty Goal:** If `(arm-empty)` is a goal fact, add 1
            to the heuristic if the arm is *not* empty in the current state.

        4.  **Unsatisfied Clear Goals:** For each block `b` where `(clear b)` is a
            goal fact, add 1 to the heuristic if `b` is *not* clear in the current
            state.

        The total heuristic value is the sum of these four terms. A heuristic value
        of 0 is achieved only when the state is the goal state.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals  # Store goal facts

        # Pre-process goal facts to build the goal structure
        self.goal_parent_map = {}
        self.goal_child_map = {}
        self.goal_on_table_set = set()
        self.goal_blocks = set()  # Blocks mentioned in on/on-table goals

        for goal_fact_str in self.goals:
            if goal_fact_str.startswith('(on '):
                # Example: '(on b1 b2)'
                parts = goal_fact_str[1:-1].split()  # ['on', 'b1', 'b2']
                child = parts[1]
                parent = parts[2]
                self.goal_parent_map[child] = parent
                self.goal_child_map[parent] = child
                self.goal_blocks.add(child)
                self.goal_blocks.add(parent)
            elif goal_fact_str.startswith('(on-table '):
                # Example: '(on-table b2)'
                parts = goal_fact_str[1:-1].split()  # ['on-table', 'b2']
                block = parts[1]
                self.goal_on_table_set.add(block)
                self.goal_blocks.add(block)

        # We only care about blocks that are part of the positional goal structure
        # (i.e., mentioned in on/on-table goals) for the 'correctly_placed' calculation.
        # Blocks only mentioned in (clear b) goals are handled separately in Term 4.
        # Blocks not mentioned in any goal fact are ignored by this heuristic.

    def __call__(self, node):
        state = node.state

        # 1. Parse current state into useful data structures
        current_on_map = {}  # child -> parent
        current_on_table_set = set()
        # current_clear_set = set() # Not strictly needed for this heuristic calculation
        # current_holding = None # Not strictly needed for this heuristic calculation

        for fact_str in state:
            if fact_str.startswith('(on '):
                parts = fact_str[1:-1].split()
                child = parts[1]
                parent = parts[2]
                current_on_map[child] = parent
            elif fact_str.startswith('(on-table '):
                parts = fact_str[1:-1].split()
                block = parts[1]
                current_on_table_set.add(block)
            # elif fact_str.startswith('(clear '):
            #     parts = fact_str[1:-1].split()
            #     block = parts[1]
            #     current_clear_set.add(block)
            # elif fact_str.startswith('(holding '):
            #     parts = fact_str[1:-1].split()
            #     current_holding = parts[1]
            # Ignore (arm-empty) for state parsing, check directly later

        # 2. Compute is_correctly_placed status for blocks in goal_blocks
        correctly_placed_memo = {}

        def is_correctly_placed(block):
            if block in correctly_placed_memo:
                return correctly_placed_memo[block]

            # If block is not part of the goal structure we care about, it cannot be correctly placed
            if block not in self.goal_blocks:
                correctly_placed_memo[block] = False
                return False

            status = False
            if block in self.goal_on_table_set:
                # Base case: should be on table
                status = '(on-table {})'.format(block) in state
            elif block in self.goal_parent_map:
                # Recursive case: should be on parent
                parent = self.goal_parent_map[block]
                # Check if block is currently on the correct parent
                is_on_correct_parent = current_on_map.get(block) == parent
                # Check if the parent is correctly placed (recursive call)
                # If parent is not in goal_blocks, it cannot be correctly placed by definition here.
                parent_is_correct = is_correctly_placed(parent) if parent in self.goal_blocks else False
                status = is_on_correct_parent and parent_is_correct
            # else: block is in goal_blocks but not in goal_on_table_set or goal_parent_map?
            # This case implies the block is only mentioned in a (clear b) goal,
            # or the goal structure is disconnected/invalid.
            # For this heuristic, such blocks are not considered 'correctly_placed'
            # in terms of position relative to the goal structure. Status remains False.

            correctly_placed_memo[block] = status
            return status

        # Compute correctly_placed_status for all blocks in goal_blocks
        # Call is_correctly_placed for each block that is part of the goal structure
        correctly_placed_status = {b: is_correctly_placed(b) for b in self.goal_blocks}

        # 3. Calculate heuristic value based on the four terms
        h_value = 0

        # Term 1: Misplaced Blocks in Goal Structure
        for b in self.goal_blocks:
            if not correctly_placed_status.get(b, False):
                h_value += 1

        # Term 2: Blocks Blocking Correctly Placed Blocks
        for fact_str in state:
            if fact_str.startswith('(on '):
                parts = fact_str[1:-1].split()
                c = parts[1]  # The block on top
                b = parts[2]  # The block below

                # Check if the block below (b) is part of the goal structure and is correctly placed
                if b in self.goal_blocks and correctly_placed_status.get(b, False):
                    # b is correctly placed. Is c the correct goal child of b?
                    goal_child_of_b = self.goal_child_map.get(b)
                    if goal_child_of_b is None or c != goal_child_of_b:
                        # There is no goal child for b (b should be clear), or c is not the correct goal child
                        h_value += 1

        # Term 3: Unsatisfied Arm-Empty Goal
        if '(arm-empty)' in self.goals and '(arm-empty)' not in state:
            h_value += 1

        # Term 4: Unsatisfied Clear Goals
        for goal_fact_str in self.goals:
            if goal_fact_str.startswith('(clear '):
                if goal_fact_str not in state:
                    h_value += 1

        return h_value
