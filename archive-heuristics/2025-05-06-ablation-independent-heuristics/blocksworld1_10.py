from heuristics.heuristic_base import Heuristic
from task import Task

class blocksworldHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for Blocksworld that estimates the distance
    to the goal state by counting the number of blocks that are not in their
    correct goal position relative to their support (another block or the table),
    plus the number of blocks that are blocking a block that is not in its
    goal position, plus the number of blocks that need to be clear but aren't,
    plus a penalty if the arm needs to be empty but isn't. This heuristic is
    designed to be non-admissible and guide a greedy best-first search.

    Assumptions:
    - The heuristic is designed for the Blocksworld domain as defined by the
      provided PDDL.
    - The state is represented as a frozenset of strings, where each string
      is a PDDL fact like '(on b1 b2)' or '(on-table b3)'.
    - The heuristic is non-admissible and intended for greedy best-first search.
    - The heuristic value is 0 if and only if the state is a goal state.
    - The heuristic value is finite for all solvable states.

    Heuristic Initialization:
    In the constructor, the goal facts from the task are processed to build
    a mapping from each block involved in an 'on' or 'on-table' goal fact
    to its desired support (the block it should be on, or 'table'). This
    mapping, `self.goal_support`, and the set of blocks involved,
    `self.goal_blocks`, are stored for efficient lookup during heuristic
    computation. Goal facts of type '(clear ?x)' and '(arm-empty)' are also
    identified and stored. Static facts are not relevant for this domain
    but a placeholder for processing them is included for generality.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is the goal state using the task's
        `goal_reached` method. If it is, the heuristic value is 0.
    2.  Parse the current state facts to determine the current configuration:
        Iterate through the state's fact strings. For each fact:
        - If it's `(on ?b ?a)`, record that block `?b` is currently on block `?a`
          in an `on_map`.
        - If it's `(on-table ?b)`, record that block `?b` is currently on the
          table in a set `on_table_blocks`.
        - If it's `(clear ?b)`, record that block `?b` is clear in a set `clear_blocks`.
        - If it's `(holding ?b)`, record that block `?b` is being held in `holding_block`.
        - If it's `(arm-empty)`, record that the arm is empty in `arm_empty_state`.
        Also, create a set `state_facts_set` for efficient lookup of facts.
    3.  Calculate `h_misplaced_support`: Initialize a count `h1` to 0. Iterate
        through each block that is part of the goal configuration (`self.goal_blocks`).
        Determine its goal support from `self.goal_support`. Determine its
        current support (the block it's on, the table, or 'held' if it's
        in the arm) using the parsed state information. If the current support
        does not match its goal support, increment `h1`. Keep track of blocks
        identified as having misplaced support in a set `misplaced_blocks`.
    4.  Calculate `h_blocking_misplaced`: Initialize a count `h2` to 0. Iterate
        through the `on_map` (representing `(on X B)` facts in the current state).
        For each fact `(on X B)`, check if `B` is in the set of `misplaced_blocks`.
        If it is, it means `X` is currently blocking a block (`B`) that is in
        the wrong place and needs to be moved. Increment `h2` for each such
        blocking block `X`.
    5.  Calculate `h_unsatisfied_clear_goals`: Initialize a count `h3` to 0.
        Iterate through the blocks that need to be clear according to the goal
        (`self.goal_clear_blocks`). If a block `B` is in this set but the fact
        `(clear B)` is not present in the `state_facts_set`, increment `h3`.
    6.  Add penalty for unsatisfied arm-empty goal: Initialize a count `h4` to 0.
        If `(arm-empty)` is a goal fact (`self.goal_arm_empty`) and the arm is
        not empty in the current state (`arm_empty_state` is False), increment `h4` by 1.
    7.  The total heuristic value is the sum `h1 + h2 + h3 + h4`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task # Store task to use goal_reached check

        # Process goal facts
        self.goal_support = {}
        self.goal_clear_blocks = set()
        self.goal_arm_empty = False

        for goal_fact_str in task.goals:
            parts = self._parse_fact(goal_fact_str)
            predicate = parts[0]
            if predicate == 'on':
                block = parts[1]
                support = parts[2]
                self.goal_support[block] = support
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_support[block] = 'table'
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear_blocks.add(block)
            elif predicate == 'arm-empty':
                self.goal_arm_empty = True
            # Ignore 'holding' goals if any, as they are transient and usually not part of final state goals

        self.goal_blocks = set(self.goal_support.keys())

        # Process static facts (placeholder - Blocksworld has none)
        # for static_fact_str in task.static:
        #     parts = self._parse_fact(static_fact_str)
        #     ... process static info ...

    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a list of strings."""
        # Remove surrounding brackets and split by space
        parts = fact_string[1:-1].split()
        return parts

    def __call__(self, node):
        state = node.state

        # 1. Check for goal state
        if self.task.goal_reached(state):
            return 0

        # 2. Parse current state
        state_info = {
            'on_map': {},
            'on_table_blocks': set(),
            'clear_blocks': set(),
            'holding_block': None,
            'arm_empty': False
        }

        # Create a set for quick lookup of state facts
        state_facts_set = set(state)

        for fact_string in state:
            parts = self._parse_fact(fact_string)
            predicate = parts[0]
            if predicate == 'on':
                block = parts[1]
                support = parts[2]
                state_info['on_map'][block] = support
            elif predicate == 'on-table':
                block = parts[1]
                state_info['on_table_blocks'].add(block)
            elif predicate == 'clear':
                block = parts[1]
                state_info['clear_blocks'].add(block)
            elif predicate == 'holding':
                block = parts[1]
                state_info['holding_block'] = block
            elif predicate == 'arm-empty':
                state_info['arm_empty'] = True

        def get_current_support(block, s_info):
            """Helper to find the current support for a block."""
            if s_info['holding_block'] == block:
                return 'held'
            if block in s_info['on_table_blocks']:
                return 'table'
            if block in s_info['on_map']:
                return s_info['on_map'][block]
            # If a block from goal_blocks is not found in state, something is wrong,
            # but for robustness, return None.
            return None


        h = 0

        # 3. Calculate h_misplaced_support
        misplaced_blocks = set()
        for block in self.goal_blocks:
            goal_sup = self.goal_support[block]
            current_sup = get_current_support(block, state_info)

            # If block is not found in state info, it's definitely not in goal position
            if current_sup is None or current_sup != goal_sup:
                h += 1
                # Add to misplaced_blocks only if we found its current support
                # Blocks not found in state info cannot be 'below' another block
                if current_sup is not None:
                    misplaced_blocks.add(block)


        # 4. Calculate h_blocking_misplaced
        # Count blocks that are on top of a misplaced block
        for block_on_top, block_below in state_info['on_map'].items():
             if block_below in misplaced_blocks:
                 h += 1

        # 5. Calculate h_unsatisfied_clear_goals
        for block in self.goal_clear_blocks:
            if '(clear ' + block + ')' not in state_facts_set:
                 h += 1

        # 6. Add penalty for unsatisfied arm-empty goal
        if self.goal_arm_empty and not state_info['arm_empty']:
             h += 1

        return h
