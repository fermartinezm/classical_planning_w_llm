from heuristics.heuristic_base import Heuristic
from task import Task

class blocksworldHeuristic(Heuristic):
    """
    Summary:
        A domain-dependent heuristic for the Blocksworld domain.
        It estimates the distance to the goal by combining two components:
        1. The number of blocks that are part of the goal configuration but are not
           immediately supported correctly (either on the wrong block or not on the table
           when they should be).
        2. The number of goal predicates that are not satisfied in the current state.
        This heuristic is not admissible but aims to guide a greedy best-first search
        efficiently by prioritizing states where blocks are closer to their correct
        positions and more goal conditions are met.

    Assumptions:
        - The input task is a valid Blocksworld task instance.
        - Goal predicates primarily consist of (on X Y), (on-table Z), and (clear W).
        - Blocks mentioned in (on X Y) or (on-table Z) goals constitute the "goal configuration".
        - The heuristic is used in a greedy best-first search, so admissibility is not required.

    Heuristic Initialization:
        The constructor processes the goal predicates from the task to build data structures
        representing the desired goal configuration:
        - goal_on_map: A dictionary mapping a block X to the block Y it should be on
                       according to the goal (from (on X Y) predicates).
        - goal_on_table_set: A set of blocks that should be on the table
                             according to the goal (from (on-table Z) predicates).
        - goal_configuration_blocks: A set containing all blocks that appear in
                                     any (on X Y) or (on-table Z) goal predicate.
        - goal_predicates: A set containing all goal predicates as strings.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state (node):
        1. Parse the current state to build data structures representing the current
           block configuration:
           - current_on_map: A dictionary mapping a block X to the block Y it is currently on
                             (from (on X Y) predicates in the state).
           - current_on_table_set: A set of blocks that are currently on the table
                                   (from (on-table Z) predicates in the state).
        2. Calculate the first component of the heuristic (incorrectly supported blocks):
           - Initialize a counter `incorrectly_supported_count` to 0.
           - Iterate through each block `b` in the `goal_configuration_blocks` set.
           - For each block `b`:
             - Check if `b` is in `goal_on_table_set`. If yes, the goal is for `b` to be on the table.
               - Check if `b` is in `current_on_table_set`. If not, increment `incorrectly_supported_count`.
             - Else (if `b` is not in `goal_on_table_set`), check if `b` is in `goal_on_map`. If yes, the goal is for `b` to be on `goal_on_map[b]`.
               - Check if `b` is in `current_on_map` AND `current_on_map[b]` is equal to `goal_on_map[b]`. If this condition is false (either `b` is not on any block, or it's on the wrong block), increment `incorrectly_supported_count`.
        3. Calculate the second component of the heuristic (unsatisfied goal predicates):
           - Initialize a counter `unsatisfied_goals_count` to 0.
           - Iterate through each goal predicate string `g_str` in the `goal_predicates` set.
           - Check if `g_str` is present in the current state set. If not, increment `unsatisfied_goals_count`.
        4. The total heuristic value for the state is the sum of `incorrectly_supported_count`
           and `unsatisfied_goals_count`.
        5. Return the total heuristic value.
        This heuristic is 0 if and only if all goal predicates are satisfied (which implies
        all blocks in the goal configuration are immediately correctly supported and all
        other goal predicates like (clear X) are met), thus correctly identifying goal states.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goal_on_map = {}
        self.goal_on_table_set = set()
        self.goal_configuration_blocks = set()
        self.goal_predicates = set(task.goals) # Store all goal predicates

        # Process static information (not typically present/needed for this heuristic in BW)
        # self.static_facts = set(task.static) # Example if static facts were relevant

        # Parse goal predicates to build goal structure maps/sets
        for goal_str in task.goals:
            # Example goal_str: '(on b1 b2)' or '(on-table b3)' or '(clear b1)'
            parts = goal_str.strip('()').split()
            if not parts: # Handle empty string after strip, though unlikely for valid predicates
                continue
            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                block_on = parts[1]
                block_under = parts[2]
                self.goal_on_map[block_on] = block_under
                self.goal_configuration_blocks.add(block_on)
                self.goal_configuration_blocks.add(block_under)
            elif predicate == 'on-table' and len(parts) == 2:
                block_on_table = parts[1]
                self.goal_on_table_set.add(block_on_table)
                self.goal_configuration_blocks.add(block_on_table)
            # 'clear', 'holding', 'arm-empty' predicates are handled by checking goal_predicates directly in __call__

    def __call__(self, node):
        state = node.state # state is a frozenset of strings

        # Parse current state predicates relevant for structural check (Part 1)
        current_on_map = {}
        current_on_table_set = set()

        for fact_str in state:
            # Example fact_str: '(on b1 b2)' or '(on-table b3)' etc.
            parts = fact_str.strip('()').split()
            if not parts:
                continue
            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                block_on = parts[1]
                block_under = parts[2]
                current_on_map[block_on] = block_under
            elif predicate == 'on-table' and len(parts) == 2:
                block_on_table = parts[1]
                current_on_table_set.add(block_on_table)
            # Ignore other predicates like clear, holding, arm-empty for this structural check

        # Calculate Part 1: Incorrectly supported blocks in goal configuration
        incorrectly_supported_count = 0
        for block in self.goal_configuration_blocks:
            is_correctly_supported = False
            if block in self.goal_on_table_set: # Goal is (on-table block)
                if block in current_on_table_set:
                    is_correctly_supported = True
            elif block in self.goal_on_map: # Goal is (on block Y)
                goal_under_block = self.goal_on_map[block]
                if block in current_on_map and current_on_map[block] == goal_under_block:
                    is_correctly_supported = True
            # If a block is in goal_configuration_blocks, it must be in either goal_on_table_set or goal_on_map.

            if not is_correctly_supported:
                incorrectly_supported_count += 1

        # Calculate Part 2: Unsatisfied goal predicates
        unsatisfied_goals_count = 0
        for goal_str in self.goal_predicates:
            if goal_str not in state:
                unsatisfied_goals_count += 1

        # Total heuristic value
        h_value = incorrectly_supported_count + unsatisfied_goals_count

        return h_value
