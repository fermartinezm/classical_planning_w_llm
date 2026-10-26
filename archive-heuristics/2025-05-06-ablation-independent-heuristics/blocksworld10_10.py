from heuristics.heuristic_base import Heuristic
from task import Task # Used for type hinting, not strictly necessary for execution if Task is globally available

class blocksworldHeuristic(Heuristic):
    """
    Summary:
        A domain-dependent heuristic for the Blocksworld domain.
        It estimates the cost to reach the goal by summing two components:
        1. The number of blocks that are not currently on the block (or table)
           they are supposed to be on according to the goal state.
        2. The total number of blocks stacked on top of blocks that are required
           to be clear in the goal state but are not clear in the current state.

    Assumptions:
        - The heuristic is designed specifically for the standard Blocksworld domain
          with 'on', 'on-table', 'clear', 'holding', 'arm-empty' predicates
          and 'pickup', 'putdown', 'stack', 'unstack' actions.
        - Goal states consist of conjunctions of 'on', 'on-table', and 'clear' facts.
        - The heuristic assumes a valid Blocksworld state representation where
          each block is either on another block or on the table, and only one
          block can be on top of another.

    Heuristic Initialization:
        In the constructor, the heuristic pre-processes the goal state to build
        data structures for efficient lookup during heuristic computation:
        - `self.goal_support`: A dictionary mapping each block that is part of
          an 'on' or 'on-table' goal fact to the block it should be on, or 'table'.
          If a block is specified in both '(on X Y)' and '(on-table X)' goal facts
          (which should not happen in valid PDDL), the 'on' fact takes precedence.
        - `self.goal_clear`: A set containing all blocks that must be clear
          in the goal state.
        Static facts from the task are also accessible but are typically empty
        or irrelevant to this heuristic in standard Blocksworld.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state, the heuristic is computed as follows:
        1. Parse the current state facts to build internal representations:
           - `state_on_map`: A dictionary mapping each block to the block it is
             currently on, or 'table'. This is constructed by first processing
             all '(on X Y)' facts, then processing '(on-table X)' facts for
             blocks not already assigned a support.
           - `state_above_map`: An inverse map of `state_on_map`, mapping each
             block (or 'table') to the set of blocks currently stacked directly on top of it.
           - `state_clear`: A set containing all blocks that are currently clear.
        2. Calculate the 'mismatched support count': Iterate through each block
           that is a key in `self.goal_support`. If the block's current support
           in `state_on_map` is different from its required goal support in
           `self.goal_support`, increment the count.
        3. Calculate the 'clear cost': Iterate through each block in `self.goal_clear`.
           If the block is not present in `state_clear`, it means it needs to be
           cleared. For each such block, recursively count the number of blocks
           currently stacked directly or indirectly on top of it using `state_above_map`.
           Sum these counts. Memoization is used in the recursive counting function
           (`_count_blocks_on_top`) to avoid redundant calculations.
        4. The final heuristic value is the sum of the 'mismatched support count'
           and the 'clear cost'.
        5. The heuristic value is 0 if and only if the state is a goal state.
           If the state is a goal, all goal supports match and all goal clear facts
           are true (implying 0 blocks on top for clear goals), resulting in a
           heuristic of 0. If the state is not a goal, at least one goal fact is
           false, which will contribute positively to either the mismatched support
           count or the clear cost, ensuring a value greater than 0.
    """

    def __init__(self, task: Task):
        super().__init__()
        # Heuristic Initialization
        self.goal_support = {}
        self.goal_clear = set()

        # Process goal facts
        for goal_fact_str in task.goals:
            parsed_fact = self._parse_fact(goal_fact_str)
            predicate = parsed_fact[0]
            if predicate == 'on':
                block = parsed_fact[1]
                support = parsed_fact[2]
                self.goal_support[block] = support
            elif predicate == 'on-table':
                block = parsed_fact[1]
                # Add to goal_support only if this block isn't already required to be on something else
                # 'on' facts take precedence over 'on-table' for a block's support goal
                if block not in self.goal_support:
                     self.goal_support[block] = 'table'
            elif predicate == 'clear':
                block = parsed_fact[1]
                self.goal_clear.add(block)
            # Ignore other goal predicates like arm-empty if they exist

        # Process static facts (typically empty in blocksworld for this heuristic)
        # self.static_info = task.static # Placeholder if static facts were relevant

    def _parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string into a tuple."""
        # Remove surrounding brackets and split by space
        # Handles facts like '(on b1 b2)', '(on-table b1)', '(clear b1)', '(arm-empty)'
        parts = fact_str.strip().replace('(', '').replace(')', '').split()
        return tuple(parts)

    def _count_blocks_on_top(self, block, state_above_map, memo):
        """
        Recursively counts blocks stacked on top of the given block in the current state.
        Uses memoization.
        """
        if block in memo:
            return memo[block]

        # Find the block directly on top of 'block'
        # In blocksworld, at most one block can be directly on top
        blocks_directly_above = state_above_map.get(block, set())

        if not blocks_directly_above:
            # Nothing is on this block
            memo[block] = 0
            return 0
        else:
            # Get the single block directly on top
            block_above = next(iter(blocks_directly_above))
            # Count the block above and everything on top of it
            count = 1 + self._count_blocks_on_top(block_above, state_above_map, memo)
            memo[block] = count
            return count


    def __call__(self, node):
        """
        Computes the blocksworld heuristic for the given state node.

        Args:
            node: The search node containing the current state.

        Returns:
            The estimated number of actions to reach the goal state.
        """
        state = node.state

        # 1. Parse the current state to build necessary maps
        state_on_map = {} # Maps block -> block_below or 'table'
        state_clear = set() # Set of blocks that are clear

        # First pass: Process 'on' facts to determine what is on what
        for state_fact_str in state:
            parsed_fact = self._parse_fact(state_fact_str)
            predicate = parsed_fact[0]
            if predicate == 'on':
                block = parsed_fact[1]
                support = parsed_fact[2]
                state_on_map[block] = support
            elif predicate == 'clear':
                block = parsed_fact[1]
                state_clear.add(block)
            # Ignore on-table, holding, arm-empty in this pass

        # Second pass: Process 'on-table' facts for blocks not already placed by 'on' facts
        for state_fact_str in state:
             parsed_fact = self._parse_fact(state_fact_str)
             predicate = parsed_fact[0]
             if predicate == 'on-table':
                 block = parsed_fact[1]
                 # If the block isn't already recorded as being on something else, it's on the table
                 if block not in state_on_map:
                     state_on_map[block] = 'table'

        # Build state_above_map from state_on_map
        state_above_map = {} # Maps block_below or 'table' -> set of blocks directly on top
        for block, support in state_on_map.items():
             if support not in state_above_map:
                 state_above_map[support] = set()
             state_above_map[support].add(block)


        # 2. Calculate the number of blocks whose immediate support is incorrect
        mismatched_support_count = 0
        for block, goal_sup in self.goal_support.items():
            current_sup = state_on_map.get(block) # Get current support, None if block not in state_on_map
            if current_sup != goal_sup:
                 mismatched_support_count += 1

        # 3. Calculate the cost associated with blocks that need to be clear but aren't
        clear_cost = 0
        on_top_memo = {} # Memoization for _count_blocks_on_top
        for block in self.goal_clear:
            if block not in state_clear:
                # Block needs to be clear but isn't. Count blocks on top.
                # _count_blocks_on_top handles cases where the block might not exist
                # or is already clear (though the outer if handles the latter).
                clear_cost += self._count_blocks_on_top(block, state_above_map, on_top_memo)

        # 4. The heuristic value is the sum of the two counts
        h_value = mismatched_support_count + clear_cost

        # 5. Return the heuristic value
        return h_value
