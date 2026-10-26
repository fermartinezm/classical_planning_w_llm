from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the input is a string and looks like a PDDL fact
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for malformed or non-string input
        return []
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed to reach the goal state by counting
    blocks that are not in their correct position within the goal stacks.

    # Summary
    The heuristic counts the number of blocks that are not currently positioned
    correctly relative to the block below them *within the context of the final
    goal stack configuration*. Each such block is estimated to require 2 actions
    (pickup/unstack + putdown/stack) to be moved towards its goal position.
    Blocks whose final position is not specified in the goal (i.e., not in an
    `(on ...)` or `(on-table ...)` goal predicate) are ignored by the heuristic.

    # Heuristic Initialization
    - Parses the goal predicates to determine the desired support (what each block
      should be directly on top of, or the table). This creates the `goal_support_map`.
    - Collects all relevant blocks mentioned in the initial state and goals for completeness,
      though the heuristic calculation focuses only on blocks in `goal_support_map`.

    # Step-By-Step Thinking for Computing Heuristic
    1. For a given state, determine the current support for each block (what it's
       directly on top of, or the table). Identify if any block is currently held.
       This creates the `current_support_map` and `held_block`.
    2. Initialize a `misplaced_count` to 0.
    3. Initialize a memoization cache `memo` for the recursive check.
    4. Iterate through each block that is part of a goal stack (i.e., each block
       that is a key in the `goal_support_map`).
       a. If the block is currently held (`block == held_block`), it is not in its
          goal stack position. Increment `misplaced_count`. Continue to the next block.
       b. If the block is not held, recursively check if it is in its correct
          goal stack position using the `_is_in_goal_stack_recursive` helper function.
          - A block B is in its goal stack position if:
            - Its goal is `(on-table B)` AND it is currently `(on-table B)`.
            - Its goal is `(on B C)` AND it is currently `(on B C)` AND block C
              is also in its goal stack position (checked recursively).
          - The recursive function uses the `memo` cache to avoid redundant calculations.
       c. If the recursive check determines the block is NOT in its goal stack
          position, increment `misplaced_count`.
    5. The heuristic value is the total `misplaced_count` multiplied by 2,
       representing an estimate of the minimum moves required (pickup/unstack +
       putdown/stack) for each block that needs to be repositioned. This is a
       non-admissible estimate suitable for greedy search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships
        and identifying all relevant blocks.
        """
        self.goals = task.goals
        # Maps a block to the block it should be directly on top of in the goal,
        # or the string 'table' if it should be on the table.
        self.goal_support_map = {}
        # Set of all blocks mentioned in the problem (init or goal)
        self.all_blocks = set()

        # Parse goals to build the goal_support_map and collect blocks
        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                block, under_block = args
                self.goal_support_map[block] = under_block
                self.all_blocks.add(block)
                self.all_blocks.add(under_block)
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                self.goal_support_map[block] = 'table'
                self.all_blocks.add(block)

            # Add any arguments from other goal predicates (like 'clear', 'holding')
            # that look like objects, just to have a comprehensive list of blocks.
            for arg in args:
                 if isinstance(arg, str) and not arg.startswith('('): # Avoid adding predicate names or malformed things
                     self.all_blocks.add(arg)


        # Parse initial state to get all blocks mentioned there as well
        for init_fact_str in task.initial_state:
             parts = get_parts(init_fact_str)
             if not parts: continue
             args = parts[1:]
             for arg in args:
                 if isinstance(arg, str) and not arg.startswith('('):
                     self.all_blocks.add(arg)


    def _is_in_goal_stack_recursive(self, block, current_support_map, memo):
        """
        Helper function to recursively check if a block is in its correct
        position within the goal stack structure. Uses memoization.

        Args:
            block (str): The block to check.
            current_support_map (dict): Maps blocks to their current support ('table' or another block).
            memo (dict): Memoization cache {block: boolean result}.

        Returns:
            bool: True if the block is in its goal stack position, False otherwise.
        """
        # If the block is not in the goal support map, it's not part of a
        # specific goal stack we are tracking. Consider it "in place" for heuristic purposes.
        if block not in self.goal_support_map:
            return True

        # Check memoization cache
        if block in memo:
            return memo[block]

        goal_under = self.goal_support_map[block]
        # Get current support. Use .get() because the block might be held,
        # in which case it won't be in current_support_map keys from 'on' or 'on-table'.
        current_under = current_support_map.get(block)

        result = False
        if goal_under == 'table':
            # Goal is on the table, check if it's currently on the table
            result = (current_under == 'table')
        else: # goal_under is a block
            # Goal is on another block, check if it's currently on that block
            # AND if the block it's on is in its goal stack position
            if current_under == goal_under:
                 # Recursively check the block below it
                 result = self._is_in_goal_stack_recursive(goal_under, current_support_map, memo)
            else:
                 # Not on the correct block
                 result = False

        # Store result in memo and return
        memo[block] = result
        return result


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node (Node): The search node containing the state.

        Returns:
            int: The estimated number of actions to reach the goal.
        """
        state = node.state

        # Build current support map and find held block
        current_support_map = {}
        held_block = None
        for state_fact_str in state:
            parts = get_parts(state_fact_str)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                block, under_block = args
                current_support_map[block] = under_block
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                current_support_map[block] = 'table'
            elif predicate == 'holding' and len(args) == 1:
                held_block = args[0]

        memo = {} # Memoization cache for recursive calls
        misplaced_count = 0

        # Iterate only through blocks that have a specified goal position
        # in the goal_support_map. Blocks not in goal_support_map are "don't care"
        # for their position and don't contribute to this heuristic.
        for block in self.goal_support_map.keys():
            # If the block is currently held, it's definitely not in its goal stack position
            if block == held_block:
                 misplaced_count += 1
                 continue # No need to check support if held

            # Check if the block is in its goal stack position recursively.
            # The recursive function handles cases where the block is not in
            # current_support_map (i.e., not on anything, like floating - though
            # only 'held' is a valid state for not being on something).
            if not self._is_in_goal_stack_recursive(block, current_support_map, memo):
                misplaced_count += 1

        # Estimate 2 actions per misplaced block (pickup/unstack + putdown/stack).
        # This is a simple, non-admissible estimate.
        return misplaced_count * 2

