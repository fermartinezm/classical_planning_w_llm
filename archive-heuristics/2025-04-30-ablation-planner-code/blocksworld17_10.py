# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy base class for testing purposes if not provided
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        # Assuming task object provides objects. If not, infer from init/goal.
        # self.objects = task.objects

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    return [] # Return empty list for invalid fact strings

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not in their final goal position relative to the
    table, and adding the number of blocks that are currently on top of any block
    that is not in its final goal position. It aims to capture the cost of moving
    misplaced blocks and clearing blocks that obstruct them.

    # Assumptions
    - Goal states consist of blocks stacked on top of each other or on the table.
    - All blocks mentioned as the lower block in 'on' goal predicates or in 'on-table'
      goal predicates are considered part of the goal structure whose position matters.
    - Blocks not part of this goal support structure are not counted as "misplaced"
      themselves, but can contribute to the heuristic if they are blocking a
      misplaced block.
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - Extracts the goal conditions to build a `goal_support_map`. This map stores,
      for each block that is the lower argument of a goal `(on A B)` or the argument
      of a goal `(on-table B)`, what its required support is (`A` or 'table').
    - Identifies all blocks that are keys in the `goal_support_map`. These are the
      blocks whose goal position relative to the table is explicitly defined by the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal conditions to create a `goal_support_map`. For a goal `(on A B)`,
       `B`'s goal support is `A`. For a goal `(on-table B)`, `B`'s goal support is 'table'.
       Store the set of blocks that are keys in this map (`goal_support_keys`).
    2. Define a recursive helper function `is_in_final_goal_pos(block, state, goal_support_map, memo)`:
       - This function checks if `block` is in its correct goal position relative to the table,
         considering only the goal support structure defined by `goal_support_map`.
       - If `block` is not a key in `goal_support_map`, it means its specific position
         relative to the goal stack structure is not defined; return `True` for this check.
       - If `block`'s goal support is 'table': Return `True` if `(on-table block)` is in the state, `False` otherwise.
       - If `block`'s goal support is `A`: Return `True` if `(on A block)` is in the state AND
         `is_in_final_goal_pos(A, state, goal_support_map, memo)` is `True`.
       - Use memoization (`memo`) to store results and prevent infinite recursion.
    3. Initialize heuristic cost `h = 0`.
    4. Identify the set of `misplaced_blocks_set`: These are the blocks that are keys
       in `goal_support_map` for which `is_in_final_goal_pos` returns `False`.
    5. Calculate `N_misplaced`: This is the count of blocks in `misplaced_blocks_set`. Add `N_misplaced` to `h`.
    6. Calculate `N_blocking_misplaced`: Iterate through all `(on X Y)` facts in the current state.
       If `Y` is in `misplaced_blocks_set`, it means `X` is blocking a block that is not
       in its final goal position. Increment `N_blocking_misplaced`. Add `N_blocking_misplaced` to `h`.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships.
        """
        super().__init__(task)
        self.goal_support_map = {}
        self.goal_support_keys = set() # Blocks whose goal support is explicitly defined

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block_on = parts[1]
                    block_under = parts[2]
                    self.goal_support_map[block_on] = block_under
                    self.goal_support_keys.add(block_on)
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_support_map[block] = 'table'
                    self.goal_support_keys.add(block)
            # Ignore other goal predicates like (clear ?) for this heuristic's structure

    def is_in_final_goal_pos(self, block, state, goal_support_map, memo):
        """
        Recursively check if a block is in its final goal position relative to the table,
        considering only blocks that are part of the goal support structure.
        Uses memoization.
        """
        # If the block is not a key in goal_support_map, its position relative
        # to the goal support structure is not defined by the map.
        # For the purpose of this recursive check, we consider it "correctly placed"
        # relative to the goal support structure. Its actual position only matters
        # if it's blocking a block that *is* in goal_support_map and is misplaced.
        if block not in goal_support_map:
             return True

        if block in memo:
            return memo[block]

        goal_support = goal_support_map[block]

        if goal_support == 'table':
            # Check if the block is on the table in the current state
            is_correct = f'(on-table {block})' in state
        else:
            # Check if the block is on its goal support in the current state
            # AND if the goal support block is in its final goal position
            # Ensure the goal_support block is also in the goal_support_map to continue recursion
            # (it should be if the goal structure is valid, but defensive check)
            is_correct = f'(on {block} {goal_support})' in state and \
                         self.is_in_final_goal_pos(goal_support, state, goal_support_map, memo)

        memo[block] = is_correct
        return is_correct

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        goal_support_map = self.goal_support_map
        goal_support_keys = self.goal_support_keys

        # Calculate N_misplaced: Count blocks in goal_support_keys that are not in final goal position
        misplaced_blocks_set = set()
        memo = {}
        for block in goal_support_keys:
             if not self.is_in_final_goal_pos(block, state, goal_support_map, memo):
                 misplaced_blocks_set.add(block)

        N_misplaced = len(misplaced_blocks_set)

        # Calculate N_blocking_misplaced: Count current 'on' facts (X on Y) where Y is misplaced
        N_blocking_misplaced = 0
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "on" and len(parts) == 3:
                block_on = parts[1]
                block_under = parts[2]
                if block_under in misplaced_blocks_set:
                     N_blocking_misplaced += 1

        h = N_misplaced + N_blocking_misplaced

        return h
