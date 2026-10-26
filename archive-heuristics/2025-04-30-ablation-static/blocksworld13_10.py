from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assuming fact is a string representation of a PDDL predicate
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # This case should ideally not happen with valid PDDL facts
         return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number
    of blocks that are not currently in their correct position within their
    goal stack, considering the blocks below them. Each such misplaced block
    is estimated to require at least 2 actions (e.g., unstack/pickup and
    stack/putdown) to be placed correctly.

    # Assumptions
    - The goal is specified as a set of `(on X Y)` and `(on-table X)` predicates,
      defining one or more desired stacks of blocks.
    - `(clear X)` and `(arm-empty)` predicates in the goal are ignored by this heuristic,
      as they are typically intermediate states or implicitly achieved.
    - The heuristic assumes a minimum cost of 2 actions to correct the position
      of a block that is not part of a correctly built goal stack segment.
    - The goal structure forms valid stacks (no cycles, no block on table and another block simultaneously).

    # Heuristic Initialization
    - Extracts the desired `on` relationships and `on-table` relationships from
      the goal predicates to build the target stack structures.
    - Stores these relationships in `self.goal_on_map` (block -> support_block)
      and `self.goal_ontable_set` (set of blocks that should be on the table).
    - Identifies all blocks involved as the 'top' block in goal `on` or `on-table` predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal configuration: Parse the goal predicates `(on X Y)` and
       `(on-table X)` to determine, for each block `X`, what it should be
       immediately on top of (`Y` or `Table`). Store this structure (e.g.,
       in `goal_on_map` and `goal_ontable_set`). Collect all blocks `X` that
       appear as the first argument in these goal predicates into `goal_blocks_to_place`.
    2. Define a recursive check `_is_correctly_stacked(block, state)`:
       - This function determines if a block `block` is in its correct goal
         position *and* everything below it in its goal stack is also correctly
         positioned relative to its goal support.
       - Base case: If `block` should be on the table (`block` in `self.goal_ontable_set`),
         it is correctly stacked if and only if `(on-table block)` is true in the `state`.
       - Recursive step: If `block` should be on `support` (`support = self.goal_on_map[block]`),
         it is correctly stacked if and only if `(on block support)` is true in the `state`
         AND `_is_correctly_stacked(support, state)` is true.
       - If `block` is a support for another block but is not itself in `self.goal_on_map`
         or `self.goal_ontable_set` as a block to be placed, its position relative
         to its own support is not specified by the goal structure we are tracking.
         The recursion stops here and assumes this part is correct relative to the goal.
    3. Compute the heuristic value:
       - Initialize a counter `incorrectly_stacked_count` to 0.
       - Iterate through all blocks that are the 'top' block in goal `on` or `on-table`
         predicates (`self.goal_blocks_to_place`).
       - For each such block, call `_is_correctly_stacked(block, state)`.
       - If the function returns `False`, increment `incorrectly_stacked_count`.
    4. The final heuristic value is `2 * incorrectly_stacked_count`. This multiplier
       represents a simplified estimate of the minimum actions (e.g., unstack/pickup + stack/putdown)
       required to correct the position of each block that is not part of a
       correctly formed goal stack segment.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal stack information.
        """
        self.goals = task.goals

        # Map block -> block it should be on in the goal
        self.goal_on_map = {}
        # Set of blocks that should be on the table in the goal
        self.goal_ontable_set = set()
        # Set of blocks that are the 'top' block in a goal on/on-table predicate
        self.goal_blocks_to_place = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    self.goal_on_map[block] = support
                    self.goal_blocks_to_place.add(block)
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_ontable_set.add(block)
                    self.goal_blocks_to_place.add(block)
            # Ignore 'clear' and 'arm-empty' goals

        # Consistency check: A block should not be in goal_on_map keys and goal_ontable_set
        # assert self.goal_on_map.keys().isdisjoint(self.goal_ontable_set)

    def _is_correctly_stacked(self, block, state):
        """
        Recursively checks if a block is in its correct goal position and
        everything below it in the goal stack is also correctly positioned.
        Assumes 'block' is one of the blocks we care about placing
        (i.e., in self.goal_blocks_to_place or a support for one).
        """
        # Case 1: Block should be on the table
        if block in self.goal_ontable_set:
            return f"(on-table {block})" in state

        # Case 2: Block should be on another block
        if block in self.goal_on_map:
            support = self.goal_on_map[block]
            # Check if block is currently on the correct support AND the support is correctly stacked
            return f"(on {block} {support})" in state and self._is_correctly_stacked(support, state)

        # Case 3: Block is a support for another block, but is not itself
        # explicitly placed relative to a lower block or the table in the goal.
        # The recursion stops here. We assume this part of the stack is correct
        # relative to the goal structure being built.
        # This happens when a block 'Y' is a value in goal_on_map (X -> Y)
        # but 'Y' is not a key in goal_on_map and not in goal_ontable_set.
        return True


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        incorrectly_stacked_count = 0

        # Iterate through all blocks that are the 'top' block in goal on/on-table predicates
        for block in self.goal_blocks_to_place:
            if not self._is_correctly_stacked(block, state):
                incorrectly_stacked_count += 1

        # Each incorrectly stacked block needs to be moved. A move typically
        # involves at least two actions (pickup/unstack + putdown/stack).
        # This is a non-admissible estimate.
        return 2 * incorrectly_stacked_count
