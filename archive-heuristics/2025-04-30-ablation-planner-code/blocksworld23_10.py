# Assuming the Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# If the base class is not available, a dummy class like the one below
# would be needed for the code to be executable standalone, but the prompt
# asks for only the heuristic code, implying the base class exists.
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#         pass
#     def __call__(self, node):
#         raise NotImplementedError("Heuristic must implement __call__")


# Helper function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and multiple spaces between parts
    return fact.strip()[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal
    predicates by summing the estimated cost for each unsatisfied goal predicate
    independently. The estimated cost for a predicate considers the effort to
    clear any blocks currently on top of the relevant blocks, plus a base cost
    for the necessary pick/unstack, stack/putdown actions.

    # Assumptions
    - The cost of each action (pickup, putdown, stack, unstack) is 1.
    - The heuristic sums costs for unsatisfied goal predicates independently,
      ignoring potential negative interactions or shared subgoals (like h_add).
    - The cost to clear a block is the number of blocks currently stacked directly
      or indirectly on top of it.
    - If (arm-empty) is a goal and the arm is not empty, one action (putdown) is needed.
    - The 'table' is treated as a special location that cannot have blocks on top.

    # Heuristic Initialization
    - Stores the goal predicates from the task.
    - Static facts are not used as the Blocksworld domain has none relevant to the heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the support relationship between blocks
       (`current_support`) and which block is directly on top of another
       (`current_on_top`). Also, identify all blocks present in the state and
       whether the arm is empty.
    2. For every block in the state, calculate the set of blocks currently stacked
       directly or indirectly on top of it (`blocks_above`). The size of this set
       represents the estimated cost to make the block clear. This is computed
       recursively using memoization.
    3. Initialize the total heuristic cost `h` to 0.
    4. Convert the state facts (frozenset) into a set of strings for efficient lookup.
    5. Iterate through each goal predicate defined in the task:
       a. If the goal predicate is already satisfied in the current state (exists in the set of state fact strings), add 0 to `h`.
       b. If the goal predicate is `(on ?x ?y)` and is not satisfied:
          - Estimate the cost to clear `?x` (number of blocks above `?x`).
          - Estimate the cost to clear `?y` (number of blocks above `?y`).
          - Add these clearing costs plus 2 (for unstack/pickup `?x` and stack `?x` on `?y`) to `h`.
       c. If the goal predicate is `(on-table ?x)` and is not satisfied:
          - Estimate the cost to clear `?x` (number of blocks above `?x`).
          - Add this clearing cost plus 2 (for unstack/pickup `?x` and putdown `?x`) to `h`.
       d. If the goal predicate is `(clear ?x)` and is not satisfied:
          - Estimate the cost to clear `?x` (number of blocks above `?x`).
          - Add this clearing cost to `h`.
       e. If the goal predicate is `(arm-empty)` and is not satisfied:
          - Add 1 (for the putdown action) to `h`.
    6. Return the total calculated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Step 1: Parse the current state
        current_support = {}
        current_on_top = {}
        all_blocks_in_state = set()
        arm_is_empty = False
        # held_block = None # Not needed for heuristic calculation

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_support[block] = support
                current_on_top[support] = block
                all_blocks_in_state.add(block)
                all_blocks_in_state.add(support)
            elif predicate == 'on-table':
                block = parts[1]
                current_support[block] = 'table'
                all_blocks_in_state.add(block)
            elif predicate == 'holding':
                block = parts[1]
                # held_block = block # Store if needed, but not for this heuristic logic
                all_blocks_in_state.add(block)
            elif predicate == 'clear':
                block = parts[1]
                all_blocks_in_state.add(block)
            elif predicate == 'arm-empty':
                arm_is_empty = True

        # Step 2: Calculate blocks above each block
        blocks_above = {}
        memo_blocks_above = {}

        def get_blocks_above_recursive(block, current_on_top, memo):
            if block in memo:
                return memo[block]

            blocks = set()
            # Find the block directly on top of 'block'
            # current_on_top maps support -> block_on_top.
            # So current_on_top.get(block) gives the block that is on top of 'block'.
            block_directly_on_top = current_on_top.get(block)

            if block_directly_on_top:
                blocks.add(block_directly_on_top)
                blocks.update(get_blocks_above_recursive(block_directly_on_top, current_on_top, memo))

            memo[block] = blocks
            return blocks

        # Compute for all blocks found in the state
        for block in all_blocks_in_state:
             blocks_above[block] = get_blocks_above_recursive(block, current_on_top, memo_blocks_above)

        # Step 3 & 6: Initialize heuristic cost and return
        h = 0

        # Step 4: Convert state facts to strings for lookup
        state_facts_strings = {str(fact) for fact in state}

        # Step 5: Iterate through goal predicates
        for goal in self.goals:
            goal_str = str(goal)
            if goal_str not in state_facts_strings:
                parts = get_parts(goal_str)
                predicate = parts[0]

                if predicate == 'on':
                    block, support = parts[1], parts[2]
                    # Get blocks above, defaulting to empty set if block/support not in state (shouldn't happen in valid PDDL)
                    cost_to_clear_block = len(blocks_above.get(block, set()))
                    cost_to_clear_support = len(blocks_above.get(support, set()))
                    h += cost_to_clear_block + cost_to_clear_support + 2
                elif predicate == 'on-table':
                    block = parts[1]
                    cost_to_clear_block = len(blocks_above.get(block, set()))
                    h += cost_to_clear_block + 2
                elif predicate == 'clear':
                    block = parts[1]
                    cost_to_clear_block = len(blocks_above.get(block, set()))
                    h += cost_to_clear_block
                elif predicate == 'arm-empty':
                    if not arm_is_empty:
                        h += 1

        return h
