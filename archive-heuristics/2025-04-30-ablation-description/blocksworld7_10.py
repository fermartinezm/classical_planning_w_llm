# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts gracefully
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Depending on expected input quality, might raise an error or log a warning
        return [] # Indicate parsing failure
    return fact[1:-1].split()

# Assuming Heuristic base is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number
    of blocks that are not in their correct goal position relative to their
    immediate support (either another block or the table).

    # Assumptions:
    - The goal specifies a desired stack configuration using 'on' and 'on-table' predicates.
    - 'clear' and 'arm-empty' predicates in the goal are typically consequences
      of the stack configuration and are not directly counted as misplaced items
      in this heuristic.
    - The heuristic counts a block as 'misplaced' if its current support
      (what it's on, on the table, or being held) is different from its
      required support in the goal state.

    # Heuristic Initialization
    - The heuristic parses the goal state to determine the required immediate
      support for each block involved in an 'on' or 'on-table' goal predicate.
      This mapping is stored in `self.goal_supports`.

    # Step-By-Step Thinking for Computing Heuristic
    1. In the `__init__` method, iterate through the goal facts provided in the `task`.
    2. For each goal fact `(on X Y)`, record that block `X` needs to be supported by block `Y`.
    3. For each goal fact `(on-table X)`, record that block `X` needs to be supported by the 'table'.
    4. In the `__call__` method, which takes a `node` (representing a state), initialize the heuristic value `h` to 0.
    5. Parse the current state facts to determine the current immediate support for each block. This involves checking for `(on X Z)`, `(on-table X)`, and `(holding X)` facts.
    6. Iterate through each block `X` for which a goal support was defined in `__init__`.
    7. Find the current support for block `X` in the given state.
    8. Compare the current support with the goal support for `X`.
    9. If the current support is different from the goal support, increment `h` by 1.
    10. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships.
        """
        # Assuming Heuristic base class exists and handles task assignment
        super().__init__(task)

        # Map block -> its required support ('table' or another block) from the goal state.
        self.goal_supports = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_supports[block] = support
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_supports[block] = 'table' # Use a special marker for table
            # Ignore other goal predicates like 'clear' or 'arm-empty' for this heuristic

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build a mapping of current supports for all blocks in the state
        current_supports = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'on' and len(parts) == 3:
                block_above, block_below = parts[1], parts[2]
                current_supports[block_above] = block_below # block_above is on block_below
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_supports[block] = 'table' # block is on the table
            elif parts[0] == 'holding' and len(parts) == 2:
                block = parts[1]
                current_supports[block] = 'holding' # block is being held

        heuristic_value = 0

        # Iterate through blocks that have a defined goal support
        for block, goal_support in self.goal_supports.items():
            # Find the current support for this block
            # If a block is in goal_supports but not in current_supports,
            # it implies it's not on anything, on table, or holding.
            # This shouldn't happen in valid states, but if it does,
            # it's definitely not in the goal support position.
            current_support = current_supports.get(block)

            # Compare current support with goal support
            if current_support != goal_support:
                 heuristic_value += 1

        return heuristic_value
