from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Should not happen with valid planner states, but defensive check
         return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic counts the number of blocks whose immediate support (the block
    they are ON or the table) is different from their required immediate support
    in the goal state. Blocks held by the arm are considered to have 'arm' as support.

    This heuristic aims to estimate the number of blocks that are not in their
    correct position relative to the object directly below them in the goal stack.
    Moving a block to its correct immediate support is a necessary step towards
    achieving the goal configuration.

    # Heuristic Initialization
    - Build a map from each block (that is ON another block or ON-TABLE in the goal)
      to its required immediate support (block or 'table') in the goal state.
      Only blocks appearing as the first argument in (on ...) or in (on-table ...) goals
      are included as keys in this map, as these are the blocks whose immediate
      support is explicitly defined as a goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Build the goal support map in the constructor by parsing the goal predicates.
       For each goal predicate `(on B A)`, the goal support for block B is A.
       For each goal predicate `(on-table B)`, the goal support for block B is 'table'.
    2. In the __call__ method, for the given state, determine the current immediate
       support for every block that is a key in the goal support map.
       - If `(on B C)` is true in the state, the current support for B is C.
       - If `(on-table B)` is true in the state, the current support for B is 'table'.
       - If `(holding B)` is true in the state, the current support for B is 'arm'.
       - (Assuming valid states, a block in the goal map will always be in one of these states).
    3. Initialize a counter for misplaced supports to 0.
    4. Iterate through each block that is a key in the goal support map.
    5. Compare the block's current immediate support (found in step 2) with its
       required goal immediate support (from the map built in step 1).
    6. If the current support is different from the goal support, increment the counter.
    7. Return the final count.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal support relationships.
        """
        self.goals = task.goals

        # Build the goal support map: block -> block_below or 'table'
        # Only include blocks whose immediate support is specified in the goal.
        self.goal_support_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_support_map[block] = support
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_support_map[block] = 'table'
            # Ignore other goal predicates like (clear ?x) for this heuristic

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Determine current support for all blocks in the state
        current_support_map = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_support_map[block] = support
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_support_map[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_support_map[block] = 'arm' # Represent held block support

        total_misplaced_support = 0
        # Iterate only over blocks whose immediate support is specified in the goal
        for block, goal_supp in self.goal_support_map.items():
            # Get current support. If a block is in the goal_support_map, it must
            # be in the state (either on-table, on another block, or held).
            # So current_supp will not be None in a valid state.
            current_supp = current_support_map.get(block)

            # Count as misplaced if its current support is different from its goal support.
            if current_supp != goal_supp:
                 total_misplaced_support += 1

        return total_misplaced_support
