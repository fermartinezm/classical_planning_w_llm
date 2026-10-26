from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure input is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    # Fallback for unexpected formats, though PDDL facts should be (pred arg...)
    return fact_str.split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions required to reach the goal state.
    The heuristic counts the number of blocks that are not in their correct
    goal position (either on the table or on the block they should be on)
    and multiplies this count by 2. This estimates the minimum number of
    pickup/unstack and putdown/stack actions needed to move each misplaced
    block, ignoring dependencies and clearing costs.

    Heuristic value is 0 only for goal states.
    Heuristic value is finite for solvable states.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal configuration.
        """
        super().__init__(task)

        # Build the goal configuration: block -> support (block or 'table')
        # This maps a block to the object it should be directly on top of in the goal.
        self.goal_pos = {}

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block_above, block_below = parts[1], parts[2]
                self.goal_pos[block_above] = block_below
            elif predicate == "on-table":
                block = parts[1]
                self.goal_pos[block] = 'table'
            # (clear ?) and (arm-empty) goals are not directly used to determine
            # a block's target support, so they are ignored for building goal_pos.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build the current configuration: block -> support (block, 'table', or 'holding')
        current_pos = {}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block_above, block_below = parts[1], parts[2]
                current_pos[block_above] = block_below
            elif predicate == "on-table":
                block = parts[1]
                current_pos[block] = 'table'
            elif predicate == "holding":
                block = parts[1]
                current_pos[block] = 'holding' # Represent 'holding' as a support type
            # Ignore (clear ?) and (arm-empty) facts for building current_pos

        # Count blocks that are part of the goal configuration but are not in their goal position
        misplaced_count = 0
        # Iterate only over blocks whose position is specified in the goal
        for block, goal_support in self.goal_pos.items():
             current_support = current_pos.get(block)

             # A block is misplaced if its current support is different from its goal support.
             # If a block is in goal_pos but not in current_pos, it's implicitly misplaced.
             # The .get(block) will return None if the block is not found in current_pos,
             # which correctly triggers the != comparison if goal_support is not None.
             # Since goal_support comes from self.goal_pos.items(), it will never be None.
             # So, if current_support is None (block not found in state config), it's counted as misplaced.
             if current_support != goal_support:
                 misplaced_count += 1

        # Simple heuristic: 2 actions per misplaced block (pickup/unstack + putdown/stack)
        # This is a non-admissible estimate.
        return 2 * misplaced_count
