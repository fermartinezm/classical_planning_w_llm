from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Define the heuristic class
class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions required to reach the goal state.
    The heuristic counts the number of blocks that are not currently
    on their correct goal parent (or table) and multiplies this count by 2.
    This is based on the idea that moving a block to its correct position
    typically requires at least two actions (pickup/unstack and stack/putdown).

    This heuristic is non-admissible but aims to be informative for greedy search.
    It does not explicitly account for the cost of clearing blocks that are
    in the way, which is a simplification that makes it efficient to compute.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for each block.
        """
        super().__init__(task)

        # Map block names to their goal parent (the block they should be on, or 'table').
        # Only includes blocks whose goal position is explicitly defined by an 'on' or 'on-table' predicate.
        self.goal_parent = {}

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_parent[block] = 'table'

        # Blocks mentioned as parents in 'on' goals but not having their own
        # 'on' or 'on-table' goal predicate are ignored by this heuristic,
        # as their final position is not specified. This is consistent with
        # iterating over self.goal_parent keys in __call__.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        The heuristic is 2 * (number of blocks not on their goal parent).
        """
        state = node.state

        # Determine the current parent for each block present in the state.
        current_parent = {}
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                current_parent[parts[1]] = parts[2]
            elif parts[0] == 'on-table':
                current_parent[parts[1]] = 'table'
            elif parts[0] == 'holding':
                held_block = parts[1]

        # If a block is held, its parent is the arm.
        if held_block is not None:
             current_parent[held_block] = 'arm'

        misplaced_count = 0

        # Iterate through blocks that have a defined goal position.
        # We only count blocks as misplaced if their goal position is known
        # and their current position is different.
        for block, goal_p in self.goal_parent.items():
            current_p = current_parent.get(block)

            # If the block is not found in the current state facts (on, on-table, holding),
            # or if its current parent is different from its goal parent, it's misplaced.
            # A block not found in current_parent means it's not on anything, on table, or held,
            # which is an invalid state in Blocksworld. We treat this as misplaced.
            if current_p != goal_p:
                 misplaced_count += 1

        # The heuristic value is 2 times the number of misplaced blocks.
        # Each misplaced block requires at least 2 actions (pickup/unstack + stack/putdown)
        # to potentially move it to its correct place.
        return 2 * misplaced_count
