from fnmatch import fnmatch
# Assume Heuristic base class is imported from heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define get_parts helper function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

# Assume Heuristic base class is imported from heuristics.heuristic_base
# If running standalone or in a different environment, you might need a mock class
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed to reach the goal state.
    The heuristic counts:
    1. Blocks that are not on their correct goal support (another block or the table).
    2. Blocks that are currently on top of another block, where this 'on' relationship
       is not part of the goal configuration. These are considered 'blocking' blocks.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by parsing the goal state to determine
        the desired support for each block.
        """
        super().__init__(task)

        # Parse goal facts to build the desired support mapping
        self.goal_support = {}
        self.goal_on = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                x, y = parts[1], parts[2]
                self.goal_support[x] = y
                self.goal_on.add((x, y))
            elif parts[0] == 'on-table':
                x = parts[1]
                self.goal_support[x] = 'table'
            # Ignore (clear ?) and (arm-empty) goals for this heuristic's structure

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Parse current state to build support mapping
        current_support = {}
        current_on = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                x, y = parts[1], parts[2]
                current_support[x] = y
                current_on.add((x, y))
            elif parts[0] == 'on-table':
                x = parts[1]
                current_support[x] = 'table'
            elif parts[0] == 'holding':
                x = parts[1]
                current_support[x] = 'arm' # Represent holding as supported by 'arm'
            # Ignore (clear ?) and (arm-empty) state facts for this heuristic's structure

        h = 0

        # 1. Count blocks not on their goal support
        # Iterate through blocks that have a defined goal support
        for block, goal_sup in self.goal_support.items():
            # If the block is in the current state and its support is wrong
            # A block is in the current state if it's on something or held.
            # This is captured by checking if it's a key in current_support.
            if block in current_support and current_support[block] != goal_sup:
                 h += 1
            # If block is in goal_support but NOT in current_support, it's an error state
            # or the block is somehow missing. Assuming valid states, this won't happen.


        # 2. Count blocking blocks
        # These are blocks C on B where (on C B) is NOT a goal fact.
        # This counts blocks that are "in the way" of forming goal stacks.
        for c, b in current_on:
            if (c, b) not in self.goal_on:
                h += 1

        # The heuristic value is the sum of misplaced blocks and blocking blocks.
        return h
