from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact string.

    Assumes fact is a string like '(predicate arg1 arg2)'.
    Returns a list of strings, e.g., ['predicate', 'arg1', 'arg2'].
    """
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost to reach the goal by counting two types
    of "misplaced" blocks:
    1. Blocks that are not in their correct goal position (relative to their
       immediate parent block or the table).
    2. Blocks that are currently sitting directly on top of another block that is
       itself not in its correct goal position.

    The heuristic value is the sum of the counts from these two categories.
    It is not admissible but aims to guide a greedy best-first search
    effectively by penalizing states where blocks are in the wrong place
    or are obstructing misplaced blocks.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and
        identifying all blocks involved in the problem.

        Args:
            task: The planning task object.
        """
        self.task_initial_state = task.initial_state
        self.goals = task.goals

        # Collect all blocks from initial state and goal facts.
        # Assumes any argument in an initial or goal predicate is a block object.
        self.all_blocks = set()
        for fact in self.task_initial_state:
             parts = get_parts(fact)
             # Add all arguments as potential objects (excluding predicate name)
             self.all_blocks.update(parts[1:])
        for fact in self.goals:
             parts = get_parts(fact)
             # Add all arguments as potential objects (excluding predicate name)
             self.all_blocks.update(parts[1:])

        # Build goal configuration (block -> parent or 'table').
        # Blocks not explicitly mentioned as being 'on' another block or 'on-table'
        # in the goal are implicitly assumed to belong on the table.
        self.goal_config = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'on':
                self.goal_config[parts[1]] = parts[2] # block -> parent
            elif parts[0] == 'on-table':
                self.goal_config[parts[1]] = 'table' # block -> 'table'

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state (a frozenset of facts).

        Returns:
            An integer heuristic value.
        """
        state = node.state

        # Build current configuration (block -> parent or 'table' or 'arm').
        # Initialize all blocks to 'unknown' and update based on state facts.
        # 'unknown' is a placeholder; a block should always be on something or held.
        current_config = {}
        for block in self.all_blocks:
             current_config[block] = 'unknown' # Default state

        for state_fact in state:
            parts = get_parts(state_fact)
            if parts[0] == 'on':
                current_config[parts[1]] = parts[2] # block -> parent
            elif parts[0] == 'on-table':
                current_config[parts[1]] = 'table' # block -> 'table'
            elif parts[0] == 'holding':
                current_config[parts[1]] = 'arm' # block -> 'arm'
            # 'clear' and 'arm-empty' facts do not define parent relationships

        # Calculate which blocks are out of place (current parent != goal parent).
        # Blocks not in goal_config are assumed to have a goal parent of 'table'.
        is_out_of_place = {}
        for block in self.all_blocks:
            goal_parent = self.goal_config.get(block, 'table')
            current_parent = current_config.get(block, 'unknown') # Should be updated by state facts

            is_out_of_place[block] = (current_parent != goal_parent)

        h = 0
        # Part 1: Count blocks that are out of place.
        for block in self.all_blocks:
            if is_out_of_place[block]:
                h += 1

        # Part 2: Count blocks that are on top of out-of-place blocks.
        # Iterate through blocks and check their current parent.
        for block in self.all_blocks:
            current_parent = current_config.get(block, 'unknown')
            # Check if the current parent is a block (i.e., not 'table', 'arm', or 'unknown')
            if current_parent in self.all_blocks:
                # If the parent block is out of place, add 1 to heuristic.
                # Use .get with False default just in case, though parent should be in all_blocks.
                if is_out_of_place.get(current_parent, False):
                    h += 1

        return h

