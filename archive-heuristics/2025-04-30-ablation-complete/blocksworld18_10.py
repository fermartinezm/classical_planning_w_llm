# from heuristics.heuristic_base import Heuristic # Assuming this is provided

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It counts blocks that are not in their correct goal position relative to their intended support,
    adding costs for clearing blocks that are in the way, and a cost for placing a held block.

    # Assumptions
    - All actions have a cost of 1.
    - The goal defines a specific configuration of blocks stacked on each other or on the table.
    - All blocks present in the initial state are expected to be part of the goal configuration (i.e., mentioned in goal 'on' or 'on-table' facts).

    # Heuristic Initialization
    - Extracts all object names from the initial state and goal facts.
    - Builds a map from each block to the block (or 'table') it should be on in the goal state, based on goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h` to 0.
    2. Identify the block currently held by the arm, if any.
    3. Build a map representing the current stack configuration (which block is immediately on top of another).
    4. For each block `B` that exists in the problem:
       a. Determine its intended support `G` in the goal state (the block or 'table' it should be on). If the block is not mentioned as being on something or on the table in the goal, skip it.
       b. Check if `B` is currently on its intended support `G` (i.e., `(on B G)` or `(on-table B)` is true in the current state).
       c. If `B` is NOT currently on its intended support:
          i. If `B` is currently held by the arm, add 1 to `h` (cost to place it).
          ii. If `B` is NOT currently held, add 2 to `h` (cost to pickup/unstack + cost to place).
          iii. If `B` is NOT currently held, iterate upwards from `B` in the current stack using the stack map: for each block `X` directly on top of the current block in this tower, add 2 to `h` (cost to move `X` out of the way).
    5. The total value of `h` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting objects and goal positions.
        """
        self.goals = task.goals
        # Blocksworld has no relevant static facts for this heuristic
        # self.static = task.static

        # Extract all objects from initial state and goal facts
        self.objects = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            # Add arguments of predicates like (on x y), (on-table x), (clear x), (holding x)
            if len(parts) > 1:
                 self.objects.update(parts[1:])
        for goal in task.goals:
             parts = get_parts(goal)
             # Add arguments of predicates like (on x y), (on-table x), (clear x)
             if len(parts) > 1:
                 self.objects.update(parts[1:])

        # Build the goal map: block -> block_below_in_goal or 'table'
        self.goal_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on' and len(parts) == 3:
                self.goal_map[parts[1]] = parts[2]
            elif parts[0] == 'on-table' and len(parts) == 2:
                self.goal_map[parts[1]] = 'table'

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Heuristic is 0 if the goal is reached
        if self.goals <= state:
            return 0

        h = 0

        # Find the held block, if any
        held_block = None
        for fact in state:
            if fact.startswith('(holding '):
                held_block = get_parts(fact)[1]
                break

        # Build current stack map for efficient lookup of blocks on top
        current_stack_map = {} # maps block_below -> block_on_top
        for fact in state:
            if fact.startswith('(on '):
                parts = get_parts(fact)
                if len(parts) == 3:
                    current_stack_map[parts[2]] = parts[1]

        for block in self.objects:
            # Find intended support G for block in the goal
            goal_pos = self.goal_map.get(block)

            # If block is not specified in goal 'on' or 'on-table' facts, ignore it for this heuristic
            if goal_pos is None:
                 continue

            # Check if block is currently on its intended support G
            is_on_goal_support = False
            if goal_pos == 'table':
                if f'(on-table {block})' in state:
                    is_on_goal_support = True
            elif goal_pos != 'holding': # Goal is on another block U
                if f'(on {block} {goal_pos})' in state:
                    is_on_goal_support = True

            if not is_on_goal_support:
                # Block is not on its intended support. It needs to be moved.
                # If it's held, it needs 1 action (stack/putdown).
                # If not held, it needs 2 actions (pickup/unstack + stack/putdown).
                if block == held_block:
                    h += 1
                else:
                    h += 2

                # Add cost for blocks on top that need to be moved out of the way
                # This only applies if the block itself is not held, as nothing can be on top of a held block.
                if block != held_block:
                    current_on_block = block
                    while current_on_block in current_stack_map:
                        block_on_top = current_stack_map[current_on_block]
                        # This block on top also needs to be moved. Add 2 actions.
                        h += 2
                        current_on_block = block_on_top

        return h
