from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to split a PDDL fact string into its components."""
    # Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal
    state by counting the number of blocks that are not in their final goal
    position according to the desired stack structure. The count is then
    multiplied by 2, representing a simplified cost of moving each misplaced
    block (e.g., unstack/pickup + stack/putdown).

    Assumptions:
    1. Blocks not explicitly mentioned as being on top of another block or
       on the table in the goal state are assumed to have a goal of being
       on the table.
    2. The cost of moving a block from an incorrect position to its correct
       position is approximately 2 actions (one to pick it up/unstack it,
       and one to put it down/stack it). This simplifies the actual costs
       which can be affected by the arm status and the need to clear other
       blocks.
    3. Clearing blocks that are blocking a desired stack is implicitly
       accounted for, as the blocking blocks themselves will likely be
       counted as not being in their final goal position.

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic parses the goal facts
    (`task.goals`) to build a `goal_support` dictionary. This dictionary
    maps each block that should be on top of another block or on the table
    in the goal to its desired support (either the block below it or the
    string 'table'). It also retrieves the set of all objects (`task.objects`)
    from the planning task.

    Step-By-Step Thinking for Computing Heuristic (`__call__`):
    1. A recursive helper function `is_in_final_goal_position(block)` is defined.
       This function determines if a given block is in its correct place
       relative to its support, and crucially, if that support is also in
       its correct place, all the way down to a block correctly placed on
       the table.
    2. The `is_in_final_goal_position` function uses memoization (`memo` dictionary)
       to avoid redundant calculations for the same block in the recursive calls.
    3. For a block `b`, its desired support (`goal_supp`) is looked up in the
       `self.goal_support` map. If `b` is not a key in `self.goal_support`, it
       defaults to 'table' (Assumption 1).
    4. If `goal_supp` is 'table', the function returns `True` if the fact
       `(on-table b)` is present in the current state, and `False` otherwise.
    5. If `goal_supp` is another block (`under_b`), the function returns `True`
       only if the fact `(on b under_b)` is present in the current state AND
       the recursive call `is_in_final_goal_position(under_b)` also returns `True`.
       Otherwise, it returns `False`.
    6. The main `__call__` method iterates through all objects (`self.all_objects`)
       in the problem instance.
    7. For each block, it calls `is_in_final_goal_position`.
    8. It counts how many blocks return `False` (i.e., are not in their final
       goal position).
    9. The final heuristic value is this count multiplied by 2 (Assumption 2).
    10. The heuristic value is 0 if and only if all blocks are in their final
        goal position, which corresponds to the goal state being reached.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the goal state to determine
        the desired support for each block and getting all objects.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals
        # Assuming task object has a list or set of all objects in the problem
        self.all_objects = task.objects

        # Parse goal to build goal_support map: block -> block_below or 'table'
        self.goal_support = {}
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'on':
                 # Goal: (on block_on_top block_below)
                 block_on_top, block_below = parts[1], parts[2]
                 self.goal_support[block_on_top] = block_below
             elif parts[0] == 'on-table':
                 # Goal: (on-table block)
                 block = parts[1]
                 self.goal_support[block] = 'table'
            # Note: (clear ?x) goals are handled implicitly by the stack structure.
            # If block X is the top of a goal stack, it should be clear.
            # If something is on X in the state, X won't be in its final position
            # according to the recursive definition if its goal is (on X Y),
            # because (on X Y) won't be true. If X's goal is (on-table X),
            # having something on it doesn't prevent (on-table X) from being true,
            # but the block on top will likely not be in its final position
            # and will be counted as not in its final position based on its own goal.


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer heuristic value estimating the cost to reach the goal.
        """
        state = node.state

        # Use memoization for the recursive function
        memo = {}

        def is_in_final_goal_position(b):
            """
            Recursively checks if block 'b' is in its final goal position.

            A block 'b' is in its final goal position if:
            1. Its goal is to be on the table, and it is currently on the table.
            2. Its goal is to be on block 'under_b', and it is currently on
               'under_b', AND 'under_b' is also in its final goal position.

            Blocks not explicitly mentioned as being on top of another block
            or on the table in the goal (i.e., not keys in self.goal_support)
            are assumed to have a goal of being on the table.
            """
            if b in memo:
                return memo[b]

            # Default goal is on table if not explicitly defined in goal_support keys
            # This handles blocks that are not part of any goal stack but exist
            # in the problem instance.
            goal_supp = self.goal_support.get(b, 'table')

            result = False
            if goal_supp == 'table':
                # Check if the block is currently on the table
                result = ('(on-table ' + b + ')') in state
            else: # goal_supp is a block name
                # Check if the block is currently on its goal support block
                # AND the goal support block is in its final position
                if ('(on ' + b + ' ' + goal_supp + ')') in state:
                     result = is_in_final_goal_position(goal_supp)

            memo[b] = result
            return result

        # Count blocks that are not in their final goal position
        count_not_in_final_position = 0
        for b in self.all_objects:
            if not is_in_final_goal_position(b):
                count_not_in_final_position += 1

        # The heuristic value is 2 times the number of blocks not in their
        # final goal position. This estimates the cost of picking up/unstacking
        # and placing each such block.
        # Each block that is not in its final position needs to be moved.
        # Moving a block typically involves unstacking/picking it up (1 action)
        # and stacking/putting it down (1 action), assuming the arm is free
        # and the target location is clear. This is a simplification, but
        # provides a reasonable non-admissible estimate.
        h = count_not_in_final_position * 2

        # The heuristic is 0 if and only if all blocks are in their final
        # goal position according to the recursive definition. This corresponds
        # to the goal state being reached, as the goal state defines the
        # complete stack structure.

        return h
