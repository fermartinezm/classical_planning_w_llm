# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Parses a PDDL fact string into a list of parts."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

# Assuming a base class like this exists in heuristics/heuristic_base.py
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

# Import the base class
from heuristics.heuristic_base import Heuristic


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Estimates the cost to reach the goal state based on the number of
    misplaced blocks within goal stacks and violated clear conditions.

    Summary:
    The heuristic counts two types of discrepancies between the current state
    and the goal state:
    1. Blocks that are part of a goal stack but are not currently positioned
       correctly relative to their intended support, considering that the
       support itself must also be correctly positioned relative to its goal
       support, and so on, down to the table.
    2. Blocks that are required to be clear in the goal state but are not
       clear in the current state.
    The total heuristic value is the sum of these two counts. A value of 0
    indicates the goal state has been reached.

    Assumptions:
    - The PDDL domain is Blocksworld with standard predicates (on, on-table,
      clear, holding, arm-empty) and actions (pickup, putdown, stack, unstack).
    - The goal state is a conjunction of (on X Y), (on-table Z), and (clear W)
      predicates, defining valid, acyclic stacks or blocks on the table.
    - The state representation includes facts like '(on X Y)', '(on-table X)',
      '(clear X)', '(holding X)', '(arm-empty)'.
    - Every block in a valid state is either on another block, on the table,
      or held by the arm.

    Heuristic Initialization:
    In the `__init__` method, the heuristic processes the goal facts provided
    in the `task` object.
    - It builds a dictionary `self.goal_pos` where keys are blocks and values
      are their required support in the goal state ('table' or another block),
      based on the `(on X Y)` and `(on-table X)` goal facts.
    - It builds a set `self.goal_clear` containing blocks that must satisfy
      the `(clear X)` predicate in the goal state.

    Step-By-Step Thinking for Computing Heuristic:
    In the `__call__` method, for a given `node` representing a state:
    1. Parse the current state facts (`node.state`) to determine the current
       position of each block. This information is stored in the `current_pos`
       dictionary (mapping block to its support: 'table', another block, or 'arm').
       Also, identify which block (if any) is currently held.
    2. Initialize a memoization dictionary `correctly_stacked_status` to store
       the computed status for the `check_stacked` helper function, avoiding
       redundant calculations and handling potential cycles (though cycles
       should not occur in valid Blocksworld states).
    3. Define the recursive helper function `check_stacked(block)`:
       - This function determines if a block is in its correct goal position
         relative to a correctly built stack below it.
       - Base Case 1: If the `block` is not a key in `self.goal_pos`, it means
         its position is not explicitly defined in the goal stacks. It is
         considered "correctly stacked" for the purpose of this part of the
         heuristic. Memoize True and return True.
       - Base Case 2: If the `block`'s status is already in `correctly_stacked_status`,
         return the memoized value.
       - Get the block's current support. If the block is currently held (`current_holding == block`), its support is 'arm'. Otherwise, look it up in `current_pos`.
       - Get the block's goal support (`self.goal_pos[block]`).
       - If the current support does not match the goal support, the block is
         misplaced. Memoize False and return False.
       - If the current support matches the goal support:
         - If the goal support is 'table', the block is correctly placed on the table.
           Memoize True and return True.
         - If the goal support is another block (say `support_block`), the block
           is correctly stacked if and only if `support_block` is also correctly
           stacked. Recursively call `check_stacked(support_block)`. Memoize
           the result of the recursive call and return it.
    4. Compute the first part of the heuristic (`misplaced_stack_count`): Iterate
       through all blocks that are keys in `self.goal_pos` (i.e., blocks whose
       position within a stack is specified in the goal). For each such block,
       call `check_stacked`. Count how many return False.
    5. Compute the second part of the heuristic (`violated_clear_count`): Iterate
       through all blocks in `self.goal_clear` (i.e., blocks that must be clear
       in the goal). For each such block, check if the fact `'(clear block)'`
       is present in the current state. Count how many are *not* clear in the state.
    6. The total heuristic value is the sum of `misplaced_stack_count` and
       `violated_clear_count`. This value is returned.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the goal state.
        """
        self.goals = task.goals

        self.goal_pos = {}
        self.goal_clear = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, support = parts[1], parts[2]
                self.goal_pos[block] = support
            elif predicate == "on-table":
                block = parts[1]
                self.goal_pos[block] = 'table'
            elif predicate == "clear":
                block = parts[1]
                self.goal_clear.add(block)
            # Ignore arm-empty goal if present

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        current_pos = {}
        current_holding = None

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, support = parts[1], parts[2]
                current_pos[block] = support
            elif predicate == "on-table":
                block = parts[1]
                current_pos[block] = 'table'
            elif predicate == "holding":
                block = parts[1]
                current_holding = block
                current_pos[block] = 'arm' # Represent holding as being on the arm
            # Ignore clear and arm-empty for current_pos mapping

        # Heuristic Part 1: Misplaced blocks in goal stacks
        correctly_stacked_status = {}

        def check_stacked(block):
            """Recursively checks if a block is correctly stacked according to the goal."""
            # If block is not part of the goal stack structure, it doesn't contribute to this part of heuristic
            if block not in self.goal_pos:
                return True # Vacuously true for this part of the heuristic

            if block in correctly_stacked_status:
                return correctly_stacked_status[block]

            # Get current support. If block is held, its support is 'arm'.
            # Otherwise, look it up in current_pos. Use .get() in case a block
            # exists in goal_pos but is not currently on/on-table/held in state
            # (e.g., problem definition error or block disappeared).
            current_support = 'arm' if current_holding == block else current_pos.get(block)

            goal_support = self.goal_pos[block]

            if current_support != goal_support:
                correctly_stacked_status[block] = False
                return False

            # Current support matches goal support
            if goal_support == 'table':
                correctly_stacked_status[block] = True
                return True
            else:
                # Check if the support block is correctly stacked
                support_block = goal_support
                status_of_support = check_stacked(support_block)
                correctly_stacked_status[block] = status_of_support
                return status_of_support

        misplaced_stack_count = 0
        # Only check blocks whose position is specified in the goal stacks
        for block in self.goal_pos.keys():
             if not check_stacked(block):
                 misplaced_stack_count += 1

        # Heuristic Part 2: Violated clear conditions
        violated_clear_count = 0
        for block in self.goal_clear:
            if f'(clear {block})' not in state:
                 violated_clear_count += 1

        total_heuristic = misplaced_stack_count + violated_clear_count

        return total_heuristic
