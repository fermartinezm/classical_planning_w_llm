from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like "(arm-empty)" which have no arguments
    fact_str = fact.strip()
    if not fact_str or len(fact_str) < 2 or fact_str[0] != '(' or fact_str[-1] != ')':
         # Handle unexpected fact format gracefully
         # print(f"Warning: Unexpected fact format: {fact}") # Optional warning
         return []
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to compare parts and args up to the length of the shorter sequence
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by summing three components:
    1. The number of blocks that are not in the correct position relative to the
       block or table immediately below them, considering the goal stack structure
       from the bottom up.
    2. The number of blocks that are required to be clear in the goal but are not
       clear in the current state.
    3. A penalty if the arm is required to be empty in the goal but is not empty
       in the current state.

    # Assumptions
    - The goal specifies the desired arrangement of blocks using `on` and `on-table`
      predicates, and potentially `clear` and `arm-empty`.
    - Blocks not mentioned in goal `on` or `on-table` facts are not considered
      part of a specific goal stack structure for the first component of the heuristic.
    - The heuristic components are additive and represent independent subgoals.

    # Heuristic Initialization
    - Parses the goal conditions to identify:
        - `goal_below`: A dictionary mapping each block to the object (another block or 'table')
          it should be immediately on top of in the goal state.
        - `all_goal_blocks`: A set of all blocks that appear in goal `on` or `on-table` facts.
        - `goal_clear`: A set of blocks that must be clear in the goal state.
        - `goal_arm_empty`: A boolean indicating if the arm must be empty in the goal state.
    - Static facts are not used by this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a given state is calculated as follows:

    1.  **Calculate the 'incorrect stack' component:**
        - Identify the current position of each block (on another block or on the table)
          by parsing the state's `on` and `on-table` facts. A block held by the arm
          is considered not on anything.
        - Determine which blocks are "correctly placed" relative to the block/table
          below them *and* are part of a correctly built stack segment from the bottom up,
          according to the goal configuration. This is done recursively or iteratively:
          A block X is correctly placed if its required base (Y or table) matches its
          actual base in the state, AND if the required base Y is itself correctly placed.
          The base case is a block correctly placed on the table.
        - Count the total number of blocks in `all_goal_blocks`.
        - Count the number of blocks identified as "correctly placed".
        - The 'incorrect stack' component is the total number of goal blocks minus the
          number of correctly placed goal blocks.

    2.  **Calculate the 'unsatisfied clear' component:**
        - Identify which blocks are currently clear in the state by parsing `clear` facts.
        - Count how many blocks in `goal_clear` are *not* in the set of currently clear blocks.

    3.  **Calculate the 'unsatisfied arm-empty' component:**
        - Check if `goal_arm_empty` is True.
        - Check if `(arm-empty)` is present in the current state facts.
        - If `goal_arm_empty` is True and `(arm-empty)` is not in the state, add 1 to the heuristic. Otherwise, add 0.

    4.  **Sum the components:** The total heuristic value is the sum of the 'incorrect stack',
        'unsatisfied clear', and 'unsatisfied arm-empty' components.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        super().__init__(task)

        self.goal_below = {}
        self.all_goal_blocks = set()
        self.goal_clear = set()
        self.goal_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            args = parts[1:]

            if predicate == "on" and len(args) == 2:
                block, below_object = args
                self.goal_below[block] = below_object
                self.all_goal_blocks.add(block)

            elif predicate == "on-table" and len(args) == 1:
                block = args[0]
                self.goal_below[block] = 'table' # Use a special string 'table'
                self.all_goal_blocks.add(block)

            elif predicate == "clear" and len(args) == 1:
                block = args[0]
                self.goal_clear.add(block)

            elif predicate == "arm-empty" and len(args) == 0:
                self.goal_arm_empty = True


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_below = {}
        current_clear = set()
        arm_is_empty = False
        holding_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "on" and len(args) == 2:
                block, below_object = args
                current_below[block] = below_object
            elif predicate == "on-table" and len(args) == 1:
                block = args[0]
                current_below[block] = 'table'
            elif predicate == "clear" and len(args) == 1:
                block = args[0]
                current_clear.add(block)
            elif predicate == "arm-empty" and len(args) == 0:
                arm_is_empty = True
            elif predicate == "holding" and len(args) == 1:
                 holding_block = args[0]
                 # A held block is not on anything, so it won't be in current_below


        # --- Component 1: Incorrect Stack Structure ---
        correctly_placed_set = set()
        # Use memoization for the recursive check
        is_correctly_placed_cache = {}

        def is_correctly_placed(block):
            """
            Recursively check if a block is correctly placed in the goal stack
            segment starting from the table.
            """
            if block in is_correctly_placed_cache:
                return is_correctly_placed_cache[block]

            # If the block is not one that needs to be in a specific goal stack position,
            # it doesn't contribute to the count of correctly placed *goal* blocks.
            # This case should ideally not be reached for blocks in self.all_goal_blocks.
            if block not in self.goal_below:
                 is_correctly_placed_cache[block] = False
                 return False

            required_below = self.goal_below[block]
            actual_below = current_below.get(block)

            # If the block is held, it's not on the required object/table
            if holding_block == block:
                 is_correctly_placed_cache[block] = False
                 return False

            # If the block is not held, check its actual position
            if actual_below is None or actual_below != required_below:
                is_correctly_placed_cache[block] = False
                return False

            # Base case: correctly placed on the table
            if required_below == 'table':
                is_correctly_placed_cache[block] = True
                correctly_placed_set.add(block) # Add to set if it's the base of a correct stack segment
                return True

            # Recursive step: check the block below
            below_block = required_below # required_below is a block name here
            if is_correctly_placed(below_block):
                is_correctly_placed_cache[block] = True
                correctly_placed_set.add(block) # Add to set if it's on a correctly placed block
                return True
            else:
                is_correctly_placed_cache[block] = False
                return False

        # Populate correctly_placed_set by checking all goal blocks
        for block in self.all_goal_blocks:
             is_correctly_placed(block) # Call recursive function to populate cache and set

        incorrect_stack_count = len(self.all_goal_blocks) - len(correctly_placed_set)

        # --- Component 2: Unsatisfied Clear Conditions ---
        unsatisfied_clear_count = 0
        for block in self.goal_clear:
            if block not in current_clear:
                unsatisfied_clear_count += 1

        # --- Component 3: Unsatisfied Arm-Empty Condition ---
        unsatisfied_arm_empty_count = 0
        if self.goal_arm_empty and not arm_is_empty:
            unsatisfied_arm_empty_count = 1

        # Total heuristic is the sum of components
        total_cost = incorrect_stack_count + unsatisfied_clear_count + unsatisfied_arm_empty_count

        return total_cost
