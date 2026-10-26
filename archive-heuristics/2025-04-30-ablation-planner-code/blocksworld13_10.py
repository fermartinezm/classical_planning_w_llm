from fnmatch import fnmatch
# Assuming Heuristic base class exists in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and ensure correct splitting
    return fact.strip()[1:-1].split()

# match function is not strictly needed for the chosen heuristic logic,
# but keeping it as it was in the examples might be expected.
# Let's keep it but ensure the core logic doesn't rely on it for performance.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Assume Heuristic base class is available in the environment
# Inherit from Heuristic if base class is provided
class blocksworldHeuristic:
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not in their goal position and existing block-on-block
    relations that are not part of the goal configuration. It also considers the arm state
    if arm-empty is a goal.

    # Assumptions
    - The goal specifies the desired support for each block (either another block or the table)
      using `(on X Y)` or `(on-table X)` predicates.
    - Each block has at most one desired support specified in the goal state.
    - The heuristic counts three types of discrepancies:
      1. Blocks whose current support differs from their goal support.
      2. Block-on-block relationships that exist in the current state but are not desired in the goal state.
      3. The arm holding a block when the goal requires the arm to be empty.

    # Heuristic Initialization
    - Parses the goal facts to determine the desired support for each block, storing it in `goal_support`.
    - Stores the set of desired `(on X Y)` goal facts for quick lookup.
    - Identifies all blocks mentioned in the goal whose position is specified.
    - Checks if `(arm-empty)` is a goal condition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current support for each block
       and identify all existing `(on X Y)` facts.
       - Create a dictionary `current_support` mapping each block to its current support (another block or 'table').
       - Create a set `state_on_facts` containing all `(on X Y)` facts from the state as strings.
       - Identify the block currently held by the arm, if any.

    2. Initialize the heuristic value to 0.

    3. Count "out-of-place" blocks:
       - Iterate through all blocks whose goal position is specified (`self.goal_blocks`).
       - For each block `b`, find its `goal_support[b]`.
       - Determine its current position: 'arm' if held, otherwise look up in `current_support` (defaulting to None if not found, implying not on table/block).
       - If the current position is different from the goal position, increment the heuristic value.

    4. Count "wrongly-stacked" relations:
       - Iterate through all `(on Z X)` facts present in the current state (`state_on_facts`).
       - For each state fact `(on Z X)`, check if the exact same fact `(on Z X)` is present in the set of goal facts (`self.goal_on_facts`).
       - If `(on Z X)` is in the state but NOT in the goal facts, increment the heuristic value.

    5. Consider the arm state:
       - If the goal includes `(arm-empty)` and the arm is currently holding a block (`held_block is not None`), increment the heuristic value.

    6. Return the total heuristic value.

    This heuristic counts the number of goal position mismatches, the number of existing
    stacking relationships that are incorrect, and the arm state if it conflicts with the goal.
    Each counted item represents something that needs to be changed, typically requiring
    at least one action.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support and goal on-facts.
        """
        self.goals = task.goals
        # Static facts are not needed for this heuristic in Blocksworld.
        # self.static = task.static # Commented out as static is not used

        # Map each block to its desired support in the goal state.
        self.goal_support = {}
        # Store the set of desired (on X Y) goal facts for quick lookup.
        self.goal_on_facts = set()
        # Keep track of all blocks whose position is specified in the goal
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
                self.goal_on_facts.add(goal)
                self.goal_blocks.add(block)
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_support[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore (clear ?) goals for this heuristic calculation
            # (arm-empty) is handled separately in __init__ and __call__

        # Check if (arm-empty) is a goal
        self.goal_arm_empty = "(arm-empty)" in self.goals


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Parse the current state
        current_support = {} # block -> support (block or 'table')
        state_on_facts = set() # set of '(on X Y)' strings in state
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                current_support[block] = support
                state_on_facts.add(fact)
            elif parts[0] == 'on-table':
                block = parts[1]
                current_support[block] = 'table'
            elif parts[0] == 'holding':
                held_block = parts[1]
            # Ignore clear and arm-empty for state parsing relevant to support/on_facts


        total_cost = 0

        # 3. Count "out-of-place" blocks
        # Iterate only through blocks whose goal position is specified
        for block in self.goal_blocks:
            goal_pos = self.goal_support[block] # Goal support is guaranteed to exist for blocks in goal_blocks

            # Determine current position/state
            if held_block == block:
                 current_pos = 'arm' # Special state for held block
            elif block in current_support:
                 current_pos = current_support[block]
            else:
                 # Block is a goal block but not held, not on table, not on another block.
                 # This indicates it's not in a standard location, thus not in its goal position.
                 # Treat its current position as None or some other indicator that doesn't match a valid goal_pos.
                 # Comparing to goal_pos directly handles this if goal_pos is 'table' or a block name.
                 current_pos = None # Represents not found in standard locations

            # Check if current position matches goal position
            if current_pos != goal_pos:
                 total_cost += 1


        # 4. Count "wrongly-stacked" relations
        for state_on_fact in state_on_facts:
            # Check if this state fact is NOT one of the goal on-facts
            if state_on_fact not in self.goal_on_facts:
                total_cost += 1

        # 5. Consider the arm state
        if self.goal_arm_empty and held_block is not None:
             total_cost += 1 # Need a putdown or stack action

        # If the state is the goal state, cost is 0.
        # This is a safety check. The calculation above *should* result in 0
        # if all goal conditions are met, but let's be explicit.
        is_goal_state = all(goal in state for goal in self.goals)
        if is_goal_state:
             return 0

        return total_cost
