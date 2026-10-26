from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and multiple spaces
    return fact.strip()[1:-1].split()

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

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by summing two components:
    1. The number of blocks that are not on their correct support (either another block or the table)
       as specified in the goal.
    2. The number of blocks that are not clear in the current state (excluding the block, if any, held by the arm).

    This heuristic is non-admissible but aims to guide the search effectively
    by penalizing blocks that are in the wrong place and blocks that are
    blocking others.

    # Assumptions
    - The goal state specifies a unique support (either another block or the table)
      for every block involved in an 'on' or 'on-table' goal predicate.
    - All blocks relevant to the problem are mentioned in the initial state or goal state facts.

    # Heuristic Initialization
    - Parse the goal conditions to determine the desired support for each block
      and identify all relevant blocks mentioned in the goal.
    - Parse the initial state conditions (if available in the task object) to identify
      all relevant blocks mentioned there.
    - Store the goal support mapping (block -> support_object or 'table').
    - Store the set of all relevant blocks.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is the goal state. If yes, return 0.
    2. Initialize heuristic value to 0.
    3. Parse the current state facts to determine:
       - The current support for each block (on another block, on the table, or held).
       - Which blocks are currently clear.
       - Which block, if any, is currently held by the arm.
    4. Iterate through all relevant blocks identified during initialization:
       a. Component 1 (Misplaced Blocks):
          - Get the block's goal support from the pre-calculated map.
          - If the block has a goal support, compare it with the block's current support.
          - If the current support is different from the goal support, increment the heuristic value.
       b. Component 2 (Uncleared Blocks):
          - Check if the block is clear in the current state.
          - If the block is NOT clear AND it is NOT currently held by the arm, increment the heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying objects.
        """
        self.goals = task.goals  # Goal conditions as a frozenset of strings.

        # Extract goal support for each block
        self.goal_supports = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == "on":
                block, under = parts[1], parts[2]
                self.goal_supports[block] = under
            elif predicate == "on-table":
                block = parts[1]
                self.goal_supports[block] = 'table'
            # Ignore 'clear' and 'arm-empty' in goal for support calculation

        # Identify all relevant blocks from initial state and goal state facts.
        self.all_blocks = set()

        # Collect objects from initial state facts (assuming task.init is available)
        if hasattr(task, 'init'):
             for fact in task.init:
                parts = get_parts(fact)
                # Predicates involving objects: on, on-table, clear, holding
                if parts[0] in ["on", "on-table", "clear", "holding"]:
                    for obj in parts[1:]:
                         self.all_blocks.add(obj)

        # Collect objects from goal state facts
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] in ["on", "on-table", "clear", "holding"]:
                 for obj in parts[1:]:
                      self.all_blocks.add(obj)

        # Ensure 'table' is not considered a block object
        if 'table' in self.all_blocks:
             self.all_blocks.remove('table')


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state  # Current world state as a frozenset of strings.

        # Check if it's the goal state first for efficiency and correctness (h=0 iff goal)
        # This relies on the search algorithm correctly identifying goal states,
        # but explicitly returning 0 here ensures the heuristic value is correct at the goal.
        is_goal_state = all(goal_fact in state for goal_fact in self.goals)
        if is_goal_state:
             return 0

        # Determine current support and clear status for each block
        current_supports = {}
        current_clear = set()
        currently_held = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, under = parts[1], parts[2]
                current_supports[block] = under
            elif predicate == "on-table":
                block = parts[1]
                current_supports[block] = 'table'
            elif predicate == "holding":
                block = parts[1]
                current_supports[block] = 'arm'
                currently_held = block
            elif predicate == "clear":
                block = parts[1]
                current_clear.add(block)
            # Ignore 'arm-empty'

        heuristic_value = 0

        # Component 1: Blocks not on correct support
        for block in self.all_blocks:
            goal_support = self.goal_supports.get(block)
            if goal_support is not None: # Only consider blocks with a goal support
                current_support = current_supports.get(block)
                # If a block is not found in current_supports, it's likely an error or
                # the block is only in the goal but not in the initial state (unlikely in BW).
                # We assume blocks are always on_table, on_block, or held.
                # If a block is in all_blocks but not on, on-table, or held, something is wrong.
                # Assuming valid states, current_supports.get(block) will be 'table', a block name, or 'arm'.
                if current_support != goal_support:
                     heuristic_value += 1

        # Component 2: Blocks that are not clear (and not held)
        for block in self.all_blocks:
             if block not in current_clear and block != currently_held:
                  heuristic_value += 1

        return heuristic_value
