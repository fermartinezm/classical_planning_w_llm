# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If Heuristic base class is not provided, a simple class can be defined:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str):
        # Handle unexpected input types, maybe log a warning or raise error
        return [] # Or handle as error

    fact = fact.strip() # Remove leading/trailing whitespace
    if not fact.startswith('(') or not fact.endswith(')'):
         # Handle facts without parentheses if necessary, though PDDL facts usually have them
         # Assuming standard PDDL fact string format like "(predicate arg1 arg2)"
         return [] # Treat as invalid format

    content = fact[1:-1].strip() # Remove outer parentheses and strip whitespace
    if not content: # Handle empty fact like "()" - unlikely but safe
        return []
    return content.split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of blocks that are either:
    1. Not in their correct goal position relative to the block below them (or the table).
    2. Obstructed (not clear) but are required to be clear in the goal state.

    # Assumptions
    - The goal state specifies the desired position (on another block or on the table)
      for every block that is part of the goal configuration. Blocks not mentioned
      in goal 'on' or 'on-table' predicates are not considered for position cost.
    - The goal state specifies which blocks must be clear.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack).
    - Each block is either on another block, on the table, or held by the arm.

    # Heuristic Initialization
    - Parses the goal state to determine the desired support (block below or table)
      for each block and identifies which blocks must be clear in the goal.
    - Collects all unique objects (blocks) mentioned in the initial state and goal
      to iterate over them during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the current support (block below, table, or hand)
       for each block and identify which blocks are currently clear.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block identified during initialization:
       a. Determine the block's goal position (its goal support: the block it should be on, or 'table').
          If the block's position is not specified in the goal ('on' or 'on-table'), skip position check for this block.
       b. Determine the block's current position (its current support: the block it is on, or 'table', or 'hand' if held).
          If the block's current position cannot be determined (e.g., not on, not on-table, not held), skip position check.
       c. If the block has a defined goal position and a determined current position, and the current position is different from the goal position, increment `h`.
       d. Check if the block is required to be clear in the goal state.
       e. Check if the block is currently clear in the current state.
       f. If the block is required to be clear in the goal but is not currently clear, increment `h`.
    4. Return the total value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and objects.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Parse goal facts to determine desired positions and clear blocks
        self.goal_support = {} # Map: block -> block_below (or 'table')
        self.goal_on_table = set() # Set of blocks that should be on the table
        self.goal_clear = set() # Set of blocks that should be clear

        all_objects = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                obj, below_obj = parts[1], parts[2]
                self.goal_support[obj] = below_obj
                all_objects.add(obj)
                all_objects.add(below_obj)
            elif predicate == 'on-table' and len(parts) == 2:
                obj = parts[1]
                self.goal_on_table.add(obj)
                all_objects.add(obj)
            elif predicate == 'clear' and len(parts) == 2:
                obj = parts[1]
                self.goal_clear.add(obj)
                all_objects.add(obj)
            # Ignore (arm-empty) goal if present

        # Collect all objects from the initial state as well
        for fact in self.initial_state:
             parts = get_parts(fact)
             # Add all arguments to all_objects
             for arg in parts[1:]:
                 all_objects.add(arg)

        self.objects = frozenset(all_objects) # Store all unique objects

        # Static facts are empty in Blocksworld, so no need to process task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Parse current state facts
        current_support = {} # Map: block -> block_below
        current_on_table = set() # Set of blocks currently on the table
        current_clear = set() # Set of blocks currently clear
        current_holding = None # The block currently held, or None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                obj, below_obj = parts[1], parts[2]
                current_support[obj] = below_obj
            elif predicate == 'on-table' and len(parts) == 2:
                obj = parts[1]
                current_on_table.add(obj)
            elif predicate == 'clear' and len(parts) == 2:
                obj = parts[1]
                current_clear.add(obj)
            elif predicate == 'holding' and len(parts) == 2:
                current_holding = parts[1]
            # Ignore (arm-empty) fact if present

        h = 0

        # Iterate through all known objects
        for obj in self.objects:
            # Determine goal position
            goal_pos = None
            if obj in self.goal_support:
                goal_pos = self.goal_support[obj]
            elif obj in self.goal_on_table:
                goal_pos = 'table'
            # If goal_pos is None, this block's position isn't explicitly specified
            # in the goal. We only count position mismatch for blocks with a goal position.

            # Determine current position
            current_pos = None
            if obj in current_support:
                current_pos = current_support[obj]
            elif obj in current_on_table:
                current_pos = 'table'
            elif current_holding == obj:
                current_pos = 'hand'
            # If current_pos is None, the block's location is unknown/invalid in the state.
            # Assuming valid states, this shouldn't happen for objects in self.objects.

            # 1. Count blocks not in correct goal position relative to support
            # Only count if both goal and current positions are determined.
            if goal_pos is not None and current_pos is not None and current_pos != goal_pos:
                h += 1

            # 2. Count blocks that should be clear but are not
            if obj in self.goal_clear and obj not in current_clear:
                h += 1

        return h
