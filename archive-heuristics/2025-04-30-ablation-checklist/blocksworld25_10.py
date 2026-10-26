from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    state by counting the number of blocks that are not in their correct
    position relative to the block directly below them (or the table) as
    defined by the goal state. Each such misplaced block is estimated to
    require 2 actions (one to pick it up/unstack, one to put it down/stack).

    # Assumptions:
    - The goal specifies the desired position for some blocks, either on
      another block `(on X Y)` or on the table `(on-table X)`.
    - Blocks not explicitly mentioned in an `(on ...)` or `(on-table ...)`
      goal predicate are assumed to have a goal position of being on the table.
    - The heuristic counts how many blocks are NOT on their correct base
      (the block or table directly below them according to the goal).
    - Each block not on its correct base is estimated to cost 2 actions
      (pickup/unstack + putdown/stack). This is a non-admissible estimate
      as it doesn't fully account for clearing blocks above or below, but
      aims to be informative.

    # Heuristic Initialization
    - Parses the goal state to determine the desired position (block or table)
      for each block mentioned in an `(on ...)` or `(on-table ...)` goal.
    - Identifies all blocks present in the initial state or mentioned in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the goal position for each block:
        - If `(on X Y)` is a goal, the goal position for X is Y.
        - If `(on-table X)` is a goal, the goal position for X is 'table'.
        - For blocks not mentioned in `(on ...)` or `(on-table ...)` goals,
          assume the goal position is 'table'.
    2.  Identify all blocks involved in the problem (those in the initial state
        or mentioned in the goal).
    3.  For the current state, determine the current position for each block:
        - If `(on X Y)` is true, the current position for X is Y.
        - If `(on-table X)` is true, the current position for X is 'table'.
        - If `(holding X)` is true, the current position for X is 'arm' (or some indicator it's not on a base).
        - If a block is not found in `on`, `on-table`, or `holding` predicates,
          this indicates an inconsistency, but we can default its current position
          to 'unknown' or handle it based on context (e.g., assume 'table' if not holding).
          A safer approach is to build the map from state facts. If a block isn't
          in `on` or `on-table` and isn't `holding`, it's likely an error in state
          representation or it implies it's on the table if arm-empty. Let's stick
          to parsing `on` and `on-table` and default to 'table' if not found,
          as `holding` is transient.
    4.  Compare the current position and goal position for each block.
    5.  Count the number of blocks where the current position is different from
        the goal position.
    6.  Multiply this count by 2, as a rough estimate of the actions needed
        (pickup/unstack + putdown/stack) for each block that is not on its
        correct base.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and identifying all blocks.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Map block -> goal_position (block below or 'table')
        self.goal_pos = {}
        # Set of all blocks involved in the problem
        self.all_blocks = set()

        # Parse goals to find desired positions and identify blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, below = parts[1], parts[2]
                    self.goal_pos[block] = below
                    self.all_blocks.add(block)
                    self.all_blocks.add(below)
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_pos[block] = 'table'
                    self.all_blocks.add(block)
            # Ignore 'clear' and 'arm-empty' goals for this heuristic's core logic

        # Add blocks from the initial state that might not be in the goals
        for fact in self.initial_state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate in ["on", "on-table", "holding", "clear"]:
                 # These predicates involve a block as the first argument
                 if len(parts) > 1:
                     self.all_blocks.add(parts[1])
                 # For 'on', the second argument is also a block
                 if predicate == "on" and len(parts) > 2:
                     self.all_blocks.add(parts[2])


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Map block -> current_position (block below or 'table' or 'arm')
        current_pos = {}
        # Track blocks currently being held
        holding_block = None

        # Parse current state to find positions
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, below = parts[1], parts[2]
                    current_pos[block] = below
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block = parts[1]
                    current_pos[block] = 'table'
            elif predicate == "holding":
                 if len(parts) == 2:
                    holding_block = parts[1]
                    current_pos[holding_block] = 'arm' # Represent block in arm

        misplaced_count = 0

        # Iterate through all known blocks and compare current vs goal position
        for block in self.all_blocks:
            # Get current base: default to 'table' if not found in 'on', 'on-table', or 'holding'
            # This handles blocks that might be implicitly on the table if not on another block or held.
            current_below = current_pos.get(block, 'table')

            # Get goal base: default to 'table' if the block is not in goal_pos
            goal_below = self.goal_pos.get(block, 'table')

            # If the current base is different from the goal base, the block is misplaced
            if current_below != goal_below:
                misplaced_count += 1

        # The heuristic is the number of misplaced blocks multiplied by 2
        # (representing pickup/unstack + putdown/stack).
        # This is a non-admissible estimate but captures the core work of moving blocks.
        return misplaced_count * 2

