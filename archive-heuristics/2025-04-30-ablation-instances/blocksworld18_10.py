from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This case should ideally not happen with valid PDDL fact strings
        # based on the problem description's state representation example.
        # Return empty list or handle gracefully based on expected input.
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing three components:
    1. The number of blocks that are part of the goal configuration but are not
       in their correct final position relative to the goal stack structure.
    2. The number of blocks that are currently stacked on top of another block
       in a way that is not required by the goal configuration.
    3. A penalty if the robot arm is currently holding a block.

    # Assumptions
    - The goal state defines a specific configuration of blocks, typically
      stacks on the table.
    - Blocks not explicitly mentioned as the first argument of an 'on' or
      'on-table' goal predicate do not have a specific goal position for themselves,
      but might need to be moved if they are blocking other blocks.
    - The cost of each action (pickup, putdown, stack, unstack) is 1.

    # Heuristic Initialization
    - The heuristic parses the goal conditions to identify:
        - `goal_pos_map`: A mapping from each block (that is the first argument
          of a goal predicate) to its target support (another block or 'table').
        - `goal_on_table`: The set of blocks that should be on the table in the goal.
        - `goal_on_block`: The set of (block, support) pairs that should have
          an 'on' relation in the goal.
        - `goal_blocks`: The set of all blocks that are the first argument of
          any goal predicate.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine:
       - `current_pos_map`: A mapping from each block to its current support
         ('table' or another block).
       - `current_on_pairs`: The set of (block, support) pairs that currently
         have an 'on' relation.
       - `current_holding`: The block currently held by the arm, or None.
    2. Calculate the set of `in_place_set`: Blocks that are in their correct
       goal position relative to their support, AND their support is also
       recursively in its correct goal position. This is done iteratively:
       - Initialize `in_place_set` with blocks from `goal_on_table` that are
         currently `on-table`.
       - Repeatedly add blocks `B` to `in_place_set` if their goal is `(on B X)`,
         they are currently `(on B X)`, and `X` is already in `in_place_set`,
         until no new blocks can be added.
    3. Calculate `h_in_place_mismatch`: Count the number of blocks in `goal_blocks`
       that are NOT in `in_place_set`. These blocks are not in their final
       recursive goal position.
    4. Calculate `h_wrong_on`: Count the number of `(block, support)` pairs in
       `current_on_pairs` that are NOT present in `goal_on_block`. These are
       blocks currently stacked incorrectly according to the goal.
    5. Calculate `h_holding`: Add 1 to the heuristic if `current_holding` is not None,
       representing the cost to free the arm.
    6. The total heuristic value is the sum of `h_in_place_mismatch`, `h_wrong_on`,
       and `h_holding`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration information.
        """
        self.goals = task.goals

        # Parse goal facts to build goal configuration maps
        self.goal_pos_map = {}
        self.goal_on_table = set()
        self.goal_on_block = set()
        self.goal_blocks = set() # Blocks that are the first argument of a goal predicate

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    self.goal_pos_map[block] = support
                    self.goal_on_block.add((block, support))
                    self.goal_blocks.add(block)
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_pos_map[block] = 'table'
                    self.goal_on_table.add(block)
                    self.goal_blocks.add(block)
            # Ignore other goal predicates like (clear ?) or (arm-empty) as they are state-dependent effects

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Parse current state
        current_pos_map = {} # block -> support ('table' or block)
        current_on_pairs = set() # (block, support) pairs
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    current_pos_map[block] = support
                    current_on_pairs.add((block, support))
            elif predicate == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    current_pos_map[block] = 'table'
            elif predicate == "holding":
                 if len(parts) == 2:
                    current_holding = parts[1]
            # Ignore 'clear' and 'arm-empty' for this heuristic calculation

        # Calculate in_place_set (recursively correct goal position)
        # A block B is in_place if:
        # 1. Goal is (on-table B) AND state has (on-table B).
        # 2. Goal is (on B X) AND state has (on B X) AND X is in_place.

        in_place_set = set()

        # Initialize with blocks satisfying condition 1
        for block in self.goal_on_table:
            if current_pos_map.get(block) == 'table':
                in_place_set.add(block)

        # Iteratively add blocks satisfying condition 2
        newly_added = True
        while newly_added:
            newly_added = False
            # Iterate through blocks that have a goal position on another block
            for block, goal_support in self.goal_pos_map.items():
                if goal_support != 'table' and block not in in_place_set:
                    # Check if the block is currently on its goal support
                    current_support = current_pos_map.get(block)
                    if current_support == goal_support:
                        # Check if the goal support is in_place
                        if goal_support in in_place_set:
                            in_place_set.add(block)
                            newly_added = True

        # --- Calculate Heuristic Components ---

        # Component 1: Blocks in goal_blocks not in their recursively correct goal position
        # These are blocks that are part of the desired stack structure but are not
        # in the right place relative to the base of the stack.
        h_in_place_mismatch = sum(1 for block in self.goal_blocks if block not in in_place_set)

        # Component 2: 'on' relations in the current state that are NOT required in the goal state
        # These are blocks that are stacked on top of others in a way that obstructs
        # achieving the goal configuration.
        h_wrong_on = sum(1 for pair in current_on_pairs if pair not in self.goal_on_block)

        # Component 3: Arm holding a block
        # If the arm is holding a block, at least one action (putdown or stack) is needed
        # to free the arm and potentially place the block.
        h_holding = 1 if current_holding else 0

        # Total heuristic value is the sum of the components.
        total_cost = h_in_place_mismatch + h_wrong_on + h_holding

        return total_cost
