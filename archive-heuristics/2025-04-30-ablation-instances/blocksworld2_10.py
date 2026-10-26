from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assuming fact is a string like '(predicate arg1 arg2)'
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle unexpected format, though PDDL facts as strings should fit
        # print(f"Warning: Unexpected fact format: {fact}") # Optional warning
        return [] # Return empty list for malformed facts

    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting:
    1. Blocks that are not in their correct goal location (on another block or on the table).
    2. Blocks that are currently on top of another block, where that block should either be clear or have a different block on top in the goal state.

    # Assumptions
    - Each action (pickup, putdown, stack, unstack) has a cost of 1.
    - The goal specifies the desired final configuration of blocks (which block is on which, or on the table). Blocks not mentioned in goal 'on' or 'on-table' facts are assumed to have a goal of being on the table.
    - The heuristic is non-admissible and designed to guide a greedy best-first search.

    # Heuristic Initialization
    - Parse the goal facts to determine the desired support (the block underneath or 'table') for each block that is explicitly mentioned in an 'on' or 'on-table' goal fact.
    - Identify all blocks present in the problem instance from the initial state.
    - Implicitly assume blocks not explicitly placed in the goal should be on the table.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state:
       - Determine the current support for each block (what it's on, or 'table').
       - Identify which block (if any) is currently being held by the arm.
       - Build a mapping of which block is immediately on top of which other block.

    2. Initialize the heuristic value to 0.

    3. Add penalty for misplaced blocks:
       - For each block in the problem:
         - Determine its goal support (from initialization, defaulting to 'table' if not explicitly in goal).
         - If the block is currently being held, it is not in its goal on/on-table position. Increment the heuristic value.
         - If the block is not being held, check its current support. If the current support is different from its goal support, increment the heuristic value.

    4. Add penalty for blocking blocks:
       - Iterate through the mapping of blocks currently on top of other blocks (e.g., for each pair (Under, Top) where Top is on Under).
       - Determine which block (if any) is supposed to be on the 'Under' block in the goal state. This is found by searching the goal positions for a block X where X's goal support is 'Under'.
       - If no block is supposed to be on 'Under' in the goal (i.e., 'Under' should be clear according to the goal structure), or if a different block is supposed to be on 'Under' in the goal, then 'Top' is blocking 'Under' from achieving its goal configuration. Increment the heuristic value.

    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and identifying all blocks.
        """
        self.goals = task.goals  # Goal conditions.
        # static_facts = task.static # Blocksworld has no static facts relevant here.

        # Map blocks to their goal support (the block underneath or 'table').
        # Blocks not explicitly mentioned in goal 'on' or 'on-table' facts
        # are implicitly assumed to have a goal of being on the table.
        self.goal_positions = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if len(parts) >= 3 and parts[0] == 'on':
                block, under = parts[1], parts[2]
                self.goal_positions[block] = under
            elif len(parts) >= 2 and parts[0] == 'on-table':
                block = parts[1]
                self.goal_positions[block] = 'table'
            # Ignore 'clear' or other potential goal facts for this heuristic's position/blocking logic

        # Identify all blocks in the problem instance from the initial state.
        self.all_blocks = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) > 1:
                 # The first argument is usually a block
                 self.all_blocks.add(parts[1])
             if len(parts) > 2 and parts[0] == 'on':
                 # The second argument of 'on' is also a block
                 self.all_blocks.add(parts[2])

        # Ensure all blocks mentioned in the goal are also in self.all_blocks
        # (This handles cases where a block might only appear in the goal, though unlikely in standard Blocksworld)
        self.all_blocks.update(self.goal_positions.keys())
        self.all_blocks.update(self.goal_positions.values())
        self.all_blocks.discard('table') # 'table' is not a block


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse the current state
        current_positions = {} # block -> support (block or 'table')
        current_block_on_top = {} # block_below -> block_on_top (immediate)
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            if len(parts) >= 3 and parts[0] == 'on':
                block, under = parts[1], parts[2]
                current_positions[block] = under
                current_block_on_top[under] = block
            elif len(parts) >= 2 and parts[0] == 'on-table':
                block = parts[1]
                current_positions[block] = 'table'
            elif len(parts) >= 2 and parts[0] == 'holding':
                current_holding = parts[1]
            # Ignore 'clear' or 'arm-empty' for this heuristic's calculation

        heuristic_value = 0

        # Penalty for misplaced blocks (location)
        for block in self.all_blocks:
            # Get the goal position, defaulting to 'table' if not specified in goal facts
            goal_pos = self.goal_positions.get(block, 'table')

            if current_holding == block:
                 # If the block is being held, it's not in its goal on/on-table position
                 heuristic_value += 1
            else:
                # Find the block's current support.
                current_pos = current_positions.get(block)

                # If a block is not holding and not in current_positions, it implies
                # it's the base of a stack that isn't fully described in the state facts
                # (e.g., only the top block and 'clear' are listed).
                # However, standard Blocksworld state representation lists all 'on' and 'on-table' facts.
                # So, current_pos should not be None for non-held blocks.
                if current_pos is not None and current_pos != goal_pos:
                    heuristic_value += 1

        # Penalty for blocking blocks (wrong block on top)
        # Iterate through all blocks that have something on top in the current state
        for under_block, top_block in current_block_on_top.items():
             # Determine which block (if any) is supposed to be on under_block in the goal state.
             correct_top_block = None
             # Find the block X such that goal_positions[X] == under_block
             for b, under in self.goal_positions.items():
                 if under == under_block:
                     correct_top_block = b
                     break # Assuming only one block can be on another in the goal

             # Check if the current top_block is the correct one according to the goal
             if correct_top_block is None:
                 # Nothing is supposed to be on under_block in the goal (it should be clear relative to goal stacks).
                 # Since top_block is on it, top_block is blocking under_block.
                 heuristic_value += 1
             elif correct_top_block != top_block:
                 # A different block (correct_top_block) is supposed to be on under_block.
                 # So, top_block is blocking under_block.
                 heuristic_value += 1

        # The heuristic is 0 if and only if all goal 'on'/'on-table' facts are met AND
        # for every block B that is currently on top of another block C, C is supposed to have B on it in the goal.
        # This means the current stack configuration matches the goal stack configuration exactly.
        # If the stack configuration matches, the 'clear' facts for the top blocks of goal stacks will also be true.
        # The 'arm-empty' fact is not directly counted but is a consequence of the final state.
        # Thus, h=0 iff the state is a goal state.

        return heuristic_value
