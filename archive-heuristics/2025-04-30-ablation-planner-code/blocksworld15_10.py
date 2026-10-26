# from heuristics.heuristic_base import Heuristic # Uncomment this line in the actual environment

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

# Assume Heuristic base class is defined elsewhere and imported as above
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError


class blocksworldHeuristic: # Assuming Heuristic base class is imported
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting the number
    of blocks that are not in their correct position relative to the block or
    the table they should be on according to the goal state.

    # Assumptions
    - The goal state is defined by a set of (on ?x ?y) and (on-table ?x) predicates,
      potentially also (clear ?x) for the top blocks of goal stacks.
    - Blocks can be on other blocks, on the table, or held by the arm.
    - The state representation is complete regarding block positions (each block
      is either on another block, on the table, or held).

    # Heuristic Initialization
    - Parses the goal predicates to determine the desired position (on which block
      or on the table) for each block that is part of a goal stack. This information
      is stored in the `goal_pos` dictionary. Blocks not explicitly placed in goal
      (on) or (on-table) predicates are ignored by the heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current position (on which block,
       on the table, or held by the arm) for every block that is currently placed
       or held. Store this in the `current_pos` dictionary.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block `block` that is explicitly placed in the goal
       (i.e., appears as the first argument in a goal (on) predicate or as the
       argument in a goal (on-table) predicate).
    4. For each such `block`, determine its desired position `goal_base` from
       the pre-calculated `goal_pos` dictionary.
    5. Determine the block's current position `current_base` from the
       `current_pos` dictionary. If the block is not found in `current_pos`,
       it means the block is not currently on anything, on the table, or held.
       In a valid Blocksworld state, this shouldned happen for existing blocks.
       Treat this case as the block being misplaced.
    6. Compare `current_base` and `goal_base`. If they are not the same,
       increment the heuristic value `h`.
    7. The final value of `h` is the heuristic estimate.

    This heuristic is 0 if and only if all blocks that are explicitly placed
    in goal stacks are in their correct position relative to the block or table
    below them as specified in the goal. This implies the goal (on) and (on-table)
    predicates are satisfied. For standard Blocksworld goals, this also implies
    the goal (clear) predicates for the top blocks are satisfied.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for blocks.
        """
        # super().__init__(task) # Call superclass constructor if inheriting
        self.goals = task.goals
        # self.static = task.static # Static facts are not used

        # Store goal positions: goal_pos[block] = block_below or 'table'
        self.goal_pos = {}

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, block_below = parts[1], parts[2]
                self.goal_pos[block] = block_below
            elif predicate == "on-table":
                block = parts[1]
                self.goal_pos[block] = 'table'
            # Ignore 'clear' goals for this heuristic calculation

        # The set of blocks whose position is explicitly defined in the goal
        self.goal_blocks_to_place = set(self.goal_pos.keys())


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Determine current positions of all blocks that are placed or held
        current_pos = {}
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, block_below = parts[1], parts[2]
                current_pos[block] = block_below
            elif predicate == "on-table":
                block = parts[1]
                current_pos[block] = 'table'
            elif predicate == "holding":
                block = parts[1]
                current_pos[block] = 'arm'
            # 'clear' and 'arm-empty' do not define a block's base position

        heuristic_value = 0

        # Count blocks that are not in their goal position relative to their base
        for block in self.goal_blocks_to_place:
            goal_base = self.goal_pos[block] # Get the desired base from goal

            # Get the current base. If the block is not found in current_pos,
            # it implies the block is not currently on anything, on the table,
            # or held. This should not happen in a valid Blocksworld state for
            # any block that exists in the problem. Treat as misplaced.
            current_base = current_pos.get(block)

            # If the block's current position doesn't match the goal position,
            # or if the block's position is not defined in the current state facts
            # (which shouldn't happen for existing blocks), increment heuristic.
            if current_base is None or current_base != goal_base:
                 heuristic_value += 1

        return heuristic_value
