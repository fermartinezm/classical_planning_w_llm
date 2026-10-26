from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting
    the blocks that are not on their correct goal base and the 'on' relationships
    in the current state that are not part of the goal state.

    # Assumptions
    - The goal state consists primarily of one or more stacks of blocks on the table,
      defined by 'on' and 'on-table' predicates.
    - The heuristic counts two types of "misplacements":
        1. A block whose required base is specified in the goal is not currently
           on that base (or table).
        2. A block is currently on top of another block, but this specific 'on'
           relationship is not part of the goal configuration.

    # Heuristic Initialization
    - The heuristic pre-processes the goal state to determine the required base
      for each block that appears as the first argument of a goal 'on' predicate
      or the argument of a goal 'on-table' predicate. This mapping is stored in
      `self.goal_base`. Blocks that are the base of a goal stack are mapped to 'table'.
    - The set of goal facts is stored for quick lookup of goal 'on' relationships.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the immediate base for every block
       (the block it is on, 'table', or 'arm' if held). Store this in `current_base`.
    2. Collect all current `(on A B)` facts as strings in a set.
    3. Initialize the heuristic value `h = 0`.
    4. Count blocks with incorrect bases: Iterate through each block `b` that is a key
       in `self.goal_base` (i.e., a block whose required base is specified in the goal).
       Look up its required base in `self.goal_base`. Look up its current base in
       `current_base` (defaulting to None if the block is not currently on anything
       or held). If the current base is different from the required goal base,
       increment `h`.
    5. Count incorrect 'on' relationships: Iterate through each `(on A B)` fact string
       currently true in the state (collected in step 2). Check if this exact fact string
       is present in the set of goal facts (`self.goals`). If it is not, increment `h`.
    6. The total heuristic value is the sum of the counts from steps 4 and 5.
    7. As a final check to ensure the heuristic is 0 only at the goal, if the current
       state is a superset of the goal facts, return 0. Otherwise, return the calculated `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals  # Store goal facts for quick lookup

        # Map blocks to their required base in the goal state
        self.goal_base = {}

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_base[block] = base
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_base[block] = 'table'

        # self.goal_base now contains entries for all blocks whose base is explicitly
        # defined in the goal state (either on another block or on the table).

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining cost to reach the goal.
        """
        state = node.state

        # Check if the goal is reached first for h=0 guarantee
        if self.goals.issubset(state):
             return 0

        # Map blocks to their current base ('table', another block, or 'arm')
        current_base = {}
        # Collect all current 'on' facts
        current_on_facts = set()
        # Track if arm is holding something (not strictly needed for this heuristic logic, but good practice)
        # holding_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, base = parts[1], parts[2]
                current_base[block] = base
                current_on_facts.add(fact)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_base[block] = 'table'
            # elif predicate == 'holding' and len(parts) == 2:
            #     holding_block = parts[1]
            #     current_base[holding_block] = 'arm' # Represent holding state

        h = 0

        # Count blocks that are not on their correct goal base
        # Iterate through blocks whose goal base is specified
        for block, target_base in self.goal_base.items():
            current_base_of_block = current_base.get(block) # Use .get for blocks not currently on anything/table

            if current_base_of_block != target_base:
                h += 1

        # Count 'on' relationships in the current state that are not in the goal
        for on_fact in current_on_facts:
            if on_fact not in self.goals:
                h += 1

        return h
