from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic counts the number of blocks that are part of the goal
    configuration but are not currently in their correct position relative
    to the block below them AND do not have the correct block on top of them.
    This provides an estimate of the number of blocks that are 'out of place'
    according to the goal structure.

    # Heuristic Initialization
    - Parses the goal facts to determine the desired position (on which block
      or on the table) and the desired block on top for every block mentioned
      in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal state to build mappings:
       - `goal_pos[block]`: The block it should be on, or 'table'.
       - `goal_above[block]`: The block that should be on top of it, or None if it should be clear.
       - `goal_objects`: The set of all blocks involved in the goal.
    2. Parse the current state to build mappings:
       - `current_pos[block]`: The block it is currently on, or 'table'.
       - `current_above[block]`: The block currently on top of it, or None if clear.
       - `holding`: Set of blocks currently held by the arm.
    3. Initialize heuristic value to 0.
    4. For each block `obj` in `goal_objects`:
       - Check if `obj` is "correctly placed" in the current state.
       - A block `obj` is correctly placed if:
         a) It is not currently being held (`obj not in holding`).
         b) Its current position matches its goal position (`current_pos.get(obj) == goal_pos.get(obj)`).
         c) The block currently on top of it matches the block that should be on top (`current_above.get(obj) == goal_above.get(obj)`).
       - If the block is NOT correctly placed, increment the heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration information.
        """
        self.goals = task.goals

        self.goal_pos = {}    # Maps block -> block_below or 'table'
        self.goal_above = {}  # Maps block_below -> block_above or None (if clear)
        self.goal_objects = set() # Set of all objects mentioned in goal facts

        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            predicate = parts[0]
            if predicate == 'on':
                obj, underob = parts[1], parts[2]
                self.goal_pos[obj] = underob
                self.goal_above[underob] = obj
                self.goal_objects.add(obj)
                self.goal_objects.add(underob)
            elif predicate == 'on-table':
                obj = parts[1]
                self.goal_pos[obj] = 'table'
                self.goal_objects.add(obj)
            elif predicate == 'clear':
                obj = parts[1]
                self.goal_above[obj] = None # Explicitly mark as should be clear
                self.goal_objects.add(obj)
            # Ignore arm-empty if it's a goal, as the heuristic focuses on block positions.
            # If arm-empty was critical and needed to be included, we could add 1 if arm not empty.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        current_pos = {}    # Maps block -> block_below or 'table'
        current_above = {}  # Maps block_below -> block_above or None (if clear)
        holding = set()     # Set of blocks currently held

        for fact_str in state:
            parts = get_parts(fact_str)
            predicate = parts[0]
            if predicate == 'on':
                obj, underob = parts[1], parts[2]
                current_pos[obj] = underob
                current_above[underob] = obj
            elif predicate == 'on-table':
                obj = parts[1]
                current_pos[obj] = 'table'
            elif predicate == 'holding':
                obj = parts[1]
                holding.add(obj)
            # 'clear' facts in state are implicitly captured by absence in current_above
            # 'arm-empty' fact is not needed for this heuristic calculation

        heuristic_value = 0

        # Iterate through all blocks that are part of the goal configuration
        for obj in self.goal_objects:
            is_holding = obj in holding
            current_pos_val = current_pos.get(obj)
            # Use .get() for goal_pos as well, just to be consistent and safe,
            # although standard Blocksworld goals should ensure obj is in goal_pos.
            goal_pos_val = self.goal_pos.get(obj)
            current_above_val = current_above.get(obj) # None if clear
            goal_above_val = self.goal_above.get(obj) # None if should be clear

            # Check if the block is correctly placed
            is_correctly_placed = True
            if is_holding:
                is_correctly_placed = False
            elif current_pos_val != goal_pos_val:
                 is_correctly_placed = False
            elif current_above_val != goal_above_val:
                 is_correctly_placed = False

            if not is_correctly_placed:
                heuristic_value += 1

        return heuristic_value
