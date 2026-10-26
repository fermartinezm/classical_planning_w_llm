from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Assuming valid PDDL fact strings like '(predicate arg1 arg2)'
         # If format is unexpected, return empty list.
         return []
    # Split by space, split() handles multiple spaces and leading/trailing spaces after stripping
    parts = fact[1:-1].split()
    return parts


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of misplaced block relationships.
    It counts, for each block relevant to the goal, whether it is on the correct
    block or table, and whether the correct block is on top of it.

    # Assumptions
    - The goal state is defined by 'on' and 'on-table' facts, and potentially 'clear' facts.
    - 'arm-empty' is not considered a primary goal fact influencing block positions.
    - The state representation is consistent (a block is either on something, on the table, or held).
    - Problems are solvable (all blocks in the goal are present in the initial state).

    # Heuristic Initialization
    - Parse the goal facts to build maps representing the desired 'below'
      relationship (what each block should be on) and the desired 'on_top'
      relationship (what block should be on top of each block).
      - `goal_below_map`: block -> block_below | 'table'
      - `goal_block_on_map`: block_below -> block_on_top | None (if block_below should be clear)
    - Identify all objects that are involved in these goal relationships.

    # Step-By-Step Thinking for Computing Heuristic
    1. For the given state, parse the facts to determine the current 'below'
       relationship for each block (what it is currently on or 'table'),
       the current 'on_top' relationship (what block is currently on top of it),
       and which block (if any) is currently held by the arm.
       - `current_below_map`: block -> block_below | 'table'
       - `current_block_on_map`: block_below -> block_on_top | None (if block_below is clear)
       - `is_holding`: block | None
    2. Initialize the heuristic cost to 0.
    3. Iterate through each block that was identified as relevant to the goal
       during initialization (`self.goal_relevant_objects`).
    4. For the current block `obj`:
        a. Retrieve its goal 'below' (`obj_goal_below`) and goal 'on_top'
           (`obj_goal_block_on`) relationships from the pre-calculated maps.
        b. Determine its current 'below' relationship (`obj_current_below`),
           handling the case where it is currently held (it has no support, effectively None).
        c. If the goal specifies what `obj` should be on (`obj_goal_below` is not None)
           AND the current 'below' relationship (`obj_current_below`) is different
           from the goal 'below' relationship, increment the heuristic cost by 1.
        d. Determine the block currently on top of `obj` (`obj_current_block_on`),
           handling the case where `obj` is currently clear or held (nothing is on top, effectively None).
        e. If the goal specifies what should be on top of `obj` (i.e., `obj` is a key in `self.goal_block_on_map`)
           AND the current 'on_top' relationship (`obj_current_block_on`) is different
           from the goal 'on_top' relationship (`obj_goal_block_on`), increment the heuristic cost by 1.
    5. The total accumulated cost is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal relationships.
        """
        self.goals = task.goals

        # Maps block -> block_below | 'table'
        self.goal_below_map = {}
        # Maps block_below -> block_on_top | None (if block_below should be clear)
        self.goal_block_on_map = {}

        # Identify goal relationships from goal facts
        blocks_mentioned_as_below_in_on_goal = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    self.goal_below_map[block_on_top] = block_below
                    self.goal_block_on_map[block_below] = block_on_top
                    blocks_mentioned_as_below_in_on_goal.add(block_below)
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block_on_table = parts[1]
                    self.goal_below_map[block_on_table] = 'table'
                    # A block on the table in the goal should be clear unless something is on it.
                    # If it's not mentioned as block_below in any 'on' goal, it should be clear.
                    # We'll handle this after parsing all 'on' facts.

        # After parsing all 'on' facts, add entries for blocks that should be clear
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == "clear":
                if len(parts) == 2:
                    block_to_be_clear = parts[1]
                    # If this block is not supposed to have anything on it in the goal
                    # (i.e., not a key in goal_block_on_map from an 'on' fact),
                    # then the goal (clear block_to_be_clear) confirms it should be clear.
                    if block_to_be_clear not in self.goal_block_on_map:
                         self.goal_block_on_map[block_to_be_clear] = None
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block_on_table = parts[1]
                    # If a block is on the table in the goal, it should be clear unless something is on it.
                    # The 'on' facts define what should be on it. If it's on the table and not a key in goal_block_on_map, it should be clear.
                    if block_on_table not in self.goal_block_on_map:
                         self.goal_block_on_map[block_on_table] = None


        # Identify all objects relevant to the goal (those involved in 'on', 'on-table', or 'clear' goals)
        # Keys of goal_below_map are blocks that are the subject of 'on' or 'on-table' goals.
        # Keys of goal_block_on_map are blocks that are the object of 'on' goals or the subject of 'clear' goals.
        self.goal_relevant_objects = set(self.goal_below_map.keys()) | set(self.goal_block_on_map.keys())


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Build current state maps
        current_below_map = {} # block -> block_below | 'table'
        current_block_on_map = {} # block_below -> block_on_top
        is_holding = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block_on_top, block_below = parts[1], parts[2]
                    current_below_map[block_on_top] = block_below
                    current_block_on_map[block_below] = block_on_top
            elif predicate == "on-table":
                 if len(parts) == 2:
                    block_on_table = parts[1]
                    current_below_map[block_on_table] = 'table'
            elif predicate == "holding":
                 if len(parts) == 2:
                    is_holding = parts[1]
            # 'clear' facts are implicitly handled by current_block_on_map.
            # If (clear X) is true, X is not a key in current_block_on_map after parsing 'on' facts.

        total_cost = 0

        # Iterate over all objects that are relevant to the goal
        for obj in self.goal_relevant_objects:
            obj_goal_below = self.goal_below_map.get(obj) # Y or 'table' or None
            obj_goal_block_on = self.goal_block_on_map.get(obj) # X or None

            # Determine current below relationship for obj
            if is_holding == obj:
                obj_current_below = None # Held block has no support
            else:
                # If obj is not held, it must be on something or on the table.
                # current_below_map will contain its support if it's on/on-table.
                # If it's not in the map and not held, it's an invalid state.
                obj_current_below = current_below_map.get(obj)


            # Determine current block on top of obj
            if is_holding == obj:
                obj_current_block_on = None # Held block has nothing on top
            else:
                # If obj is a key in current_block_on_map, something is on it. Otherwise, it's clear.
                obj_current_block_on = current_block_on_map.get(obj) # block or None


            # Condition 1: Is the block in the correct position relative to what's below it?
            # Only count if the goal specifies the support for this block.
            if obj_goal_below is not None:
                 if obj_current_below != obj_goal_below:
                     total_cost += 1

            # Condition 2: Is the correct block on top of this block?
            # Only count if the goal specifies what should be on top of this block (including nothing).
            # This is true if obj is a key in self.goal_block_on_map.
            if obj in self.goal_block_on_map:
                 if obj_current_block_on != obj_goal_block_on:
                      total_cost += 1

        return total_cost

