from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or non-string input gracefully, although facts are expected to be strings.
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting the number of
    misplaced blocks and blocks that are obstructing the placement of other blocks.
    Specifically, it counts:
    1. Blocks that are not in their correct immediate goal location (on the table or on the correct block below them).
    2. Blocks that are currently being held by the arm (as they are not in their final resting place).
    3. Blocks that are placed on top of other blocks where that 'on' relationship is not part of the goal state (these blocks are obstructing).

    # Assumptions
    - The goal state specifies the desired position for all blocks, either on the table or on another specific block.
    - The goal state implies a set of desired stacks.
    - The goal state is reachable.
    - Standard Blocksworld goals where achieving the correct stacks and having the arm empty implies the goal is reached (including clear predicates for stack tops).

    # Heuristic Initialization
    - Parse the goal conditions to determine the desired block-on-block relationships (`goal_stack_below`) and blocks that should be on the table (`goal_on_table`).
    - Store the set of all desired `(on X Y)` facts (`goal_on_facts`).
    - Identify all blocks present in the problem from the task's initial facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state to determine the immediate location of each block (on-table, on another block, or held). Create a mapping `current_location`. Also, identify all current `(on C B)` facts.
    2. Initialize a cost counter to 0.
    3. Iterate through all blocks identified in the problem:
       - Determine the block's current location using the `current_location` map.
       - Determine the block's desired goal location (on-table or on a specific block) based on the parsed goal structure. If a block is not explicitly mentioned in the goal 'on' or 'on-table' predicates, assume its goal is to be on the table.
       - If the block is currently being held, increment the cost by 1 (it's not in its final place).
       - If the block is not held, compare its current location (on-table or on block C) with its goal location (on-table or on block U):
         - If the current location does not match the goal location, increment the cost by 1.
    4. Iterate through all `(on C B)` facts present in the current state:
       - If the fact `(on C B)` is *not* one of the desired `(on X Y)` facts in the goal state, it means block C is wrongly placed on block B. This block C must be moved to allow block B (or blocks below B) to be moved or cleared. Increment the cost by 1.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal structure and all objects.
        """
        self.goals = task.goals

        # Parse goal predicates to build the desired stack structure
        self.goal_stack_below = {}  # block -> block_it_should_be_on
        self.goal_on_table = set()  # set of blocks that should be on the table
        self.goal_on_facts = set()  # set of goal '(on X Y)' fact strings

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'on':
                if len(parts) == 3:
                    block_above, block_below = parts[1], parts[2]
                    self.goal_stack_below[block_above] = block_below
                    self.goal_on_facts.add(goal)
            elif parts and parts[0] == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_on_table.add(block)
            # Ignore (clear ?x) and (arm-empty) goals for structural analysis here,
            # they are implicitly handled by penalizing blocks in wrong places or held.

        # Extract all objects from task facts and goals
        self.all_objects = set()
        # Extract from initial state facts
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) > 1:
                 # Assuming arguments after predicate are objects
                 for obj in parts[1:]:
                     # Simple check to avoid adding predicate names or 'table' if it appears as an object
                     # In blocksworld, objects are typically b1, b2, etc.
                     # A more robust parser would distinguish objects by type or definition.
                     # For this domain, arguments after the predicate are objects.
                     self.all_objects.add(obj)

        # Also add objects mentioned in goals, in case they are not in the initial state
        for block in self.goal_stack_below.keys():
             self.all_objects.add(block)
        for block in self.goal_stack_below.values():
             self.all_objects.add(block)
        for block in self.goal_on_table:
             self.all_objects.add(block)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        cost = 0

        # Map current locations and identify current 'on' facts
        current_location = {} # block -> 'table' or block_below or 'held'
        current_on_facts = set() # set of current '(on C B)' fact strings

        # Populate current_location and current_on_facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_location[block] = 'table'
            elif predicate == 'on' and len(parts) == 3:
                block_above_name, block_below_name = parts[1], parts[2]
                current_location[block_above_name] = block_below_name
                current_on_facts.add(fact)
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                current_location[block] = 'held'

        # Heuristic Part 1 & 2: Blocks in wrong immediate location or held
        for block in self.all_objects:
            current_loc = current_location.get(block, None)

            # Determine goal location for this block
            goal_loc = self.goal_stack_below.get(block, None)
            if goal_loc is None and block in self.goal_on_table:
                 goal_loc = 'table'

            # If a block is not in the goal structure, assume its goal is on the table.
            # This handles partial goals gracefully.
            if goal_loc is None:
                 goal_loc = 'table'

            if current_loc == 'held':
                cost += 1 # Penalize holding any block
            elif current_loc is not None and current_loc != goal_loc:
                 cost += 1 # Block is on table/another block, but the wrong one

        # Heuristic Part 3: Blocks wrongly placed on top
        for fact in current_on_facts:
            if fact not in self.goal_on_facts:
                cost += 1 # This 'on' relationship is not desired in the goal

        # The heuristic is 0 if and only if:
        # 1. No blocks are held (cost from Part 2 is 0).
        # 2. All blocks are in their correct immediate goal location (cost from Part 1 is 0).
        # 3. All current 'on' facts are goal 'on' facts (cost from Part 3 is 0).
        # These conditions together imply that the blocks form the exact stacks specified in the goal,
        # and the arm is empty. This is the definition of the goal state in standard Blocksworld.
        # Therefore, the heuristic is 0 iff the state is a goal state.

        return cost

