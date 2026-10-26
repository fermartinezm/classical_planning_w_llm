# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """
    Parses a PDDL fact string into its predicate and objects.
    e.g., '(on b1 b2)' -> ('on', ['b1', 'b2'])
    e.g., '(on-table b1)' -> ('on-table', ['b1'])
    e.g., '(arm-empty)' -> ('arm-empty', [])
    """
    # Remove surrounding parentheses and split by whitespace
    parts = fact_str.strip("()").split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

# Helper function to build the configuration map (block -> block_below_it or 'table' or 'arm')
def build_config_refined(facts):
    """
    Builds a dictionary mapping each block to the object directly below it,
    'table' if it's on the table, or 'arm' if it's being held.
    """
    config = {}
    holding_block = None
    for fact_str in facts:
        predicate, objects = parse_fact(fact_str)
        if predicate == 'on':
            above, below = objects
            config[above] = below
        elif predicate == 'on-table':
            block = objects[0]
            config[block] = 'table'
        elif predicate == 'holding':
            holding_block = objects[0]
            config[holding_block] = 'arm' # Represent block being held
    return config, holding_block

# Helper function to build the clear status map (block -> True/False)
def build_clear_status(facts, all_objects):
    """
    Builds a dictionary mapping each block to its clear status (True if clear, False otherwise).
    Requires the set of all possible objects in the domain.
    """
    # Initialize all objects as not clear by default
    clear_status = {obj: False for obj in all_objects}
    # Update status based on explicit clear facts
    for fact_str in facts:
        predicate, objects = parse_fact(fact_str)
        if predicate == 'clear':
            block = objects[0]
            clear_status[block] = True
    return clear_status

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by counting
        the number of blocks that are not in their correct final position
        relative to the block below them (considering the entire stack below),
        adding penalties for the arm holding a block and for incorrectly
        stacked blocks that are not clear (as they need to be cleared before
        being moved or stacked upon).

    Assumptions:
        - The input state and task objects follow the structure described
          in the problem description (frozenset of fact strings).
        - The PDDL domain uses standard blocksworld predicates:
          (on ?x ?y), (on-table ?x), (clear ?x), (arm-empty), (holding ?x).
        - The goal is specified using (on ?x ?y) and (on-table ?x) facts.
        - The goal does not require the arm to be holding a block.
        - The set of all objects is consistent across initial state, goal, and static facts.

    Heuristic Initialization:
        The constructor pre-parses the goal facts from the task object to build
        the `goal_base_map`. This map stores, for each block that is part of
        a goal stack or is required to be on the table in the goal, the object
        it should be directly on top of, or the string 'table'. This
        information is static for a given task and is computed only once.
        It also identifies all objects present in the initial state, goal,
        and static facts for use in determining clear status.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Parse the current state facts to determine the current configuration
            of blocks (`current_config_refined`), which block is being held
            (`current_holding_block`), and the clear status of each block
            (`current_clear_status`). The set of all objects identified during
            initialization is used to ensure all blocks have a clear status entry.
        2.  Define a recursive helper function `is_correctly_stacked_recursive`
            that checks if a given block and the entire stack below it match
            the goal configuration stored in `self.goal_base_map`. This function
            uses memoization to avoid redundant calculations. A block is
            considered correctly stacked if its current base matches its goal
            base AND the block below it (if any) is also correctly stacked.
            Blocks not explicitly present as keys in `self.goal_base_map`
            (i.e., not part of the specified goal stacks) are considered
            correct for the purpose of the recursive check, as their specific
            base is not a goal requirement.
        3.  Initialize the heuristic value `h` to 0.
        4.  Iterate through each block that is required to be in a specific
            position relative to a base in the goal state (i.e., blocks that
            are keys in `self.goal_base_map`). If the `is_correctly_stacked_recursive`
            function returns False for a block, increment `h`. This counts
            blocks that are not in their correct final stack configuration.
        5.  If the arm is currently holding a block (`current_holding_block` is
            not None), increment `h` by 1. This penalizes the state of holding
            a block, as it typically requires an action to resolve.
        6.  Identify the set of blocks that are part of the goal configuration
            (keys in `self.goal_base_map`) but were found to be incorrectly
            stacked in step 4.
        7.  Iterate through this set of incorrectly stacked goal blocks. For
            each block, check its clear status in the current state using
            `current_clear_status`. If the block is NOT clear, increment `h`
            by 1. This penalizes blocks that are "in the way" of achieving the
            goal stacks and need to be cleared first.
        8.  Return the final calculated value of `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-parsing goal facts and collecting all objects.

        @param task: The planning task object.
        """
        self.goal_base_map = {} # block -> block_below_it or 'table'
        self.all_objects = set()

        # Parse goal facts to build goal_base_map and collect objects
        for fact_str in task.goals:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'on':
                above, below = objects
                self.goal_base_map[above] = below
                self.all_objects.update(objects)
            elif predicate == 'on-table':
                block = objects[0]
                self.goal_base_map[block] = 'table'
                self.all_objects.update(objects)

        # Collect all objects from initial state
        for fact_str in task.initial_state:
             predicate, objects = parse_fact(fact_str)
             self.all_objects.update(objects)

        # Collect all objects from static facts
        for fact_str in task.static:
             predicate, objects = parse_fact(fact_str)
             self.all_objects.update(objects)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The heuristic value (integer).
        """
        # 1. Parse current state facts
        current_config_refined, current_holding_block = build_config_refined(state)
        current_clear_status = build_clear_status(state, self.all_objects)

        # 2. Define recursive helper with memoization
        memo = {}
        def is_correctly_stacked_recursive(block):
            if block in memo:
                return memo[block]

            goal_base = self.goal_base_map.get(block)

            # If the block is not part of the goal stack structure we care about,
            # it's considered "correct" relative to this goal structure.
            if goal_base is None:
                memo[block] = True
                return True

            current_base = current_config_refined.get(block)

            # If the block is not found in the current config (e.g., not on anything, not held - should not happen for objects in state)
            # or if the current base doesn't match the goal base, it's not correctly stacked here.
            if current_base != goal_base:
                 memo[block] = False
                 return False

            # Case 1: Goal is on-table and current is on-table
            if goal_base == 'table':
                memo[block] = True
                return True

            # Case 2: Goal is on another block and current is on the same block
            # Check if the block below is also correctly stacked
            result = is_correctly_stacked_recursive(goal_base)
            memo[block] = result
            return result

        # 3. Initialize heuristic value
        h_value = 0

        # 4. Add penalty for blocks in goal_base_map.keys() that are not correctly stacked
        incorrectly_stacked_goal_blocks = set()
        for block in self.goal_base_map.keys():
            if not is_correctly_stacked_recursive(block):
                h_value += 1
                incorrectly_stacked_goal_blocks.add(block)

        # 5. Add penalty if arm is holding a block
        if current_holding_block:
            h_value += 1

        # 6. Add penalty for incorrectly stacked blocks that are not clear
        for block in incorrectly_stacked_goal_blocks:
            # Check if the block is NOT clear in the current state
            # Use .get(block, False) in case the block is not in all_objects (shouldn't happen if all_objects is built correctly)
            if not current_clear_status.get(block, False):
                 h_value += 1

        return h_value
