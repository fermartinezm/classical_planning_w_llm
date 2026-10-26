# Add necessary imports at the top
from collections import deque

# Helper functions (defined outside the class)
def parse_fact(fact_string):
    """Helper function to parse a PDDL fact string."""
    # Remove leading '(' and trailing ')'
    cleaned_string = fact_string[1:-1]
    # Split by spaces
    parts = cleaned_string.split()
    return parts

def get_objects_from_facts(facts):
    """Helper function to extract all object names from a set of facts."""
    objects = set()
    for fact_string in facts:
        parts = parse_fact(fact_string)
        # Skip predicate name (parts[0])
        for obj in parts[1:]:
            objects.add(obj)
    return objects

# The heuristic class
class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of blocks that are not currently in their correct
    position within the goal stack structure. A block is considered "correctly
    stacked" if it is on its required goal support (another block or the table)
    AND if its goal support is itself correctly stacked (recursively), AND the
    block that should be on top of it in the goal is indeed on top of it
    in the current state (or it should be clear and is clear). The heuristic
    value is the total number of blocks whose final position is specified in
    the goal minus the number of blocks that are correctly stacked. To ensure
    the heuristic is zero only in goal states, the value is capped at a minimum
    of 1 for any non-goal state.

    Assumptions:
    - The input task is a valid Blocksworld task.
    - Goal facts primarily consist of (on ?x ?y) and (on-table ?x) predicates
      defining the desired block configuration. (clear ?x) goals are handled
      implicitly by checking the block on top. (arm-empty) and (holding ?x)
      goals are ignored for the core block configuration count but contribute
      to the non-zero heuristic value if the state is not the full goal.
    - Every block present in the initial state or goal facts that is relevant
      to the goal configuration appears as the first argument in exactly one
      (on ?x ?y) or (on-table ?x) goal fact. Blocks not mentioned as the first
      argument in such a goal fact are not considered for the core heuristic count.

    Heuristic Initialization:
    The heuristic is initialized with the planning task. It stores the set of
    goal facts for the goal state check. It precomputes the goal configuration
    by parsing the goal facts:
    - Identifies all objects involved in the problem from initial and goal states.
    - Creates a mapping `goal_support` from each block to its required support
      (another block or the string 'table') based on (on ?x ?y) and (on-table ?x)
      goal facts.
    - Creates a mapping `goal_on_top` from each block to the block that should
      be directly on top of it in the goal state, or None if the block should
      be clear.
    - Identifies the set of `goal_blocks`, which are the blocks whose final
      position is specified in the goal facts (i.e., blocks appearing as the
      first argument in an (on ?x ?y) or (on-table ?x) goal fact).
    Static facts are accepted but not used in this Blocksworld heuristic.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the current state is the actual goal state (i.e., if all goal
       facts are true in the state). If yes, return 0.
    2. Parse the current state facts to determine the current configuration:
       - Create a mapping `current_support` from each block to its current
         support (another block, 'table', or 'arm' if held). Initialize to 'unknown'.
       - Create a mapping `current_on_top` from each block to the block
         currently directly on top of it, or None if clear. Initialize to None.
    3. Identify blocks from `goal_blocks` that are "correctly stacked" according
       to the goal configuration and the current state. This is done iteratively,
       starting from the blocks that should be on the table in the goal:
       - Initialize an empty set `correctly_stacked_blocks`.
       - Initialize a queue with all blocks from `goal_blocks` that should
         be on the table in the goal (`goal_support.get(block) == 'table'`).
       - Use a set `queued_for_check` to track blocks added to the queue to avoid
         redundancy.
       - While the queue is not empty:
         - Dequeue a `block`.
         - Check if this `block` meets the conditions to be correctly stacked:
           a. Its current support (`current_support.get(block)`) must match its
              goal support (`goal_support.get(block)`).
           b. If its goal support is another block `Y` (`goal_support.get(block) != 'table'`),
              then `Y` must already be in the `correctly_stacked_blocks` set.
           c. The block currently on top of `block` (`current_on_top.get(block)`)
              must match the block that should be on top of it in the goal
              (`goal_on_top.get(block)`).
         - If all applicable conditions (a, and optionally b, and c) are met,
           add `block` to `correctly_stacked_blocks`.
         - Find the block `next_block_up` from `goal_blocks` such that `next_block_up`
           should be directly on top of `block` in the goal (`goal_support.get(next_block_up) == block`).
           If such a block exists and is not already in `correctly_stacked_blocks`
           or `queued_for_check`, enqueue `next_block_up` and add it to `queued_for_check`.
    4. Calculate a base heuristic value as the total number of blocks in
       `goal_blocks` minus the number of blocks in `correctly_stacked_blocks`.
    5. Return the maximum of 1 and the base heuristic value. This ensures the
       heuristic is positive for any non-goal state, satisfying the requirement
       that the heuristic is zero only for goal states.
    """
    def __init__(self, task, static):
        """
        Initializes the heuristic with goal information.

        Args:
            task: The planning task object.
            static: Set of static facts (not used in this heuristic).
        """
        self.task_goals_set = task.goals # Store the original goal set

        # Collect all objects from initial and goal states
        self.all_objects = get_objects_from_facts(task.initial_state | self.task_goals_set)

        # Precompute goal structure
        self.goal_support = {}
        self.goal_on_top = {obj: None for obj in self.all_objects}
        self.goal_blocks = set() # Blocks whose final position is specified

        for fact_string in self.task_goals_set:
            parts = parse_fact(fact_string)
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    obj1, obj2 = parts[1], parts[2]
                    self.goal_support[obj1] = obj2
                    self.goal_on_top[obj2] = obj1
                    self.goal_blocks.add(obj1)
            elif predicate == 'on-table':
                if len(parts) == 2:
                    obj = parts[1]
                    self.goal_support[obj] = 'table'
                    self.goal_blocks.add(obj)
            # Ignore other goal predicates like 'clear', 'arm-empty', 'holding'
            # for the core stack structure definition.

        # Defensive check: Ensure all objects mentioned as supports in goal_support
        # are in all_objects and have a goal_on_top entry.
        for support in list(self.goal_support.values()): # Iterate over a copy
             if support != 'table' and support not in self.all_objects:
                 # This case indicates an object is a support in a goal fact
                 # but wasn't found in initial or goal facts as an object itself.
                 # This shouldn't happen in valid PDDL but handle defensively.
                 self.all_objects.add(support)
                 self.goal_on_top[support] = None # Initialize goal_on_top


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The heuristic value (integer).
        """
        # 1. Check if the state is the actual goal state
        if self.task_goals_set.issubset(state):
            return 0

        # 2. Parse current state facts
        current_support = {obj: 'unknown' for obj in self.all_objects}
        current_on_top = {obj: None for obj in self.all_objects}

        for fact_string in state:
            parts = parse_fact(fact_string)
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3: # Ensure correct arity
                    obj1, obj2 = parts[1], parts[2]
                    if obj1 in self.all_objects and obj2 in self.all_objects:
                        current_support[obj1] = obj2
                        current_on_top[obj2] = obj1
            elif predicate == 'on-table':
                if len(parts) == 2: # Ensure correct arity
                    obj = parts[1]
                    if obj in self.all_objects:
                        current_support[obj] = 'table'
            elif predicate == 'holding':
                 if len(parts) == 2: # Ensure correct arity
                    obj = parts[1]
                    if obj in self.all_objects:
                        current_support[obj] = 'arm' # Block is held by the arm

            # 'clear' facts are implicitly captured by current_on_top being None

        # 3. Identify correctly stacked blocks iteratively
        correctly_stacked_blocks = set()
        q = deque()

        # Start with blocks from goal_blocks that should be on the table in the goal
        for block in self.goal_blocks:
            if self.goal_support.get(block) == 'table':
                 q.append(block)

        # Keep track of blocks already added to queue to avoid redundant checks
        queued_for_check = set(q)

        while q:
            block = q.popleft()

            # Check conditions for block to be correctly stacked
            is_correct = True

            # a. Check current support matches goal support
            if current_support.get(block) != self.goal_support.get(block):
                is_correct = False

            # b. If goal support is a block, check if it's correctly stacked
            if is_correct: # Only check if (a) passed
                support = self.goal_support.get(block)
                if support != 'table': # If the block should be on another block
                    # Check if the block it should be on is correctly stacked
                    if support not in correctly_stacked_blocks:
                        is_correct = False

            # c. Check block on top matches goal on top
            if is_correct: # Only check if (a) and (b) passed
                 if current_on_top.get(block) != self.goal_on_top.get(block):
                     is_correct = False

            if is_correct:
                correctly_stacked_blocks.add(block)
                # Find the block that should be on top of this one in the goal
                # and add it to the queue if it's a goal block and not already queued/processed
                for next_block_up in self.goal_blocks:
                    if self.goal_support.get(next_block_up) == block:
                        if next_block_up not in correctly_stacked_blocks and next_block_up not in queued_for_check:
                             q.append(next_block_up)
                             queued_for_check.add(next_block_up)


        # 4. Calculate base heuristic value
        # Count goal blocks that were NOT found to be correctly stacked
        base_heuristic = len(self.goal_blocks) - len(correctly_stacked_blocks)

        # 5. Ensure heuristic is > 0 for non-goal states.
        # This handles cases where the block configuration is correct but other
        # goal facts (like arm-empty or clear) are not met).
        return max(1, base_heuristic)
