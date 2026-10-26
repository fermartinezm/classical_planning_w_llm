from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this exists based on examples

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like "(arm-empty)" which have no arguments
    if fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return [] # Should not happen with valid PDDL facts


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct position relative to the block directly below them in the goal stack.
    A block is considered "correctly stacked" if it is on the correct block
    (or the table) according to the goal, AND the block below it is also
    correctly stacked. The heuristic counts the number of blocks that fail
    this recursive condition.

    # Assumptions
    - The goal specifies the desired position (on another block or on the table)
      for every block involved in the goal configuration.
    - The heuristic focuses only on the 'on' and 'on-table' goal conditions,
      assuming that 'clear' and 'arm-empty' conditions will be met naturally
      when the block stacking is correct.
    - Goal stacks are linear (at most one block directly on top of another).

    # Heuristic Initialization
    - Parses the goal facts to determine the desired block stacking configuration:
      - `self.goal_pos`: A dictionary mapping each block to the block it should
        be directly on top of in the goal, or 'table' if it should be on the table.
      - `self.goal_above`: A dictionary mapping each block (or 'table') to the
        block that should be directly on top of it in the goal. Assumes at most
        one block on top in the goal configuration.
    - Collects all unique objects from the initial state and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current position of each block from the state:
       - Is it on the table? `(on-table B)`
       - Is it on another block? `(on B C)`
       - Is it being held? `holding(B)`
    2. Initialize a status tracker `correctly_stacked_status` for each block to None.
    3. Initialize the heuristic value `h = 0`.
    4. Identify the base blocks in the goal configuration: those that should be
       on the table according to `self.goal_pos`. Add these to a queue for processing.
    5. Process blocks layer by layer based on the goal stacks using a queue:
       - Dequeue a block `B`.
       - If its status is already determined, skip.
       - Determine if `B` is correctly stacked relative to its goal base (`goal_p = self.goal_pos[B]`):
         - If `goal_p` is 'table': `B` is correctly stacked if `(on-table B)` is true in the current state.
         - If `goal_p` is a block `C`: `B` is correctly stacked if `C` is already determined to be correctly stacked (i.e., `correctly_stacked_status[C]` is True) AND `(on B C)` is true in the current state. (The queue processing order ensures `C`'s status is known).
       - Set `correctly_stacked_status[B]` based on this determination.
       - If `B` is *not* correctly stacked, increment `h`.
       - If there is a block `D` that should be directly on top of `B` in the goal
         (according to `self.goal_above`), add `D` to the queue if not already added.
    6. Continue processing until the queue is empty.
    7. The final value of `h` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and objects.
        """
        # Call parent constructor to get goals and static facts
        super().__init__(task)

        self.goal_pos = {} # block -> base_block or 'table'
        self.goal_above = {} # base_block or 'table' -> block_on_top (assuming linear stacks in goal)

        # Parse goal facts to build goal configuration maps
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty facts like (arm-empty) if they are goals

            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1], parts[2]
                self.goal_pos[obj] = underob
                self.goal_above[underob] = obj # Assume only one block on top in goal
            elif predicate == "on-table":
                obj = parts[1]
                self.goal_pos[obj] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for this heuristic

        # Collect all objects from initial state and goals
        self.objects = set()
        for fact in task.initial_state:
             self.objects.update(get_parts(fact)[1:]) # Add all arguments
        for goal in task.goals:
             self.objects.update(get_parts(goal)[1:]) # Add all arguments
        self.objects = list(self.objects) # Convert to list for consistent iteration


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Build current configuration maps
        current_pos = {} # block -> base_block or 'table' or 'holding'
        # current_above is not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1], parts[2]
                current_pos[obj] = underob
            elif predicate == "on-table":
                obj = parts[1]
                current_pos[obj] = 'table'
            elif predicate == "holding":
                obj = parts[1]
                current_pos[obj] = 'holding'

        # Initialize status and heuristic
        correctly_stacked_status = {obj: None for obj in self.objects}
        h = 0

        # Queue for layered processing, starting with blocks whose goal base is 'table'
        q = deque()
        added_to_q = set()

        for obj in self.objects:
            # Only consider blocks that are part of the goal configuration (have a goal_pos)
            if obj in self.goal_pos and self.goal_pos[obj] == 'table':
                 q.append(obj)
                 added_to_q.add(obj)

        # Process blocks layer by layer
        while q:
            block = q.popleft()

            # If status is already determined, skip
            if correctly_stacked_status.get(block) is not None:
                continue

            # Blocks in the queue must have a goal_pos defined
            goal_p = self.goal_pos[block]

            # Determine if the block is correctly stacked relative to its goal base
            is_correct = False
            if goal_p == 'table':
                # Base case: Block should be on the table
                is_correct = (current_pos.get(block) == 'table')
            else: # goal_p is a block (C)
                # Recursive case: Block should be on C.
                # It's correctly stacked only if C is correctly stacked AND block is currently on C.
                # We assume C's status is already determined because C is below block in the goal stack.
                # If C's status is None, there's an issue with goal_pos or processing order.
                base_block_status = correctly_stacked_status.get(goal_p)
                if base_block_status is True:
                     is_correct = (current_pos.get(block) == goal_p)
                # If base_block_status is False or None, is_correct remains False.

            correctly_stacked_status[block] = is_correct

            # If the block is NOT correctly stacked, increment heuristic
            if not is_correct:
                h += 1

            # Add the block that should be on top of this block in the goal to the queue
            # This ensures the next layer of the stack is processed.
            block_on_top_in_goal = self.goal_above.get(block)
            if block_on_top_in_goal is not None and block_on_top_in_goal not in added_to_q:
                 q.append(block_on_top_in_goal)
                 added_to_q.add(block_on_top_in_goal)

        # The heuristic is the count of blocks that are not correctly stacked
        return h
