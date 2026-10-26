import re
from collections import deque

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by counting
        the number of blocks that are not "correctly stacked". A block is
        considered correctly stacked if it is on its designated goal support
        (either the table or another block) AND the block it is supported by
        (if any) is also correctly stacked. This effectively counts the number
        of blocks that are not part of a correctly built tower segment starting
        from the table up according to the goal configuration.

    Assumptions:
        - The input state is a frozenset of ground fact strings.
        - The state representation for any given block includes exactly one
          fact specifying its location: either '(on ?x ?y)', '(on-table ?x)',
          or '(holding ?x)'.
        - The goal state is provided as a frozenset of ground fact strings,
          primarily specifying '(on ?x ?y)' and '(on-table ?x)' relationships.
          It is assumed that the goal facts define the required position
          (on another block or on the table) for all blocks that are relevant
          to the goal configuration. Blocks not mentioned as the first argument
          of an '(on ?x ?y)' fact or the argument of an '(on-table ?x)' fact
          in the goal are not considered in the heuristic count.
        - The heuristic does not explicitly count the cost of achieving
          '(clear ?x)' or '(arm-empty)' goals, as these are often
          intermediate states or derived properties that are implicitly
          handled by achieving the correct block placements.

    Heuristic Initialization:
        The heuristic is initialized with the planning task. During
        initialization, it extracts:
        1.  The set of all objects (blocks) present in the task by parsing
            all possible ground facts provided in `task.facts`. This set is
            used to ensure we consider all relevant blocks when determining
            current positions.
        2.  The desired goal support for each block by parsing the goal facts
            (`task.goals`). For each block `B` that appears as the first
            argument of an '(on B A)' goal fact or the argument of an
            '(on-table B)' goal fact, its goal support is recorded (block `A`
            or the string 'table'). Blocks not explicitly given a support in
            the goal facts are not included in the `goal_support` mapping and
            do not contribute to the heuristic count.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1.  Check if the state is the goal state using `self.task.goal_reached(state)`.
            If yes, the heuristic is 0.
        2.  Determine the current support for each block in the given state.
            This involves iterating through the state facts and identifying
            '(on ?x ?y)', '(on-table ?x)', and '(holding ?x)' facts to build
            a mapping from each block to its current support ('table', another
            block, or 'arm' if held). Blocks not found in these facts are
            implicitly assumed to be on the table if the state is complete,
            but the implementation relies on the assumption that state facts
            cover all blocks' locations.
        3.  Initialize a boolean flag `is_correctly_stacked` for each block
            (from the set of all objects) to `False`.
        4.  Identify the base cases for "correctly stacked": Any block `B`
            whose goal support is 'table' (i.e., `B` is a key in `goal_support`
            and `goal_support[B] == 'table'`) AND whose current support is also
            'table' (`current_support.get(B) == 'table'`) is correctly stacked.
            Set their `is_correctly_stacked` flag to `True` and add them to a
            queue for propagation.
        5.  Propagate the "correctly stacked" status: While the queue is not
            empty, dequeue a block `A`. This block `A` is known to be correctly
            stacked. Find all blocks `B` whose goal support is `A` (i.e., `B`
            is a key in `goal_support` and `goal_support[B] == A`). For each
            such block `B`, if its `is_correctly_stacked[B]` flag is currently
            `False` AND its current support is `A` (`current_support.get(B) == A`),
            then `B` is now correctly stacked on a correctly stacked block.
            Set `is_correctly_stacked[B]` to `True` and enqueue `B`.
        6.  After propagation finishes, count the number of blocks `B` that are
            included in the `goal_support` mapping (i.e., their goal position
            is explicitly defined) for which `is_correctly_stacked[B]` is still
            `False`. This count represents the number of blocks whose goal
            position is defined but are not part of a correctly built tower
            segment from the bottom up.
        7.  Return this count as the heuristic value.
    """

    def __init__(self, task):
        self.task = task
        self.goal_support = {}
        self.all_objects = set()

        # 1. Extract all objects from task.facts
        # task.facts contains all possible ground facts, e.g., '(on b1 b2)', '(on-table b1)', '(clear b1)', '(holding b1)', '(arm-empty)'
        for fact_str in task.facts:
            # Remove leading/trailing parens and split by space
            parts = fact_str[1:-1].split()
            # The first part is the predicate, the rest are arguments (objects)
            if len(parts) > 1: # Ignore facts like '(arm-empty)'
                for obj in parts[1:]:
                    self.all_objects.add(obj)

        # 2. Build goal_support mapping from task.goals
        # We only care about 'on' and 'on-table' facts in the goal
        for goal_fact_str in task.goals:
            parts = goal_fact_str[1:-1].split()
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for this heuristic calculation

        # Note: Blocks in self.all_objects that are NOT keys in self.goal_support
        # are not explicitly placed in the goal configuration. They do not
        # contribute to the heuristic count based on this definition.

    def __call__(self, state):
        # 1. Check if goal reached
        if self.task.goal_reached(state):
            return 0

        # 2. Determine current support for each block
        current_support = {}
        # Initialize all objects to have no known support yet
        # This handles cases where a block might not be mentioned in on/on-table/holding
        # facts in a state (though this shouldn't happen in valid states).
        # Let's rely on the assumption that state is complete.
        for fact_str in state:
            parts = fact_str[1:-1].split()
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_support[block] = support
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'arm' # Block is held

        # 3. Compute is_correctly_stacked
        # Initialize for all objects found in the task
        is_correctly_stacked = {obj: False for obj in self.all_objects}
        q = deque()

        # Base cases: Blocks correctly on the table in the goal
        # Iterate through blocks whose goal position is defined
        for block, goal_sup in self.goal_support.items():
             if goal_sup == 'table':
                 # Check if the block is currently on the table
                 if current_support.get(block) == 'table':
                     is_correctly_stacked[block] = True
                     q.append(block)

        # Propagation: Blocks correctly stacked on correctly stacked blocks
        while q:
            supported_block = q.popleft() # This block is correctly stacked

            # Find blocks that should be on top of supported_block in the goal
            # Iterate through blocks whose goal position is defined
            blocks_to_check = [
                block for block, goal_sup in self.goal_support.items()
                if goal_sup == supported_block
            ]

            for block in blocks_to_check:
                # If block is not already marked correctly stacked
                # AND its current support is the supported_block
                if not is_correctly_stacked[block] and current_support.get(block) == supported_block:
                    is_correctly_stacked[block] = True
                    q.append(block)

        # 4. Calculate heuristic: Count blocks that are NOT correctly stacked
        #    Consider only blocks whose goal position is defined (i.e., are keys in goal_support).
        h_value = 0
        for block in self.goal_support.keys():
             if not is_correctly_stacked[block]:
                 h_value += 1

        return h_value
