import re

class blocksworldHeuristic:
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the distance to the goal by counting the number
    of blocks that are not part of a correctly built stack segment from the
    bottom up, according to the goal configuration. A block is considered
    correctly stacked relative to below if it is on the table in the current
    state and should be on the table in the goal, OR if it is on the correct
    block below it in the current state and that block is itself correctly
    stacked relative to below according to the goal.

    Assumptions:
    - The goal state consists primarily of (on ?x ?y) and (on-table ?x) facts,
      defining desired stacks. (clear ?x) facts in the goal are implicitly
      handled as achieving the correct stack structure usually results in the
      correct blocks being clear.
    - The goal state defines a set of stacks where each block has a unique
      block below it or is on the table.
    - The input state and goal facts are strings in the format '(predicate arg1 ...)'
      and object names do not contain spaces or parentheses.
    - The heuristic does not need to be admissible, but aims to be informative
      for greedy best-first search.

    Heuristic Initialization:
    1. Parses the goal facts from the task to identify all objects involved
       in the goal and their desired positions relative to the block below
       them or the table.
    2. Stores this desired configuration in a dictionary `self.goal_on`,
       mapping a block to the block it should be on, or to the string 'table'.
    3. Collects all unique objects mentioned in the goal facts into `self.all_objects`.

    Step-By-Step Thinking for Computing Heuristic:
    1. For a given state, the heuristic calculates how many blocks are *not*
       in their correct position relative to the block below them in the goal
       stack, considering dependencies (i.e., a block is only correctly placed
       on another block if that block is also correctly placed relative to its
       goal position below).
    2. A recursive helper function `_is_correctly_stacked(block, state, goal_mapping, memo_dict)`
       is used to determine if a block is part of a correctly built goal stack
       segment from the bottom up.
    3. The base case for the recursion is when a block should be on the table
       in the goal; it is correctly stacked if it is on the table in the current state.
    4. The recursive step is when a block should be on another block `y` in the goal;
       it is correctly stacked if it is on `y` in the current state AND `y` is
       correctly stacked relative to its goal position below.
    5. Memoization (`memo_dict`) is used within the recursive function to store
       results for blocks that have already been checked, preventing redundant
       computation and handling potential (though unlikely in valid goals) cycles.
    6. The heuristic iterates through all objects that are keys in the `self.goal_on`
       mapping (i.e., blocks whose position relative to below is specified in the goal).
    7. For each such object, it calls `_is_correctly_stacked`.
    8. The total count of such objects is `len(self.goal_on)`.
    9. The number of correctly stacked objects is counted.
    10. The heuristic value is the total number of goal-positioned objects minus
        the number of correctly stacked objects. This represents the number of
        objects that are "out of place" relative to the goal stack structure.
    11. A heuristic value of 0 is achieved if and only if all blocks that have
        a specified position relative to below in the goal are correctly stacked
        according to the recursive definition, which implies the goal stacking
        configuration is achieved.
    """
    def __init__(self, task):
        self.goal_on = {}
        self.all_objects = set()

        # Parse goal facts to build the goal_on mapping and collect all objects
        for fact_str in task.goals:
            # Remove parentheses and split by space
            parts = fact_str.strip('()').split()
            if not parts: # Skip empty strings or malformed facts
                continue
            predicate = parts[0]

            if predicate == 'on':
                if len(parts) == 3:
                    obj1, obj2 = parts[1], parts[2]
                    self.goal_on[obj1] = obj2
                    self.all_objects.add(obj1)
                    self.all_objects.add(obj2)
            elif predicate == 'on-table':
                if len(parts) == 2:
                    obj = parts[1]
                    self.goal_on[obj] = 'table'
                    self.all_objects.add(obj)
            # Ignore other predicates like 'clear', 'arm-empty', 'holding' in goal for this heuristic's structure logic

        # Note: Static facts are available in task.static but are not needed
        # for this heuristic which focuses on the dynamic state of blocks.

    def __call__(self, state):
        memo = {} # Memoization dictionary for _is_correctly_stacked

        def _is_correctly_stacked(block, current_state, goal_mapping, memo_dict):
            """
            Recursive helper to check if a block is part of a correctly built
            goal stack segment from the bottom up.
            """
            if block in memo_dict:
                return memo_dict[block]

            target_below = goal_mapping.get(block)

            # If the block is not in the goal_on mapping, it means its position
            # relative to below is not specified in the goal. It cannot be
            # "correctly stacked relative to below" according to this definition.
            # This case should ideally not be reached for blocks that are keys
            # in goal_on, but is a safe guard.
            if target_below is None:
                 memo_dict[block] = False
                 return False

            result = False
            if target_below == 'table':
                # Check if the block is on the table in the current state
                if f'(on-table {block})' in current_state:
                    result = True
            else:
                # Check if the block is on the correct block below it in the current state
                # AND the block below is correctly stacked.
                if f'(on {block} {target_below})' in current_state:
                    # Recursively check the block below
                    result = _is_correctly_stacked(target_below, current_state, goal_mapping, memo_dict)

            memo_dict[block] = result
            return result

        # Get the list of objects whose position relative to below is specified in the goal
        goal_positioned_objects = list(self.goal_on.keys())

        # If there are no goal positions defined (e.g., empty goal or only clear/arm-empty goals),
        # the heuristic is 0.
        if not goal_positioned_objects:
             return 0

        # Calculate the number of blocks that are NOT correctly stacked relative to below
        # by summing 1 for each block that _is_correctly_stacked returns False for.
        # Use a fresh memo for the calculation pass.
        memo_final = {}
        heuristic_value = sum(1 for obj in goal_positioned_objects if not _is_correctly_stacked(obj, state, self.goal_on, memo_final))

        return heuristic_value
