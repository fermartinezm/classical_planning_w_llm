# from fnmatch import fnmatch # Not needed with direct parsing
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assuming valid PDDL fact strings like '(predicate arg1 arg2)' or '(predicate)'
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Fallback or handle unexpected formats, though PDDL facts from parser should be consistent
    return fact.split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of blocks that are not in their correct position
    relative to the block or table below them, considering the goal stack
    structure from the bottom up. It also adds a cost if the arm is not empty.
    Specifically, it counts blocks that are part of the goal configuration
    but are not "correctly stacked", where "correctly stacked" means being
    on the correct base AND that base is also correctly stacked recursively.

    # Assumptions
    - The goal state is defined primarily by the desired `on` and `on-table`
      predicates, forming specific stacks of blocks.
    - `clear` predicates in the goal are typically satisfied when the stack
      below is correctly built and nothing extra is placed on top.
    - The cost of achieving the goal is related to fixing blocks that are
      not in their correct place within the goal stack structure and freeing the arm.

    # Heuristic Initialization
    - Parses the goal conditions (`task.goals`) to determine the desired support
      (block or table) for each block that is part of a goal stack. This creates
      the `goal_support` mapping (block -> base).
    - Identifies all blocks that are mentioned as arguments in the goal `on` or
      `on-table` predicates (`goal_blocks`). These are the blocks whose position
      is constrained by the goal structure.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse Current State:
       - Iterate through the facts in the current state (`node.state`).
       - For each `on` fact `(on ?x ?y)`, record that `?x`'s current support is `?y`.
       - For each `on-table` fact `(on-table ?x)`, record that `?x`'s current support is "table".
       - For each `holding` fact `(holding ?x)`, record that `?x`'s current support is "arm".
       - Determine if the `arm-empty` fact is present.
       - Store these current supports in a dictionary `current_support` (block -> base/table/arm).

    2. Define and Use `is_correctly_stacked` Function:
       - Define a recursive helper function `is_correctly_stacked(block)` that checks
         if a given `block` is in its correct position relative to its base, and if
         that base is also correctly positioned recursively, according to the goal structure.
       - The base cases for the recursion are:
         - If the block is not in `goal_support` (its position isn't specified in the goal structure), it's considered correctly stacked relative to the goal structure.
         - If the block's goal base is "table" and its current base is also "table".
       - The recursive step is: If the block's goal base is another block `X`, its current
         base must match `X`, AND `is_correctly_stacked(X)` must be true.
       - Use memoization (`memo` dictionary) within the function to store results for
         already computed blocks, preventing redundant work and handling cycles (though
         cycles shouldn't occur in valid blocksworld states/goals).

    3. Count Misplaced Blocks:
       - Initialize a counter `misplaced_count` to 0.
       - Iterate through each `block` in the set of `goal_blocks` (blocks involved in
         the goal `on`/`on-table` predicates).
       - Call `is_correctly_stacked(block)`. If it returns `False`, increment `misplaced_count`.

    4. Add Arm Cost:
       - Initialize `arm_cost` to 0.
       - If the `arm-empty` fact is not present in the current state, set `arm_cost` to 1.
         This represents the cost of putting down the currently held block.

    5. Calculate Total Heuristic Value:
       - The total heuristic value is `misplaced_count + arm_cost`.

    6. Handle Goal State:
       - As a final check, if the current state is the actual goal state (`self.goals <= state`),
         the heuristic value must be 0. Return 0 in this case, overriding the calculated value.
         This ensures the heuristic is 0 if and only if the goal is reached.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships.
        """
        self.goals = task.goals

        # Parse goal predicates to determine the desired support for each block
        self.goal_support = {}
        self.goal_blocks = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, base = parts[1], parts[2]
                self.goal_support[block] = base
                self.goal_blocks.add(block)
                self.goal_blocks.add(base)
            elif parts[0] == "on-table":
                block = parts[1]
                self.goal_support[block] = "table" # Use string "table" for table support
                self.goal_blocks.add(block)
            # Ignore (clear ?) and (arm-empty) goals for support mapping in __init__

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 6. Handle Goal State (Check first for correctness)
        if self.goals <= state:
             return 0

        # 1. Parse Current State
        current_support = {}
        arm_empty = False
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, base = parts[1], parts[2]
                current_support[block] = base
            elif parts[0] == "on-table":
                block = parts[1]
                current_support[block] = "table" # Use string "table" for table support
            elif parts[0] == "holding":
                block = parts[1]
                current_support[block] = "arm" # Use string "arm" for arm support
            elif parts[0] == "arm-empty":
                arm_empty = True

        # Memoization dictionary for the recursive correctly_stacked check
        memo = {}

        # 2. Define and Use `is_correctly_stacked` Function (recursive helper)
        def is_correctly_stacked(block):
            """
            Checks if a block is in its correct goal position relative to its base,
            and if that base is also correctly positioned recursively.
            """
            if block in memo:
                return memo[block]

            # If the block doesn't have a specified base in the goal, it's "correctly stacked"
            # relative to the goal structure requirements.
            if block not in self.goal_support:
                 memo[block] = True
                 return True

            goal_base = self.goal_support[block]
            current_base = current_support.get(block) # Use .get, returns None if block is not on anything/held

            # If the current base doesn't match the goal base, it's not correctly stacked
            if current_base != goal_base:
                memo[block] = False
                return False

            # If the goal base is the table, and the current base matches, it's correctly stacked
            if goal_base == "table":
                memo[block] = True
                return True

            # If the goal base is another block, check if that block is correctly stacked
            # current_base must be goal_base (a block) here
            result = is_correctly_stacked(goal_base)
            memo[block] = result
            return result

        # 3. Count Misplaced Blocks
        misplaced_count = 0
        # Iterate through all blocks that are part of the goal configuration
        for block in self.goal_blocks:
             if not is_correctly_stacked(block):
                 misplaced_count += 1

        # 4. Add Arm Cost
        arm_cost = 0
        if not arm_empty:
             arm_cost = 1 # Need an action to put down the held block

        # 5. Calculate Total Heuristic Value
        total_cost = misplaced_count + arm_cost

        return total_cost
