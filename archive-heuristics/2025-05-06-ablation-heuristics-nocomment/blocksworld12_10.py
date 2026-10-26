# from fnmatch import fnmatch # Not used in this heuristic
from heuristics.heuristic_base import Heuristic # Assuming this base class exists as per examples

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal by counting the
        number of blocks that are not part of a correctly built stack prefix
        from the bottom up, and adding a penalty related to the number of
        blocks currently stacked on top of them that need to be cleared.
        Specifically, for each block that is not correctly stacked according
        to the goal configuration, it adds 2 (representing typical actions
        like unstack/pickup and stack/putdown needed to move this block)
        plus the number of blocks currently stacked directly or indirectly
        on top of it in the current state. This heuristic is non-admissible
        and designed to guide a greedy best-first search effectively by
        prioritizing states where more blocks are in their correct relative
        positions within the goal stacks or require less clearing.

    Assumptions:
        - The goal state is defined by a conjunction of (on ?x ?y) and
          (on-table ?z) predicates, potentially with (clear ?w) and (arm-empty).
          The heuristic primarily focuses on the (on ?x ?y) and (on-table ?z)
          goals to define the required stack structure.
        - Blocks not mentioned in (on ?x ?y) or (on-table ?z) goals do not
          have a required final position for the purpose of the "correctly
          stacked" check, but they contribute to the cost if they are
          blocking a block that *is* part of the goal structure.
        - The heuristic assumes standard Blocksworld actions (pickup, putdown,
          stack, unstack).
        - The goal predicates define a valid, non-cyclic stack structure.

    Heuristic Initialization:
        The constructor parses the task's goal predicates to build the target
        stack structure.
        - `goal_on_map`: A dictionary mapping a block to the block it should
          be directly on top of in the goal state, derived from (on ?x ?y) goals.
        - `goal_on_table_set`: A set of blocks that should be directly on the
          table in the goal state, derived from (on-table ?z) goals.
        - `goal_blocks`: A set of all blocks that appear in any (on ?x ?y)
          or (on-table ?z) goal predicate. These are the blocks whose positions
          are constrained by the goal stack structure.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Parse the current state to determine the current position of each block:
           - `current_on_map`: Maps a block to the block it is currently on.
           - `current_on_table_set`: Set of blocks currently on the table.
           - `current_block_above_map`: Maps a block to the block currently
             directly on top of it (inverse of current_on_map).
           - `current_holding`: The block currently held by the arm, or None.
           - `arm_is_empty`: Boolean indicating if the arm is empty.
        2. Identify which blocks are "correctly stacked from the bottom up".
           A block `b` is correctly stacked if:
           - It is supposed to be on the table in the goal (`b` is in `goal_on_table_set`)
             AND it is currently on the table (`b` is in `current_on_table_set`).
           - OR it is supposed to be on block `b_under` in the goal (`goal_on_map[b] == b_under`)
             AND it is currently on `b_under` (`current_on_map.get(b) == b_under`)
             AND `b_under` is also correctly stacked from the bottom up.
           This check is performed using a recursive helper function with memoization
           (`is_correctly_stacked`) to avoid redundant calculations. Only blocks
           in `goal_blocks` are checked for being correctly stacked.
        3. Calculate the "cost to clear above" for any block in the current state.
           This is the number of blocks currently stacked directly or indirectly
           on top of the given block. This is computed using a recursive helper
           function with memoization (`cost_to_clear_above`). This function works
           for any block, regardless of whether it's in the goal structure.
        4. Initialize the heuristic value `h` to 0.
        5. Iterate through each block `b` in the set of `goal_blocks`.
        6. For each block `b`, check if it is correctly stacked using the
           `is_correctly_stacked` function.
        7. If `b` is NOT correctly stacked:
           - Add 2 to the heuristic value (representing the minimum actions
             like unstack/pickup and stack/putdown needed to move this block
             to its correct position relative to its base).
           - Add the result of `cost_to_clear_above(b, ...)` to the heuristic value.
             This accounts for the effort needed to remove blocks currently
             on top of `b` before `b` can be moved.
        8. Return the final heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        # static_facts = task.static # Blocksworld typically has no static facts

        self.goal_on_map = {}
        self.goal_on_table_set = set()
        self.goal_blocks = set() # Blocks mentioned in on/on-table goals

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                obj, under_obj = parts[1], parts[2]
                self.goal_on_map[obj] = under_obj
                self.goal_blocks.add(obj)
                self.goal_blocks.add(under_obj)
            elif predicate == "on-table":
                obj = parts[1]
                self.goal_on_table_set.add(obj)
                self.goal_blocks.add(obj)
            # Ignore 'clear' and 'arm-empty' goals for stack structure definition

    def __call__(self, node):
        state = node.state

        # 1. Parse current state
        current_on_map = {}
        current_on_table_set = set()
        current_block_above_map = {} # Maps block_below -> block_above
        current_holding = None
        arm_is_empty = False

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                obj, under_obj = parts[1], parts[2]
                current_on_map[obj] = under_obj
                current_block_above_map[under_obj] = obj
            elif predicate == "on-table":
                obj = parts[1]
                current_on_table_set.add(obj)
            elif predicate == "holding":
                current_holding = parts[1]
            elif predicate == "arm-empty":
                arm_is_empty = True

        # Helper function for step 2 (Correctly Stacked)
        correctly_stacked_memo = {}
        def is_correctly_stacked(block):
            if block in correctly_stacked_memo:
                return correctly_stacked_memo[block]

            # A block is correctly stacked if it's in its goal position relative
            # to its base AND its base is correctly stacked.

            # Find the block this block should be on in the goal
            goal_base = self.goal_on_map.get(block)

            if goal_base is None:
                # This block should be on the table in the goal (it's a goal base)
                # Check if it's actually on the table
                result = block in current_on_table_set
            else:
                # This block should be on goal_base in the goal
                # Check if it's actually on goal_base
                current_base = current_on_map.get(block)
                if current_base == goal_base:
                    # It's on the correct block, now check if the block below is correctly stacked
                    result = is_correctly_stacked(goal_base)
                else:
                    # It's not on the correct block
                    result = False

            correctly_stacked_memo[block] = result
            return result

        # Helper function for step 3 (Cost to Clear Above)
        cost_above_memo = {}
        def cost_to_clear_above(block):
            if block in cost_above_memo:
                return cost_above_memo[block]

            # Find the block currently directly on top of this block
            block_above = current_block_above_map.get(block)

            if block_above is None:
                # Nothing is on top
                cost = 0
            else:
                # Cost is 1 (to move the block above) + cost to clear above that block
                cost = 1 + cost_to_clear_above(block_above)

            cost_above_memo[block] = cost
            return cost

        # 4. Initialize heuristic
        h = 0

        # 5. Iterate through blocks constrained by the goal stack structure
        for block in self.goal_blocks:
            # 6. Check if the block is correctly stacked
            if not is_correctly_stacked(block):
                # 7. If not correctly stacked, add cost
                # Add 2 for the block itself (pickup/unstack + putdown/stack)
                h += 2
                # Add the cost to clear anything currently on top of this block
                h += cost_to_clear_above(block)

        # 8. Return heuristic value
        return h
