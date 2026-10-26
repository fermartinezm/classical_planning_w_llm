from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if fact.startswith('(') and fact.endswith(')'):
        content = fact[1:-1].strip()
        if content:
            return tuple(content.split())
    return tuple() # Should not happen for valid facts like (on x y) or (on-table x) etc.

# The match function from examples is not strictly necessary if parsing directly,
# but keeping it as it was provided in examples.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct final position within a correctly built stack segment, rooted
    at the table, as defined by the goal state. It also adds the number
    of blocks that are currently on top of a block that is not in its
    correct final position, representing the cost of clearing.

    # Assumptions
    - The goal state defines a specific configuration of blocks stacked
      on top of each other or on the table.
    - All blocks present in the problem instance are either part of the
      explicit goal configuration (defined by 'on' or 'on-table' goals)
      or, if not explicitly mentioned, their implicit goal is to be on
      the table and clear.
    - The heuristic is non-admissible and designed to guide a greedy
      best-first search efficiently.

    # Heuristic Initialization
    - Extracts the goal configuration from the task's goal facts, building
      a mapping `goal_on` where `goal_on[block]` is the block (or the string
      'table') that `block` should be directly on top of in the goal state.
    - Identifies all blocks present in the problem instance by examining
      both the initial state and the goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all blocks present in the problem instance by collecting
       all arguments from 'on', 'on-table', 'holding', and 'clear' facts
       in the initial state and 'on', 'on-table' facts in the goal.
    2. Parse the goal facts to create the `goal_on` mapping: for each
       `(on x y)` goal, `goal_on[x] = y`; for each `(on-table x)` goal,
       `goal_on[x] = 'table'`.
    3. For a given state, parse the state facts to create the `current_pos`
       mapping: for each `(on x y)` fact, `current_pos[x] = y`; for each
       `(on-table x)` fact, `current_pos[x] = 'table'`; for each `(holding x)`
       fact, `current_pos[x] = 'arm'`.
    4. Define a recursive helper function `is_in_final_position(block, current_pos, memo)`:
       - This function checks if `block` is in its correct final position *and*
         if the block it is supposed to be on is also in its correct final position,
         recursively down to the table.
       - Use memoization (`memo`) to store results for blocks that have already
         been checked.
       - If `block` is in the `goal_on` map:
         - Get its `goal_support = goal_on[block]`.
         - Get its `current_support = current_pos.get(block)`.
         - If `current_support != goal_support`, return False.
         - If `current_support == goal_support`:
           - If `goal_support == 'table'`, return True.
           - If `goal_support` is a block, recursively call `is_in_final_position`
             on `goal_support`. Return the result.
       - If `block` is NOT in the `goal_on` map (it's an extra block):
         - Assume its goal is to be on the table.
         - Return True if `current_pos.get(block) == 'table'`, False otherwise.
    5. Initialize heuristic components `h1 = 0` and `h2 = 0`.
    6. Compute `h1`: Iterate through all blocks identified in step 1. For each block,
       call `is_in_final_position`. If it returns False, increment `h1`. This counts
       blocks that are not in their correct place relative to their goal stack chain.
    7. Compute `h2`: Iterate through all blocks identified in step 1. For each block `b`,
       get its `current_support = current_pos.get(b)`. If `current_support` is a block
       (i.e., `b` is on top of another block), check if `is_in_final_position(current_support)`
       is False. If it is, increment `h2`. This counts blocks that are currently
       sitting on top of a block that is misplaced.
    8. The total heuristic value is `h1 + h2`.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal configuration and all blocks.
        """
        self.goals = task.goals

        # Map block -> block_it_should_be_on (or 'table') in the goal state.
        self.goal_on = {}

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "on":
                if len(parts) == 3:
                    block, under_block = parts[1], parts[2]
                    self.goal_on[block] = under_block
            elif parts and parts[0] == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_on[block] = 'table'
            # Ignore (clear ?x) and (arm-empty) goals for this mapping.

        # Identify all blocks present in the problem instance.
        # Collect objects from initial state facts and goal facts.
        all_objects = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) > 1:
                 all_objects.update(parts[1:]) # Add all arguments as objects

        for goal in task.goals:
             parts = get_parts(goal)
             if len(parts) > 1:
                 all_objects.update(parts[1:]) # Add all arguments as objects

        # Filter out predicate names or special terms like 'table', 'arm' if they
        # somehow ended up in the object list (unlikely with correct parsing).
        # 'table' is a valid support, but not a block object itself.
        # 'arm' is a state property, not a block object.
        self.all_blocks = {obj for obj in all_objects if obj not in ['table', 'arm', 'arm-empty']}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings).

        # Map block -> block_it_is_on (or 'table' or 'arm') in the current state.
        current_pos = {}
        # We don't need current_above for this heuristic calculation.

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "on":
                if len(parts) == 3:
                    block, under_block = parts[1], parts[2]
                    current_pos[block] = under_block
            elif parts and parts[0] == "on-table":
                if len(parts) == 2:
                    block = parts[1]
                    current_pos[block] = 'table'
            elif parts and parts[0] == "holding":
                 if len(parts) == 2:
                    block = parts[1]
                    current_pos[block] = 'arm'
            # Ignore (clear ?x) and (arm-empty) for position mapping.

        # Memoization for the recursive check.
        memo = {}

        def is_in_final_position(block):
            """
            Recursive helper to check if a block is in its correct final position
            within a correctly built stack segment, or is on the table if its
            goal position is not specified.
            """
            if block in memo:
                return memo[block]

            current_support = current_pos.get(block)

            if block in self.goal_on:
                # This block is part of the defined goal stacks/table positions.
                goal_support = self.goal_on[block]

                if current_support != goal_support:
                    memo[block] = False
                    return False
                else: # current_support == goal_support
                    if goal_support == 'table':
                        memo[block] = True
                        return True
                    else: # goal_support is a block (under_block)
                        # Block is on the correct block, now check if the block below is correct.
                        # Ensure goal_support is a block object before recursive call.
                        if goal_support in self.all_blocks:
                            result = is_in_final_position(goal_support)
                            memo[block] = result
                            return result
                        else:
                             # Goal support is not a recognized block object. This indicates a problem definition issue.
                             # Treat this block as misplaced.
                             memo[block] = False
                             return False
            else:
                # This block is in the problem instance but not explicitly in goal_on.
                # Assume its goal is to be on the table.
                result = (current_support == 'table')
                memo[block] = result
                return result

        # Heuristic h1: Count blocks from self.all_blocks where is_in_final_position is False.
        # These are blocks that are not part of a correctly built goal stack segment from the table up.
        h1 = 0
        for block in self.all_blocks:
             if not is_in_final_position(block):
                 h1 += 1

        # Heuristic h2: Count blocks from self.all_blocks that are currently on top
        # of a block that is not in its final position. These blocks need to be cleared.
        h2 = 0
        for block in self.all_blocks:
             current_support = current_pos.get(block)
             # Check if the block is currently on another block (not table or arm)
             if current_support in self.all_blocks: # Ensure current_support is a block object
                 # Check if the block it is on is not in its final position
                 if not is_in_final_position(current_support):
                     h2 += 1

        # The total heuristic is the sum of blocks in the wrong place (relative to goal stack)
        # and blocks that are blocking misplaced blocks.
        total_cost = h1 + h2

        return total_cost
