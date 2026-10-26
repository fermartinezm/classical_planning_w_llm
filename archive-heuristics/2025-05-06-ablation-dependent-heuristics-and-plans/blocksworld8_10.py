from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential malformed fact strings defensively
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

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
    This heuristic estimates the number of actions required to reach the goal state
    by summing up estimated costs for achieving each unsatisfied goal predicate.
    It considers the cost of placing blocks in their target positions (on another
    block or on the table), the cost of clearing blocks that are blocking desired
    clear conditions, and the cost of emptying the arm if required by the goal.
    This heuristic is non-admissible and designed to guide a greedy best-first search.

    # Assumptions
    - The goal specifies a configuration of blocks using `on`, `on-table`, and `clear` predicates, and potentially `arm-empty`.
    - The cost of moving a block to its target position (either onto another block or onto the table) is estimated as 1 action if the block is already held, and 2 actions otherwise (representing a pickup/unstack followed by a stack/putdown).
    - The cost of achieving a `(clear X)` goal predicate when X is not clear is estimated by counting the number of blocks directly on top of X. Each block on top is assumed to require 2 actions to move out of the way (unstack + putdown). This simplifies the clearing cost and ignores dependencies among blocks on top.
    - The cost of achieving an `(arm-empty)` goal predicate when the arm is not empty is estimated as 1 action (putdown the held block).
    - The heuristic assumes valid Blocksworld states and goals where blocks are either on the table, on another block, or held.

    # Heuristic Initialization
    - The heuristic stores the set of goal predicates (`task.goals`) for quick access during the heuristic computation.
    - Static facts (`task.static`) are typically empty in Blocksworld and are not used by this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the total heuristic cost `h` to 0.
    2. Pre-process the current `state` to quickly determine:
       - Which block, if any, is currently held (`holding_block`).
       - The block immediately underneath each block (`current_on` mapping).
       - Which blocks are currently on the table (`current_on_table` set).
       - Which blocks are currently clear (`current_clear` set).
    3. Check if the current `state` is the goal state. If `self.goals` is a subset of `state`, return 0. This ensures `h=0` if and only if the goal is reached.
    4. Iterate through each predicate `goal_pred` in the set of goal predicates (`self.goals`).
    5. If `goal_pred` is already true in the current `state`, it contributes 0 to the heuristic. Skip to the next goal predicate.
    6. If `goal_pred` is not true in the current `state`, estimate the cost to achieve it based on its form:
       - If `goal_pred` is of the form `(on X Y)`:
         - This goal is not satisfied. To achieve it, block X needs to be placed on block Y.
         - If the robot is currently holding block X (`holding_block == X`), the immediate next step towards this goal is likely `stack X Y`. This costs 1 action. Add 1 to `h`.
         - Otherwise (X is on the table or on another block), X first needs to be picked up or unstacked. This typically takes 1 action (pickup/unstack), followed by 1 action to stack it on Y. Estimate this cost as 2 actions. Add 2 to `h`.
       - If `goal_pred` is of the form `(on-table X)`:
         - This goal is not satisfied. To achieve it, block X needs to be placed on the table.
         - If the robot is currently holding block X (`holding_block == X`), the immediate next step towards this goal is likely `putdown X`. This costs 1 action. Add 1 to `h`.
         - Otherwise (X is on another block), X first needs to be unstacked. This typically takes 1 action (unstack), followed by 1 action to put it down. Estimate this cost as 2 actions. Add 2 to `h`.
       - If `goal_pred` is of the form `(clear X)`:
         - This goal is not satisfied, meaning `(clear X)` is not in `state`. This implies there is at least one block on top of X.
         - To make X clear, all blocks directly on top of it must be moved. Count the number of blocks `Y` such that `(on Y X)` is true in the current `state`.
         - For each such block Y, estimate the cost to move it out of the way as 2 actions (unstack Y from X + putdown Y somewhere else, e.g., on the table). Add `count * 2` to `h`. This is a simplified cost and doesn't account for needing to clear blocks on top of Y.
       - If `goal_pred` is `(arm-empty)`:
         - This goal is not satisfied, meaning `(arm-empty)` is not in `state`. This implies the robot is holding a block.
         - To achieve `arm-empty`, the held block must be put down. This costs 1 action (`putdown`). Add 1 to `h`.
    7. Return the total computed heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal predicates."""
        self.goals = task.goals
        # Static facts are empty in Blocksworld and not needed for this heuristic.
        # self.static = task.static

    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # If the state is a goal state, the heuristic is 0.
        # This check is important for correctness (h=0 iff goal).
        if self.goals <= state:
             return 0

        h = 0

        # Pre-process current state for faster lookups
        holding_block = None
        current_on = {} # Maps block -> block_it_is_on
        # current_on_table = set() # Not strictly needed for the logic below, but good for completeness
        # current_clear = set() # Not strictly needed for the logic below, but good for completeness

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "holding" and len(parts) == 2:
                holding_block = parts[1]
            elif predicate == "on" and len(parts) == 3:
                block_on = parts[1]
                block_under = parts[2]
                current_on[block_on] = block_under
            # elif predicate == "on-table" and len(parts) == 2:
            #     block_on_table = parts[1]
            #     current_on_table.add(block_on_table)
            # elif predicate == "clear" and len(parts) == 2:
            #      block_clear = parts[1]
            #      current_clear.add(block_clear)


        # Iterate through goal predicates and estimate cost for unsatisfied ones
        for goal_pred in self.goals:
            # Check if the goal predicate is already satisfied in the current state
            if goal_pred in state:
                continue # Goal predicate is already satisfied, contributes 0

            # Goal predicate is not satisfied, estimate cost
            parts = get_parts(goal_pred)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "on" and len(parts) == 3:
                x, y = parts[1], parts[2]
                # Goal: (on X Y) is not true. Cost to put X on Y.
                if holding_block == x:
                    h += 1 # X is held, need to stack
                else:
                    h += 2 # X is not held, need to pickup/unstack + stack

            elif predicate == "on-table" and len(parts) == 2:
                x = parts[1]
                # Goal: (on-table X) is not true. Cost to put X on table.
                if holding_block == x:
                    h += 1 # X is held, need to putdown
                else:
                    # X must be on some block Z. Need to unstack X from Z + putdown X.
                    h += 2 # Need to unstack + putdown

            elif predicate == "clear" and len(parts) == 2:
                x = parts[1]
                # Goal: (clear X) is not true. Cost to clear X.
                # Count blocks directly on top of X
                num_on_top = sum(1 for block_on, block_under in current_on.items() if block_under == x)
                h += num_on_top * 2 # Each block on top needs unstack + putdown

            elif predicate == "arm-empty" and len(parts) == 1:
                 # Goal: (arm-empty) is not true. Cost to empty arm.
                 # This means the arm is holding something.
                 h += 1 # Need to putdown the held block

            # Ignore other goal predicates if any (like object types)

        return h

