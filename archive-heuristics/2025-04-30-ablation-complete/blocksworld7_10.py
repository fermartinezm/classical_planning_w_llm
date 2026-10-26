# from heuristics.heuristic_base import Heuristic # Assuming this is available

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle unexpected fact format, maybe log a warning or raise error
         return [] # Return empty list for malformed facts
    return fact[1:-1].split()

# Helper to find what's directly on top of a block
def get_direct_block_on_top(block, current_state):
    """Find the block directly on top of the given block in the current state."""
    for fact in current_state:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] == 'on' and parts[2] == block:
            return parts[1]
    return None # Nothing directly on top

# Helper to count blocks stacked on top (recursive count of the stack above)
def count_blocks_stacked_on_top(block, current_state):
    """Count the number of blocks stacked directly or indirectly on top of the given block."""
    count = 0
    current = get_direct_block_on_top(block, current_state)
    while current:
        count += 1
        current = get_direct_block_on_top(current, current_state)
    return count

# Helper to check if the arm is holding anything
def is_arm_holding_anything(current_state):
    """Check if the arm is currently holding any block."""
    for fact in current_state:
        parts = get_parts(fact)
        if len(parts) == 2 and parts[0] == 'holding':
            return True
    return False

# Helper to get the object being held
def get_held_object(current_state):
    """Get the object currently held by the arm, or None if arm is empty."""
    for fact in current_state:
        parts = get_parts(fact)
        if len(parts) == 2 and parts[0] == 'holding':
            return parts[1]
    return None


class blocksworldHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy
    each unsatisfied goal condition independently, summing up these estimates.
    It considers the cost of clearing blocks and performing the final move
    (pickup/unstack + stack/putdown).

    # Assumptions
    - The cost of each action (pickup, putdown, stack, unstack) is 1.
    - The heuristic sums the estimated costs for each unsatisfied goal fact
      independently, ignoring potential negative interactions (e.g., unstacking
      a block might satisfy a 'clear' goal but also put a block in a wrong place)
      or positive interactions (e.g., unstacking might clear multiple blocks).
    - The heuristic assumes that necessary preconditions (like arm-empty, clear)
      can be met by performing the counted actions.
    - (holding B) goals are not explicitly handled as final goals, focusing on
      (on X Y), (on-table Z), (clear W), and (arm-empty).

    # Heuristic Initialization
    - Stores the goal conditions from the task.
    - Static facts are not used as Blocksworld domain typically has none.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:
    1. Initialize total_cost = 0.
    2. Check if the current state is a goal state. If yes, return 0.
    3. Pre-calculate if the arm is holding anything and which object is held for efficiency.
    4. Iterate through each goal fact in the task's goals.
    5. If a goal fact `g` is already true in the current state, it contributes 0 to the heuristic.
    6. If a goal fact `g` is not true in the current state, estimate the cost to achieve it:
       a. If `g` is `(on B Y)`:
          - Cost to clear B: Count how many blocks are currently stacked on B. Add this count.
          - Cost to clear Y: Count how many blocks are currently stacked on Y. Add this count.
          - Cost to move B into position: If B is currently held, it needs 1 action (stack). If B is not held, it's on table or another block. It needs 1 action (pickup or unstack) to get into the hand, and 1 action (stack) to be placed on Y. Total 2 actions. Add this cost (1 or 2).
          - Add the total estimated cost for this `(on B Y)` goal to `total_cost`.
       b. If `g` is `(on-table B)`:
          - Cost to clear B: Count how many blocks are currently stacked on B. Add this count.
          - Cost to move B to the table: If B is currently held, it needs 1 action (putdown). If B is not held, it must be on another block (since not on table and goal not met). It needs 1 action (unstack) to get into the hand, and 1 action (putdown) to be placed on the table. Total 2 actions. Add this cost (1 or 2).
          - Add the total estimated cost for this `(on-table B)` goal to `total_cost`.
       c. If `g` is `(clear B)`:
          - Cost to clear B: Count how many blocks are currently stacked on B. Add this count.
          - Add the total estimated cost for this `(clear B)` goal to `total_cost`.
       d. If `g` is `(arm-empty)`:
          - Cost to empty the arm: If the arm is not empty (something is held), add 1 (for putdown or stack).
          - Add the total estimated cost for this `(arm-empty)` goal to `total_cost`.
    7. Return `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        # Blocksworld has no static facts, so task.static is not used.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Check if the current state is a goal state
        if self.goals <= state:
             return 0

        # Pre-calculate if arm is holding anything and what object
        arm_is_holding = is_arm_holding_anything(state)
        held_object = get_held_object(state)

        # Iterate through each goal fact
        for goal in self.goals:
            # If goal is already satisfied, it contributes 0
            if goal in state:
                continue

            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "on" and len(parts) == 3:
                B, Y = parts[1], parts[2]
                # Goal: (on B Y) is not satisfied

                # Cost to clear B: Need to unstack everything on B
                cost_clear_b = count_blocks_stacked_on_top(B, state)
                total_cost += cost_clear_b

                # Cost to clear Y: Need to unstack everything on Y
                cost_clear_y = count_blocks_stacked_on_top(Y, state)
                total_cost += cost_clear_y

                # Cost to move B into position (get in hand + stack)
                # If B is already held, cost is 1 (stack).
                # If B is not held, it's on table or another block. Cost is 1 (pickup/unstack) + 1 (stack) = 2.
                if held_object == B:
                    total_cost += 1 # Cost to stack B on Y
                else:
                    total_cost += 2 # Cost to get B in hand + stack B on Y


            elif predicate == "on-table" and len(parts) == 2:
                B = parts[1]
                # Goal: (on-table B) is not satisfied

                # Cost to clear B: Need to unstack everything on B
                cost_clear_b = count_blocks_stacked_on_top(B, state)
                total_cost += cost_clear_b

                # Cost to move B to the table (get in hand + putdown)
                # If B is already held, cost is 1 (putdown).
                # If B is not held, it must be on another block (since not on table and goal not met).
                # Cost is 1 (unstack) + 1 (putdown) = 2.
                if held_object == B:
                    total_cost += 1 # Cost to putdown B
                else:
                    total_cost += 2 # Cost to get B in hand (unstack) + putdown B


            elif predicate == "clear" and len(parts) == 2:
                B = parts[1]
                # Goal: (clear B) is not satisfied (something is on B)

                # Cost to clear B: Count how many blocks are currently stacked on B.
                # Each block on top needs to be unstacked.
                cost_clear_b = count_blocks_stacked_on_top(B, state)
                total_cost += cost_clear_b

            elif predicate == "arm-empty" and len(parts) == 1:
                # Goal: (arm-empty) is not satisfied (something is held)

                # Cost to empty the arm: Need to put down or stack the held block. Assume 1 action.
                if arm_is_holding:
                    total_cost += 1

            # Ignoring (holding B) goals as per docstring assumption

        return total_cost
