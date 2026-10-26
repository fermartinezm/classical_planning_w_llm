from heuristics.heuristic_base import Heuristic
# fnmatch is not strictly needed if we parse facts manually
# from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # This should not happen with valid PDDL facts, but as a safeguard
         # or for non-predicate facts like ':init', ':goal', etc.
         # For this heuristic, we only care about predicate facts.
         return []
    
    # Remove parentheses and split by whitespace
    return fact_str[1:-1].split()

# No need for a match function if using get_parts directly
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#     """
#     parts = get_parts(fact)
#     # Simple check: number of parts must match number of args
#     if len(parts) != len(args):
#         return False
#     # Check if each part matches the corresponding arg (with fnmatch support)
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by summing penalties
    for unsatisfied goal conditions related to block positions and for
    undesired block positions in the current state (obstructions).

    # Heuristic Initialization
    - Extracts the desired `(on X Y)` and `(on-table X)` facts from the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h` to 0.
    2. Identify the set of `(on X Y)` facts required in the goal (`goal_on`).
    3. Identify the set of `(on-table X)` facts required in the goal (`goal_on_table`).
    4. Identify the set of `(on X Y)` facts true in the current state (`current_on`).
    5. Identify the set of `(on-table X)` facts true in the current state (`current_on_table`).
    6. Check if the arm is holding a block (`held_block`).
    7. Add a penalty for each goal `(on X Y)` fact that is not true in the current state.
       - Each such fact requires `X` to be moved onto `Y`, which typically involves
         getting `X` into the arm (pickup/unstack) and then stacking it on `Y`.
         Estimate cost: 2 actions (e.g., pickup + stack).
    8. Add a penalty for each goal `(on-table X)` fact that is not true in the current state.
       - Each such fact requires `X` to be moved onto the table, involving
         getting `X` into the arm and then putting it down.
         Estimate cost: 2 actions (e.g., unstack + putdown).
    9. Add a penalty for each `(on X Y)` fact that is true in the current state
       but is *not* a goal `(on X Y)` fact.
       - This represents an obstruction (`X` is on `Y`, but shouldn't be).
         `X` needs to be moved out of the way, involving unstacking it from `Y`
         and putting it down or stacking it elsewhere.
         Estimate cost: 2 actions (e.g., unstack + putdown).
    10. Add a penalty if the arm is currently holding a block.
        - The arm is busy and needs one action (putdown or stack) to become free.
        Estimate cost: 1 action.
    11. The total heuristic value is the sum of all penalties.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are empty in Blocksworld, so no need to process task.static

        # Extract goal (on X Y) and (on-table X) facts
        self.goal_on = set()
        self.goal_on_table = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip non-predicate facts like ':goal'
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                self.goal_on.add((parts[1], parts[2]))
            elif predicate == "on-table" and len(parts) == 2:
                self.goal_on_table.add(parts[1])
            # We ignore (clear X) goals for this heuristic's calculation,
            # as the cost of achieving clearance is often related to
            # removing obstructing blocks, which is penalized separately.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # Parse current state facts
        current_on = set()
        current_on_table = set()
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip non-predicate facts
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                current_on.add((parts[1], parts[2]))
            elif predicate == "on-table" and len(parts) == 2:
                current_on_table.add(parts[1])
            elif predicate == "holding" and len(parts) == 2:
                held_block = parts[1]
            # We don't need current_clear for this heuristic calculation.
            # We also don't need arm-empty explicitly, as held_block covers the arm state.


        # If the state is the goal state, the heuristic is 0.
        # This check is implicitly handled by the calculation below if the goal
        # only contains 'on', 'on-table', and 'clear' facts and the arm is empty
        # in the goal (standard Blocksworld). If the goal contains other facts
        # or the arm is not empty in the goal (unusual), this might need adjustment.
        # Assuming standard Blocksworld goals, if all goal_on and goal_on_table
        # are satisfied, there are no 'wrong' current_on facts (as they would
        # contradict the goal state), and the arm is empty, the heuristic will be 0.
        # We can rely on the calculation.

        h = 0

        # Penalty for goal (on X Y) facts not satisfied
        for (X, Y) in self.goal_on:
            if (X, Y) not in current_on:
                # X needs to be stacked on Y. Requires getting X in hand + stack.
                h += 2

        # Penalty for goal (on-table X) facts not satisfied
        for X in self.goal_on_table:
            if X not in current_on_table:
                 # X needs to be put on the table. Requires getting X in hand + putdown.
                 h += 2

        # Penalty for blocks that are currently on top of other blocks but shouldn't be
        # These are obstructions that need to be moved.
        for (X, Y) in current_on:
            # If this (on X Y) relation is not desired in the goal
            if (X, Y) not in self.goal_on:
                # X is on Y, but shouldn't be. X needs to be moved.
                # Requires unstacking X from Y + putting X down or stacking elsewhere.
                h += 2

        # Penalty for holding a block
        if held_block is not None:
            # The arm is busy. It needs one action (putdown or stack) to become free.
            h += 1

        return h

