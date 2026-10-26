from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or non-fact strings gracefully, though state facts are expected format
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of blocks that are not in their correct goal position
    and the number of blocks that need to be clear according to the goal but are not.
    Each such discrepancy adds 1 to the heuristic value.

    # Assumptions
    - The goal specifies the desired position (on another block or on the table)
      for all blocks involved in the goal configuration.
    - The goal may also specify that certain blocks must be clear.
    - The heuristic counts two types of discrepancies:
        1. A block's immediate support (what it's directly on or if it's on the table/held)
           does not match its required goal support.
        2. A block is required to be clear in the goal, but something is currently on it.
    - Each discrepancy adds 1 to the heuristic value. This is a relaxation, as
      fixing one discrepancy might require multiple actions or might fix multiple
      discrepancies.

    # Heuristic Initialization
    - Parses the goal conditions (`task.goals`) to determine the required support
      for each block (which block it should be on, or if it should be on the table).
      This is stored in the `self.goal_support` dictionary, mapping block names
      to their required support ('table' or another block name).
    - Parses the goal conditions to identify which blocks must be clear in the goal state.
      This is stored in the `self.goal_clear` set.
    - Static facts (`task.static`) are not used as Blocksworld has no static predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the heuristic value `h` to 0.
    2. Determine the current support for every block present in the state by iterating
       through the state facts:
       - If `(on B C)` is true, block B's current support is C.
       - If `(on-table B)` is true, block B's current support is 'table'.
       - If `(holding B)` is true, block B's current support is 'holding'.
       Store these in a dictionary `current_support`, mapping block names to their
       current support.
    3. Identify which blocks are currently clear by iterating through the state facts
       and collecting blocks from `(clear B)` predicates into a set `current_clear`.
    4. Iterate through each block `B` for which a goal support is defined in `self.goal_support`:
       - Get the required goal support `goal_sup = self.goal_support[B]`.
       - Get the current support `current_sup = current_support.get(block, None)`.
       - If `current_sup` is different from `goal_sup`, increment `h` by 1.
       (Note: A block being held will have 'holding' as current support, which will
        always be different from its goal support ('table' or another block),
        correctly contributing to the heuristic).
    5. Iterate through each block `B` that is required to be clear in the goal
       (`self.goal_clear` set):
       - Check if `B` is present in the `current_clear` set.
       - If `B` is not in `current_clear` (meaning `(clear B)` is false in the state),
         increment `h` by 1.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support and goal clear conditions.
        """
        # Store goals and static facts as per example heuristics
        self.goals = task.goals
        self.static = task.static # Blocksworld has no static facts

        self.goal_support = {}
        self.goal_clear = set()

        # Parse goal conditions
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: # Skip malformed facts
                continue

            predicate = parts[0]
            if predicate == "on":
                # Goal is (on ?x ?y)
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    self.goal_support[block] = support
            elif predicate == "on-table":
                # Goal is (on-table ?x)
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_support[block] = 'table' # Use a special value for table
            elif predicate == "clear":
                # Goal is (clear ?x)
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_clear.add(block)
            # Ignore other potential goal predicates if any (like arm-empty)

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current support map and find current clear blocks
        current_support = {}
        current_clear = set()
        # currently_holding = None # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue

            predicate = parts[0]
            if predicate == "on":
                # (on ?x ?y)
                if len(parts) == 3:
                    block, support = parts[1], parts[2]
                    current_support[block] = support
            elif predicate == "on-table":
                # (on-table ?x)
                if len(parts) == 2:
                    block = parts[1]
                    current_support[block] = 'table' # Use a special value for table
            elif predicate == "holding":
                # (holding ?x)
                if len(parts) == 2:
                    block = parts[1]
                    current_support[block] = 'holding' # Use a special value for holding
                    # currently_holding = block # Not strictly needed
            elif predicate == "clear":
                 # (clear ?x)
                 if len(parts) == 2:
                     block = parts[1]
                     current_clear.add(block)
            # Ignore other predicates like arm-empty

        h = 0

        # Heuristic part 1: Count blocks with wrong support
        # Iterate through blocks that have a defined goal position
        for block, goal_sup in self.goal_support.items():
            current_sup = current_support.get(block, None) # Get current support, default None if block not found

            # If block is not found in current_support, it's an unexpected state
            # (e.g., block disappeared), but for robustness, treat as mismatch.
            # In Blocksworld, a block is always either on something, on the table, or held.
            # So it should always be in current_support unless the state is invalid.
            # A block not in current_support implies it's not on anything, not on table, and not held.
            # This shouldn't happen in valid states.
            # If it does happen, its support is definitely not the goal support.
            if current_sup is None:
                 h += 1
            elif current_sup != goal_sup:
                h += 1

        # Heuristic part 2: Count blocks that should be clear but are not
        for block in self.goal_clear:
            if block not in current_clear:
                 h += 1

        return h
