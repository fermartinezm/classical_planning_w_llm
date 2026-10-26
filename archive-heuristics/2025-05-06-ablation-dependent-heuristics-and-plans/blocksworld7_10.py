from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
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
    This heuristic estimates the number of unsatisfied goal conditions related
    to block positions (on/on-table) and block clearness. It counts how many
    blocks are not on their correct base (or table) according to the goal,
    plus how many blocks that need to be clear are not clear. It also adds
    a cost if the arm needs to be empty but isn't.

    # Assumptions
    - The goal specifies the desired stack configuration (on/on-table) for
      some or all blocks, and potentially which blocks should be clear.
    - The heuristic counts each misplaced block relative to its goal base,
      each uncleared block that should be clear, and the state of the arm
      if arm-empty is a goal, as a unit of cost.
    - This heuristic is not admissible but aims to be informative for
      greedy best-first search by capturing key differences from the goal state.

    # Heuristic Initialization
    - Extracts the goal configuration from the task:
        - `goal_below`: A dictionary mapping a block to the block it should be
          on, or 'table' if it should be on the table.
        - `goal_clear`: A set of blocks that should be clear.
        - `goal_arm_empty`: A boolean indicating if `(arm-empty)` is a goal.
    - Collects all unique objects involved in the initial state and goals
      (although not strictly needed for this specific heuristic calculation,
       it's good practice).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the position of each block:
       - Identify which block is on which other block (`on ?x ?y`).
       - Identify which blocks are on the table (`on-table ?x`).
       - Identify which block is being held (`holding ?x`).
       - Identify which blocks are clear (`clear ?x`).
       - Create a map `current_below` from block to its base ('table', another block, or 'holding').
       - Create a set `current_clear` of blocks that are clear.
       - Check if the arm is empty (`arm-empty`).
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through all blocks that have a specified goal position in `goal_below`.
       a. Get the target base `target_base = goal_below[B]`.
       b. Find the current base `current_base = current_below.get(B)`.
       c. If `current_base` is not equal to `target_base`, increment `h`.
          (A block being held ('holding') will not match 'table' or any block,
           correctly contributing to the cost if its goal position is not 'holding').
    4. Iterate through all blocks that are in the `goal_clear` set.
       a. If `(clear B)` is not in the current state (i.e., `B` is not in `current_clear`),
          increment `h`.
    5. If `goal_arm_empty` is True and `(arm-empty)` is not in the current state,
       increment `h`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        self.goal_below = {}
        self.goal_clear = set()
        self.goal_arm_empty = False
        self.all_objects = set() # Collect all objects mentioned in goals/initial state

        # Parse goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                obj, underob = parts[1], parts[2]
                self.goal_below[obj] = underob
                self.all_objects.add(obj)
                self.all_objects.add(underob)
            elif predicate == "on-table" and len(parts) == 2:
                obj = parts[1]
                self.goal_below[obj] = 'table'
                self.all_objects.add(obj)
            elif predicate == "clear" and len(parts) == 2:
                obj = parts[1]
                self.goal_clear.add(obj)
                self.all_objects.add(obj)
            elif predicate == "arm-empty" and len(parts) == 1:
                 self.goal_arm_empty = True


        # Add objects from initial state that might not be in goals
        # (This part is not strictly necessary for the current heuristic logic
        # but is included for completeness if needed later)
        # for fact in task.initial_state:
        #      parts = get_parts(fact)
        #      if not parts: continue # Skip malformed facts
        #      predicate = parts[0]
        #      if predicate in ["on", "on-table", "clear", "holding"]:
        #          for obj in parts[1:]:
        #              self.all_objects.add(obj)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        current_below = {} # block -> block_below | 'table' | 'holding'
        current_clear = set()
        current_arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                obj, underob = parts[1], parts[2]
                current_below[obj] = underob
            elif predicate == "on-table" and len(parts) == 2:
                obj = parts[1]
                current_below[obj] = 'table'
            elif predicate == "clear" and len(parts) == 2:
                obj = parts[1]
                current_clear.add(obj)
            elif predicate == "holding" and len(parts) == 2:
                 obj = parts[1]
                 current_below[obj] = 'holding' # Represent holding state
            elif predicate == "arm-empty" and len(parts) == 1:
                 current_arm_empty = True


        total_cost = 0

        # Heuristic Component 1: Misplaced blocks relative to goal base
        # Iterate over blocks that have a specified goal position
        for obj, target_base in self.goal_below.items():
            current_base = current_below.get(obj) # Get current base, None if block is not in state

            # If the block is not currently on its target base (or table), increment cost.
            # This covers cases where it's on the wrong block, on the table when it shouldn't be,
            # on a block when it should be on the table, or being held.
            if current_base != target_base:
                 total_cost += 1

        # Heuristic Component 2: Blocks that should be clear but aren't
        for obj in self.goal_clear:
            if obj not in current_clear:
                total_cost += 1

        # Heuristic Component 3: Arm needs to be empty but isn't
        if self.goal_arm_empty and not current_arm_empty:
             total_cost += 1

        # Ensure heuristic is 0 at goal state:
        # If state is a goal state, all goal facts are in state.
        # - For (on B Y) goal: (on B Y) in state -> current_below[B] == Y == target_base. No cost.
        # - For (on-table B) goal: (on-table B) in state -> current_below[B] == 'table' == target_base. No cost.
        # - For (clear B) goal: (clear B) in state -> B in current_clear. No cost.
        # - For (arm-empty) goal: (arm-empty) in state -> current_arm_empty is True. No cost.
        # If all goal facts are in state, total_cost is 0. Correct.

        # Ensure heuristic is finite for solvable states:
        # The heuristic counts discrete mismatches based on goal predicates.
        # The number of goal predicates is finite. The count is always a finite integer. Correct.

        return total_cost

