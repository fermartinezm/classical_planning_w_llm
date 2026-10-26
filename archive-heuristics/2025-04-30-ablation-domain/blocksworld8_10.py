from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

# Helper function to check if a fact matches a pattern (not strictly needed for this heuristic but good practice)
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#
#     - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
#     - `args`: The expected pattern (wildcards `*` allowed).
#     - Returns `True` if the fact matches the pattern, else `False`.
#     """
#     parts = get_parts(fact)
#     if len(parts) != len(args):
#         return False
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting three types of discrepancies between the current state and the goal:
    1. Blocks that are not on their correct base (table or another block) according to the goal.
    2. Blocks that are currently stacked on top of another block, but are not supposed
       to be directly on that specific block in the goal state.
    3. The arm is holding a block (as the goal state typically requires the arm to be empty).

    # Assumptions
    - The goal state specifies the desired position (on another block or on the table)
      for a subset of blocks using `(on ?x ?y)` and `(on-table ?x)` predicates.
    - Blocks not explicitly given an `on` or `on-table` position in the goal are assumed
      to belong on the table in the goal state.
    - Standard Blocksworld actions (pick-up, put-down, stack, unstack) with unit cost.
    - The goal state includes `(arm-empty)`. `(clear ?x)` goals are assumed to be
      satisfied if the stack structure below `?x` is correct and nothing is on `?x`.

    # Heuristic Initialization
    - Parses the goal facts to build a map `goal_support` indicating what each block
      explicitly mentioned in an `on` or `on-table` goal fact should be directly on top of
      in the goal state.
    - Identifies all blocks present in either the initial state or the goal state.
    - For blocks not explicitly mentioned in `on` or `on-table` goal facts, their
      goal support is implicitly set to 'table' when the heuristic is computed.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state facts to build:
       - `current_support`: A map indicating what each block is currently directly on top of
         (another block, 'table', or 'holding').
       - `current_on_pairs`: A set of tuples `(T, B)` for every fact `(on T B)` in the state.
       - `arm_is_holding`: A boolean indicating if the arm is holding any block.
    2. Initialize the heuristic value `h` to 0.
    3. **Part 1: Count blocks in the wrong position relative to their base.**
       Iterate through all blocks identified during initialization (`self.all_blocks`). For each block `B`:
       - Determine its goal support (`goal_below`) from the pre-computed `self.goal_support` map,
         defaulting to 'table' if the block is not explicitly in the map.
       - Determine its current support (`current_below`) from the `current_support` map.
         If a block is not found in `current_support` (e.g., not on/on-table/holding),
         it's in an unexpected state; for heuristic purposes, we can treat its current
         support as distinct from any valid goal support. However, in typical blocksworld
         states, every block is either on another block, on the table, or held.
         We rely on `current_support.get(block)` returning None if the block isn't in
         the relevant state facts, and compare this to `goal_below`.
       - If `current_below` is different from `goal_below`, increment `h`.
    4. **Part 2: Count blocks that are on top of another block when they shouldn't be.**
       Iterate through each pair `(T, B)` in `current_on_pairs` (meaning block `T` is currently on block `B`).
       - Determine the goal support for block `T` (`goal_below_T`) from the `self.goal_support` map,
         defaulting to 'table' if `T` is not explicitly in the map.
       - If `goal_below_T` is different from `B` (i.e., block `T` is not supposed to be directly on block `B`
         in the goal state, including the case where `T` should be on the table), increment `h`.
    5. **Part 3: Penalize the arm holding a block.**
       If `arm_is_holding` is true, increment `h` by 1.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying all blocks.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Build goal_support map: block -> block_below or 'table'
        self.goal_support = {}
        # Identify all blocks mentioned in goal predicates
        goal_blocks_set = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty or invalid facts
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, block_below = parts[1], parts[2]
                self.goal_support[block] = block_below
                goal_blocks_set.add(block)
                goal_blocks_set.add(block_below)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
                goal_blocks_set.add(block)
            # Ignore (clear ?) and (arm-empty) goals for goal_support map

        # Identify all blocks present in the initial state
        initial_blocks_set = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             if not parts: continue # Skip empty or invalid facts
             predicate = parts[0]
             # Consider objects in relevant predicates
             if predicate in ["on", "on-table", "holding", "clear"]:
                 for obj in parts[1:]:
                     initial_blocks_set.add(obj)
             # Also consider objects in arm-empty (though no object is mentioned)
             # or other potential predicates if the domain were extended.
             # For standard blocksworld, the above is sufficient.

        # Combine blocks from goal and initial state
        self.all_blocks = list(goal_blocks_set | initial_blocks_set)

        # Note: Blocks not explicitly in self.goal_support are assumed to go on the table.
        # This is handled by the .get(block, 'table') call in __call__.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Build current_support map and current_on_pairs from the state
        current_support = {}
        current_on_pairs = set()
        arm_is_holding = False

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, block_below = parts[1], parts[2]
                current_support[block] = block_below
                current_on_pairs.add((block, block_below))
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                # Assuming only one block can be held at a time
                block = parts[1]
                current_support[block] = 'holding'
                arm_is_holding = True
            # Ignore (clear ?) and (arm-empty) facts for support/on maps

        h = 0

        # Part 1: Count blocks in the wrong position relative to their base.
        # Consider all blocks identified during initialization.
        for block in self.all_blocks:
            # Get goal support, default to 'table' if block not explicitly in goal_support
            goal_below = self.goal_support.get(block, 'table')
            # Get current support. Returns None if block is not found in current_support map.
            current_below = current_support.get(block)

            # If the block's current support is different from its goal support
            # This handles blocks on wrong block, on table when shouldn't be, held when shouldn't be,
            # or not on/on-table/holding when they should be on something/table.
            if current_below != goal_below:
                 h += 1

        # Part 2: Count blocks that are on top of another block when they shouldn't be.
        for block_t, block_b in current_on_pairs:
            # Get goal support for the block on top (T), default to 'table'
            goal_below_t = self.goal_support.get(block_t, 'table')
            # If T is not supposed to be directly on B in the goal state
            if goal_below_t != block_b:
                h += 1

        # Part 3: Penalize the arm holding a block.
        if arm_is_holding:
            h += 1

        # The heuristic should be 0 iff the state is a goal state.
        # If the state is a goal state, all (on) and (on-table) goal facts are true,
        # meaning Part 1 and Part 2 counts will be 0. (arm-empty) goal means Part 3 is 0.
        # If the state is NOT a goal state, at least one goal fact is false.
        # If an (on X Y) or (on-table X) goal is false, Part 1 or Part 2 will be > 0.
        # If an (arm-empty) goal is false, Part 3 will be > 0.
        # If a (clear X) goal is false, it means (on T X) is true for some T.
        # If T is supposed to be on X in the goal, this contradicts (clear X) goal (assuming consistent goals).
        # If T is not supposed to be on X in the goal, Part 2 counts it.
        # So, h=0 iff the state satisfies all (on), (on-table), and (arm-empty) goals,
        # which implies (clear X) for goal top blocks. Thus, h=0 iff it's a goal state.

        return h
