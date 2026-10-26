import re
# Assuming the heuristic base class is available like this:
# from heuristics.heuristic_base import Heuristic
# We define a dummy base class here for the code to be self-contained.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

from typing import Dict, Set, Optional, Tuple, FrozenSet, List

# Helper to parse PDDL facts (can be simplified if format is guaranteed)
def parse_fact(fact: str) -> Tuple[Optional[str], List[str]]:
    """Parses a PDDL fact string into predicate and arguments."""
    fact = fact.strip()
    # Basic check for parentheses and non-empty content
    if not fact.startswith("(") or not fact.endswith(")") or len(fact) <= 2:
        return None, []
    # Split the content within parentheses
    parts = fact[1:-1].split()
    if not parts:
        return None, []
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class blocksworldHeuristic(Heuristic):
    """
    Heuristic for the Blocksworld domain.

    Summary:
    Estimates the cost to reach the goal by summing costs for each block that is
    not in its final correct position relative to the block below it (or the table),
    considering the entire tower down to the table. It accounts for the cost of
    moving the block itself (pickup/unstack + putdown/stack) and the cost of
    moving any blocks currently stacked on top of it that must also move.

    Assumptions:
    - The goal is specified by `(on blockA blockB)`, `(on-table blockA)`, and potentially `(clear blockA)` predicates.
    - Blocks not mentioned in the goal's `on`/`on-table` predicates do not contribute to the heuristic unless they are part of a tower that needs moving because a block below them is misplaced.
    - Each move cycle costs 2 actions (pickup/unstack + putdown/stack), except when starting with a held block (cost 1 for the required putdown/stack).

    Heuristic Initialization:
    - Parses the goal predicates to build a `goal_on` map (block -> target_support).
    - Extracts all unique object (block) names from the goal and initial state predicates.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state (`node.state`) to determine:
       - `current_on`: Which block is on which block.
       - `current_on_table`: Which blocks are on the table.
       - `held_block`: Which block (if any) is held by the arm.
       - `on_what`: A reverse map (support -> block_above) for efficient upward tower traversal.
    2. Initialize heuristic value `h = 0` and `processed_blocks = set()`.
    3. Define a recursive helper function `_check_correct_position(block, memo)`:
       - Checks if `block` is correctly placed relative to its goal support AND if the support block itself is correctly placed (recursively down to the table).
       - Uses memoization based on `(block, current_state_tuple)` to avoid redundant calculations within a single state evaluation.
       - Returns `True` if the block and the tower below it match the goal configuration down to the table. Returns `False` otherwise.
       - Blocks not explicitly mentioned in the `goal_on` map are considered 'correct' by default in this check, as their final position isn't constrained by an `on` or `on-table` goal. They might still be moved if they obstruct a block below them.
    4. Iterate through all known blocks `B`:
       - If `B` has already been processed (i.e., its cost was included when processing a block below it in a tower), skip it.
       - Call `_check_correct_position(B)` to determine if `B` or any block supporting it is misplaced relative to the goal.
       - If `_check_correct_position(B)` returns `False`:
         - This signifies that `B` and all blocks currently stacked directly or indirectly above it must eventually move.
         - Traverse the tower upwards from `B`. For each block `X` encountered in this upward traversal (including `B` itself) that hasn't been processed yet:
           - Mark `X` as processed.
           - Add cost to `h`: Add 1 if `X` is the `held_block` (cost for putdown/stack), otherwise add 2 (cost for pickup/unstack + putdown/stack).
           - Continue traversing upwards.
    5. After checking all blocks based on goal positions: If there is a `held_block` that was *not* marked as processed during the main loop (this typically happens if the held block has no specific goal position or if its goal position check returned True), it still needs one action (putdown/stack). Add 1 to `h` in this case.
    6. Ensure `h` is 0 if the current state satisfies all goal predicates (this happens naturally if the logic is correct, but explicitly checked via `self.goals <= state`).
    7. Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic.
        - Parses goal predicates into goal_on map.
        - Collects all unique block names.
        """
        self.goals = task.goals
        self.goal_on: Dict[str, str] = {} # block -> block_below / "TABLE"
        self.objects: Set[str] = set() # Store all unique block names

        # Extract goal configuration and objects
        for fact in task.goals:
            pred, args = parse_fact(fact)
            if pred == "on" and len(args) == 2:
                obj, underobj = args
                self.goal_on[obj] = underobj
                self.objects.add(obj)
                self.objects.add(underobj)
            elif pred == "on-table" and len(args) == 1:
                obj = args[0]
                self.goal_on[obj] = "TABLE"
                self.objects.add(obj)
            elif pred == "clear" and len(args) == 1:
                 # Ensure block mentioned in clear goal is known
                 self.objects.add(args[0])

        # Add objects from initial state too, ensuring all are known
        # This assumes objects are arguments to predicates.
        for fact in task.initial_state:
             pred, args = parse_fact(fact)
             # Consider arguments of relevant predicates as objects
             if pred in ["on", "on-table", "clear", "holding"]:
                 for arg in args:
                     # Basic check: Add any non-predicate string as an object
                     # A more robust method would use :objects from PDDL if available
                     self.objects.add(arg)

        # Static facts are not used in this heuristic
        self.static = task.static


    def _get_current_support(self, block: str, current_on: Dict[str, str],
                             current_on_table: Set[str], held_block: Optional[str]) -> Optional[str]:
        """Helper to find what the block is currently supported by."""
        if block == held_block: return "ARM"
        # Check 'on' before 'on_table' as a block can't be both 'on' another block and 'on-table'
        if block in current_on: return current_on[block]
        if block in current_on_table: return "TABLE"
        # Block might exist but be clear and not placed (only 'clear' fact exists)
        # Or it might be an invalid block name passed. Return None.
        return None

    def _check_correct_position(self, block: str, current_on: Dict[str, str],
                                current_on_table: Set[str], held_block: Optional[str],
                                memo: Dict[Tuple, bool]) -> bool:
        """
        Recursive check if block and tower below it are correct relative to goal.
        Uses memoization to avoid re-calculating for the same block in the same state context.
        """
        # Base cases for recursion
        if block == "TABLE": return True
        # If block is None (invalid query) or ARM (cannot be a goal support), it's incorrect.
        if block is None or block == "ARM": return False

        # Memoization key: Use a tuple that uniquely identifies the state relevant to this block's check.
        # Includes the block name and frozensets of state components.
        state_tuple = (
            block,
            frozenset(current_on.items()),
            frozenset(current_on_table),
            held_block
        )
        if state_tuple in memo:
            return memo[state_tuple]

        goal_support = self.goal_on.get(block) # Target block/TABLE or None if not in goal

        # If block B is not mentioned in the goal's on/on-table predicates,
        # it doesn't have a defined "correct" position relative to the goal structure.
        # Consider it "correctly placed" by default for this check.
        if goal_support is None:
            memo[state_tuple] = True
            return True

        current_support = self._get_current_support(block, current_on, current_on_table, held_block)

        # If the current support doesn't match the required goal support, it's incorrect.
        if current_support != goal_support:
            memo[state_tuple] = False
            return False

        # Current support matches goal support. Now, recursively check if the support itself is correctly placed.
        # The block 'block' is only considered correct if its support 'goal_support' is also correctly placed.
        result = self._check_correct_position(goal_support, current_on, current_on_table, held_block, memo)
        memo[state_tuple] = result
        return result

    def __call__(self, node) -> int:
        """
        Calculates the heuristic value for the given state node.
        Represents the estimated number of actions to reach the goal.
        """
        state = node.state
        # Memoization cache specific to this state evaluation
        memo: Dict[Tuple, bool] = {}

        # Check if goal is already reached - heuristic value is 0
        if self.goals <= state:
            return 0

        # --- Parse current state ---
        current_on: Dict[str, str] = {}
        current_on_table: Set[str] = set()
        held_block: Optional[str] = None
        # on_what maps a support (block/table) to the block directly on top of it
        on_what: Dict[str, str] = {} # block_below -> block_above

        for fact in state:
            pred, args = parse_fact(fact)
            if pred == "on" and len(args) == 2:
                # Assume args[0] is the block on top, args[1] is the support
                current_on[args[0]] = args[1]
                on_what[args[1]] = args[0]
            elif pred == "on-table" and len(args) == 1:
                current_on_table.add(args[0])
            elif pred == "holding" and len(args) == 1:
                held_block = args[0]

        # --- Calculate heuristic value ---
        h: int = 0
        # Keep track of blocks whose cost contribution has already been calculated
        processed_blocks: Set[str] = set()

        # Iterate through all known objects (blocks) to check their position
        for block in self.objects:
            if block in processed_blocks:
                continue # Skip if already processed as part of a lower block's tower

            # Check if the block or the tower supporting it is incorrect relative to the goal
            is_correct = self._check_correct_position(block, current_on, current_on_table, held_block, memo)

            if not is_correct:
                # If 'block' or something below it is wrong, then 'block' and everything
                # currently stacked above it must eventually move. Calculate the cost for this tower segment.
                cost_for_tower: int = 0
                # Traverse upwards from 'block'
                curr: Optional[str] = block
                while curr is not None and curr != "TABLE" and curr != "ARM":
                     # Stop if we encounter a block already processed (part of another tower costed earlier)
                     if curr in processed_blocks:
                         break

                     processed_blocks.add(curr)
                     # Calculate the cost to move this block 'curr'
                     # Cost is 1 (putdown/stack) + 1 (pickup/unstack, unless held) = 2
                     move_cost: int = 1 # Base cost for the final putdown/stack action
                     if curr != held_block:
                         move_cost += 1 # Add cost for the initial pickup/unstack action
                     cost_for_tower += move_cost

                     # Move to the block directly above 'curr' for the next iteration
                     curr = on_what.get(curr)

                h += cost_for_tower

        # After checking all blocks based on goal positions:
        # If the arm is holding a block, and that block wasn't processed (meaning it and its
        # potential support tower were considered 'correct' or it has no goal position),
        # we still need one action to put it down or stack it.
        if held_block is not None and held_block not in processed_blocks:
             h += 1 # Cost for the final putdown/stack action

        return h
