from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Summary:
    ----------
    This heuristic estimates the number of actions needed to rearrange blocks from the current state
    to the goal configuration. The estimate is not necessarily admissible but tries to be accurate
    and efficient so as to minimize node expansions during greedy best-first search.

    Assumptions:
    -------------
    - The arm can hold only one block at a time.
    - Blocks can be either on the table or stacked on another block.
    - Actions allowed are pickup, putdown, stack, and unstack.
    - The goal is fully specified as a conjunction of clear predicates, on relations, and on-table predicates.
    - We assume the only moving unit is the arm holding exactly one block.

    Heuristic Initialization:
    --------------------------
    - Extract goal on-relations: for each block, determine its goal support (the block it should be on or None for on-table).
    - Extract the set of blocks mentioned in goals.
    - Store goal clear blocks, goal on-table blocks.
    - No static facts to process in this domain.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    For each block involved in the goal:
      1. Determine where the block currently is (on what block or on-table).
      2. Determine where the block should be according to the goal.
      3. Estimate moves required if the block is not yet at goal:
        - A block is "correctly placed" if it is on its goal support,
          its goal support is either on the table or itself correctly placed,
          and the block is clear.
        - If the block is "correctly placed," no action needed for it.
        - Otherwise, at minimum two actions are needed to place it correctly (unstack or pickup + stack or putdown).
        (This approximates a lower bound on fixing this block's position.)
      4. Additionally, count how many blocks are currently held by the arm (one or zero).
      5. If the arm is holding a block that is not supposed to be held in the goal (goal never holds a block), add at least one action to put it down.

    The heuristic sums the estimated moves for all blocks not correctly placed plus a possible correction if the arm is holding a block.

    This heuristic tries to identify blocks out of place and counts minimal rearrangement actions per block,
    understanding that stacking/unstacking is at minimum 2 actions per misplaced block,
    plus actions related to holding the wrong block.

    The heuristic is zero iff the state satisfies all goal conditions.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static  # unused, but stored if needed later

        # Extract goal info: for each block, track what it should be on in the goal.
        self.goal_on = {}          # block -> supporting block or None if on-table
        self.goal_clear = set()    # blocks that should be clear in goal
        self.goal_on_table = set() # blocks that should be on-table in goal

        for fact in self.goals:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "on":
                b, support = parts[1], parts[2]
                self.goal_on[b] = support
            elif pred == "on-table":
                b = parts[1]
                self.goal_on[b] = None
                self.goal_on_table.add(b)
            elif pred == "clear":
                b = parts[1]
                self.goal_clear.add(b)
            elif pred == "holding":
                # In blocksworld goals, usually should not hold a block at goal.
                # If it occurs, we track here for completeness.
                pass

        # Collect all blocks mentioned in the goal
        self.goal_blocks = set(self.goal_on.keys()) | self.goal_clear | self.goal_on_table

    def __call__(self, node):
        state = node.state

        # Quick check: if goal satisfied, heuristic = 0
        if self.goals <= state:
            return 0

        # Helper: find supports and clear for all blocks in current state
        # on[x] = y means block x currently on y, or None if on table
        current_on = {}
        current_clear = set()
        holding_block = None
        arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]

            if pred == "on":
                b, support = parts[1], parts[2]
                current_on[b] = support
            elif pred == "on-table":
                b = parts[1]
                current_on[b] = None
            elif pred == "clear":
                b = parts[1]
                current_clear.add(b)
            elif pred == "holding":
                holding_block = parts[1]
            elif pred == "arm-empty":
                arm_empty = True

        # Function to check if a block is correctly placed according to heuristic criteria:
        # - On correct support (goal)
        # - If on another block, that block is correctly placed recursively
        # - Block is clear
        # We memoize this to avoid recursion overhead.
        memo_correct = {}

        def is_correct(b):
            if b not in self.goal_on:
                # No goal info for this block: assume it's irrelevant for heuristic
                return True
            if b not in current_on:
                # Block not on any support? Should not happen, treat as incorrect
                return False
            if b not in current_clear:
                return False  # not clear means block on top of it, so can't be in final position
            
            goal_support = self.goal_on[b]  # None if on table goal
            curr_support = current_on[b]

            if curr_support != goal_support:
                return False

            if goal_support is None:
                # on table goal reached
                return True

            # Recursively check the support block correctness
            if goal_support not in memo_correct:
                memo_correct[goal_support] = is_correct(goal_support)

            return memo_correct[goal_support]

        total_cost = 0

        # For all blocks relevant for goal, estimate moves needed if not correctly placed
        for b in self.goal_blocks:
            if b not in current_on:
                # no info about this block in state? Penalize by 2 actions as heuristic
                total_cost += 2
                continue

            if not is_correct(b):
                # At least 2 actions to fix this block: (pick/unstack + stack/putdown)
                total_cost += 2

        # If arm holds a block but goal never requires holding (typical for blocksworld),
        # then count an additional action to put it down.
        if holding_block is not None:
            # In the goals shown, holding is never part of goal,
            # so if holding something, need at least one putdown or stack action.
            total_cost += 1

        return total_cost
