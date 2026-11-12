from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def parse_fact(fact):
    """Extract the predicate and arguments from a PDDL fact string."""
    parts = fact[1:-1].split()
    return parts[0], parts[1:]


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The fact as a string (e.g. "(on b1 b2)").
    - `args`: The pattern parts, wildcards '*' allowed.
    Returns True if all parts match the pattern.
    """
    parts = fact[1:-1].split()
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal stack 
    configuration from the current state. It is designed to guide greedy best-first 
    search efficiently, prioritizing states closer in terms of block stacking.

    Assumptions
    -----------
    - Only one arm that can hold a single block at a time.
    - Blocks can be on the table or stacked on other blocks.
    - Actions are pickup, putdown, stack, and unstack as per the domain.
    - The heuristic need not be admissible and may overestimate, but should be informative.
    - The arm is either empty or holding one block at a time.
    - Goal conditions specify a desired stacking of blocks (on and on-table predicates),
      and which blocks must be clear.

    Heuristic Initialization
    ------------------------
    - Parse and store the goal facts, partitioning them into:
      * Goal "on" relationships: which block should be on which block.
      * Goal "on-table" blocks.
      * Goal "clear" blocks.
    - Extract the set of blocks appearing in goals to focus estimation on relevant blocks.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the current arm holding state and which block is held (if any).
    2. For each block appearing in the goal:
       - Determine its current "support": either another block it is on, or the table, or if held.
       - Determine the goal support for that block (on which block or on table).
    3. For the held block, if it differs from the goal support, at least one move is needed to place it correctly.
    4. For blocks not held:
       - If the current support differs from the goal support, that block must be moved.
         Count 1 for moving the block (pickup + putdown/stack = at least 2 actions, but heuristic counts 
         moves conservatively).
       - If the block is currently "buried" under incorrectly stacked blocks, estimate more moves:
         * Blocks that are misplaced above or below goal stacks cause additional unstacking, stacking.
    5. Count the number of blocks that must be repositioned because their current configuration does not match the goal.
    6. Add 1 if the arm is holding a block that still must be placed correctly.
    7. The heuristic approximately counts blocks out of place plus the cost to fix the arm.

    This heuristic tries to minimize expanded nodes by reflecting how far blocks are from their
    goal supports, and whether the arm is busy holding a block that needs to be repositioned.

    Returns
    -------
    An integer estimate of actions needed to reach the goal.
    Zero if the goal is reached (goal facts are all true in the current state).
    """

    def __init__(self, task):
        """
        Extract and store goal predicates, especially block support and clear conditions.
        """
        self.goals = task.goals

        # Parse goal "on" relations => dict block -> block it should be on
        self.goal_on = {}
        self.goal_on_table = set()
        self.goal_clear = set()

        for fact in self.goals:
            pred, args = parse_fact(fact)
            if pred == "on":
                b, under = args
                self.goal_on[b] = under
            elif pred == "on-table":
                (b,) = args
                self.goal_on_table.add(b)
            elif pred == "clear":
                (b,) = args
                self.goal_clear.add(b)

        # All blocks mentioned in the goal on/on-table
        self.goal_blocks = set(self.goal_on.keys()) | set(self.goal_on.values()) | self.goal_on_table | self.goal_clear

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Map block to its current support (block or table or None if held)
        current_support = {}  # block -> supporting block or "table"
        clear_blocks = set()
        holding_block = None
        arm_empty = False

        # Extract information from current state:
        # Track what block is held, which blocks are clear, and what supports each block.
        # Facts can be: (holding b), (on b b2), (on-table b), (clear b), (arm-empty)
        for fact in state:
            pred, args = parse_fact(fact)
            if pred == "holding":
                (b,) = args
                holding_block = b
            elif pred == "arm-empty":
                arm_empty = True
            elif pred == "on":
                b, under = args
                current_support[b] = under
            elif pred == "on-table":
                (b,) = args
                current_support[b] = "table"
            elif pred == "clear":
                (b,) = args
                clear_blocks.add(b)

        # Blocks that appear in the goal but not currently on anything (should not happen but be safe)
        for b in self.goal_blocks:
            if b not in current_support and b != holding_block:
                current_support[b] = None  # unknown or held

        # Heuristic calculation:

        # 1) Count how many blocks are NOT on their goal support.
        #    For blocks held, this is special-cased below.
        blocks_out_of_place = 0
        for b in self.goal_blocks:
            if b == holding_block:
                continue  # Held block dealt with separately
            if b not in current_support:
                # block is neither held nor on something - assume out of place
                blocks_out_of_place += 1
                continue

            current_supp = current_support[b]
            goal_supp = None
            if b in self.goal_on:
                goal_supp = self.goal_on[b]
            elif b in self.goal_on_table:
                goal_supp = "table"
            else:
                # No goal positioning known for this block: maybe clear only
                goal_supp = None

            if current_supp != goal_supp:
                blocks_out_of_place += 1

        # 2) Add cost if holding a block that is not yet placed on goal support.
        holding_cost = 0
        if holding_block is not None:
            goal_supp = None
            if holding_block in self.goal_on:
                goal_supp = self.goal_on[holding_block]
            elif holding_block in self.goal_on_table:
                goal_supp = "table"
            else:
                goal_supp = None

            # The block is held, so current support is None.
            # If it still needs to be placed somewhere, cost at least 1 (putting it down).
            if goal_supp is not None:
                holding_cost = 1

        # 3) Account for misplaced blocks that are blocking goal blocks.
        #    If a block that should be clear is not clear, requires extra unstacking.
        misplaced_blocking_cost = 0
        for b in self.goal_clear:
            if b not in clear_blocks:
                # b should be clear but is not -> at least one block on top must be moved
                misplaced_blocking_cost += 1

        # 4) Combine all estimates.
        # The heuristic sums:
        # - number of blocks out of place (to be moved and stacked on correct support)
        # - cost of correctly placing the held block, if any
        # - extra cost penalizing blocks that should be clear but are not (additional unstacking)
        estimate = blocks_out_of_place + holding_cost + misplaced_blocking_cost

        return estimate
