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
    Domain-dependent heuristic for the Blocksworld domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to reach the goal state from a given state.
    It is designed to guide a greedy best-first search efficiently in the blocksworld domain,
    where the robot arm manipulates blocks by picking up, putting down, stacking, and unstacking.

    Assumptions
    -----------
    - The arm can hold at most one block at a time.
    - Block predicates in the state are of the forms: 
      (on ?x ?y), (on-table ?x), (clear ?x), (holding ?x), and (arm-empty).
    - The goal imposes conditions of the forms (on ?x ?y), (on-table ?x), and (clear ?x).
    - There are no extraneous objects irrelevant to the goal.
    - The heuristic does not have to be admissible, but should be computationally efficient and informative.

    Heuristic Initialization
    ------------------------
    - Extract goal facts and parse goal "on" relationships to form goal "supporting" maps.
    - Extract the set of all blocks mentioned in the problem from goal and initial facts.
    - Store goal requirements for on(?x ?y), on-table(?x), and clear(?x) predicates for quick checks.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goal facts exactly, return 0.

    2. Identify which block (if any) the arm is currently holding. 
       Actions needed to put it down or stack it onto the correct support will be estimated.

    3. For each block:
       - Determine the current support:
         * If holding(?b): support = None (held in hand)
         * Else if on(?b ?under): support = under
         * Else if on-table(?b): support = "table"
       - Determine the goal support for this block:
         * If goal has (on ?b ?under): goal_support = under
         * Else if goal has (on-table ?b): goal_support = "table"
         * Else no support requirement (rare in goals, but safe to handle).

    4. The heuristic estimates the minimal number of block moves needed:
       - For every block whose current support does not match the goal support:
         * A minimum of two moves are needed (unstack/pickup + putdown/stack).
       - If the arm is holding a block:
         * Count 1 action if the block needs to be placed correctly.
       - Additionally, certain goal clear conditions that are not true add at least 1 action.

    5. The heuristic sums twice the number of misplaced blocks plus the cost for the current held block
       and cost for unsatisfied clear goals. This roughly approximates the number of moves needed and helps
       guide the search efficiently.

    6. The heuristic never returns ∞ and is zero if and only if the goal is reached.

    This heuristic is domain-dependent and aims at balancing informativeness and computational efficiency.
    """

    def __init__(self, task):
        # Extract goal facts
        self.goals = task.goals

        # Parse the goal "on" relationships into a dict: block -> support (block or "table")
        self.goal_support = {}
        self.goal_clear = set()
        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block = parts[1]
                support = parts[2]
                self.goal_support[block] = support
            elif pred == "on-table":
                block = parts[1]
                self.goal_support[block] = "table"
            elif pred == "clear":
                block = parts[1]
                self.goal_clear.add(block)

        # Set of all blocks referenced in goal or static (assuming blocks come from goal_support keys)
        # If needed, could parse from static or initial, but goal blocks are usually sufficient here.
        self.goal_blocks = set(self.goal_support.keys()) | self.goal_clear

    def __call__(self, node):
        state = node.state

        # Shortcut: if goal is satisfied, heuristic is zero
        if self.goals <= state:
            return 0

        # Determine which block (if any) is currently held by the arm
        held_block = None
        for fact in state:
            if fact.startswith("(holding "):
                held_block = get_parts(fact)[1]
                break

        # Build current support mapping: block -> support (block or "table" or None if held)
        current_support = {}

        # If block is held by the arm, support = None
        # Else find "on"/"on-table" predicate for it
        blocks_seen = set()
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "on":
                block, under = parts[1], parts[2]
                current_support[block] = under
                blocks_seen.add(block)
                blocks_seen.add(under)
            elif pred == "on-table":
                block = parts[1]
                current_support[block] = "table"
                blocks_seen.add(block)
            elif pred == "holding":
                block = parts[1]
                current_support[block] = None
                blocks_seen.add(block)

        # Also include any blocks that may appear only in goal but missing here
        all_blocks = blocks_seen | self.goal_blocks

        # Count how many blocks are misplaced (support does not match goal support)
        misplaced_blocks = 0

        for block in all_blocks:
            goal_supp = self.goal_support.get(block, None)  # Can be None if no goal support requirement
            curr_supp = current_support.get(block, None)  # None if held or unknown

            if goal_supp is None:
                # No explicit goal support requirement for this block
                # We assume no cost for support for this block
                continue

            # If current support differs from goal support, count as misplaced
            # Treat None (held) as different from any support other than None
            if curr_supp != goal_supp:
                misplaced_blocks += 1

        # Count how many goal-clear conditions are violated
        # A block needs to be clear in goal but is not clear now
        clear_missing = 0
        current_clear = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "clear", "*")
        }
        for block in self.goal_clear:
            if block not in current_clear:
                clear_missing += 1

        # Cost estimation:
        # Each misplaced block requires at least 2 moves: unstack/pickup + stack/putdown
        # The held block (if misplaced) costs 1 action to put down correctly
        # Each missing clear goal costs at least 1 action to fix (e.g., unstack clears)

        cost = 2 * misplaced_blocks + clear_missing

        # If arm holds a block that is misplaced, it costs 1 action to put it down properly;
        # subtract one placement from misplaced_blocks if held holds a misplaced block to avoid double count
        if held_block is not None:
            goal_supp = self.goal_support.get(held_block, None)
            curr_supp = current_support.get(held_block, None)  # Should be None, since held
            if goal_supp is not None and curr_supp != goal_supp:
                # Held block misplaced: remove one misplaced block penalty (2 moves) replace by 1 move penalty
                cost -= 2
                cost += 1

        # Heuristic must be at least 1 if any misplaced or missing clear exists and cost > 0,
        # but if cost=0 (goal), that was filtered above.
        if cost < 0:
            cost = 0

        return cost
