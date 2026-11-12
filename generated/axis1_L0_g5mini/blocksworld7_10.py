from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the classical Blocksworld (STRIPS) domain.
    The heuristic estimates the number of block-moving actions required to
    satisfy goal (on / on-table / clear / arm-empty) facts from a given state.

    Assumptions
    -----------
    - Only the predicates (on ?x ?y), (on-table ?x), (clear ?x), (holding ?x)
      and (arm-empty) appear in states (as in the provided domain).
    - Exactly one block can be directly on another block (standard Blocksworld).
    - The heuristic is allowed to be non-admissible (it may overestimate).
    - Static facts are present in task.static but Blocksworld has no useful static
      facts; nonetheless they are stored for completeness.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - A mapping (goal_support) from every block mentioned in (on / on-table)
      goal facts to its required support: either another block-name or 'table'.
    - A set of blocks that are required to be clear in the goals (goal_clear).
    - Whether the goal requires the arm to be empty (goal_arm_empty).
    - The static facts (task.static) are stored (even if empty).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic performs the following steps:

    1. Immediate goal check:
       - If all goal facts hold in the state, return 0.

    2. Build a compact representation of the current state:
       - under[block] = immediate support of block (another block name or 'table').
         If a block is being carried (holding), it will not appear in under.
       - holding_block = the block currently held (if any).
       - collect the set of blocks present in the state and goals.

    3. Compute blocks_above[b] for each block b:
       - Number of blocks stacked (transitively) on top of b in the current state.
         For example, if a is on b and c is on a, then blocks_above[b] == 2.

    4. For each goal relation (on b s) or (on-table b):
       - If the goal fact is already true in the state, it contributes 0.
       - Otherwise:
         - To make b movable, all blocks on top of b must be removed:
           cost += blocks_above[b] (each requires an unstack).
         - To move b into position requires one action:
           - If the planner is already holding b, that is counted as 1 (stack/putdown).
           - Otherwise, we count 1 for picking/moving-and-placing b.
         - If the goal requires b to be on another block s (not table), then
           s must be clear before stacking b onto it. We pessimistically add
           blocks_above[s] to the cost (to clear the target).
       - This is a conservative (possibly over-counting) estimate of needed
         block removals and placements but is informative for greedy search.

    5. For goal clear(x) facts that are not true, add blocks_above[x] to the cost
       (every block above x needs to be removed).

    6. For a goal (arm-empty) not satisfied in the state, add 1 (we need to put
       down the currently held block).

    Notes on Design Choices
    -----------------------
    - The heuristic intentionally double-counts some removals (e.g., a block that
      must be removed to free multiple targets may be counted multiple times).
      This is acceptable because admissibility is not required and such
      overestimates can still strongly guide greedy best-first search.
    - The heuristic is efficient: it only inspects facts once and computes
      transitive "above" counts in O(n^2) worst-case for n blocks.
    """

    def __init__(self, task):
        # Save goal facts and static facts
        self.goals = task.goals
        self.static = task.static

        # Map each block that appears in an on/on-table goal to its required support.
        # support is either another block name or the string 'table'.
        self.goal_support = {}
        # Track blocks that must be clear in the goal.
        self.goal_clear = set()
        # Track if goal requires arm-empty
        self.goal_arm_empty = "(arm-empty)" in self.goals

        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]
            if pred == "on":
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
            elif pred == "on-table":
                block = parts[1]
                self.goal_support[block] = "table"
            elif pred == "clear":
                self.goal_clear.add(parts[1])
            # other goal predicates (holding, arm-empty) are handled similarly if needed

    def __call__(self, node):
        state = node.state

        # If all goal facts are already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Build current "under" mapping: under[block] = what block (or 'table') it is on.
        under = {}  # block -> support ('table' or another block)
        holding_block = None

        # collect blocks appearing in state
        blocks = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                top, bottom = parts[1], parts[2]
                under[top] = bottom
                blocks.add(top)
                blocks.add(bottom)
            elif pred == "on-table":
                top = parts[1]
                under[top] = "table"
                blocks.add(top)
            elif pred == "holding":
                holding_block = parts[1]
                blocks.add(holding_block)
            elif pred == "clear":
                blocks.add(parts[1])
            # arm-empty and other predicates do not add new block names

        # Also include blocks mentioned in goal support mapping (exclude 'table')
        for b, sup in self.goal_support.items():
            blocks.add(b)
            if sup != "table":
                blocks.add(sup)

        # Remove 'table' if somehow present
        blocks.discard("table")

        # Helper to check whether x is (transitively) above b in current state
        def is_above(x, b):
            cur = under.get(x, None)
            while cur is not None:
                if cur == b:
                    return True
                if cur == "table":
                    return False
                cur = under.get(cur, None)
            return False

        # Precompute blocks_above counts (how many blocks are stacked on top of each block)
        blocks_above = {}
        block_list = list(blocks)
        for b in block_list:
            cnt = 0
            for x in block_list:
                if x == b:
                    continue
                if is_above(x, b):
                    cnt += 1
            blocks_above[b] = cnt

        heuristic = 0

        # Cost contribution for each on/on-table goal
        for b, target in self.goal_support.items():
            # Build the exact goal fact string to check current satisfaction
            if target == "table":
                desired_fact = f"(on-table {b})"
            else:
                desired_fact = f"(on {b} {target})"

            if desired_fact in state:
                continue  # already satisfied

            # Cost to free b: unstack every block above b (blocks_above[b])
            # If planner currently holds b, then we don't need to unstack it nor pick it;
            # only one action (stack or putdown) is required to place it.
            if holding_block == b:
                cost = 1  # one action to place b correctly (stack/putdown)
            else:
                # unstack each blocking block + one action to move/place b
                cost = blocks_above.get(b, 0) + 1

            # If target is a block, we pessimistically add cost to clear that target:
            # every block on top of the target must be removed before stacking b.
            if target != "table":
                cost += blocks_above.get(target, 0)

            heuristic += cost

        # Cost contribution for clear goals that are not satisfied
        for x in self.goal_clear:
            desired_fact = f"(clear {x})"
            if desired_fact not in state:
                heuristic += blocks_above.get(x, 0)

        # If goal requires arm-empty but currently the arm holds a block, add 1
        if self.goal_arm_empty and "(arm-empty)" not in state:
            heuristic += 1

        # Ensure integer result
        return int(heuristic)