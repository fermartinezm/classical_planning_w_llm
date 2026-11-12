from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the classic STRIPS Blocksworld.

    The heuristic estimates the number of actions required to reach the goal
    by reasoning about which blocks must be moved (and which blocks block them).
    The estimate counts block-move operations: each block that must be moved
    is assumed to require a pick/unstack and then a put/stack or putdown (2 actions).
    If the planner is already holding a block that needs to be moved, one pick
    action is saved (estimate reduced by 1). Additionally, simple goal predicates
    such as (clear ...) or (arm-empty) that are not satisfied contribute a small
    fixed penalty to guide the search when those goals are the only remaining ones.

    Assumptions
    -----------
    - The PDDL facts in states are strings like "(on b1 b2)", "(on-table b1)",
      "(clear b1)", "(holding b1)", "(arm-empty)".
    - Goals are a (possibly partial) description of the final stacking
      configuration. They may contain (on X Y), (on-table X), and also
      other predicates like (clear X) or (arm-empty).
    - Moving a block (either via pickup/unstack followed by stack/putdown)
      is counted as 2 actions. This is a simplification that yields a fast,
      informative heuristic (not necessarily admissible).
    - If the arm is already holding a block that must be moved, one pick
      action is already done and the estimated cost is reduced by 1.
    - Static facts are not used by this heuristic; all needed information is
      present in the initial state and goal facts.

    Heuristic Initialization
    ------------------------
    The constructor extracts goals and organizes goal "support" relationships:
    - goal_support: map block -> its desired support ("TABLE" for on-table).
    - goal_stacks: a list of goal stacks, each represented as a list of blocks
      from bottom (on-table) to top. Only stacks that can be inferred from the
      set of (on ...) and (on-table ...) goal facts are constructed.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node (with a state), the heuristic performs the following steps:

    1. Immediate goal check:
       - If all goal facts hold in the current state return 0.

    2. Parse the current state to determine:
       - current_support: map block -> what it rests on ("TABLE" for on-table,
         "HAND" if currently held).
       - under_to_above: the inverse mapping for direct "above" links; used to
         walk up stacks and count / enumerate blocks above a given block.
       - which block (if any) is currently held by the arm.

    3. For each goal stack (bottom-to-top):
       - Construct the current stack that sits on the same bottom block (only
         when the goal bottom is actually on the table in the current state).
       - Compare the current stack and the goal stack from the bottom upward,
         and compute the length of the matching prefix (bottom prefix that is
         already correct).
       - All goal blocks above the matched prefix are "misplaced" and will need
         to be moved into position.

    4. Identify blockers:
       - For every misplaced block, any blocks currently stacked on top of it
         (in the current state) must be removed as well; they are collected
         as "blockers". Blockers are only added when they are not already in
         the misplaced set (to avoid double counting).

    5. Compute a coarse cost:
       - Each block in (misplaced U blockers) is assumed to require two actions
         (pick/unstack + put/stack or putdown) -> 2 actions per block.
       - If the arm currently holds a block that is in that set, reduce the
         cost by 1 (one pick action already done).
       - Additionally, any goal facts that are not satisfied and are NOT of the
         form (on ...) or (on-table ...) (e.g. (clear X), (arm-empty)) add a
         +1 penalty each. This helps to make the heuristic non-zero when the
         only remaining unsatisfied goals are such predicates.

    6. Return the total estimated cost (integer). The heuristic is always finite
       for solvable states and returns 0 exactly on goal states.
    """

    def __init__(self, task):
        # Store all goal facts (frozenset of strings)
        self.goals = task.goals

        # Build goal_support mapping: block -> support (block name or "TABLE")
        self.goal_support = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "on":
                self.goal_support[parts[1]] = parts[2]
            elif parts[0] == "on-table":
                self.goal_support[parts[1]] = "TABLE"

        # Build goal stacks (bottom -> top) from goal_support information.
        # Find bottoms: blocks that have support "TABLE" in the goal.
        self.goal_stacks = []
        # Inverse mapping for goal: support -> block that is on it in goal
        goal_under_to_above = {}
        for block, support in self.goal_support.items():
            goal_under_to_above.setdefault(support, []).append(block)

        # Build stacks starting from each goal bottom that is explicitly on-table.
        # If multiple blocks appear to be on the same support in goals we follow
        # one possible chain; goal descriptions normally encode a unique stack.
        for bottom_block, support in list(self.goal_support.items()):
            if support == "TABLE":
                stack = [bottom_block]
                # Grow upward as long as there is a uniquely specified block
                # that sits on the current top in the goal description.
                while True:
                    top = stack[-1]
                    above_candidates = goal_under_to_above.get(top, [])
                    if len(above_candidates) == 1:
                        stack.append(above_candidates[0])
                    else:
                        # stop if none or ambiguous (ambiguous goal part is not
                        # further expanded).
                        break
                self.goal_stacks.append(stack)

    def __call__(self, node):
        state = node.state

        # Quick goal check: zero when all goal facts are true in the state.
        if self.goals <= state:
            return 0

        # Parse current state to build current_support and under_to_above mappings.
        current_support = {}  # block -> support block name, "TABLE" or "HAND"
        under_to_above = {}  # support -> directly above block (single)
        holding_block = None
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, support = parts[1], parts[2]
                current_support[block] = support
                # record direct above relation: support -> block
                under_to_above[support] = block
            elif pred == "on-table":
                block = parts[1]
                current_support[block] = "TABLE"
                # don't record table->block here because table may have multiple bottoms
            elif pred == "holding":
                block = parts[1]
                holding_block = block
                current_support[block] = "HAND"
            # clear and arm-empty are not needed here for structure building

        misplaced = set()  # blocks that are goal-specified but not in correct place
        # For each goal stack, compute the matched prefix length and collect misplaced
        for goal_stack in self.goal_stacks:
            if not goal_stack:
                continue
            bottom = goal_stack[0]
            # To have any match, bottom must currently be on the table in the current state
            if current_support.get(bottom) != "TABLE":
                # no bottom match at all; entire goal stack must be placed
                misplaced.update(goal_stack)
                continue
            # Build current stack sitting on that bottom by walking under_to_above
            current_stack = [bottom]
            while True:
                top = current_stack[-1]
                if top in under_to_above:
                    current_stack.append(under_to_above[top])
                else:
                    break
            # Compare bottom-up
            match_len = 0
            max_compare = min(len(current_stack), len(goal_stack))
            for i in range(max_compare):
                if current_stack[i] == goal_stack[i]:
                    match_len += 1
                else:
                    break
            # Blocks above the matched prefix in the goal stack are misplaced
            for b in goal_stack[match_len:]:
                misplaced.add(b)

        # Identify blockers: blocks sitting on top of any misplaced block in the current state,
        # but that are not themselves already counted as misplaced.
        blockers = set()

        def collect_blockers_above(block):
            """Collect blocks that are above `block` in the current state (directly or indirectly)."""
            cur = block
            while cur in under_to_above:
                above = under_to_above[cur]
                if above in misplaced:
                    # if the above block is also misplaced, it will be moved as part of misplaced set
                    cur = above
                    continue
                blockers.add(above)
                cur = above

        for b in list(misplaced):
            collect_blockers_above(b)

        blocks_to_move = misplaced | blockers

        # Base cost: assume 2 actions (pick/unstack + put/stack or putdown) per block to move.
        cost = 2 * len(blocks_to_move)

        # If the arm already holds a block that needs to be moved, one pick is saved.
        if holding_block and holding_block in blocks_to_move and cost > 0:
            cost -= 1

        # Account for simple unsatisfied goals that are not represented by (on ...) or (on-table ...)
        # e.g., (clear X), (arm-empty) etc. Add a small penalty for each such unsatisfied predicate.
        extra_unsat = 0
        for g in self.goals:
            if g not in state:
                parts = get_parts(g)
                if parts[0] not in ("on", "on-table"):
                    extra_unsat += 1
        cost += extra_unsat

        # Ensure non-negative integer
        if cost < 0:
            cost = 0
        return int(cost)