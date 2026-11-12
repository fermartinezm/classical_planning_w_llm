from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A fast domain-dependent heuristic for the classical Blocksworld STRIPS domain.
    The heuristic estimates the number of actions required to reach the goal by
    counting how many goal-specified block placements are not yet in place and
    approximating the cost to move each misplaced block. It exploits the fact
    that moving a block typically requires a pickup/unstack (1 action) and a
    stack/putdown (1 action). If the planner is already holding a block that
    must be moved, we discount the pickup/unstack action.

    Assumptions
    -----------
    - Goals are stated using "on" and "on-table" predicates (possibly a subset
      of all blocks). Only those goal predicates are considered required.
    - Blocks not mentioned in the goal do not need to be placed specifically,
      except insofar as they block goal blocks; blocking effects are captured
      by insisting on matching bottom-up sequences of goal stacks.
    - There is at most one block being held (as per the domain).
    - The heuristic does not need to be admissible; it prioritizes informative
      estimates while being cheap to compute.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
    - Stores the goal facts.
    - Extracts the partial ordering of blocks from goal facts (a mapping from a
      block to what it should be on in the goal: either another block or 'table').
    - Builds goal stacks (lists of blocks bottom -> top) for every goal base
      (blocks that should be on the table in the goal).
    - Keeps static facts (if any) for potential future use (none required here).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic performs:
    1. If all goal facts are already satisfied in the state, return 0.

    2. Parse the current state and extract:
       - For every block that is "on" another block or "on-table" what it is
         currently on (a mapping `below[block] = support` where support is a
         block name or the string 'table').
       - The inverse mapping `above[support] = block` for quick stack traversal.
       - Which block (if any) is currently being held.

    3. Build current stacks: for each block resting on the table (below == 'table')
       follow the `above` pointers to obtain the stack as a list bottom -> top.

    4. For each goal stack (built from the goal facts during initialization),
       find the corresponding current stack that contains the same goal base
       block (if any). Compute how many blocks match in sequence from the
       bottom upwards. The idea is that a contiguous correct bottom-up suffix
       of a goal stack does not require moving.

    5. Let total_goal_blocks be the number of distinct blocks that appear in
       the goal's "on" or "on-table" predicates. Let matched_blocks be the sum
       of matching bottom-up counts over all goal stacks. Then misplaced_blocks
       = total_goal_blocks - matched_blocks.

    6. Estimate action cost:
       - Conservatively assume each misplaced block requires two actions:
         a pickup/unstack and a stack/putdown -> cost 2 per misplaced block.
       - If the planner is already holding a block that is counted among the
         misplaced blocks, it already paid the pickup/unstack, so subtract 1
         (we can directly stack it where needed).
       - Finally ensure the heuristic is never negative.

    Rationale
    ---------
    The heuristic is cheap (linear in the number of facts) and captures two
    important guiding signals:
    - How many goal-specified placements are already correct in a bottom-up
      sense (these reduce necessary work dramatically).
    - A simple but effective estimate of the cost to move each misplaced block,
      with a small bonus if the block is already in-hand.
    This tends to produce informative estimates that reduce the number of
    expanded nodes for greedy best-first search while remaining fast to compute.
    """

    def __init__(self, task):
        """Extract goal structures and static facts from the planning task."""
        # Store the exact goal facts (a set of fact strings).
        self.goals = set(task.goals)
        # Static facts (not used here but kept for completeness).
        self.static = set(task.static)

        # Map each block that appears in goals to what it should be on:
        # goal_below[block] = either another block name or 'table'
        self.goal_below = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "on" and len(parts) == 3:
                block, under = parts[1], parts[2]
                self.goal_below[block] = under
            elif pred == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_below[block] = "table"

        # Build inverse mapping goal_above so we can construct goal stacks
        # bottom -> top. goal_above[support] = block that should be directly above support
        self.goal_above = {}
        for block, under in self.goal_below.items():
            # only register if under is a block (or 'table' which we treat specially)
            self.goal_above.setdefault(under, []).append(block)

        # Build goal stacks: find all blocks that should be on the table in the goal
        # and for each follow upward (there may be at most one block above each
        # block in well-formed goals; if multiple are present we build all branches).
        self.goal_stacks = []  # list of lists bottom->top
        # Roots are blocks that goal says should be on the table.
        roots = [b for b, under in self.goal_below.items() if under == "table"]
        # In some formulations a goal might omit explicit on-table facts. For robustness,
        # include blocks that are not listed as being on top of any block as potential roots.
        all_goal_blocks = set(self.goal_below.keys())
        blocks_with_goal_parent = {under for under in self.goal_below.values() if under != "table"}
        # Add any goal block that doesn't appear as a child (no one under it in goal) as root candidate.
        candidate_roots = set(roots) | (all_goal_blocks - blocks_with_goal_parent)
        visited = set()

        # We build stacks by walking upward deterministically if possible; if the goal
        # description contains branching (ill-formed for typical blocksworld goals),
        # we create multiple stacks for each branch.
        for root in candidate_roots:
            if self.goal_below.get(root, None) != "table":
                # Only consider true bases (on-table) as canonical roots; if root isn't on-table
                # skip to maintain conventional stack interpretation.
                continue
            # Depth-first expansion for possible branches (rare in normal goals).
            stack_paths = [[root]]
            while stack_paths:
                path = stack_paths.pop()
                top = path[-1]
                visited.add(top)
                # children that should be directly above "top" in the goal
                children = self.goal_above.get(top, [])
                if not children:
                    self.goal_stacks.append(path)
                else:
                    for c in children:
                        # avoid cycles in malformed goals
                        if c in path:
                            continue
                        stack_paths.append(path + [c])

        # Final set of goal-related blocks for convenience
        self.goal_blocks = set(self.goal_below.keys())

    def __call__(self, node):
        """Compute the heuristic estimate (an integer >= 0)."""
        state = node.state

        # 1) If all goal facts are already true, heuristic is zero.
        if self.goals <= state:
            return 0

        # 2) Parse current state to obtain below/above relations and held block.
        below = {}  # below[block] = support (block name or 'table')
        above = {}  # above[support] = block directly on top of support
        held_block = None

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "on" and len(parts) == 3:
                a, b = parts[1], parts[2]
                below[a] = b
                above[b] = a
            elif pred == "on-table" and len(parts) == 2:
                a = parts[1]
                below[a] = "table"
            elif pred == "holding" and len(parts) == 2:
                held_block = parts[1]

        # 3) Build current stacks bottom -> top for blocks that are on the table.
        current_stacks = []  # list of lists bottom->top
        # find table bases
        bases = [b for b, under in below.items() if under == "table"]
        for base in bases:
            stack = [base]
            top = base
            while top in above:
                top = above[top]
                stack.append(top)
            current_stacks.append(stack)

        # Helper: find current stack (list) that contains a given block
        block_to_stack = {}
        for stack in current_stacks:
            for b in stack:
                block_to_stack[b] = stack

        # 4) For each goal stack, compute how many blocks match bottom-up with the current stack.
        matched_blocks = 0
        # If goal_stacks is empty (e.g., goals only include atomic facts not forming stacks),
        # fall back to pairwise checks of single-block goals.
        if self.goal_stacks:
            for gstack in self.goal_stacks:
                # choose the current stack that contains the goal base (gstack[0]) if any
                base = gstack[0]
                cstack = block_to_stack.get(base, [])
                # If base not in any current stack, cstack will be [] and match length 0.
                # Compare bottom-up.
                match_len = 0
                for i in range(min(len(gstack), len(cstack))):
                    if gstack[i] == cstack[i]:
                        match_len += 1
                    else:
                        break
                matched_blocks += match_len
        else:
            # No structured goal stacks; just count exact matches of on/on-table facts.
            for block in self.goal_blocks:
                desired_under = self.goal_below.get(block)
                actual_under = below.get(block)
                if desired_under == actual_under:
                    matched_blocks += 1

        total_goal_blocks = len(self.goal_blocks)
        misplaced_blocks = max(0, total_goal_blocks - matched_blocks)

        # 5) Cost estimate: 2 actions per misplaced block (unstack/pickup + stack/putdown).
        # If the planner is already holding a block that is among the misplaced blocks,
        # we can subtract 1 because the pickup/unstack step is already done.
        est = misplaced_blocks * 2

        if held_block and held_block in self.goal_blocks:
            # Determine if the held block contributed to the matched_blocks:
            # If it is not currently counted as matched (i.e., it's not placed in the correct
            # bottom-up position), then reduce the estimate by 1.
            # A held block cannot be part of current stacks, so it won't be counted in matched_blocks.
            # Therefore, if it's in the goal set, it is effectively misplaced.
            est -= 1

        # Safety: never return negative heuristic
        if est < 0:
            est = 0

        return int(est)