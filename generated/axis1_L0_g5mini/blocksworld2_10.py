from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract parts of a PDDL fact string by removing the outer parentheses
    and splitting on whitespace.

    Example:
      "(on b1 b2)" -> ["on", "b1", "b2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the Blocksworld domain that estimates the
    number of actions required to reach the goal by comparing the current
    block arrangement to the goal arrangement (the goal "stacks"). The
    heuristic counts how many blocks are not yet in their correct position
    in the goal stacks (including blocks that must be removed because they
    are on top of a block that should be clear) and converts this count into
    an estimated number of actions. The heuristic is intentionally
    non-admissible but designed to be informative and cheap to compute.

    Assumptions
    -----------
    - Actions have unit cost and include:
        * pickup (from table), putdown (to table), unstack (from another block),
          and stack (onto another block). From the STRIPS operators in the
          domain, moving a block typically requires ~2 actions (pickup/unstack
          + stack/putdown). If a block is already held, stacking it costs 1.
    - The goal specification uses only predicates from the domain: on, on-table,
      clear and possibly arm-empty. The heuristic focuses primarily on placing
      blocks according to on/on-table goals; explicit clear and arm-empty goals
      are handled implicitly by how blocks must be moved.
    - The PDDL facts in "node.state" are strings like "(on b1 b2)" and we
      extract parts by removing surrounding parentheses.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Read the task goals and static facts (static facts are unused for
      Blocksworld but are stored for completeness).
    - Build a mapping `goal_parent` from each block to its goal parent:
        - If the goal contains "(on X Y)" then goal_parent[X] = Y
        - If the goal contains "(on-table X)" then goal_parent[X] = "TABLE"
    - Build a `child_of` mapping that maps a parent (block or "TABLE") to the
      child block that should be directly on it in the goal. This lets us
      reconstruct goal stacks bottom-to-top.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the current state already satisfies the goals (task.goal_reached),
       return 0.

    2. Parse the current state and build `current_parent`:
       - For "(on X Y)" set current_parent[X] = Y
       - For "(on-table X)" set current_parent[X] = "TABLE"
       - For "(holding X)" set current_parent[X] = "HAND"
       Blocks that do not appear in the current facts (rare) are treated as
       not being in the correct position.

    3. Reconstruct the goal stacks:
       - Identify bottoms as blocks whose goal parent is "TABLE".
       - Using `child_of`, build each goal stack as a list bottom -> top.

    4. For each goal stack, walk from bottom to the top and count how many
       contiguous blocks are already correctly placed in the current state:
       - A bottom block is correct if current_parent[bottom] == "TABLE".
       - For the next block on top, it is correct only if current_parent[next]
         == previous_block, etc.
       - Stop at the first mismatch in a stack. The number of matched blocks
         across all goal stacks is `correct_count`.

    5. Let `total_goal_blocks` be the number of distinct blocks that appear in
       the goal (either as subjects of on/on-table or as parents in on-goals).
       The number of misplaced blocks is `misplaced = total_goal_blocks - correct_count`.

    6. Convert misplaced blocks to an action estimate:
       - We use a simple conversion: most misplaced blocks require two actions
         (pickup/unstack + stack/putdown), so we start with `2 * misplaced`.
       - If a block is currently held (current_parent[block] == "HAND") and
         it is counted as misplaced, we reduce the cost by 1 because we save
         the pickup/unstack action (stacking while holding costs only 1 action).
       - The result is clipped to be non-negative and returned as an integer.

    Rationale
    ---------
    - This heuristic captures both: blocks that are in the wrong place and
      blocks that are obstructing others (since blocks on top of places that
      should be clear will be counted as misplaced).
    - It is cheap to compute (linear in number of facts) and provides a
      sensible estimate of remaining work in terms of actions.
    - It returns 0 exactly when the planner's goal check is satisfied.
    """

    def __init__(self, task):
        # Store goal facts and static facts.
        self.goals = task.goals
        self.static = task.static

        # Map each block to its desired parent in the goal ("TABLE" if on-table)
        # Example: goal_parent['b2'] = 'b5' or 'TABLE'
        self.goal_parent = {}
        # Map parent -> child according to the goal (parent can be 'TABLE' or a block).
        self.child_of = {}

        # Also collect all blocks that appear in the goal (either as child or parent)
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]
            if pred == "on" and len(parts) >= 3:
                child, parent = parts[1], parts[2]
                self.goal_parent[child] = parent
                self.child_of[parent] = child
                self.goal_blocks.add(child)
                self.goal_blocks.add(parent)
            elif pred == "on-table" and len(parts) >= 2:
                child = parts[1]
                self.goal_parent[child] = "TABLE"
                # parent is TABLE, record child_of for TABLE if needed; there may
                # be multiple table bottoms -- child_of will be updated for each.
                # To build stacks we will start from all children whose parent is TABLE.
                # but child_of mapping will only keep the last one per parent if multiple;
                # we do not rely solely on child_of to detect all children -- instead
                # we'll build stacks by finding parent relationships below.
                # Still store a mapping for TABLE if this child is the bottom of a single stack.
                # Note: the domain's goal is assumed to be sensible (no multiple children on same parent).
                self.child_of.setdefault("TABLE", child)
                self.goal_blocks.add(child)
            elif pred == "clear" and len(parts) >= 2:
                # clear goals do not directly affect parent-child maps, but ensure block included
                self.goal_blocks.add(parts[1])
            elif pred == "holding" and len(parts) >= 2:
                # If the goal demands holding a block, include it
                self.goal_blocks.add(parts[1])
            elif pred == "arm-empty":
                # Nothing to add to goal_blocks
                pass

        # Attempt to build explicit stacks (bottom->top) using goal_parent.
        # Find all bottoms (blocks whose goal_parent == "TABLE").
        self.goal_stacks = []
        bottoms = [b for b, p in self.goal_parent.items() if p == "TABLE"]
        # There could be bottoms that are parents in goal_parent but not keys (if they are only parents),
        # ensure we consider those in building stacks by scanning all goal_blocks for which goal_parent is TABLE.
        # Build stacks by repeatedly finding the child that has the current as parent.
        # First build reverse mapping parent->child(s) for robust stack construction.
        rev = {}
        for child, parent in self.goal_parent.items():
            rev.setdefault(parent, []).append(child)
        # For each bottom, follow a chain while there is exactly one child for the parent.
        visited = set()
        for bottom in bottoms:
            stack = [bottom]
            visited.add(bottom)
            cur = bottom
            # While there is exactly one child under cur in goal, continue the chain.
            while True:
                children = rev.get(cur, [])
                # choose the child that continues the single chain; if multiple children, stop.
                if len(children) != 1:
                    break
                nxt = children[0]
                if nxt in visited:
                    break
                stack.append(nxt)
                visited.add(nxt)
                cur = nxt
            self.goal_stacks.append(stack)

        # For any goal block not yet included in a stack, create a singleton stack entry.
        for b in self.goal_blocks:
            if b not in visited:
                # singleton stack: it might have a parent that is not TABLE; we tolerate this.
                # But ensure we add it so it will be counted when misplaced.
                self.goal_stacks.append([b])
                visited.add(b)

    def __call__(self, node):
        state = node.state

        # If already goal, return 0.
        # (Task.goal_reached isn't available here, but self.goals is set and the planner uses that check.)
        # To ensure consistency with the planner, use set inclusion.
        if self.goals <= state:
            return 0

        # Build current_parent mapping for blocks:
        # current_parent[block] = 'TABLE' | parent_block | 'HAND' (if holding)
        current_parent = {}
        # Track whether something is being held and which block it is.
        held_blocks = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on" and len(parts) >= 3:
                child, parent = parts[1], parts[2]
                current_parent[child] = parent
            elif pred == "on-table" and len(parts) >= 2:
                child = parts[1]
                current_parent[child] = "TABLE"
            elif pred == "holding" and len(parts) >= 2:
                child = parts[1]
                current_parent[child] = "HAND"
                held_blocks.add(child)
            # clear and arm-empty facts are ignored for parent mapping

        # Ensure all goal blocks have an entry in current_parent (if not, mark as unknown -> not correct)
        for b in self.goal_blocks:
            if b not in current_parent:
                # It's safer to treat it as not in place (mark as some invalid parent).
                current_parent[b] = "UNKNOWN"

        # Count how many blocks are already in correct positions from bottoms upward.
        correct_count = 0
        counted = set()  # blocks already counted as correct

        # We will use the explicit goal stacks we built in __init__.
        for stack in self.goal_stacks:
            # For each stack (bottom->top), check contiguous correctness
            prev_parent = None
            match_ok = True
            for i, block in enumerate(stack):
                # desired parent for this block
                desired_parent = self.goal_parent.get(block, "TABLE")
                # current parent of this block
                cur_parent = current_parent.get(block, "UNKNOWN")
                # A bottom block is correct only if it's on the table
                if i == 0:
                    if cur_parent == "TABLE" and desired_parent == "TABLE":
                        correct_count += 1
                        counted.add(block)
                    else:
                        match_ok = False
                        break
                else:
                    # desired parent is previous block in the stack
                    needed_parent = stack[i - 1]
                    if cur_parent == needed_parent:
                        correct_count += 1
                        counted.add(block)
                    else:
                        match_ok = False
                        break
            # continue with other stacks

        total_goal_blocks = len(self.goal_blocks)
        misplaced = total_goal_blocks - correct_count

        # Base estimate: assume ~2 actions per misplaced block (pickup/unstack + stack/putdown)
        estimate = 2 * misplaced

        # If a misplaced block is currently held, we can save one action (no pickup/unstack).
        held_savings = 0
        for hb in held_blocks:
            # If held block is part of goal_blocks and is not counted as already correct -> it's misplaced
            if hb in self.goal_blocks and hb not in counted:
                held_savings += 1

        estimate -= held_savings

        # Ensure non-negative and integer.
        if estimate < 0:
            estimate = 0
        return int(estimate)