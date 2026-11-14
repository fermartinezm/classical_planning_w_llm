from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the token list of a PDDL fact string '(pred a b)' -> ['pred','a','b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the classic Blocksworld STRIPS domain.
    The heuristic estimates the number of actions required to transform the
    current arrangement of blocks into the goal arrangement. It reasons
    about goal "stacks" (chains of on / on-table relations) and counts how
    many blocks in those goal stacks are already in the correct position
    (in the correct parent relation). Blocks that are out of place are
    assumed to require a small constant number of actions to be relocated.
    Special treatment is given to blocks currently held by the arm (they
    may require fewer actions to place correctly).

    Assumptions
    -----------
    - The goal description consists of (on <b> <parent>), (on-table <b>)
      and possibly (clear <b>) facts. We construct goal stacks from the
      on/on-table facts.
    - Every block that appears in the goal on/on-table facts is intended
      to be placed somewhere in the goal stacks. Blocks not mentioned in
      any goal on/on-table are ignored (they do not constrain the goal
      stacks considered by this heuristic).
    - Moving a misplaced block is approximately two actions (pick/unstack
      + stack/putdown). If the planner is already holding that block and
      the desired parent is clear (or the parent is the table), placing
      it costs approximately one action (a single stack/putdown).
    - The heuristic is not required to be admissible; its purpose is to
      provide a good, cheaply computable guide for greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Extract goal facts from the task:
      * Build a mapping goal_parent[block] = parent (a block name or 'table').
      * From that mapping build goal stacks (lists of blocks bottom->top).
    - Collect explicit clear-goals (set of blocks that should be clear
      in the goal). These are considered additionally when they are not
      already satisfied in the current state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node (with current state expressed as PDDL facts):
    1. If the goal is already reached (all goal facts true in the state)
       return 0.
    2. Parse the current state to determine, for each block, its current
       parent:
         - If (on b p) in state -> parent is p
         - If (on-table b) in state -> parent is 'table'
         - If (holding b) in state -> parent is 'hand'
       (All blocks mentioned in the goal mapping are considered; missing
       blocks are treated as "not in place".)
    3. For each goal stack (bottom -> ... -> top) compute how many blocks
       from the bottom are already on their desired parent (a correct
       contiguous prefix). These matched blocks are considered "already
       in place" for that stack.
    4. For the remaining (unmatched) blocks in that goal stack we add
       cost estimates:
         - Default cost 2 per misplaced block (one action to pick/unstack
           it, one to place it).
         - If the block is currently held by the arm (parent == 'hand')
           and its desired parent is the table or is currently clear in
           the state, count 1 instead of 2 (one stack/putdown action).
    5. Additionally, for any explicitly required (clear X) goal that is
       not satisfied in the current state, add a small penalty (1),
       because some block must be moved to make X clear. The moved block
       will often also be counted in the per-stack mismatches, so this
       penalty is deliberately small and acts as a lightweight guide.
    6. Sum the estimated costs for all stacks and clear requirements and
       return the total as the heuristic value.
    """

    def __init__(self, task):
        """
        Extract goal structure from 'task' and prepare data structures used
        during heuristic computation.

        Parameters
        ----------
        task : Task
            The planning task object with .goals and .static attributes.
        """
        # Keep a reference to the goal facts and static facts
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Parse goal facts to build goal_parent: block -> parent ('table' or another block)
        self.goal_parent = {}
        self.goal_clear = set()  # blocks that must be clear in the goal
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "on":
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
            elif parts[0] == "on-table":
                block = parts[1]
                self.goal_parent[block] = "table"
            elif parts[0] == "clear":
                self.goal_clear.add(parts[1])

        # Build inverse mapping to discover children for each parent in the goal
        inv = {}
        for child, parent in self.goal_parent.items():
            inv[parent] = inv.get(parent, []) + [child]

        # Build goal stacks: start from every block whose goal parent is 'table'
        self.goal_stacks = []
        bottoms = [b for b, p in self.goal_parent.items() if p == "table"]
        # Also consider isolated stacks whose bottom is not table but still present:
        # (this is defensive; typical goal formulations use on-table for bottoms)
        # Add any block not appearing as a child in inv (i.e., not under any other)
        # as a potential bottom if it was omitted above.
        child_set = set(self.goal_parent.keys())
        parents_that_are_children = {p for p in self.goal_parent.values() if p != "table"}
        implicit_bottoms = [b for b in child_set if b not in parents_that_are_children and b not in bottoms]
        bottoms.extend(implicit_bottoms)

        # For each bottom, follow upward (there should be at most one child per parent
        # in standard Blocksworld goal encodings, but handle multiple children by
        # stopping when multiple successors exist).
        for bottom in bottoms:
            stack = [bottom]
            while True:
                successors = inv.get(stack[-1], [])
                if len(successors) == 1:
                    stack.append(successors[0])
                else:
                    # stop if zero or multiple successors (ambiguous / multi-stack)
                    break
            self.goal_stacks.append(stack)

        # If no stacks were found but there are goal_parent entries, create single-block stacks
        if not self.goal_stacks and self.goal_parent:
            for b in self.goal_parent:
                self.goal_stacks.append([b])

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Parameters
        ----------
        node : SearchNode
            Node exposing .state (a frozenset of fact strings).

        Returns
        -------
        int
            Estimated number of actions to reach a goal state from node.state.
        """
        state = node.state

        # If goal already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Build current parent mapping: block -> parent ('table', block name, or 'hand')
        current_parent = {}
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                child, parent = parts[1], parts[2]
                current_parent[child] = parent
            elif pred == "on-table":
                child = parts[1]
                current_parent[child] = "table"
            elif pred == "holding":
                child = parts[1]
                current_parent[child] = "hand"

        # For convenience, ability to check if a block is clear in the current state:
        def is_clear(block):
            return f"(clear {block})" in state

        heuristic_cost = 0

        # Estimate cost for misplaced blocks in each goal stack.
        for stack in self.goal_stacks:
            # Count matched prefix from the bottom of the stack
            matched = 0
            for i, blk in enumerate(stack):
                desired_parent = "table" if i == 0 else stack[i - 1]
                if current_parent.get(blk) == desired_parent:
                    matched += 1
                else:
                    break

            # Remaining blocks in this goal stack are considered misplaced
            for j in range(matched, len(stack)):
                blk = stack[j]
                desired_parent = "table" if j == 0 else stack[j - 1]
                loc = current_parent.get(blk)
                # If the planner already holds the block and the desired parent is easy to place onto,
                # placing it may cost only one action.
                if loc == "hand":
                    if desired_parent == "table" or is_clear(desired_parent):
                        heuristic_cost += 1
                    else:
                        heuristic_cost += 2
                else:
                    # Default cost to move a block from an arbitrary place to its goal parent:
                    # one action to pick/unstack and one to place (stack/putdown).
                    heuristic_cost += 2

        # Handle explicit clear goals (if a block must be clear but currently isn't).
        for blk in self.goal_clear:
            if f"(clear {blk})" not in state:
                # Small penalty: at least one move is necessary to clear blk.
                heuristic_cost += 1

        # Defensive: for any goal that requires a block to be on the table but it is not,
        # ensure an additional cost is counted (if not already captured by stacks).
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "on-table":
                b = parts[1]
                if current_parent.get(b) != "table":
                    # If currently held, placing costs 1, otherwise approx 2.
                    if current_parent.get(b) == "hand":
                        heuristic_cost += 1
                    else:
                        heuristic_cost += 2

        # Return integer heuristic value
        return int(heuristic_cost)