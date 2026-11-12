from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern with wildcards `*` allowed.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Summary:
    ----------
    This heuristic estimates the number of actions required to transform the current block
    configuration into the goal configuration. It leverages the discrepancies between the 
    current and goal stacks of blocks, counting the minimal rebuild efforts required.
    The heuristic is designed to be efficiently computable and to guide greedy best-first search
    effectively, without requiring admissibility.

    Assumptions:
    ------------
    - The goal specifies the desired stacking relations (on and clear facts).
    - The initial state and goal states contain only valid block configurations.
    - Each block appears exactly once in the stacks (no duplicates).
    - The arm can hold only one block at a time.
    - The state is represented as PDDL facts including (on ?x ?y), (on-table ?x), (clear ?x), etc.
    - The arm-empty and holding status do not significantly affect the heuristic estimation.
    - The goal may specify partial stacks; blocks not mentioned in the goal can be ignored for the heuristic.

    Heuristic Initialization:
    -------------------------
    - Extract all blocks present in the task (from facts).
    - Extract the goal stacking relations: for each block, record what it should be on (another block or table).
    - Organize the goal into goal_pos: mapping block -> its intended support (block or 'table').
    - Precompute a set of all blocks appearing in goal conditions for focused heuristic calculations.
    - Static facts are not used here.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. For each block mentioned in the goal:
       a. Determine its current support (block it is on or 'table').
       b. Compare it to the goal support.
       c. If they differ, this block must be unstacked and restacked correctly, so add cost.

    2. Account for blocks that are stacked in correct order from the bottom upwards:
       a. For each goal stack (from base block on table upward), count how many blocks at the bottom of 
          the stack are in correct position and stable (correct order).
       b. Blocks above the first mismatch contribute to actions because they need to be moved.

    3. Compute minimal moves based on:
       - One pickup/unstack action per block moved,
       - One putdown/stack action per block moved,
       leading to roughly twice the number of misplaced blocks positioned incorrectly.

    4. Additionally, if the arm is holding a block in the current state that is incorrect, add cost 
       to put it down before proceeding.

    5. Return the sum of these costs as the heuristic value.

    This approach approximates the number of moves needed to fix misplaced blocks and rebuild 
    the stacks according to the goal.

    Note: The heuristic returns 0 if the current state matches the goal conditions exactly.
    """

    def __init__(self, task):
        # Extract all blocks from the union of facts and goals by scanning predicates.
        blocks = set()
        for fact in task.facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("on", "on-table", "clear", "holding"):
                # Potentially blocks in positions 1 or 2
                for part in parts[1:]:
                    if part.isalnum():
                        blocks.add(part)
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            if pred in ("on", "on-table", "clear"):
                for part in parts[1:]:
                    if part.isalnum():
                        blocks.add(part)

        self.blocks = blocks

        # Parse goal stacking: map each block to what it should be on ('table' if on-table)
        # This includes also blocks that are only clear in goal but no on/on-table
        self.goal_pos = dict()  # block -> block under it or 'table'
        # We build it from goal facts:
        # Gather (on ?x ?y) and (on-table ?x)
        # For blocks only appearing in (clear ?b) goal but no position, treat as unknown, ignore
        for fact in task.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, under = parts[1], parts[2]
                self.goal_pos[block] = under
            elif pred == "on-table":
                block = parts[1]
                self.goal_pos[block] = "table"

        # For blocks not explicitly positioned in goal (no on or on-table), assume 'table' by default:
        for b in blocks:
            if b not in self.goal_pos:
                self.goal_pos[b] = "table"

    def __call__(self, node):
        state = node.state

        # Helper: get current support of block in state:
        # Returns block under it, or 'table' if on-table
        # If block is held, treat as unsupported (None)
        def current_support(block):
            # Check if block is holding
            if match_in_state(state, "holding", block):
                return None
            # Check "on" facts with block as first argument
            for fact in state:
                if match(fact, "on", block, "*"):
                    parts = get_parts(fact)
                    return parts[2]
            # Check if block on table:
            if match_in_state(state, "on-table", block):
                return "table"
            # Unknown, treat as None
            return None

        # Helper: check if a pattern exists in state (with wildcards)
        def match_in_state(state, pred, *args):
            for fact in state:
                if match(fact, pred, *args):
                    return True
            return False

        # 1) Check whether current arm holding a block:
        held_block = None
        for fact in state:
            if fact.startswith("(holding "):
                parts = get_parts(fact)
                if len(parts) == 2:
                    held_block = parts[1]
                break

        # 2) Build mapping: current support for each block that appears in goal
        # Ignore blocks not in goal since they are irrelevant for heuristic.
        # Note: blocks held have no support.
        current_pos = {}
        for block in self.goal_pos:
            current_pos[block] = current_support(block)

        # 3) Identify bottom blocks for goal stacks:
        # Blocks with goal support == 'table' are roots of stacks.
        # For each root block, traverse the goal stack upward, counting matching suffix in current_pos.
        # The heuristic cost will be twice the number of misplaced blocks (unstack + stack)
        # plus one if arm holding wrong block.

        # Invert goal_pos to get stacks: support -> list of blocks on top
        support_to_blocks = {}
        for b, under in self.goal_pos.items():
            support_to_blocks.setdefault(under, []).append(b)

        heuristic_cost = 0

        # To avoid counting blocks multiple times, keep track of blocks matched in correct position
        matched_blocks = set()

        # Depth-first process each goal stack from base (on-table)
        def count_correct_stack(base):
            correct_count = 0
            current_block = base
            while True:
                # Check if current_block is in correct position:
                if current_pos.get(current_block) == self.goal_pos[current_block]:
                    correct_count += 1
                    # Find block stacked on top of current_block in goal (only one allowed on a block)
                    # If multiple blocks on same support, pick one arbitrarily (there should be at most one in blocksworld)
                    next_blocks = support_to_blocks.get(current_block, [])
                    if len(next_blocks) == 1:
                        current_block = next_blocks[0]
                    else:
                        break
                else:
                    break
            return correct_count

        # For each goal root (blocks on table)
        goal_roots = support_to_blocks.get("table", [])
        total_goal_blocks = len(self.goal_pos)

        correctly_placed = 0
        for root in goal_roots:
            correct_stack_len = count_correct_stack(root)
            # Mark these blocks as matched to avoid double counting
            current_block = root
            for _ in range(correct_stack_len):
                matched_blocks.add(current_block)
                next_blocks = support_to_blocks.get(current_block, [])
                if len(next_blocks) == 1:
                    current_block = next_blocks[0]
                else:
                    break
            correctly_placed += correct_stack_len

        misplaced_blocks = total_goal_blocks - correctly_placed
        # Each misplaced block costs roughly 2 actions (unstack + stack)
        heuristic_cost = 2 * misplaced_blocks

        # If arm is holding a block that is misplaced or even not in goal, add cost to put it down first
        if held_block is not None:
            # If held block not positioned correctly according to goal, add cost for putting it down
            goal_support = self.goal_pos.get(held_block)
            curr_supp = current_pos.get(held_block)
            if curr_supp != goal_support:
                heuristic_cost += 1  # putdown or stack action cost

        # Heuristic is 0 only if the goal is reached: all blocks correctly placed and arm empty or holding correct block
        # If goal reached, heuristic must be zero.
        if heuristic_cost == 0:
            # Verify goal fully reached:
            # The state must contain all goal facts
            if not self.goal_pos:
                return 0
            if node.task.goal_reached(state):
                # If arm is holding something incorrectly, heuristic is not zero.
                if held_block is not None:
                    # But maybe held block is in correct position? That cannot be if holding
                    # Because holding means block not on table
                    return heuristic_cost  # non-zero
                return 0

        return heuristic_cost
