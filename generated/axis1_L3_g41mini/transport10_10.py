from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on-table item10 item6)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (blocksworld variant).

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal conditions
    (mainly arm-empty predicates with given arguments). It analyzes the current state with
    respect to the goal arm-empty facts and the block configuration (on, on-table, clear, holding)
    to estimate how many pickups, stacks, and unstacks are needed.

    Assumptions
    -----------
    - The true goal is always expressed as some set of (arm-empty ?v3 ?v4) facts.
    - Objects are blocks and arm entities interacting with them.
    - The heuristic does not require admissibility but aims to minimize node expansions.
    - The domain actions allow pickup, stack, unstack with given preconditions.
    - The relationship between blocks and arms implies that to achieve arm-empty,
      blocks may need to be moved/stored appropriately.

    Heuristic Initialization
    ------------------------
    - Extract the goal arm-empty predicates and store their (arm, object) pairs.
    - Extract static facts to build a quick-access structure of block relations:
      - which blocks are on which blocks (on),
      - which blocks are on the table with which arm (on-table),
      - which arms are empty on which blocks (arm-empty),
      - which blocks are clear (clear),
      - which blocks are held (holding).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state satisfies all goal arm-empty facts, return 0.

    2. Count how many goal (arm-empty ?v3 ?v4) facts are unsatisfied in the current state.
       Each missing arm-empty likely requires at least one action (putdown or release).

    3. Identify blocks currently held by any arm and those that are not held.
       Blocks held but not matching goal arm-empty conditions indicate necessary puts/stacks.

    4. For each goal arm-empty fact, check if that arm is free in the current state:
       - If the arm is not free but must be empty at that block, it suggests an unstack or putdown is needed.

    5. For blocks that must be on the table at certain arms but are currently obstructed,
       estimate the number of moves to clear their stack (count blocks above).

    6. Sum the above components into a heuristic estimate:
       - Number of missing arm-empty facts (at least one action per missing).
       - Number of currently held blocks that must be put down or stacked.
       - Number of blocks that must be cleared for the arm to be empty at the goal block.

    7. Return the summed estimate as the heuristic value.

    The heuristic trades perfect precision for computational efficiency by focusing mainly on:
    - arm-empty goals unmet,
    - blocks held but needing release,
    - blocks blocking arm-empty conditions.
    """

    def __init__(self, task):
        """Preprocess goal arm-empty conditions and static facts for quick access."""
        self.goals = task.goals

        # Extract goal arm-empty facts: store as set of tuples (arm, block)
        self.goal_arm_empty = set()
        for fact in self.goals:
            p, *args = get_parts(fact)
            if p == "arm-empty" and len(args) == 2:
                self.goal_arm_empty.add((args[0], args[1]))

        # From static facts, extract sets for quick membership checks
        self.static_on = set()
        self.static_on_table = set()
        self.static_arm_empty = set()
        self.static_clear = set()
        self.static_holding = set()

        for fact in task.static:
            p, *args = get_parts(fact)
            if p == "on" and len(args) == 2:
                self.static_on.add((args[0], args[1]))
            elif p == "on-table" and len(args) == 2:
                self.static_on_table.add((args[0], args[1]))
            elif p == "arm-empty" and len(args) == 2:
                self.static_arm_empty.add((args[0], args[1]))
            elif p == "clear" and len(args) == 2:
                self.static_clear.add((args[0], args[1]))
            elif p == "holding" and len(args) == 2:
                self.static_holding.add((args[0], args[1]))

    def __call__(self, node):
        """
        Estimate the approximate number of actions to reach the goal from the current state.
        """
        state = node.state

        # Extract the following sets from the current state
        on = set()
        on_table = set()
        arm_empty = set()
        clear = set()
        holding = set()

        for fact in state:
            p, *args = get_parts(fact)
            if p == "on" and len(args) == 2:
                on.add((args[0], args[1]))
            elif p == "on-table" and len(args) == 2:
                on_table.add((args[0], args[1]))
            elif p == "arm-empty" and len(args) == 2:
                arm_empty.add((args[0], args[1]))
            elif p == "clear" and len(args) == 2:
                clear.add((args[0], args[1]))
            elif p == "holding" and len(args) == 2:
                holding.add((args[0], args[1]))

        # 1) If the state satisfies all goal arm-empty facts, heuristic is zero
        if self.goal_arm_empty <= arm_empty:
            return 0

        # 2) Count how many goal arm-empty facts are missing
        missing_arm_empty = len(self.goal_arm_empty - arm_empty)

        # 3) Count the number of currently held blocks
        held_blocks = {b for _, b in holding}
        num_held_blocks = len(held_blocks)

        # 4) Estimate needed clears for goal arms that must be empty at blocks
        # If an arm must be empty at block B, but block B is not clear at that arm,
        # then some blocks above B must be moved (each such block adds actions).
        clears_needed = 0
        for (arm, block) in self.goal_arm_empty:
            # Only care about arms that are not empty at the goal block in current state
            if (arm, block) not in arm_empty:
                # Check if block is clear at the arm: if not, estimate the stack height above it
                if (arm, block) not in clear:
                    # Count how many blocks are on top of `block` at `arm`
                    # In the domain, clear is defined over (item-5 item-4), so
                    # approximate by counting blocks stacked on top iteratively:
                    # but here we only have (clear arm block) and (on block1 block2).
                    # We interpret "blocks above" as blocks on top iteratively.

                    # We'll estimate the stack height above block ignoring arms because types are unusual:
                    # We calculate a small BFS starting from the block counting blocks on top.

                    def count_blocks_above(b):
                        count = 0
                        # A simple recursive count of blocks on top
                        blocks_above = {b_top for (b_top, b_under) in on if b_under == b}
                        for b_above in blocks_above:
                            count += 1 + count_blocks_above(b_above)
                        return count

                    blocks_above_count = count_blocks_above(block)
                    # Each blocking block requires at least one unstack and one stack, so 2 actions minimum
                    clears_needed += 2 * blocks_above_count

        # 5) Also, any held blocks not at their arm-empty goal likely require at least one putdown or stack
        held_blocks_to_putdown = 0
        # Get set of arms that should be empty for those blocks
        goal_arms_for_held_blocks = set(a for a, b in self.goal_arm_empty if b in held_blocks)
        for (arm, block) in holding:
            # If holding block b at arm a, but (a,b) not a goal arm-empty, consider putdown
            if (arm, block) not in self.goal_arm_empty:
                held_blocks_to_putdown += 1

        # 6) Sum heuristic components:
        # - missing arm-empty facts (minimum 1 action each)
        # - blocks currently held that need to be put down or stacked
        # - clears needed to make arms empty at the goal block
        heuristic_value = (
            missing_arm_empty +
            held_blocks_to_putdown +
            clears_needed
        )

        # Return at least 1 if goal not reached and heuristic nonzero,
        # to avoid zero in non-goal states.
        return max(heuristic_value, 1)
