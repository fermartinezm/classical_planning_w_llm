from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not part of a correctly built goal stack prefix,
    plus costs for unsatisfied clear and arm-empty goal conditions.

    # Assumptions
    - The goal specifies the desired positions (on or on-table) for relevant blocks.
    - The goal may specify clear conditions for the top blocks of stacks.
    - The goal may specify the arm should be empty.
    - Blocks not mentioned in goal predicates do not have a specific required position or state.

    # Heuristic Initialization
    - Extract goal positions (on/on-table) for blocks from goal facts.
    - Identify blocks that must be clear in the goal.
    - Check if arm-empty is a goal condition.
    - Identify all blocks that are part of the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts to build:
       - `goal_on_map`: maps a block U to the block B that should be directly on top of it (if (on B U) is a goal).
       - `goal_on_table_set`: set of blocks that should be on the table in the goal.
       - `goal_clear_set`: set of blocks that should be clear in the goal.
       - `goal_arm_empty`: boolean indicating if arm-empty is a goal.
       - `goal_blocks`: set of all blocks mentioned in any of the above goal conditions.

    2. Parse the current state facts to build:
       - `state_on_map`: maps a block U to the block B currently directly on top of it (if (on B U) is in state).
       - `state_on_table_set`: set of blocks currently on the table.
       - `state_clear_set`: set of blocks currently clear.
       - `state_holding`: the block currently held, or None.
       - `state_arm_empty`: boolean indicating if arm-empty is in state.

    3. Identify blocks that are "correctly positioned in the goal stack". A block B is correctly positioned if:
       - (on-table B) is a goal fact AND (on-table B) is true in the state.
       - OR (on B U) is a goal fact AND (on B U) is true in the state AND U is correctly positioned.
       Compute the set `correctly_positioned_blocks` iteratively.

    4. Initialize heuristic cost `h = 0`.

    5. Add cost for blocks not correctly positioned:
       For each block `b` in the set `goal_blocks`:
         If `b` is not in `correctly_positioned_blocks`:
           `h += 2` (Estimate 2 actions: pickup/unstack + stack/putdown).

    6. Add cost for unsatisfied clear goals:
       For each block `b` in `goal_clear_set`:
         # Check if the goal clear predicate is not in the state facts
         if f'(clear {b})' not in state:
             h += 1 # Estimate 1 action to move the block currently on top

    7. Add cost for unsatisfied arm-empty goal:
       if self.goal_arm_empty and '(arm-empty)' not in state:
            h += 1 # Estimate 1 action: putdown

    8. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # Call the base class constructor, assuming it handles task
        super().__init__(task) # This should store self.goals, self.static, self.objects

        self.goal_on_map = {} # maps base -> block on top
        self.goal_on_table_set = set()
        self.goal_clear_set = set()
        self.goal_arm_empty = False
        self.goal_blocks = set() # Collect all blocks mentioned in goal

        # Parse goal facts (using self.goals provided by super().__init__)
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_on_map[base] = block
                self.goal_blocks.add(block)
                self.goal_blocks.add(base)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_on_table_set.add(block)
                self.goal_blocks.add(block)
            elif predicate == "clear" and len(parts) == 2:
                block = parts[1]
                self.goal_clear_set.add(block)
                self.goal_blocks.add(block)
            elif predicate == "arm-empty" and len(parts) == 1:
                self.goal_arm_empty = True
            # Ignore other goal predicates if any (e.g., type predicates)


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state # state is a frozenset of fact strings

        # Parse current state facts into convenient data structures
        state_on_map = {} # maps base -> block on top
        state_on_table_set = set()
        state_clear_set = set()
        state_holding = None
        state_arm_empty = False

        # Convert state frozenset to a set for faster lookups if needed,
        # but direct lookup in frozenset is also O(1) on average.
        # Let's stick to frozenset for now.

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                state_on_map[base] = block
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                state_on_table_set.add(block)
            elif predicate == "clear" and len(parts) == 2:
                block = parts[1]
                state_clear_set.add(block)
            elif predicate == "holding" and len(parts) == 2:
                state_holding = parts[1]
            elif predicate == "arm-empty" and len(parts) == 1:
                state_arm_empty = True

        # 3. Identify correctly positioned blocks (part of a correct bottom-up goal stack prefix)
        correctly_positioned_blocks = set()
        
        # Add blocks that are correctly on the table according to the goal
        for block in self.goal_on_table_set:
            if block in state_on_table_set:
                 correctly_positioned_blocks.add(block)

        # Iteratively add blocks that are correctly stacked on correctly positioned blocks
        changed = True
        while changed:
            changed = False
            # Iterate through goal 'on' facts (base -> block)
            for base, block in self.goal_on_map.items():
                # Check if the block is currently on the correct base
                if base in state_on_map and state_on_map[base] == block:
                    # Check if the base is correctly positioned
                    if base in correctly_positioned_blocks:
                        # If both are true, the block is correctly positioned
                        if block not in correctly_positioned_blocks:
                            correctly_positioned_blocks.add(block)
                            changed = True

        # 4. Initialize heuristic cost
        h = 0

        # 5. Add cost for blocks not correctly positioned
        # Iterate through blocks that are part of the goal configuration
        for block in self.goal_blocks:
             if block not in correctly_positioned_blocks:
                 h += 2 # Cost to move the block (pickup/unstack + stack/putdown)

        # 6. Add cost for unsatisfied clear goals
        for block in self.goal_clear_set:
            # Check if the goal clear predicate is not in the state facts
            if f'(clear {block})' not in state:
                 h += 1 # Estimate 1 action to move the block currently on top

        # 7. Add cost for unsatisfied arm-empty goal
        if self.goal_arm_empty and '(arm-empty)' not in state:
            h += 1 # Estimate 1 action: putdown

        return h
