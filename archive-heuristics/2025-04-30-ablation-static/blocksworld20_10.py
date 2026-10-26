from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return [] # Return empty list for malformed or non-string input
    return fact[1:-1].split()

# Helper function for checking if a block is correctly stacked
def is_correctly_stacked(block, state_set, goal_base):
    """
    Checks if a block is in its correct goal position relative to its base,
    and if the entire stack below it (according to the goal) is also correct.
    Uses the goal_base map to traverse the desired stack downwards.
    Assumes valid goal_base structure (terminates at 'table', no cycles).
    """
    current = block
    while current in goal_base: # Loop terminates when base is 'table' or block not in goal_base
        base = goal_base[current]
        if base == 'table':
            # Check if the current block is on the table as required
            return f"(on-table {current})" in state_set
        else:
            # Check if the current block is on the correct base block
            if f"(on {current} {base})" not in state_set:
                return False
            # Move down to check the base block
            current = base
    # If the loop finishes, it means the initial 'block' was not a key in goal_base.
    # This function is intended to check blocks that *are* part of the goal stack structure.
    # If called for a block not in goal_base, it cannot be "correctly stacked" within that structure.
    # Returning False is appropriate in this context.
    return False


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by summing several components related to unsatisfied goal conditions and misplaced blocks:
    1. Blocks that are part of the goal stack structure but are not in their correct recursive position relative to their base.
    2. Blocks that are currently on top of other blocks in a configuration that is not part of the goal stack.
    3. Goal predicates of the form (clear ?x) that are not satisfied.
    4. The goal predicate (arm-empty), if it exists, is not satisfied.

    # Assumptions
    - The goal defines one or more specific stacks of blocks and possibly blocks on the table.
    - The goal structure derived from (on ?x ?y) and (on-table ?x) predicates is acyclic and terminates on the table for all stacks.
    - Goal predicates are well-formed PDDL facts.
    - The goal does not contain contradictory conditions like requiring a block to be on another AND clear simultaneously.

    # Heuristic Initialization
    - Parses the goal predicates to build the desired stack structure (`self.goal_base`) mapping a block to the block it should be directly on top of, or 'table'.
    - Stores the set of goal predicates of the form (on ?x ?y) for efficient lookup (`self.goal_on_facts`).
    - Stores the set of goal predicates of the form (clear ?x) for efficient lookup (`self.goal_clear_facts`).
    - Checks if (arm-empty) is a goal (`self.goal_arm_empty`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Convert the state (frozenset of facts) into a set for efficient lookups.
    2. Calculate H1 (Misplaced blocks in goal stacks):
       - Initialize H1 = 0.
       - For each block `B` that is a key in `self.goal_base` (i.e., part of the goal stack structure):
         - Check if `B` is correctly stacked according to the goal structure and the current state using the `is_correctly_stacked` helper function.
         - If `B` is *not* correctly stacked, increment H1.
    3. Calculate H2 (Obstructing blocks):
       - Initialize H2 = 0.
       - Iterate through all facts `(on X Y)` present in the current state.
       - For each such fact, check if `(on X Y)` is present in `self.goal_on_facts`.
       - If `(on X Y)` is *not* a goal predicate, it means `X` is wrongly on Y. Increment H2.
    4. Calculate H3 (Unsatisfied clear goals):
       - Initialize H3 = 0.
       - For each goal fact `(clear B)` in `self.goal_clear_facts`:
         - Check if `(clear B)` is present in the current state set.
         - If `(clear B)` is *not* in the state, increment H3.
    5. Calculate H4 (Unsatisfied arm-empty goal):
       - Initialize H4 = 0.
       - If `self.goal_arm_empty` is True and "(arm-empty)" not in state_set:
            h4 += 1
    6. The total heuristic value is H1 + H2 + H3 + H4.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal structure and specific goal facts."""
        self.goals = task.goals  # Goal conditions (frozenset of strings)

        # Build the goal stack structure: block -> block_it_should_be_on or 'table'
        self.goal_base = {}
        # Store specific goal fact types for quick lookup
        self.goal_on_facts = set()
        self.goal_clear_facts = set()
        self.goal_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_base[block] = base
                self.goal_on_facts.add(goal)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_base[block] = 'table'
            elif predicate == "clear" and len(parts) == 2:
                self.goal_clear_facts.add(goal)
            elif predicate == "arm-empty" and len(parts) == 1:
                self.goal_arm_empty = True
            # Ignore other potential goal predicates if any


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset)
        state_set = set(state) # Convert to set for faster lookups

        # H1: Count blocks not in their correct goal stack position
        h1 = 0
        # We only care about blocks that are part of the goal stack structure defined by 'on'/'on-table' goals
        blocks_to_check_h1 = set(self.goal_base.keys()) # Use keys of goal_base

        for block in blocks_to_check_h1:
             if not is_correctly_stacked(block, state_set, self.goal_base):
                 h1 += 1

        # H2: Count blocks that are currently on top of others but shouldn't be
        h2 = 0
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                # This is an (on X Y) fact in the current state
                # Check if this specific (on X Y) fact is a goal fact
                if fact not in self.goal_on_facts:
                    # If it's not a goal fact, then X is wrongly on Y
                    h2 += 1

        # H3: Count unsatisfied clear goals
        h3 = 0
        for goal_clear_fact in self.goal_clear_facts:
            if goal_clear_fact not in state_set:
                h3 += 1

        # H4: Count unsatisfied arm-empty goal
        h4 = 0
        if self.goal_arm_empty and "(arm-empty)" not in state_set:
            h4 += 1

        # The heuristic is the sum of these components
        return h1 + h2 + h3 + h4
