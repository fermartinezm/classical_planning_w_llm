# Import necessary modules
# The problem description implies a Heuristic base class exists.
# Assuming it's available in 'heuristics.heuristic_base'
from heuristics.heuristic_base import Heuristic

# Helper function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct final position relative to the block directly below them in the
    goal configuration. A block is considered correctly placed only if it
    is on the correct block (or table) according to the goal, AND the block
    below it is also correctly placed.

    # Assumptions
    - The goal specifies the desired 'on' relationships and 'on-table'
      relationships for a subset of blocks.
    - Blocks not mentioned in goal 'on' or 'on-table' facts are not
      considered for their final position in this heuristic calculation.
      (Their position might still matter if they block a goal block, but
       this heuristic simplifies by ignoring them).
    - The heuristic value is the count of blocks that are not correctly
      placed according to the recursive definition.

    # Heuristic Initialization
    - Extracts the goal 'on' and 'on-table' facts to build a map
      (`goal_support`) indicating what each block should be directly on
      in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts to create a `goal_support` map: For each goal
       `(on X Y)`, record that `X` should be on `Y`. For each goal
       `(on-table Z)`, record that `Z` should be on the 'table'.
    2. Parse the current state facts to create a `current_support_map`:
       For each state `(on X Y)`, record that `X` is on `Y`. For each state
       `(on-table Z)`, record that `Z` is on the 'table'. For each state
       `(holding W)`, record that `W` is on the 'arm'.
    3. Define a recursive helper function `is_correctly_placed(block)`:
       - This function checks if the given `block` is in its correct final
         position relative to the structure below it, AND if the block below it
         is also correctly placed (recursively).
       - Use memoization (`correctly_placed_status` dictionary) to store results for blocks already checked to avoid redundant computation and infinite recursion on cyclic dependencies (though goal structures should be acyclic).
       - Base case: If the block is not in the `goal_support` map, it doesn't have a specific target position we are tracking. Assume it's 'correct' for the purpose of this heuristic.
       - If the goal is `(on-table block)`, it's correctly placed if and only if it is currently `(on-table block)`.
       - If the goal is `(on block Y)`, it's correctly placed if and only if it is currently `(on block Y)` AND `is_correctly_placed(Y)` is true.
    4. Initialize the total heuristic cost to 0.
    5. Iterate through all blocks that appear as keys in the `goal_support` map.
    6. For each such block, call `is_correctly_placed(block)`.
    7. If `is_correctly_placed(block)` returns False, increment the total cost.
    8. Return the total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal support relationships.
        """
        self.goals = task.goals

        # Build the goal_support map: block -> what it should be on ('table' or another block)
        self.goal_support = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                # Goal is (on X Y)
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
            elif predicate == "on-table" and len(parts) == 2:
                # Goal is (on-table Z)
                block = parts[1]
                self.goal_support[block] = 'table'
            # Ignore other goal predicates like (clear ...) or (arm-empty) for this heuristic

        # Static facts are not used in this Blocksworld heuristic
        # static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build the current_support_map: block -> what it is currently on ('table', another block, or 'arm')
        current_support_map = {}
        # Iterate through state facts to find support for each block
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                # State is (on X Y)
                block, support = parts[1], parts[2]
                current_support_map[block] = support
            elif predicate == "on-table" and len(parts) == 2:
                # State is (on-table Z)
                block = parts[1]
                current_support_map[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                # State is (holding W)
                block = parts[1]
                current_support_map[block] = 'arm'
            # Ignore other state predicates like (clear ...) or (arm-empty)

        # Memoization dictionary for the recursive check
        correctly_placed_status = {}

        def is_correctly_placed(block):
            """
            Recursive helper to check if a block is in its correct goal position
            relative to the structure below it.
            """
            # If already computed, return the stored result
            if block in correctly_placed_status:
                return correctly_placed_status[block]

            # If the block is not in the goal_support map, it doesn't have a
            # specific target position we are tracking. Assume it's 'correct'
            # for the purpose of this heuristic.
            if block not in self.goal_support:
                correctly_placed_status[block] = True
                return True

            goal_pos = self.goal_support[block]
            current_pos = current_support_map.get(block)

            status = False

            if goal_pos == 'table':
                # Goal is (on-table block)
                status = (current_pos == 'table')
            else: # goal_pos is another block Y
                # It's correctly placed only if it's currently on Y AND Y is correctly placed
                if current_pos == goal_pos:
                    # Check if the block it's on (goal_pos) is itself correctly placed
                    status = is_correctly_placed(goal_pos)

            correctly_placed_status[block] = status
            return status

        # Calculate the total heuristic cost
        total_cost = 0
        # We only care about blocks that have a defined goal position in goal_support
        for block in self.goal_support:
            if not is_correctly_placed(block):
                total_cost += 1

        return total_cost
