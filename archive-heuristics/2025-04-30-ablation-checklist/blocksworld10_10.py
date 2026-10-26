from fnmatch import fnmatch
# Assuming heuristic_base.py is available in a 'heuristics' directory
# If running standalone, you might need a dummy Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of arguments in the pattern
    if len(parts) != len(args):
         return False
    # Check if each part matches the corresponding pattern using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the domain-dependent heuristic class
class blocksworldHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the difficulty of reaching the goal by counting:
    1. Blocks that are not on their correct base according to the goal.
    2. 'on' relationships in the current state that are not part of the goal.
    3. A penalty if the arm is holding a block.

    # Heuristic Initialization
    - Parses goal facts to identify the desired base for each block and the set of goal 'on' relationships.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal configuration: For each block mentioned in an 'on' or 'on-table' goal, determine what should be directly below it. Store goal 'on' facts.
    2. Identify the current configuration: For each block in the state, determine what is directly below it ('on' or 'on-table') or if it's being held.
    3. Initialize heuristic value `h = 0`.
    4. For each block that has a specified base in the goal: If its current base is different from its goal base, increment `h`.
    5. For each 'on' relationship currently true in the state: If this exact 'on' relationship is not required in the goal, increment `h`.
    6. If the robot's arm is currently holding a block, increment `h`.
    7. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.

        Args:
            task: The planning task object containing goals and static facts.
        """
        self.goals = task.goals

        # Parse goal facts to build goal configuration
        self.goal_base = {} # Maps block -> block_below or 'table'
        self.goal_on_facts = set() # Set of goal (on X B) facts as strings
        self.goal_blocks = set() # Blocks explicitly mentioned in goal on/on-table facts

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'on':
                # Ensure fact has correct structure before accessing parts
                if len(parts) == 3:
                    block, under_block = parts[1], parts[2]
                    self.goal_base[block] = under_block
                    self.goal_on_facts.add(goal)
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(under_block)
            elif parts and parts[0] == 'on-table':
                 # Ensure fact has correct structure before accessing parts
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_base[block] = 'table'
                    self.goal_blocks.add(block)
            # Ignore other goal predicates like (clear ?) or (arm-empty) for this heuristic's base calculation

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the cost to reach the goal.
        """
        state = node.state

        # Parse state facts to build current configuration
        current_base = {} # Maps block -> block_below or 'table' or 'arm'
        held_block = None
        state_on_facts = set() # Set of state (on X B) facts as strings

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'on':
                 # Ensure fact has correct structure
                if len(parts) == 3:
                    block, under_block = parts[1], parts[2]
                    current_base[block] = under_block
                    state_on_facts.add(fact)
            elif parts and parts[0] == 'on-table':
                 # Ensure fact has correct structure
                if len(parts) == 2:
                    block = parts[1]
                    current_base[block] = 'table'
            elif parts and parts[0] == 'holding':
                 # Ensure fact has correct structure
                if len(parts) == 2:
                    held_block = parts[1]
                    current_base[held_block] = 'arm'
            # Ignore clear and arm-empty predicates for this heuristic calculation

        h = 0

        # 1. Count blocks that are in the wrong position relative to their base *if* they have a goal base
        # Iterate through blocks that are relevant to the goal configuration
        for block in self.goal_blocks:
            # Get the current base for this block. Use .get() with None as default
            # in case a goal block is somehow not represented in the state facts (unlikely in valid states).
            current = current_base.get(block)
            goal = self.goal_base[block] # This block is in goal_blocks, so it must have a goal_base

            # If the block's current base is different from its goal base
            # This covers cases where it's on the wrong block, on the table when it shouldn't be,
            # or on a block when it should be on the table.
            # It also covers the case where a goal block is being held ('arm' != goal_base).
            if current != goal:
                 h += 1

        # 2. Count (on X B) relationships currently true in the state that are not goal relationships
        # These represent blocks that are part of incorrect stacks.
        for fact in state_on_facts:
             if fact not in self.goal_on_facts:
                 h += 1

        # 3. Add penalty if the arm is holding a block
        # This penalizes the arm being busy, as it must put down or stack the block
        # before it can perform other necessary actions.
        if held_block is not None:
            h += 1

        return h

