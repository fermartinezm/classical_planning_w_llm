# Add the necessary import
from heuristics.heuristic_base import Heuristic

# Define the helper function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    return fact[1:-1].split()

# Define the heuristic class
class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of necessary actions by summing two components:
    1. Blocks that are part of a goal stack but are not in their correct position relative to the goal structure below them.
    2. Blocks that are currently on top of another block that is either part of a goal stack or is required to be clear, and the block on top is not the one that is supposed to be there according to the goal.

    # Assumptions
    - The goal specifies the desired positions of some blocks (on other blocks or on the table) forming stacks, and which blocks should be clear.
    - Blocks not mentioned in 'on' or 'on-table' goal predicates do not have a specific required final position in the stack structure.

    # Heuristic Initialization
    - Parses the goal predicates to identify:
        - `goal_parent`: A mapping from a block to the block it should be directly on top of in the goal, or 'table' if it should be on the table.
        - `goal_children`: A mapping from a block to the block that should be directly on top of it in the goal.
        - `goal_clear`: A set of blocks that must be clear in the goal state.
        - `goal_blocks`: A set of blocks whose position is specified in an 'on' or 'on-table' goal predicate.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current position of every block (on another block, on the table, or held).
    2. Determine which blocks are "in goal position chain": A block `b` is in the goal position chain if its goal parent is the table and it is currently on the table, OR if its goal parent is block `y`, it is currently on `y`, AND `y` is also in the goal position chain. Blocks not part of any 'on' or 'on-table' goal are considered trivially in the goal position chain for the purpose of not breaking a chain.
    3. Count the number of blocks in `goal_blocks` that are *not* in the goal position chain. Each such block represents a misplaced block within the desired stack structure.
    4. Count the number of "blocking" blocks: A block `x` is blocking if `(on x y)` is true in the state, and (`y` is in `goal_blocks` OR `y` is in `goal_clear`), AND `x` is *not* the block that should be on `y` according to `goal_children` (i.e., `goal_children.get(y)` is not `x`).
    5. The total heuristic value is the sum of the counts from steps 3 and 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Map block to its desired parent block or 'table'
        self.goal_parent = {}
        # Map block to its desired child block (the one directly on top in the goal)
        self.goal_children = {}
        # Set of blocks that must be clear in the goal
        self.goal_clear = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, parent = parts[1], parts[2]
                self.goal_parent[block] = parent
                # Assuming only one block is specified to be directly on top of 'parent' in the goal
                self.goal_children[parent] = block
            elif predicate == "on-table":
                block = parts[1]
                self.goal_parent[block] = 'table'
            elif predicate == "clear":
                block = parts[1]
                self.goal_clear.add(block)
            # Ignore 'arm-empty' goal if present

        # Set of blocks whose position is specified in the goal (keys in goal_parent)
        self.goal_blocks = set(self.goal_parent.keys())

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Helper to find the current position of a block
        def current_pos(block, current_state):
            for fact in current_state:
                parts = get_parts(fact)
                if parts[0] == 'on' and parts[1] == block:
                    return parts[2] # Returns the block it's on
                if parts[0] == 'on-table' and parts[1] == block:
                    return 'table'
                if parts[0] == 'holding' and parts[1] == block:
                    return 'holding'
            # If the block is not found in any of these predicates, its position is unknown
            # based on the provided state facts. This might indicate an issue with the state
            # representation or the block is not currently relevant to position/holding.
            # Returning None signifies its position isn't explicitly defined in the state facts
            # we are checking.
            return None


        # Helper to check if a block is in the goal position chain (recursive with memoization)
        memo = {}
        def is_in_goal_chain(block, current_state, goal_parent_map, memo_dict):
            # If the block is not part of any 'on' or 'on-table' goal, it doesn't break a chain
            # by being somewhere specific.
            if block not in goal_parent_map:
                return True

            if block in memo_dict:
                return memo_dict[block]

            desired_parent = goal_parent_map[block]
            actual_parent = current_pos(block, current_state)

            # If block's position is unknown, it's not in the chain
            if actual_parent is None:
                 memo_dict[block] = False
                 return False

            result = False
            if desired_parent == 'table':
                # Block should be on the table
                result = (actual_parent == 'table')
            else:
                # Block should be on another block (desired_parent)
                result = (actual_parent == desired_parent) and \
                         is_in_goal_chain(desired_parent, current_state, goal_parent_map, memo_dict)

            memo_dict[block] = result
            return result

        # 1. Count blocks not in the goal position chain
        misplaced_goal_blocks_count = 0
        for block in self.goal_blocks:
            if not is_in_goal_chain(block, state, self.goal_parent, memo):
                misplaced_goal_blocks_count += 1

        # 2. Count blocking blocks
        blocking_blocks_count = 0
        # Find all 'on' facts in the current state
        current_on_facts = {} # Map child -> parent
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                child, parent = parts[1], parts[2]
                current_on_facts[child] = parent

        for child, parent in current_on_facts.items():
            # Check if the parent block is relevant (part of goal stack or needs to be clear)
            is_relevant_parent = parent in self.goal_blocks or parent in self.goal_clear

            if is_relevant_parent:
                # Check if the child block is NOT the one that is supposed to be on the parent in the goal
                desired_child = self.goal_children.get(parent) # Returns None if parent has no desired child in goal
                if desired_child != child:
                     blocking_blocks_count += 1

        # The heuristic is the sum of these two counts
        return misplaced_goal_blocks_count + blocking_blocks_count
