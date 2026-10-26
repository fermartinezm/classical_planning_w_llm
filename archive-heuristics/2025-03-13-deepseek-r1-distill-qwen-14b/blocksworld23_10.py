from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack blocks into the target configuration by comparing the current state with the goal state.

    # Assumptions:
    - The goal is to achieve a specific stack of blocks.
    - Each block that is not in its correct position in the goal stack contributes to the heuristic value.

    # Heuristic Initialization
    - Extracts the goal conditions to determine the target stack configuration.
    - Processes static facts to understand the domain structure.

    # Step-by-Step Thinking for Computing Heuristic
    1. Extract the goal stack by parsing the goal conditions.
    2. Parse the current state to build the current stack of blocks.
    3. Compare the current stack with the goal stack.
    4. Count the number of blocks that are not in their correct position in the goal stack.
    5. The heuristic value is the count of such misplaced blocks.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract the goal stack
        self.goal_stack = []
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'on':
                on_block = parts[2]
                self.goal_stack.append(on_block)

        # Build a reverse map for faster lookups
        self.reverse_goal_map = {}
        for i, block in enumerate(self.goal_stack):
            self.reverse_goal_map[block] = i

    def __call__(self, node):
        """Compute the heuristic value based on the current state."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        current_stack = []
        clear_table = True
        for fact in state:
            if match(fact, 'clear', '*'):
                clear_table = False
            if match(fact, 'on', '*', '*'):
                _, obj, under = get_parts(fact)
                current_stack.append(obj)

        # Find the base block (the one on the table)
        base_block = None
        for fact in state:
            if match(fact, 'on-table', '*'):
                base_block = get_parts(fact)[1]
                break

        # Build the current stack starting from the base
        current_stack = []
        under = base_block
        while True:
            for fact in state:
                if match(fact, 'on', '*', under):
                    obj = get_parts(fact)[1]
                    current_stack.append(obj)
                    under = obj
                    break
            else:
                break

        # Calculate the heuristic
        heuristic = 0
        min_length = min(len(current_stack), len(self.goal_stack))
        for i in range(min_length):
            if current_stack[i] != self.goal_stack[i]:
                heuristic += 1
        # Add remaining blocks that are extra in the current stack
        heuristic += abs(len(current_stack) - len(self.goal_stack))

        return heuristic
