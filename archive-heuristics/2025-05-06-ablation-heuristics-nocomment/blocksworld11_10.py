from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by counting
        the number of blocks that are currently not in their specified goal
        position (either on another block or on the table). Each such
        "misplaced" block is assumed to require approximately 2 actions
        (one to pick it up/unstack it, and one to stack it/put it down)
        to move it towards its goal location. The heuristic returns
        twice the count of misplaced blocks. It is non-admissible but aims
        to guide a greedy best-first search efficiently.

    Assumptions:
        - The input task is a valid Blocksworld planning task.
        - State facts are represented as strings like '(predicate arg1 arg2)'.
        - Goal facts include 'on', 'on-table', and potentially 'clear'.
        - Objects are blocks that can be stacked or placed on the table.
        - Any block not explicitly given an 'on' or 'on-table' goal is assumed
          to have a goal of being on the table.
        - In any valid state, each block is either on another block, on the table,
          or being held ('holding').

    Heuristic Initialization:
        In the constructor (__init__), the heuristic processes the task's initial
        state and goals to identify all objects (blocks) in the problem.
        It then builds a dictionary `self.goal_pos` mapping each object to its
        desired location in the goal state. This location is either the name
        of the block it should be on, or the string 'table' if it should be
        on the table. Blocks not mentioned in any 'on' or 'on-table' goal
        predicate are assigned a goal location of 'table'.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Get the current state from the provided search node.
        2.  Build a dictionary `current_pos` mapping each object to its current
            location. This location is determined by looking for facts like
            '(on X Y)' (X is on Y), '(on-table X)' (X is on the table), or
            '(holding X)' (X is in the arm, represented as 'arm').
        3.  Initialize a counter `misplaced_count` to 0.
        4.  Iterate through all objects identified during initialization.
        5.  For each object, retrieve its goal location from `self.goal_pos`
            and its current location from `current_pos`.
        6.  If the object's current location is different from its goal location,
            increment `misplaced_count`. Being in the 'arm' is never a goal
            location, so any object being held is always counted as misplaced.
        7.  The heuristic value is `2 * misplaced_count`. This scales the count
            by 2, reflecting the typical two-action sequence (pickup/unstack +
            stack/putdown) required to move a misplaced block. This count
            provides an estimate of the minimum number of block movements needed.
    """
    def __init__(self, task):
        self.goals = task.goals
        # static_facts = task.static # Blocksworld has no static facts

        # Collect all objects from initial state and goal state
        all_objects = set()
        def collect_objects_from_facts(facts):
            for fact in facts:
                parts = self._get_parts(fact)
                # Assume arguments after predicate are objects
                for part in parts[1:]:
                    all_objects.add(part)

        collect_objects_from_facts(task.initial_state)
        collect_objects_from_facts(task.goals)
        self.all_objects = list(all_objects) # Store as list

        # Build goal positions dictionary
        self.goal_pos = {}
        for goal in self.goals:
            parts = self._get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1:]
                self.goal_pos[obj] = underob
            elif predicate == "on-table":
                obj = parts[1]
                self.goal_pos[obj] = 'table'

        # For objects not explicitly placed by 'on' or 'on-table' goals, assume they should be on the table
        for obj in self.all_objects:
            if obj not in self.goal_pos:
                self.goal_pos[obj] = 'table'

    def _get_parts(self, fact):
        """Helper to parse a fact string into predicate and arguments."""
        # Removes leading/trailing parens and splits by space
        return fact[1:-1].split()

    def __call__(self, node):
        state = node.state

        # Build current positions dictionary
        current_pos = {}
        for fact in state:
            parts = self._get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1:]
                current_pos[obj] = underob
            elif predicate == "on-table":
                obj = parts[1]
                current_pos[obj] = 'table'
            elif predicate == "holding":
                obj = parts[1]
                current_pos[obj] = 'arm' # Represent being held

        misplaced_count = 0
        for obj in self.all_objects:
            goal_loc = self.goal_pos[obj] # Guaranteed to exist

            # Find current location. A block must be somewhere.
            current_loc = current_pos.get(obj)

            # If a block is not found in current_pos, it's not in any on/on-table/holding fact.
            # This indicates an invalid state representation for this domain.
            # However, for heuristic computation, we must return a value.
            # Assume such a block is misplaced.
            if current_loc is None:
                 # This case indicates a potentially malformed state or domain interaction.
                 # In a strictly valid blocksworld state, every block is either on, on-table, or held.
                 # Counting it as misplaced is a reasonable fallback.
                 misplaced_count += 1
            elif current_loc != goal_loc:
                 misplaced_count += 1

        # Each misplaced block typically requires at least 2 actions (pickup/unstack + stack/putdown)
        # to move it towards its goal position.
        return 2 * misplaced_count
