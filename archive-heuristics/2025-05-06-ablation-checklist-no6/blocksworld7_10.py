# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of blocks that are either in the wrong position
    relative to their goal parent or are blocking a block that is in its
    correct goal position relative to its parent. Each such 'unhappy' block
    contributes 2 to the heuristic, representing the estimated cost of
    moving it (unstack/pickup + stack/putdown).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        super().__init__(task)

        # Extract all objects mentioned in goal predicates ('on' or 'on-table').
        # These are the objects whose final position is specified by the goal.
        goal_objects = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] in ["on", "on-table"]:
                 # Add all arguments of these predicates as goal objects
                 goal_objects.update(parts[1:])
        self.goal_objects = list(goal_objects) # Use a list for consistent iteration order

        # Build goal configuration: goal_parent and goal_child
        self.goal_parent = {}
        self.goal_child = {} # block -> block_on_top_in_goal

        # First pass to identify goal parents (on or on-table) for goal_objects
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1], parts[2]
                # Only consider relations between goal objects
                if obj in self.goal_objects and underob in self.goal_objects:
                     self.goal_parent[obj] = underob
                     self.goal_child[underob] = obj # This assumes only one block can be on top
            elif predicate == "on-table":
                obj = parts[1]
                if obj in self.goal_objects:
                    self.goal_parent[obj] = 'table'

        # For goal objects that are not mentioned as being under another block in the goal,
        # they must be the top of a goal stack. Their goal_child is None.
        for obj in self.goal_objects:
             if obj not in self.goal_child:
                 self.goal_child[obj] = None # Nothing should be on top in the goal


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state  # Current world state is a frozenset of fact strings.

        # Build current configuration: current_parent and current_child
        current_parent = {}
        current_child = {} # block -> block_on_top_in_current

        # Parse state facts to find current parent/child relationships
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1], parts[2]
                # Only track relationships involving goal objects for consistency
                if obj in self.goal_objects:
                    current_parent[obj] = underob
                if underob in self.goal_objects:
                     current_child[underob] = obj # Assumes only one block on top
            elif predicate == "on-table":
                obj = parts[1]
                if obj in self.goal_objects:
                    current_parent[obj] = 'table'
            elif predicate == "holding":
                obj = parts[1]
                if obj in self.goal_objects:
                    current_parent[obj] = 'arm'

        # For goal objects, if nothing is currently on them, their current_child is None.
        for obj in self.goal_objects:
             if obj not in current_child:
                 current_child[obj] = None

        # Calculate heuristic value
        h = 0
        for block in self.goal_objects:
            # Get goal and current parent/child relationships
            goal_p = self.goal_parent.get(block)
            current_p = current_parent.get(block) # Use .get() in case block is not in state facts (shouldn't happen in valid states)

            # If a goal object is not found in the current state's location facts,
            # it's misplaced. Assign penalty and skip child check.
            if current_p is None:
                 h += 2
                 continue

            # Condition 1: Block is in the wrong place relative to its parent
            if current_p != goal_p:
                h += 2 # Estimated cost to move this block
            else: # Block is in the correct place relative to its parent
                # Condition 2: Block has a wrong block on top
                goal_c = self.goal_child.get(block)
                current_c = current_child.get(block)

                # Penalty if there is *any* block on top (current_c is not None)
                # AND the block on top is not the correct one (current_c != goal_c).
                # This covers cases where something is on top but shouldn't be (goal_c is None)
                # and cases where the wrong block is on top (current_c != goal_c).
                if current_c is not None and current_c != goal_c:
                     h += 2 # Estimated cost to move the block on top

        # The heuristic is 0 if and only if for every goal object:
        # 1. Its current parent matches its goal parent.
        # 2. AND it has the correct block (or nothing) on top.
        # This implies the entire goal stack structure is satisfied for all goal objects.
        # This seems correct for a goal state.

        return h
