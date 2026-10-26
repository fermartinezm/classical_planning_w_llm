# from heuristics.heuristic_base import Heuristic # Assuming this base class exists

class floortileHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic counts the number of goal conditions that are not yet satisfied
    in the current state. Specifically, it counts the number of tiles that are
    required to be painted with a specific color in the goal state but are
    either not painted or painted with a different color in the current state.

    # Assumptions
    - The goal is a conjunction of (painted tile color) facts.
    - Tiles that need to be painted are initially clear or painted incorrectly.
    - There is no action to unpaint a tile.
    - All goal tiles must be painted correctly.

    # Heuristic Initialization
    - The heuristic stores the set of goal facts from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state from the search node. The state is represented as a frozenset of fact strings.
    2. Initialize a counter for unsatisfied goal conditions to 0.
    3. Iterate through each goal fact stored during initialization (these are the facts that must be true in the goal state).
    4. For each goal fact (which is expected to be of the form '(painted tile color)'):
       a. Check if this exact fact string exists in the current state.
       b. If the goal fact is NOT present in the current state, it means this specific painting goal is not yet achieved. Increment the counter.
    5. The heuristic value is the final count of unsatisfied goal conditions. This value represents the number of individual painting tasks that still need to be completed.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal conditions."""
        # The set of facts that must hold in goal states.
        # In the floortile domain, goals are typically (painted tile color) facts.
        self.goals = task.goals
        # Static facts are not needed for this specific heuristic calculation.
        # static_facts = task.static

    def __call__(self, node):
        """Estimate the minimum cost to reach the goal state."""
        state = node.state # Current world state as a frozenset of fact strings.

        # The heuristic is the number of goal facts not present in the current state.
        # This works because goal facts are positive literals and the goal is a conjunction.
        unsatisfied_goals_count = 0
        for goal_fact in self.goals:
            # Check if the goal fact string is present in the current state set.
            # This implicitly handles the structure like '(painted tile_X_Y color_Z)'
            # without needing explicit parsing of objects, as the goal facts
            # are provided as exact strings in the task.goals set.
            if goal_fact not in state:
                unsatisfied_goals_count += 1

        return unsatisfied_goals_count
