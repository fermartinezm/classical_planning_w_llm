from heuristics.heuristic_base import Heuristic

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their respective goal locations by summing the Manhattan distances from each box's current position to its goal.

    # Assumptions:
    - Each box must be moved to a specific goal location.
    - The Manhattan distance between a box's current position and its goal provides a good estimate of the required actions.
    - The robot can move freely and reach each box without considering obstacles.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goals.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each box, determine its current location from the state.
    2. Calculate the Manhattan distance between the box's current location and its goal location.
    3. Sum all these distances to get the total estimated number of actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                loc = parts[2]
                self.goals[box] = loc

    def __call__(self, node):
        """Compute the heuristic value based on the current state."""
        state = node.state
        heuristic = 0

        for box, goal_loc in self.goals.items():
            current_loc = None
            for fact in state:
                if fact.startswith('(at ') and box in fact:
                    current_loc = fact.split()[-1]
                    break
            if current_loc is None:
                continue  # Box is not present, should not happen in Sokoban

            # Parse current and goal locations into coordinates
            current_x, current_y = map(int, current_loc.split('_')[1:])
            goal_x, goal_y = map(int, goal_loc.split('_')[1:])

            # Calculate Manhattan distance
            distance = abs(current_x - goal_x) + abs(current_y - goal_y)
            heuristic += distance

        return heuristic
