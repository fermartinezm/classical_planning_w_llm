from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.

    # Assumptions
    - The heuristic assumes that the robot can always move to push a box.
    - It does not consider deadlocks or other complex constraints.
    - It assumes each push action takes one step.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goal conditions.
    - Store the adjacency information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of each box from the state.
    2. For each box, calculate the Manhattan distance between its current location and its goal location.
       The Manhattan distance is calculated as the sum of the absolute differences of the x and y coordinates.
       To get the x and y coordinates, we extract them from the location names. For example, loc_2_3 has x=2 and y=3.
    3. Sum the Manhattan distances for all boxes. This sum is the heuristic value.
    4. If the state is a goal state, the heuristic value is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.box_goals = {}
        for goal in task.goals:
            if goal.startswith('(at'):
                parts = goal[1:-1].split()
                box = parts[1]
                location = parts[2]
                self.box_goals[box] = location

    def __call__(self, node):
        """Estimate the number of actions needed to move all boxes to their goal locations."""
        state = node.state

        if self.goal_reached(state):
            return 0

        total_distance = 0
        for fact in state:
            if fact.startswith('(at ') and 'box' in fact:
                parts = fact[1:-1].split()
                box = parts[1]
                current_location = parts[2]
                if box in self.box_goals:
                    goal_location = self.box_goals[box]
                    total_distance += self.manhattan_distance(current_location, goal_location)

        return total_distance

    def manhattan_distance(self, loc1, loc2):
        """Calculate the Manhattan distance between two locations."""
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)

    def goal_reached(self, state):
        """Check if the state is a goal state."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True
