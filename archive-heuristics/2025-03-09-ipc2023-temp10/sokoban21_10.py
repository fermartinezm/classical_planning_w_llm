from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.

    # Assumptions:
    - The heuristic assumes that the robot can always reach a position to push a box towards its goal.
    - It does not consider obstacles or walls, providing a lower-bound estimate.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goal conditions.
    - Store the adjacency information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the state.
    2. For each box, find its goal location.
    3. Calculate the Manhattan distance between the box's current location and its goal location.
       The Manhattan distance is calculated as the sum of the absolute differences in the x and y coordinates of the two locations.
       To calculate the x and y coordinates, we split the location name into its row and column.
    4. Sum the Manhattan distances for all boxes. This sum is the heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                self.box_goals[box] = location

        # Extract adjacency information (not used in this simple heuristic, but could be used for more advanced heuristics)
        self.adjacencies = {}
        for fact in static_facts:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                direction = parts[3]
                if loc1 not in self.adjacencies:
                    self.adjacencies[loc1] = {}
                self.adjacencies[loc1][direction] = loc2

    def __call__(self, node):
        """Estimate the number of actions needed to move all boxes to their goal locations."""
        state = node.state
        total_distance = 0

        # Extract current box locations
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] != 'robot':
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # If the state is a goal state, return 0
        goal_state = True
        for box, goal_location in self.box_goals.items():
            if box not in box_locations or box_locations[box] != goal_location:
                goal_state = False
                break
        if goal_state:
            return 0

        # Calculate the sum of Manhattan distances
        for box, current_location in box_locations.items():
            if box in self.box_goals:
                goal_location = self.box_goals[box]

                # Extract row and column from location names (e.g., loc_2_3 -> row=2, col=3)
                current_row, current_col = map(int, current_location.split('_')[1:])
                goal_row, goal_col = map(int, goal_location.split('_')[1:])

                # Calculate Manhattan distance
                distance = abs(current_row - goal_row) + abs(current_col - goal_col)
                total_distance += distance

        return total_distance
