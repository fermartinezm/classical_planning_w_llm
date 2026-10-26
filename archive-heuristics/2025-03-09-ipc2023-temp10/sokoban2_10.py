from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.

    # Assumptions
    - Moving the robot is less costly than moving a box.
    - The heuristic does not consider blocking boxes or walls.
    - Each box has a unique goal location.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goal conditions.
    - Store the adjacency information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of each box from the state.
    2. For each box, retrieve its goal location.
    3. Calculate the Manhattan distance between the box's current location and its goal location.
       - Manhattan distance is calculated as the sum of the absolute differences of the x and y coordinates.
       - The x and y coordinates are extracted from the location names.
    4. Sum the Manhattan distances for all boxes.
    5. Return the total sum as the heuristic value.
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

        # Extract current box locations
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] != 'robot':
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # If the state is a goal state, return 0
        goal_reached = True
        for box, goal_location in self.box_goals.items():
            if box not in box_locations or box_locations[box] != goal_location:
                goal_reached = False
                break
        if goal_reached:
            return 0

        # Calculate the sum of Manhattan distances
        total_distance = 0
        for box, current_location in box_locations.items():
            if box in self.box_goals:
                goal_location = self.box_goals[box]

                # Extract x and y coordinates from location names
                current_x, current_y = map(int, current_location.split('_')[1:])
                goal_x, goal_y = map(int, goal_location.split('_')[1:])

                # Calculate Manhattan distance
                distance = abs(current_x - goal_x) + abs(current_y - goal_y)
                total_distance += distance

        return total_distance
