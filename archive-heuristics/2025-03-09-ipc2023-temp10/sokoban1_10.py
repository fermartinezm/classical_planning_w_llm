from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of moves required to push all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.

    # Assumptions
    - The heuristic assumes that each box can be moved independently to its goal location, ignoring potential blockages
      by other boxes or walls.
    - It also assumes that the robot can always reach a position to push a box in the right direction.

    # Heuristic Initialization
    - The heuristic initialization extracts the goal locations for each box and the adjacency information
      between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the state.
    2. Extract the goal locations of all boxes from the task goals.
    3. For each box, calculate the Manhattan distance between its current location and its goal location.
       The Manhattan distance is calculated as the sum of the absolute differences of the x and y coordinates
       of the two locations.
    4. Sum the Manhattan distances for all boxes. This sum is the heuristic estimate of the number of moves
       required to solve the puzzle.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
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

    def __call__(self, node):
        """
        Estimate the number of moves required to push all boxes to their goal locations.
        """
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
        goal_state = True
        for box, goal_location in self.box_goals.items():
            if box not in box_locations or box_locations[box] != goal_location:
                goal_state = False
                break
        if goal_state:
            return 0

        # Calculate the sum of Manhattan distances
        total_distance = 0
        for box, goal_location in self.box_goals.items():
            if box in box_locations:
                current_location = box_locations[box]

                # Extract x and y coordinates from location names
                current_x, current_y = map(int, current_location.split('_')[1:])
                goal_x, goal_y = map(int, goal_location.split('_')[1:])

                # Calculate Manhattan distance
                distance = abs(current_x - goal_x) + abs(current_y - goal_y)
                total_distance += distance
            else:
                # Box is not in the state, which means the state is unsolvable
                return float('inf')

        return total_distance
