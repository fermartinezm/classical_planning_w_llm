from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import heapq

class sokoban10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the sokoban domain.

    # Summary
    This heuristic estimates the cost to solve a Sokoban problem by calculating the sum of Manhattan distances
    between each box and its closest goal location, plus an estimate of the number of moves required to clear a path
    for each box to its goal.

    # Assumptions
    - The heuristic assumes that the robot can always reach a position to push a box towards its goal.
    - It does not explicitly account for deadlocks or situations where a box is trapped.
    - It assumes that moving a box closer to its goal is always beneficial.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal locations for each box and the adjacency information
      between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract Box and Robot Locations:
       - Identify the current location of each box and the robot from the state.
    2. Extract Goal Locations:
       - Identify the goal location for each box from the task goals.
    3. Calculate Manhattan Distances:
       - For each box, calculate the Manhattan distance to its closest goal location.
    4. Estimate Path Clearing Cost:
       - For each box, estimate the cost of clearing a path towards its goal. This is a simplified estimate
         based on the number of boxes blocking a direct path.  This part is omitted to reduce computational complexity.
    5. Sum Costs:
       - Sum the Manhattan distances for all boxes to obtain the final heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                self.box_goals[box] = location

        self.adjacencies = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                direction = parts[3]
                if loc1 not in self.adjacencies:
                    self.adjacencies[loc1] = {}
                self.adjacencies[loc1][direction] = loc2

    def __call__(self, node):
        """
        Estimate the cost to solve the Sokoban problem from the given state.
        """
        state = node.state
        box_locations = {}
        robot_location = None

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at':
                if parts[1] != 'robot':
                    box = parts[1]
                    location = parts[2]
                    box_locations[box] = location
            elif parts[0] == 'at-robot':
                robot_location = parts[1]

        if self.goal_reached(state):
            return 0

        total_distance = 0
        for box, box_location in box_locations.items():
            goal_location = self.box_goals[box]
            
            # Extract coordinates from location names (e.g., loc_1_1 becomes (1, 1))
            box_coords = tuple(map(int, box_location.split('_')[1:]))
            goal_coords = tuple(map(int, goal_location.split('_')[1:]))
            
            # Calculate Manhattan distance
            distance = abs(box_coords[0] - goal_coords[0]) + abs(box_coords[1] - goal_coords[1])
            total_distance += distance

        return total_distance

    def goal_reached(self, state):
        """
        Check if the goal has been reached.
        """
        return self.goals <= state
