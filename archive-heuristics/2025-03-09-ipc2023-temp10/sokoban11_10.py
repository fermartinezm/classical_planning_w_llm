from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import heapq

class sokoban11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to solve a Sokoban problem by calculating the sum of Manhattan distances
    from each box to its closest goal location, plus an estimate of the robot's distance to the nearest box.
    It also considers potential deadlocks by checking if any box is in a corner and not a goal.

    # Assumptions
    - The cost of moving a box is proportional to the Manhattan distance to its goal.
    - The robot needs to be adjacent to a box to push it.
    - Deadlock states are highly undesirable and should be avoided.

    # Heuristic Initialization
    - Extract the initial box locations, goal locations, and adjacency information from the task.
    - Identify corner locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of boxes and the robot from the state.
    2. Calculate the Manhattan distance from each box to every goal location.
    3. Assign each box to its closest goal location and sum the Manhattan distances.
    4. Estimate the robot's distance to the nearest box by calculating the Manhattan distance from the robot
       to each box location and subtracting 1 (since the robot needs to be adjacent to the box).
    5. Add a penalty if any box is in a corner location and is not a goal location, indicating a potential deadlock.
    6. Return the sum of the box distances, robot distance, and deadlock penalty.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and corner locations."""
        self.goals = task.goals
        self.static = task.static

        self.box_goals = {}
        self.box_locations = set()
        self.robot_location = None

        # Extract goal locations for each box
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                self.box_goals.setdefault(box, []).append(location)

        # Extract adjacency information
        self.adj = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                self.adj.setdefault(loc1, []).append(loc2)
                self.adj.setdefault(loc2, []).append(loc1)

        # Identify corner locations (locations with only two adjacent locations)
        self.corners = set()
        for loc, adj_locs in self.adj.items():
            if len(adj_locs) == 2:
                self.corners.add(loc)

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state

        box_locations = {}
        robot_location = None

        # Extract box and robot locations from the current state
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] != 'robot':
                box = parts[1]
                location = parts[2]
                box_locations[box] = location
            elif parts[0] == 'at-robot':
                robot_location = parts[1]

        # If it is the goal state, return 0
        if node.state >= self.goals:
            return 0

        # Calculate the sum of Manhattan distances from each box to its closest goal
        total_box_distance = 0
        for box, box_location in box_locations.items():
            min_distance = float('inf')
            for goal_location in self.box_goals[box]:
                distance = self.manhattan_distance(box_location, goal_location)
                min_distance = min(min_distance, distance)
            total_box_distance += min_distance

        # Estimate the robot's distance to the nearest box
        min_robot_distance = float('inf')
        for box, box_location in box_locations.items():
            distance = self.manhattan_distance(robot_location, box_location) - 1 # Robot needs to be adjacent
            min_robot_distance = min(min_robot_distance, distance)

        # Add a penalty if any box is in a corner and not a goal
        deadlock_penalty = 0
        for box, box_location in box_locations.items():
            if box_location in self.corners and box_location not in self.box_goals.get(box, []):
                deadlock_penalty += 10  # High penalty to avoid deadlocks

        # Return the estimated cost
        return total_box_distance + min_robot_distance + deadlock_penalty

    def manhattan_distance(self, loc1, loc2):
        """Calculate the Manhattan distance between two locations."""
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)
