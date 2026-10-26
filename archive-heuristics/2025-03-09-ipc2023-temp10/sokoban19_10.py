from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box and its closest goal location.
    Additionally, it estimates the number of moves required for the robot to reach each box before pushing it.

    # Assumptions
    - The heuristic assumes that boxes can be moved independently, which is not always true in Sokoban.
    - It only considers the Manhattan distance, ignoring obstacles and the need for clear paths.
    - It assumes the robot can always reach a box to push it, which might not be possible due to walls or other boxes.

    # Heuristic Initialization
    - Extract the goal locations for each box.
    - Store the adjacency information between locations to estimate robot move costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes and the robot from the state.
    2. For each box, find the Manhattan distance to its closest goal location.
    3. Estimate the number of moves required for the robot to reach a position from which it can push the box towards its goal.
       - This is done by calculating the Manhattan distance between the robot and the box's current location.
    4. Sum the Manhattan distances for all boxes and the estimated robot move costs.
    5. If the state is a goal state, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                self.box_goals.setdefault(box, []).append(location)

        # Extract adjacency information
        self.adjacencies = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                direction = parts[3]
                self.adjacencies.setdefault(loc1, []).append((loc2, direction))
                # Assuming adjacency is bidirectional, though the PDDL definition doesn't enforce it.
                self.adjacencies.setdefault(loc2, []).append((loc1, self.reverse_direction(direction)))

    def reverse_direction(self, direction):
        if direction == 'up':
            return 'down'
        elif direction == 'down':
            return 'up'
        elif direction == 'left':
            return 'right'
        elif direction == 'right':
            return 'left'
        return None

    def __call__(self, node):
        """
        Estimate the number of actions needed to move all boxes to their goal locations.
        """
        state = node.state

        # Check if the state is a goal state
        if self.is_goal_state(state):
            return 0

        # Extract box locations
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] not in ['robot']:
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # Extract robot location
        robot_location = None
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at-robot':
                robot_location = parts[1]
                break

        total_cost = 0
        for box, current_location in box_locations.items():
            # Find the closest goal location for the box
            min_distance = float('inf')
            for goal_location in self.box_goals.get(box, []):
                distance = self.manhattan_distance(current_location, goal_location)
                min_distance = min(min_distance, distance)
            total_cost += min_distance

            # Estimate the cost for the robot to reach the box
            robot_distance = self.manhattan_distance(robot_location, current_location)
            total_cost += robot_distance

        return total_cost

    def manhattan_distance(self, loc1, loc2):
        """
        Calculate the Manhattan distance between two locations.
        """
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)

    def is_goal_state(self, state):
        """
        Check if the current state is a goal state.
        """
        return self.goals <= state
