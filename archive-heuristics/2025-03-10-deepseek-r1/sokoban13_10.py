from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class sokoban13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It calculates the sum of the minimal pushes needed for each box to reach its goal and adds the robot's minimal distance to any box not yet at its goal.

    # Assumptions
    - The robot can move freely between adjacent locations as per the adjacency graph.
    - Each box's path to its goal is approximated by the shortest path in the adjacency graph, ignoring other boxes.
    - The robot needs to reach a box to push it, and each push moves the box one step towards its goal.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task's goals.
    - Builds an adjacency graph from the static 'adjacent' facts.
    - Precomputes all-pairs shortest paths using BFS for efficient lookup during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the robot's current location from the state.
    2. Extract each box's current location from the state.
    3. For each box not at its goal:
        a. Compute the shortest path distance from the box's current location to its goal.
        b. Add this distance to the sum of box distances.
        c. Compute the shortest path from the robot's current location to the box's current location.
        d. Track the minimum robot-to-box distance.
    4. The heuristic value is the sum of all box distances plus the minimum robot-to-box distance.
    """

    def __init__(self, task):
        self.goal_locations = {}
        # Extract goal locations for each box
        for goal in task.goals:
            if goal.startswith('(at '):
                parts = goal[1:-1].split()
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Build adjacency list from static facts
        self.adjacency = defaultdict(list)
        for fact in task.static:
            if fact.startswith('(adjacent '):
                parts = fact[1:-1].split()
                l1 = parts[1]
                l2 = parts[2]
                self.adjacency[l1].append(l2)

        # Precompute all-pairs shortest paths using BFS
        self.distances = defaultdict(dict)
        locations = set(self.adjacency.keys())
        for loc in locations:
            # BFS to find shortest paths from loc
            visited = {loc: 0}
            queue = deque([loc])
            while queue:
                current = queue.popleft()
                current_dist = visited[current]
                for neighbor in self.adjacency[current]:
                    if neighbor not in visited or current_dist + 1 < visited.get(neighbor, float('inf')):
                        visited[neighbor] = current_dist + 1
                        queue.append(neighbor)
            # Update distances for this source location
            for node, dist in visited.items():
                self.distances[loc][node] = dist

    def __call__(self, node):
        state = node.state
        # Extract robot's current location
        robot_loc = None
        for fact in state:
            if fact.startswith('(at-robot '):
                parts = fact[1:-1].split()
                robot_loc = parts[1]
                break
        if not robot_loc:
            return float('inf')  # Robot has no location (invalid state)

        # Extract current box locations
        current_boxes = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                if parts[0] == 'at' and parts[1].startswith('box'):
                    box = parts[1]
                    loc = parts[2]
                    current_boxes[box] = loc

        sum_box_dist = 0
        min_robot_dist = float('inf')
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_boxes.get(box)
            if current_loc == goal_loc:
                continue  # Box is already at goal

            # Calculate box to goal distance
            box_to_goal = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
            if box_to_goal == float('inf'):
                return float('inf')  # Box cannot reach goal
            sum_box_dist += box_to_goal

            # Calculate robot to box distance
            robot_to_box = self.distances.get(robot_loc, {}).get(current_loc, float('inf'))
            if robot_to_box < min_robot_dist:
                min_robot_dist = robot_to_box

        if sum_box_dist == 0:
            return 0  # All boxes are at their goals

        if min_robot_dist == float('inf'):
            return float('inf')  # Robot cannot reach any box

        return sum_box_dist + min_robot_dist
