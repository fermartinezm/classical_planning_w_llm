from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class Sokoban24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to solve a Sokoban problem by summing two components for each box:
    1. The minimal number of pushes required to move the box from its current position to its goal, based on precomputed shortest paths ignoring other boxes.
    2. The minimal number of moves required for the robot to reach the box's current position, adjusted to account for needing to be adjacent to push.

    # Assumptions
    - The robot can move freely between adjacent locations as long as they are clear (ignores dynamic obstacles like other boxes in path calculations).
    - Each box's path to its goal is computed independently, ignoring other boxes which might block the path.
    - The static adjacency between locations defines the movement graph, and shortest paths are precomputed once during initialization.

    # Heuristic Initialization
    - Extract goal locations for each box from the task's goal conditions.
    - Build an adjacency graph from static 'adjacent' facts.
    - Precompute the shortest path between all pairs of locations using BFS to enable efficient distance lookups.

    # Step-By-Step Thinking for Computing Heuristic
    1. For the current state, determine the robot's location and each box's current location.
    2. For each box not at its goal:
        a. Compute the shortest path distance (number of pushes) from the box's current location to its goal.
        b. Compute the shortest path distance from the robot's current location to the box's current location.
        c. Adjust the robot's distance by subtracting 1 (since the robot must be adjacent to push).
        d. Sum the adjusted robot distance and the box's push distance.
    3. Sum the values from step 2d for all boxes to get the total heuristic estimate.
    """

    def __init__(self, task):
        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == "at":
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts
        self.adjacency = defaultdict(list)
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == "adjacent":
                l1, l2 = parts[1], parts[2]
                self.adjacency[l1].append(l2)

        # Precompute shortest paths between all locations using BFS
        self.distance = defaultdict(dict)
        all_locations = set(self.adjacency.keys())
        for loc in all_locations:
            self.distance[loc] = {}
            queue = deque([(loc, 0)])
            visited = set()
            while queue:
                current, dist = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                self.distance[loc][current] = dist
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state
        robot_loc = None
        box_locs = {}

        # Extract current robot and box positions from the state
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at":
                box = parts[1]
                loc = parts[2]
                box_locs[box] = loc

        if not robot_loc:
            return float("inf")

        total = 0
        for box, current_loc in box_locs.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue  # Box is already at goal

            # Get minimal pushes for box to reach goal
            pushes = self.distance[current_loc].get(goal_loc, float("inf"))
            if pushes == float("inf"):
                return float("inf")

            # Get robot's distance to the box's current location
            robot_dist = self.distance[robot_loc].get(current_loc, float("inf"))
            if robot_dist == float("inf"):
                return float("inf")
            robot_adjacent_cost = max(0, robot_dist - 1)

            total += pushes + robot_adjacent_cost

        return total
