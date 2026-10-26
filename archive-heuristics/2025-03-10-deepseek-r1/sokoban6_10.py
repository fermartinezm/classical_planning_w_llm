from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class sokoban6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It calculates the sum of the robot's distance to each box plus the box's distance to its goal, considering the shortest paths in the directed adjacency graph.

    # Assumptions:
    - The robot can move freely through empty cells.
    - Boxes can be pushed in any order, ignoring other boxes (relaxed problem).
    - The shortest path between locations is precomputed using BFS on the directed adjacency graph.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task's goals.
    - Parses static adjacency facts to build a directed graph of locations.
    - Precomputes shortest paths between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Precompute Shortest Paths**:
       - Build a directed adjacency graph from static 'adjacent' facts.
       - For each location, perform BFS to find the shortest path to all other locations.

    2. **Extract Current State Information**:
       - Find the robot's current location from 'at-robot' facts.
       - For each box, determine its current location from 'at' facts.

    3. **Calculate Distances**:
       - For each box not at its goal:
         - Compute the robot's shortest path distance to the box's current location.
         - Compute the box's shortest path distance to its goal location.
         - Sum these two distances for the box.

    4. **Sum All Box Costs**:
       - The heuristic value is the sum of all individual box costs.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information and precompute distances."""
        self.goal_locations = {}
        # Extract box goals from task's goals
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and parts[2].startswith('loc_'):
                box = parts[1]
                location = parts[2]
                self.goal_locations[box] = location

        # Build directed adjacency graph from static facts
        self.adjacency = defaultdict(list)
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                from_loc = parts[1]
                to_loc = parts[2]
                self.adjacency[from_loc].append(to_loc)

        # Precompute shortest paths between all locations
        self.distances = defaultdict(dict)
        locations = set(self.adjacency.keys())
        for loc in locations:
            self.bfs(loc)

    def bfs(self, start):
        """Perform BFS from start location and record distances to all reachable locations."""
        queue = deque([(start, 0)])
        visited = {start: 0}
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        self.distances[start] = visited

    def get_distance(self, from_loc, to_loc):
        """Return the shortest distance from from_loc to to_loc, or a large number if unreachable."""
        return self.distances.get(from_loc, {}).get(to_loc, float('inf'))

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        robot_loc = None
        box_locations = {}

        # Extract robot's current location
        for fact in state:
            if fact.startswith('(at-robot '):
                parts = fact[1:-1].split()
                robot_loc = parts[1]
                break

        # Extract box locations
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        total = 0
        for box, current_loc in box_locations.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue  # Box is already at goal

            # Robot's distance to the box's current location
            robot_dist = self.get_distance(robot_loc, current_loc)
            # Box's distance to its goal
            box_dist = self.get_distance(current_loc, goal_loc)

            total += robot_dist + box_dist

        return total
