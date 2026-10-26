from collections import deque
from collections import defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It calculates the minimal robot movement to reach each box and the minimal pushes needed for each box to reach its goal.

    # Assumptions
    - Each box must be pushed to a specific goal location.
    - The robot can move freely to any adjacent clear cell.
    - The shortest path for the robot to reach a box and for the box to reach its goal is computed using BFS on the adjacency graph.

    # Heuristic Initialization
    - Extract adjacency relationships from static facts to build a graph of locations.
    - Extract goal locations for each box from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Compute the minimal distance for the robot to reach any adjacent cell of the box's current position.
        b. Compute the minimal distance for the box to be pushed from its current position to its goal.
        c. Sum these distances for all boxes to get the total estimated actions.
    2. If all boxes are at their goals, the heuristic is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic with adjacency graph and goal locations."""
        self.adjacency = defaultdict(set)
        self.goal_locations = {}

        # Build adjacency graph from static facts
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'adjacent':
                l1, l2 = parts[1], parts[2]
                self.adjacency[l1].add(l2)
                self.adjacency[l2].add(l1)

        # Extract goal locations for each box
        for goal in task.goals:
            parts = goal.strip('()').split()
            if parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        robot_loc = None
        box_locs = {}

        # Extract current robot and box positions from the state
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1] in self.goal_locations:
                box = parts[1]
                loc = parts[2]
                box_locs[box] = loc

        if not robot_loc:
            return 0  # Should not happen in valid states

        total = 0
        for box, current_loc in box_locs.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue

            # Robot's distance to adjacent cells of the box's current location
            adjacent_cells = self.adjacency.get(current_loc, set())
            if not adjacent_cells:
                robot_dist = float('inf')
            else:
                robot_dist = self.bfs_min_distance(robot_loc, adjacent_cells)

            # Box's distance to its goal
            box_dist = self.bfs_min_distance(current_loc, {goal_loc})

            total += robot_dist + box_dist

        return total

    def bfs_min_distance(self, start, targets):
        """Compute the minimal distance from start to any target using BFS."""
        if start in targets:
            return 0
        visited = set([start])
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor in targets:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')  # No path found
