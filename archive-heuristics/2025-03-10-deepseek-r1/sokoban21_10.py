from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class sokoban21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It computes the minimal robot movement to reach each box and the minimal pushes needed for each box to reach its goal.

    # Assumptions
    - The robot can move freely between adjacent cells as per the static adjacency.
    - Each box's path to the goal is computed based on the static adjacency, ignoring dynamic obstacles (other boxes).
    - The heuristic sums the robot's distance to each box's push start position and the box's distance to its goal.

    # Heuristic Initialization
    - Extracts static adjacency facts to build a directed graph of locations.
    - Precomputes all-pairs shortest paths for robot movement and box pushes using BFS.
    - Extracts goal positions for each box from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, return 0.
    2. Extract the robot's current position from the state.
    3. Extract each box's current position from the state.
    4. For each box not at its goal:
        a. Compute the shortest path (B_dist) from the box's current position to its goal using precomputed paths.
        b. Find all possible predecessor locations (adjacent cells) from which the robot can push the box.
        c. Compute the minimal robot distance (R_dist) from the robot's current position to any predecessor.
        d. Add B_dist + R_dist to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information and precompute shortest paths."""
        self.goals = task.goals
        self.static = task.static

        # Extract all box goals (at boxX locY)
        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and parts[1].startswith('box'):
                self.box_goals[parts[1]] = parts[2]

        # Build adjacency list and reverse adjacency list
        self.adjacency = defaultdict(list)
        self.adjacency_reverse = defaultdict(list)
        locations = set()
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                l1, l2 = parts[1], parts[2]
                self.adjacency[l1].append(l2)
                self.adjacency_reverse[l2].append(l1)
                locations.update([l1, l2])

        self.all_locations = list(locations)

        # Precompute all-pairs shortest paths using BFS for each location
        self.shortest_paths = {}
        for loc in self.all_locations:
            self.shortest_paths[loc] = self.bfs(loc)

    def bfs(self, start):
        """Perform BFS from start location and return distances to all reachable locations."""
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Check if all goals are satisfied
        if all(goal in state for goal in self.goals):
            return 0

        # Extract robot's current position
        robot_pos = None
        for fact in state:
            if fact.startswith('(at-robot '):
                robot_pos = fact[1:-1].split()[1]
                break
        if not robot_pos:
            return float('inf')

        # Extract current box positions
        box_positions = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                box = parts[1]
                if box in self.box_goals:
                    box_positions[box] = parts[2]

        total = 0
        for box, goal_loc in self.box_goals.items():
            current_loc = box_positions.get(box, None)
            if not current_loc or current_loc == goal_loc:
                continue  # Skip if box is not present or already at goal

            # Compute B_dist: box's distance to goal
            b_dist = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))

            # Compute R_dist: minimal distance from robot to any predecessor of current_loc
            predecessors = self.adjacency_reverse.get(current_loc, [])
            r_dist = float('inf')
            for pred in predecessors:
                pred_dist = self.shortest_paths.get(robot_pos, {}).get(pred, float('inf'))
                if pred_dist < r_dist:
                    r_dist = pred_dist

            # Add to total, use a large value if unreachable
            if r_dist == float('inf') or b_dist == float('inf'):
                total += 1000  # Penalize unreachable paths
            else:
                total += r_dist + b_dist

        return total
