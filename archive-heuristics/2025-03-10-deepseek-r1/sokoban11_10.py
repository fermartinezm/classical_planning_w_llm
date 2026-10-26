from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

class sokoban11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to solve a Sokoban problem by considering:
    1. The minimal number of pushes required to move each box to its goal.
    2. The minimal number of moves required for the robot to reach the nearest box not yet at its goal.

    # Assumptions
    - The Sokoban grid is based on the adjacency relations provided in the static facts.
    - Boxes can only be pushed to adjacent cells if the path is clear, but the heuristic assumes optimal paths ignoring other boxes.
    - The heuristic does not account for the need to reposition the robot between multiple pushes but focuses on the most immediate required actions.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task's goal conditions.
    - Builds an adjacency graph from static 'adjacent' facts.
    - Precomputes all-pairs shortest path distances using BFS for efficient lookups.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current State**: Identify the robot's current location and the positions of all boxes.
    2. **Check Box Goals**: For each box, determine if it is already at its goal location.
    3. **Box Push Distances**: For boxes not at their goals, sum the shortest path distances from their current positions to their goals.
    4. **Robot Movement Cost**: Compute the minimal distance for the robot to reach any adjacent cell of a box not at its goal, adding one for the push action.
    5. **Combine Costs**: The heuristic value is the sum of all box push distances and the minimal robot movement cost.
    """

    def __init__(self, task):
        self.goal_locations = {}
        self.adjacency = {}
        self.distance = {}

        # Extract goal locations for each box
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                from_loc, to_loc = parts[1], parts[2]
                self.adjacency.setdefault(from_loc, set()).add(to_loc)
                self.adjacency.setdefault(to_loc, set()).add(from_loc)

        # Precompute all-pairs shortest paths using BFS
        all_locations = set(self.adjacency.keys())
        for neighbors in self.adjacency.values():
            all_locations.update(neighbors)
        all_locations = list(all_locations)

        for start in all_locations:
            self.distance[start] = {}
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for loc in all_locations:
                self.distance[start][loc] = visited.get(loc, float('inf'))

    def __call__(self, node):
        state = node.state
        robot_loc = None
        box_locs = {}

        # Extract current robot and box positions
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                box_locs[box] = parts[2]

        if not robot_loc:
            return float('inf')

        sum_box_dist = 0
        boxes_not_done = []
        for box, current_loc in box_locs.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue
            boxes_not_done.append(box)
            sum_box_dist += self.distance[current_loc].get(goal_loc, float('inf'))

        if not boxes_not_done:
            return 0

        # Calculate minimal robot distance to any box's adjacent cell + 1 for push
        min_robot_cost = float('inf')
        for box in boxes_not_done:
            current_loc = box_locs[box]
            for adj in self.adjacency.get(current_loc, []):
                dist = self.distance[robot_loc].get(adj, float('inf'))
                if dist + 1 < min_robot_cost:
                    min_robot_cost = dist + 1

        heuristic_value = sum_box_dist + (min_robot_cost if min_robot_cost != float('inf') else 0)
        return heuristic_value
