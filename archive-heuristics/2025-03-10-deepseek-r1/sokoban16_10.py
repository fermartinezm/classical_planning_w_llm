from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class sokoban16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It combines the minimal robot movement to reach the nearest box and the sum of the minimal pushes required for each box.

    # Assumptions
    - The Sokoban grid is based on the static adjacency facts, allowing bidirectional movement between connected locations.
    - Each box's path to its goal is computed ignoring other boxes, considering only static obstacles.
    - The robot's movement to a box's adjacent cell is the minimal path from its current position.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task's goals.
    - Builds an adjacency graph from static 'adjacent' facts.
    - Precomputes all-pairs shortest paths for static locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, calculate the shortest path from its current location to its goal (number of pushes needed).
    2. For each box, compute the robot's minimal distance to any adjacent cell of the box's current location.
    3. Sum the pushes for all boxes and add the minimal robot movement cost to initiate the first push.
    4. If all boxes are at their goals, return 0. If any box or robot path is unreachable, return infinity.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = self._get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('box'):
                self.goal_locations[parts[1]] = parts[2]

        self.adjacency = defaultdict(list)
        for fact in task.static:
            if self._match(fact, 'adjacent', '*', '*', '*'):
                l1, l2 = self._get_parts(fact)[1], self._get_parts(fact)[2]
                self.adjacency[l1].append(l2)
                self.adjacency[l2].append(l1)

        self.distances = defaultdict(dict)
        all_locs = set(self.adjacency.keys())
        for loc in all_locs:
            visited = {loc: 0}
            queue = deque([loc])
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacency[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for node, dist in visited.items():
                self.distances[loc][node] = dist

    def _get_parts(self, fact):
        return fact[1:-1].split()

    def _match(self, fact, *args):
        parts = self._get_parts(fact)
        return len(parts) == len(args) and all(fnmatch(p, a) for p, a in zip(parts, args))

    def __call__(self, node):
        state = node.state
        robot_loc = None
        for fact in state:
            if self._match(fact, 'at-robot', '*'):
                robot_loc = self._get_parts(fact)[1]
                break
        if not robot_loc:
            return float('inf')

        box_locs = {}
        for fact in state:
            if self._match(fact, 'at', 'box*', '*'):
                parts = self._get_parts(fact)
                box_locs[parts[1]] = parts[2]

        total_box_dist = 0
        min_robot_cost = float('inf')

        for box, current_loc in box_locs.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue

            box_dist = self.distances[current_loc].get(goal_loc, float('inf'))
            if box_dist == float('inf'):
                return float('inf')
            total_box_dist += box_dist

            adjacent = self.adjacency.get(current_loc, [])
            if not adjacent:
                return float('inf')
            min_adj_dist = min([self.distances[robot_loc].get(adj, float('inf')) for adj in adjacent], default=float('inf'))
            if min_adj_dist == float('inf'):
                return float('inf')
            robot_cost = min_adj_dist + 1  # Move to adjacent + push

            if robot_cost < min_robot_cost:
                min_robot_cost = robot_cost

        if total_box_dist == 0:
            return 0
        if min_robot_cost == float('inf'):
            return float('inf')

        return total_box_dist + min_robot_cost
