from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

class sokoban18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal positions.
    For each box, it calculates the minimal robot moves to reach a position from which the box can be pushed,
    adds one action for the push, and adds the minimal pushes required to move the box to its goal.

    # Assumptions:
    - The maze structure (adjacent cells) is static and precomputed.
    - Boxes can only be pushed, not pulled.
    - The heuristic assumes paths are available (ignores dynamic obstacles like other boxes).

    # Heuristic Initialization
    - Extracts box goals from the task.
    - Builds forward and reverse adjacency graphs from static 'adjacent' facts.
    - Precomputes shortest paths between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Determine all possible positions the robot can be in to push the box (reverse adjacency).
        b. Find the minimal distance from the robot's current position to any of these push positions.
        c. Calculate the minimal number of pushes needed to move the box from its current position to the goal.
        d. Sum the robot's distance, one push action, and the box's distance to the goal.
    2. Sum the values for all boxes to get the total heuristic estimate.
    """

    def __init__(self, task):
        self.goals = {}
        for goal in task.goals:
            if goal.startswith('(at '):
                parts = goal[1:-1].split()
                if parts[0] == 'at' and parts[2] == 'box':
                    box = parts[1]
                    loc = parts[3]
                    self.goals[box] = loc
                else:
                    box = parts[1]
                    loc = parts[2]
                    self.goals[box] = loc

        self.forward_adj = defaultdict(list)
        self.reverse_adj = defaultdict(list)
        for fact in task.static:
            if fact.startswith('(adjacent '):
                parts = fact[1:-1].split()
                from_loc = parts[1]
                to_loc = parts[2]
                self.forward_adj[from_loc].append(to_loc)
                self.reverse_adj[to_loc].append(from_loc)

        self.distances = defaultdict(dict)
        all_locations = set(self.forward_adj.keys()).union(set(self.reverse_adj.keys()))
        for start in all_locations:
            visited = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for neighbor in self.forward_adj[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for loc, dist in visited.items():
                self.distances[start][loc] = dist

    def __call__(self, node):
        state = node.state
        robot_loc = None
        for fact in state:
            if fact.startswith('(at-robot '):
                robot_loc = fact[1:-1].split()[1]
                break
        if not robot_loc:
            return float('inf')

        boxes = {}
        for fact in state:
            if fact.startswith('(at ') and not fact.startswith('(at-robot'):
                parts = fact[1:-1].split()
                box = parts[1]
                loc = parts[2]
                boxes[box] = loc

        total = 0
        for box, box_loc in boxes.items():
            goal_loc = self.goals.get(box)
            if not goal_loc or box_loc == goal_loc:
                continue

            push_positions = self.reverse_adj.get(box_loc, [])
            if not push_positions:
                return float('inf')

            min_robot_dist = min(
                [self.distances[robot_loc].get(pos, float('inf')) for pos in push_positions],
                default=float('inf')
            )
            if min_robot_dist == float('inf'):
                return float('inf')

            box_to_goal = self.distances[box_loc].get(goal_loc, float('inf'))
            if box_to_goal == float('inf'):
                return float('inf')

            total += min_robot_dist + 1 + box_to_goal

        return total
