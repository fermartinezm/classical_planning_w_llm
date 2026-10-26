from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class sokoban2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It calculates the minimal robot movement to reach each box and the minimal pushes needed for each box to reach its goal.

    # Assumptions
    - Each box must be pushed to its specific goal location.
    - The robot can move freely through empty cells.
    - The shortest path for the robot and boxes is computed using precalculated distances based on the static adjacency information.

    # Heuristic Initialization
    - Extracts adjacency information from static facts to build directed forward and reverse adjacency lists.
    - Precomputes shortest paths between all pairs of locations using BFS for efficient lookup.
    - Extracts goal locations for each box from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, check if it's already at its goal. If not:
        a. Compute the minimal distance (D) from the box's current location to its goal using precomputed shortest paths.
        b. Determine all adjacent cells from which the robot can push the box (reverse adjacency).
        c. Compute the minimal distance (R) from the robot's current location to any of these adjacent cells.
        d. Add R + D to the heuristic value.
    2. Sum the values for all boxes to get the total heuristic estimate.
    """

    def __init__(self, task):
        self.goal_locations = {}
        self.forward_adjacency = defaultdict(list)
        self.reverse_adjacency = defaultdict(list)
        self.distances = {}
        self.locations = set()

        # Extract goal locations for each box
        for goal in task.goals:
            parts = goal.strip('()').split()
            if parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Extract adjacency from static facts
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'adjacent':
                from_loc = parts[1]
                to_loc = parts[2]
                self.forward_adjacency[from_loc].append(to_loc)
                self.reverse_adjacency[to_loc].append(from_loc)
                self.locations.add(from_loc)
                self.locations.add(to_loc)

        # Precompute shortest paths for forward adjacency (robot and box movement)
        for start in self.locations:
            queue = deque([start])
            visited = {start: 0}
            while queue:
                current = queue.popleft()
                for neighbor in self.forward_adjacency.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for node, dist in visited.items():
                self.distances[(start, node)] = dist

    def __call__(self, node):
        state = node.state
        robot_loc = None
        box_locations = {}

        # Extract current robot and box positions from state
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                box_locations[box] = loc

        if not robot_loc:
            return float('inf')

        heuristic_value = 0
        for box, current_loc in box_locations.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue

            # Compute box's distance to goal (D)
            D = self.distances.get((current_loc, goal_loc), float('inf'))

            # Find adjacent cells from which the robot can push the box
            adjacent_cells = self.reverse_adjacency.get(current_loc, [])
            if not adjacent_cells:
                R = float('inf')
            else:
                # Compute robot's minimal distance to any adjacent cell
                R = min([self.distances.get((robot_loc, adj), float('inf')) for adj in adjacent_cells])

            if R == float('inf') or D == float('inf'):
                heuristic_value += 1000000  # Penalize unreachable states
            else:
                heuristic_value += R + D

        return heuristic_value
