from collections import deque
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push each box to its target location. It considers both the shortest path for the robot to reach each box and the Manhattan distance each box needs to be pushed.

    # Assumptions:
    - The robot can move freely between clear locations.
    - Each box can be pushed one step at a time.
    - The target location for each box is known and clear.

    # Heuristic Initialization
    - Extract the target location for each box from the goal conditions.
    - Build an adjacency graph from the static facts to model the grid structure.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, determine its current location and target location.
    2. Compute the shortest path from the robot's current location to the box's current location using BFS on the adjacency graph.
    3. Compute the Manhattan distance from the box's current location to its target location.
    4. Sum these two distances for each box to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # Extract goal locations for each box
        self.goals = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                loc = parts[2]
                self.goals[box] = loc

        # Build adjacency graph from static facts
        self.adjacency_graph = {}
        for fact in task.static:
            if fact.startswith('(adjacent '):
                loc1, loc2, _ = fact[9:-1].split()
                if loc1 not in self.adjacency_graph:
                    self.adjacency_graph[loc1] = []
                self.adjacency_graph[loc1].append(loc2)
                if loc2 not in self.adjacency_graph:
                    self.adjacency_graph[loc2] = []
                self.adjacency_graph[loc2].append(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Extract robot's current location
        current_robot_loc = None
        for fact in state:
            if fact.startswith('(at-robot '):
                current_robot_loc = fact[9:-1].split()[0]
                break
        if current_robot_loc is None:
            return float('inf')  # Robot not found, invalid state

        # Extract current locations of boxes
        box_locations = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[4:-1].split()
                if len(parts) == 2:
                    box, loc = parts
                    box_locations[box] = loc

        # For each box, compute the heuristic contribution
        for box, target in self.goals.items():
            current_box_loc = box_locations.get(box, None)
            if current_box_loc is None:
                continue  # Box not present in state
            if current_box_loc == target:
                continue  # Box is already at target

            # Compute shortest path from robot to box
            robot_to_box = self.bfs_shortest_path(current_robot_loc, current_box_loc)
            if robot_to_box == float('inf'):
                return float('inf')  # No path, state is unsolvable

            # Compute Manhattan distance from box to target
            box_to_target = self.manhattan_distance(current_box_loc, target)

            total_cost += robot_to_box + box_to_target

        return total_cost

    def bfs_shortest_path(self, start, end):
        """Compute the shortest path length using BFS."""
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current == end:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.adjacency_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')  # No path found

    def manhattan_distance(self, loc1, loc2):
        """Compute the Manhattan distance between two locations."""
        def parse_loc(loc):
            parts = loc.split('_')
            return int(parts[1]), int(parts[2])
        x1, y1 = parse_loc(loc1)
        x2, y2 = parse_loc(loc2)
        return abs(x1 - x2) + abs(y1 - y2)
