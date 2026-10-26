from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

class floortile9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their required colors. It considers the movement costs for robots to reach adjacent tiles of each target, necessary color changes, and the paint actions.

    # Assumptions
    - Robots can move freely between adjacent tiles (ignoring 'clear' conditions for efficiency).
    - Each paint action requires the robot to be on an adjacent tile.
    - Color changes are possible if the required color is available.
    - Robots can work in parallel, but the heuristic sums individual costs for simplicity.

    # Heuristic Initialization
    - Extracts adjacency relations between tiles from static facts.
    - Precomputes shortest paths between all pairs of tiles using BFS.
    - Extracts available colors and goal painting conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Identify Unpainted Tiles**: Check which goal tiles are not yet painted correctly.
    2. **Robot Positions and Colors**: Track each robot's current position and color.
    3. **Adjacent Tiles**: For each target tile, find all adjacent tiles (movement destinations).
    4. **Movement Cost**: Compute the minimal distance from each robot to any adjacent tile of the target.
    5. **Color Change Cost**: Add 1 action if the robot's color doesn't match the target and the color is available.
    6. **Paint Action**: Add 1 action for painting.
    7. **Sum Costs**: Sum the minimal costs for all target tiles.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information."""
        self.static = task.static
        self.adjacency = defaultdict(list)
        self.available_colors = set()
        self.goal_painted = {}
        self.distances = {}

        # Extract adjacency from static facts
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] in ['up', 'down', 'left', 'right']:
                a, b = parts[1], parts[2]
                self.adjacency[a].append(b)
                self.adjacency[b].append(a)  # assuming undirected for movement possibilities

        # Precompute shortest paths between all tiles
        tiles = set(self.adjacency.keys())
        for tile in tiles:
            self.distances[tile] = self.bfs(tile)

        # Extract available colors
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])

        # Extract goal painted conditions
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

    def bfs(self, start):
        """Compute shortest paths from 'start' to all other tiles using BFS."""
        queue = deque([(start, 0)])
        distances = {start: 0}
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_painted = {}
        robots = []

        # Extract current painted tiles and robot states
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            elif parts[0] == 'robot-at':
                robot = parts[1]
                pos = parts[2]
                # Find robot's current color
                color = None
                for f in state:
                    fparts = f[1:-1].split()
                    if fparts[0] == 'robot-has' and fparts[1] == robot:
                        color = fparts[2]
                        break
                robots.append((robot, pos, color))

        # Determine tiles that still need painting
        tiles_to_paint = []
        for tile, req_color in self.goal_painted.items():
            if current_painted.get(tile) != req_color:
                tiles_to_paint.append((tile, req_color))

        total_cost = 0

        for tile, req_color in tiles_to_paint:
            min_cost = float('inf')
            adjacent_tiles = self.adjacency.get(tile, [])
            if not adjacent_tiles:
                continue  # Skip if no adjacent tiles (invalid problem)

            for robot in robots:
                r_pos, r_color = robot[1], robot[2]
                if r_pos not in self.distances:
                    continue  # Robot's position not in precomputed distances

                # Find minimal distance to any adjacent tile
                min_dist = min(
                    (self.distances[r_pos].get(adj, float('inf')) for adj in adjacent_tiles),
                    default=float('inf')
                )
                if min_dist == float('inf'):
                    continue  # Skip unreachable

                # Calculate color change cost
                color_cost = 1 if r_color != req_color and req_color in self.available_colors else 0

                total_robot_cost = min_dist + color_cost + 1  # +1 for paint
                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost

            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
