from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class sokoban20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal positions. For each box, it calculates the minimal robot distance to reach an adjacent clear cell and the box's shortest path to its goal. The sum of these values for all boxes gives the heuristic estimate.

    # Assumptions:
    - The robot can move freely between adjacent cells.
    - Each box requires being pushed along its shortest path to the goal.
    - The robot's movement to the box's adjacent cell is required once per box, assuming sequential pushing.

    # Heuristic Initialization
    - Precompute the adjacency graph from static facts.
    - Compute all-pairs shortest paths (APSP) for efficient distance lookups.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current positions of the robot and boxes from the state.
    2. For each box not at its goal:
        a. Find all adjacent cells to the box that are clear.
        b. Compute the minimal robot distance to any of these cells.
        c. Compute the box's shortest path distance to its goal.
        d. Add these two distances to the total heuristic.
    3. Sum the values for all boxes to get the final heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic with static data and precompute shortest paths."""
        self.goal_locations = {}
        self.static = task.static
        self.adjacency_graph = self._build_adjacency_graph()
        self.shortest_paths = self._compute_all_shortest_paths()

        # Extract goal locations for each box from the task's goals
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc

    def _build_adjacency_graph(self):
        """Build an undirected graph from the static adjacency facts."""
        graph = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                from_loc, to_loc = parts[1], parts[2]
                if from_loc not in graph:
                    graph[from_loc] = set()
                graph[from_loc].add(to_loc)
                if to_loc not in graph:
                    graph[to_loc] = set()
                graph[to_loc].add(from_loc)
        return graph

    def _compute_all_shortest_paths(self):
        """Precompute shortest paths between all pairs of locations using BFS."""
        shortest_paths = {}
        for location in self.adjacency_graph:
            shortest_paths[location] = self._bfs(location)
        return shortest_paths

    def _bfs(self, start):
        """Compute shortest paths from start to all other locations."""
        visited = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.adjacency_graph.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        return visited

    def _get_distance(self, from_loc, to_loc):
        """Return the shortest path distance between two locations, or infinity if unreachable."""
        if from_loc not in self.shortest_paths:
            return float('inf')
        return self.shortest_paths[from_loc].get(to_loc, float('inf'))

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        robot_pos = None
        boxes = {}
        clear_locations = set()

        # Extract current state information
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at-robot':
                robot_pos = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                boxes[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                clear_locations.add(parts[1])

        if not robot_pos:
            return float('inf')

        total = 0
        for box, current_loc in boxes.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue  # Box is already at goal

            # Find all adjacent cells to the box's current location
            adjacent = self.adjacency_graph.get(current_loc, [])
            # Filter to clear adjacent cells
            clear_adjacent = [loc for loc in adjacent if loc in clear_locations]

            if not clear_adjacent:
                # No clear adjacent cells, box is stuck
                return float('inf')

            # Find minimal robot distance to any clear adjacent cell
            min_robot_dist = min(self._get_distance(robot_pos, adj) for adj in clear_adjacent)

            # Box's distance to goal
            box_dist = self._get_distance(current_loc, goal_loc)

            total += min_robot_dist + box_dist

        return total
