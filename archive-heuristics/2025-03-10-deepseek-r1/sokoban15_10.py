from fnmatch import fnmatch
from collections import deque
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact."""
    return fact[1:-1].split()

def parse_location(loc_name):
    """Parse a location name like 'loc_x_y' into coordinates (x, y)."""
    parts = loc_name.split('_')
    return (int(parts[1]), int(parts[2]))

class sokoban15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes to their goal positions. For each box, it calculates the Manhattan distance to its goal and the minimal robot path to an adjacent clear cell, summing these values for all boxes.

    # Assumptions
    - Each box's goal is a single static location.
    - The robot can move through any clear location.
    - The Sokoban grid is based on the adjacency relations from static facts.
    - Boxes can only be pushed to adjacent cells if the robot can reach the opposite side.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task's goals.
    - Builds an adjacency map from static 'adjacent' facts to represent possible movements.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the robot's current location and clear cells from the state.
    2. For each box not at its goal:
        a. Compute Manhattan distance from current position to the goal.
        b. Identify adjacent cells to the box that are clear.
        c. Use BFS to find the shortest path from the robot's location to each clear adjacent cell.
        d. Sum the Manhattan distance and the minimal robot path for each box.
    3. Return the total sum as the heuristic value.
    """

    def __init__(self, task):
        """Initialize with goal locations and adjacency map from static data."""
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        self.adjacency_map = defaultdict(set)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                from_loc, to_loc = parts[1], parts[2]
                self.adjacency_map[from_loc].add(to_loc)
                self.adjacency_map[to_loc].add(from_loc)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        robot_loc = None
        clear_locs = set()
        boxes = {}

        # Extract robot location, clear cells, and box positions
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'clear':
                clear_locs.add(parts[1])
            elif parts[0] == 'at' and parts[1] in self.goal_locations:
                boxes[parts[1]] = parts[2]

        if not robot_loc:
            return 0

        # BFS to compute distances from robot to clear locations
        distances = {robot_loc: 0}
        queue = deque([robot_loc])
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            for neighbor in self.adjacency_map.get(current, []):
                if neighbor not in distances and neighbor in clear_locs:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        total = 0
        for box, current_loc in boxes.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue

            try:
                current_x, current_y = parse_location(current_loc)
                goal_x, goal_y = parse_location(goal_loc)
                manhattan = abs(current_x - goal_x) + abs(current_y - goal_y)
            except:
                continue

            adjacent = self.adjacency_map.get(current_loc, [])
            clear_adjacent = [loc for loc in adjacent if loc in clear_locs]
            if not clear_adjacent:
                continue

            min_dist = min((distances.get(loc, float('inf')) for loc in clear_adjacent), default=float('inf'))
            if min_dist == float('inf'):
                continue

            total += manhattan + min_dist

        return total
