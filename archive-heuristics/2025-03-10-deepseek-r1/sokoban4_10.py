from fnmatch import fnmatch
from collections import deque
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokoban4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It calculates the minimal robot movement to reach each box and the minimal pushes needed for each box.

    # Assumptions
    - Each box requires a series of pushes to reach its goal.
    - The robot can move freely through clear cells (locations without boxes).
    - The minimal path for the robot to reach a box's adjacent cell is computed via BFS.
    - The number of pushes for a box is approximated by the Manhattan distance between its current position and its goal.

    # Heuristic Initialization
    - Extract static adjacency information between locations to build a movement graph.
    - Parse location coordinates from their names (e.g., loc_x_y).
    - Extract goal locations for each box from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current State Information**:
       - Identify the robot's current location.
       - Determine the current location of each box.
       - Identify all clear cells (locations without boxes).

    2. **Robot Movement BFS**:
       - Perform BFS from the robot's current location, considering only clear cells as passable.
       - This computes the minimal steps required for the robot to reach any clear cell.

    3. **Box Processing**:
       - For each box not at its goal:
         a. Find all adjacent cells to the box's current location.
         b. Filter adjacent cells to those that are clear (robot can move there).
         c. Find the minimal robot steps to reach any clear adjacent cell using BFS results.
         d. Compute Manhattan distance from the box's current location to its goal.
         e. Add the sum of robot steps and Manhattan distance to the total heuristic.

    4. **Summing Costs**:
       - The total heuristic is the sum of costs for all boxes, providing an estimate of the total actions needed.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts
        self.adjacency = defaultdict(list)
        self.location_coords = {}
        locations = set()
        for fact in task.static:
            if match(fact, 'adjacent', '*', '*', '*'):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.adjacency[l1].append(l2)
                locations.update([l1, l2])

        # Parse coordinates for each location
        for loc in locations:
            parts = loc.split('_')
            if len(parts) >= 3:
                x, y = int(parts[1]), int(parts[2])
                self.location_coords[loc] = (x, y)

    def __call__(self, node):
        state = node.state
        robot_loc = None
        boxes = {}
        clear_cells = set()

        # Extract robot location, box positions, and clear cells
        for fact in state:
            if match(fact, 'at-robot', '*'):
                robot_loc = get_parts(fact)[1]
            elif match(fact, 'at', '*', '*'):
                parts = get_parts(fact)
                boxes[parts[1]] = parts[2]
            elif match(fact, 'clear', '*'):
                clear_cells.add(get_parts(fact)[1])

        if not robot_loc:
            return 0  # No robot, invalid state

        # BFS to compute robot distances to all clear cells
        visited = {}
        queue = deque()
        if robot_loc in clear_cells:
            visited[robot_loc] = 0
            queue.append(robot_loc)
            while queue:
                current = queue.popleft()
                current_dist = visited[current]
                for neighbor in self.adjacency.get(current, []):
                    if neighbor in clear_cells and neighbor not in visited:
                        visited[neighbor] = current_dist + 1
                        queue.append(neighbor)

        total = 0
        for box, current_loc in boxes.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue  # Box is at goal or no goal defined

            # Compute Manhattan distance for the box
            current_coords = self.location_coords.get(current_loc)
            goal_coords = self.location_coords.get(goal_loc)
            if not current_coords or not goal_coords:
                continue  # Invalid coordinates
            md = abs(current_coords[0] - goal_coords[0]) + abs(current_coords[1] - goal_coords[1])

            # Find adjacent cells to the box's current location
            adjacent = self.adjacency.get(current_loc, [])
            clear_adjacent = [cell for cell in adjacent if cell in clear_cells]
            if not clear_adjacent:
                total += 1000 + md  # Penalize if no adjacent clear cells
                continue

            # Find minimal robot distance to any clear adjacent cell
            min_dist = min(visited.get(cell, float('inf')) for cell in clear_adjacent)
            if min_dist == float('inf'):
                total += 1000 + md  # No path found
                continue

            total += min_dist + md

        return total
