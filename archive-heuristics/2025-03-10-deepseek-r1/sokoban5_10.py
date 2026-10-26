from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def parse_location(loc):
    """Parse location name into (x, y) coordinates."""
    parts = loc.split('_')
    return (int(parts[1]), int(parts[2]))


class Sokoban5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed by summing the minimal robot movement to reach each box's adjacent cell and the box's distance to its goal. The minimal robot movement is computed based on the shortest Manhattan distance to any adjacent cell of the box. The box's distance is the Manhattan distance to its goal.

    # Assumptions
    - Each box must be pushed in a straight path to its goal (Manhattan distance).
    - The robot can move freely to any adjacent cell (ignoring obstacles for heuristic purposes).
    - Static adjacency information is used to determine valid adjacent cells for each location.

    # Heuristic Initialization
    - Extract adjacency information between locations from static facts.
    - Extract goal locations for each box from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, determine its current location and goal location.
    2. Calculate the Manhattan distance from the box's current location to its goal.
    3. Identify all adjacent cells to the box's current location using precomputed adjacency data.
    4. Compute the minimal Manhattan distance from the robot's current location to any of these adjacent cells.
    5. Sum the minimal robot distance and box distance for all boxes to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with adjacency map and goal locations."""
        self.adjacent_map = defaultdict(list)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                self.adjacent_map[loc1].append(loc2)

        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Find robot's current location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
                break
        if not robot_loc:
            return float('inf')

        total = 0
        robot_x, robot_y = parse_location(robot_loc)

        for box, goal_loc in self.goal_locations.items():
            # Find current location of the box
            current_box_loc = None
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == box and len(parts) == 3:
                    current_box_loc = parts[2]
                    break
            if not current_box_loc or current_box_loc == goal_loc:
                continue

            # Get adjacent locations to the box
            adjacent_locs = self.adjacent_map.get(current_box_loc, [])
            if not adjacent_locs:
                return float('inf')

            # Calculate minimal robot distance to any adjacent cell of the box
            min_robot_dist = float('inf')
            for adj_loc in adjacent_locs:
                adj_x, adj_y = parse_location(adj_loc)
                distance = abs(robot_x - adj_x) + abs(robot_y - adj_y)
                if distance < min_robot_dist:
                    min_robot_dist = distance

            # Calculate box's distance to goal
            current_x, current_y = parse_location(current_box_loc)
            goal_x, goal_y = parse_location(goal_loc)
            box_dist = abs(current_x - goal_x) + abs(current_y - goal_y)

            total += min_robot_dist + box_dist

        return total
