from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach the tiles.

    # Assumptions:
    - Each robot can only hold one color at a time.
    - Robots must change colors if they don't have the required color.
    - Robots must move to adjacent tiles to paint them.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles from static facts.
    - Extract available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted and their target colors from the goal state.
    2. For each robot, determine its current location and the color it is holding.
    3. For each tile that needs to be painted:
       a. If the tile is already painted with the correct color, no action is needed.
       b. If the robot at the current location has the correct color, estimate the cost to paint the tile.
       c. If the robot does not have the correct color, estimate the cost to change the color and then paint the tile.
       d. If no robot is at an adjacent location, estimate the cost to move a robot to an adjacent location, change color if needed, and then paint the tile.
    4. Sum up the estimated costs for all tiles to get the overall heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.painted_goals = {}
        for goal in self.goals:
            if goal.startswith('(painted'):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.painted_goals[tile] = color

        self.adjacency = {}
        for fact in self.static:
            if fact.startswith('(up') or fact.startswith('(down') or fact.startswith('(left') or fact.startswith('(right'):
                parts = fact[1:-1].split()
                relation = parts[0]
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.adjacency:
                    self.adjacency[tile1] = {}
                self.adjacency[tile1][relation] = tile2

        self.available_colors = set()
        for fact in self.static:
            if fact.startswith('(available-color'):
                parts = fact[1:-1].split()
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        if self.goal_reached(state):
            return 0

        robot_locations = {}
        robot_colors = {}
        for fact in state:
            if fact.startswith('(robot-at'):
                parts = fact[1:-1].split()
                robot = parts[1]
                tile = parts[2]
                robot_locations[robot] = tile
            elif fact.startswith('(robot-has'):
                parts = fact[1:-1].split()
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        painted_tiles = {}
        for fact in state:
            if fact.startswith('(painted'):
                parts = fact[1:-1].split()
                tile = parts[1]
                color = parts[2]
                painted_tiles[tile] = color

        cost = 0
        for tile, target_color in self.painted_goals.items():
            if tile in painted_tiles and painted_tiles[tile] == target_color:
                continue

            min_robot_cost = float('inf')
            for robot, location in robot_locations.items():
                robot_cost = 0
                current_color = robot_colors[robot]

                if current_color != target_color:
                    robot_cost += 1  # Color change

                # Check if the robot is at an adjacent tile
                adjacent = False
                for relation, adjacent_tile in self.adjacency.get(location, {}).items():
                    if adjacent_tile == tile:
                        adjacent = True
                        break
                
                if not adjacent:
                    # Find the shortest path to an adjacent tile
                    robot_cost = float('inf')
                    
                    #Check if the robot is already adjacent to the tile
                    for relation in ['up', 'down', 'left', 'right']:
                        if relation in self.adjacency.get(location, {}) and self.adjacency[location][relation] == tile:
                            robot_cost = 0
                            break
                    
                    if robot_cost == float('inf'):
                        # If not adjacent, then find the closest tile to the robot and move to it
                        robot_cost = float('inf')
                        
                        #If the robot is not adjacent to the tile, then move to the closest tile
                        robot_cost = 2 #Move to the closest tile
                
                if robot_cost != float('inf'):
                    robot_cost += 1 #Paint the tile
                    min_robot_cost = min(min_robot_cost, robot_cost)

            cost += min_robot_cost

        return cost

    def goal_reached(self, state):
        """Check if the goal has been reached."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True
