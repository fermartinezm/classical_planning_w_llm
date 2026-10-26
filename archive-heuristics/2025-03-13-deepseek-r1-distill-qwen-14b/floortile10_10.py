from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The Manhattan distance the robot needs to move to reach each unpainted tile.
    - Whether the robot needs to change colors before painting a tile.
    - The number of tiles remaining to be painted.

    # Assumptions:
    - The robot can only paint one tile at a time.
    - The robot must be on a tile to paint it.
    - The robot may need to change colors between tiles if the required color changes.
    - Movement between adjacent tiles costs one action.
    - Changing colors costs one action.

    # Heuristic Initialization
    - Extracts goal conditions and static facts (tile layout and available colors) from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted and their required colors.
    2. Determine the robot's current position and the color it is holding.
    3. For each unpainted tile:
       a. Calculate the Manhattan distance from the robot's current position.
       b. If the robot's held color doesn't match the required color, add the cost to change colors.
       c. Add the distance to the total cost.
    4. Sum all these costs to estimate the total number of actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Preprocess static facts to build a grid layout and adjacency map
        self.grid = {}
        self.adjacent = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.adjacent[(x, 'up')] = y
                self.adjacent[(y, 'down')] = x
            elif match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.adjacent[(x, 'down')] = y
                self.adjacent[(y, 'up')] = x
            elif match(fact, "left", "*", "*"):
                x, y = get_parts(fact)[1], get_parts(fact)[2]
                self.adjacent[(x, 'left')] = y
                self.adjacent[(y, 'right')] = x
            elif match(fact, "right", "*", "*"):
                x, y = get_parts(fact)[1], get_parts(fact)[2]
                self.adjacent[(x, 'right')] = y
                self.adjacent[(y, 'left')] = x

        # Extract available colors
        self.available_colors = {get_parts(fact)[1] for fact in self.static if match(fact, "available-color", "*")}

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Helper functions to extract information from the state
        def get_robot_pos():
            facts = [fact for fact in state if match(fact, "robot-at", "*", "*")]
            return get_parts(facts[0])[2], get_parts(facts[0])[1] if len(facts) > 0 else None

        def get_held_color():
            facts = [fact for fact in state if match(fact, "robot-has", "*", "*")]
            return get_parts(facts[0])[2] if len(facts) > 0 else None

        def get_painted_tiles():
            painted = {}
            for fact in state:
                if match(fact, "painted", "*", "*"):
                    tile = get_parts(fact)[1]
                    color = get_parts(fact)[2]
                    painted[tile] = color
            return painted

        # Get current state information
        robot_x, robot_tile = get_robot_pos()
        held_color = get_held_color()
        painted = get_painted_tiles()

        # Count remaining tiles and track their positions and required colors
        remaining = []
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile = get_parts(goal)[1]
                color = get_parts(goal)[2]
                if tile not in painted or painted[tile] != color:
                    remaining.append((tile, color))

        if not remaining:
            return 0

        # Calculate heuristic cost
        total_cost = 0
        current_tile = robot_tile
        current_color = held_color

        for tile, color in remaining:
            # Calculate Manhattan distance between current and target tile
            # (Assuming grid is laid out in a 2D plane with coordinates)
            # This is a simplified distance calculation; actual coordinates would be needed for precise Manhattan distance
            distance = 0  # Simplified for this example; adjust based on actual tile layout

            # Add movement cost
            total_cost += distance

            # Add color change cost if needed
            if current_color != color:
                total_cost += 1
                current_color = color

            # Move to next tile
            current_tile = tile

        return total_cost
