from heuristics.heuristic_base import Heuristic
from collections import defaultdict

# Helper function to extract parts of a PDDL fact
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Simple split is sufficient for this domain's fact structure.
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums the cost of painting each unpainted goal tile,
    the cost of changing robot colors when needed, and a minimal movement cost
    for tiles that no robot is currently adjacent to.

    # Assumptions
    - The goal is to paint a specific set of tiles with specific colors.
    - Tiles must be clear to be painted or moved onto.
    - Robots can only paint adjacent tiles.
    - Robots carry one color at a time and can change color if available.
    - The grid structure is defined by 'up', 'down', 'left', 'right' facts.
    - Problems are solvable and do not require repainting tiles painted with the wrong color (as goal only specifies correct color).

    # Heuristic Initialization
    - Extracts the set of goal (painted tile, color) pairs.
    - Builds an adjacency map of the grid from static 'up', 'down', 'left', 'right' facts.
    - Stores available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that are not currently painted with the required color.
    2. Initialize the heuristic value to 0.
    3. Count the number of unique colors required by the unpainted goal tiles. For each such color, if no robot currently possesses this color, add 1 to the heuristic (representing the cost of a 'change_color' action by some robot). This counts the cost per color type, not per robot or per tile.
    4. For each unpainted goal tile:
       a. Add 1 to the heuristic (representing the cost of the 'paint' action).
       b. Check if any robot is currently located on a tile directly adjacent to this goal tile (using the pre-computed adjacency map).
       c. If no robot is adjacent to the goal tile, add 1 to the heuristic (representing a minimal movement cost required for *some* robot to reach an adjacent tile). This is a simplified movement cost that avoids complex pathfinding or assignment problems.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, grid structure,
        and available colors from the task.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal tiles and their required colors
        self.goal_tiles = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles.add((tile, color))

        # Build adjacency map from static facts
        self.adj_map = defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                self.adj_map[tile1].add(tile2)
                self.adj_map[tile2].add(tile1) # Grid is undirected for movement

        # Store available colors
        self.available_colors = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "available-color":
                 self.available_colors.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify unpainted goal tiles
        unpainted_goal_tiles = set()
        current_painted = {} # Map tile -> color if painted
        robot_locations = {} # Map robot -> tile
        robots_with_color = {c: set() for c in self.available_colors} # Map color -> set of robots

        # Extract relevant information from the current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            elif parts[0] == "robot-at":
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                if color in robots_with_color: # Only track available colors
                    robots_with_color[color].add(robot)

        # Determine which goal tiles are not yet painted correctly
        for goal_tile, goal_color in self.goal_tiles:
            # A goal tile needs painting if it's not currently painted with the correct color.
            # This includes tiles that are not painted at all, or painted with a different color.
            # Assuming solvable problems, tiles painted with the wrong color are not goal tiles,
            # or there's an implicit way to clear them (not shown in domain).
            # We proceed assuming we only care about tiles that should be (painted T C)
            # and currently are not.
            if goal_tile not in current_painted or current_painted[goal_tile] != goal_color:
                 unpainted_goal_tiles.add((goal_tile, goal_color))


        # If all goal tiles are painted correctly, the heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        h = 0

        # Calculate color change cost
        needed_colors_for_unpainted = {color for tile, color in unpainted_goal_tiles}
        colors_to_acquire = {color for color in needed_colors_for_unpainted if not robots_with_color.get(color, set())} # Use .get for safety
        h += len(colors_to_acquire) # Add 1 for each color that no robot currently has

        # Calculate paint and minimal movement cost per tile
        for goal_tile, goal_color in unpainted_goal_tiles:
            h += 1 # Cost for the paint action itself

            # Check if any robot is currently adjacent to this goal tile
            any_robot_adjacent = False
            adjacent_tiles = self.adj_map.get(goal_tile, set())
            if adjacent_tiles: # Check if the tile has any neighbors defined
                for robot, location in robot_locations.items():
                    if location in adjacent_tiles:
                        any_robot_adjacent = True
                        break

            # If no robot is adjacent, add a minimal movement cost (e.g., 1)
            # This is a simple estimate that a robot needs at least one move to get adjacent.
            if not any_robot_adjacent:
                 h += 1

        return h
