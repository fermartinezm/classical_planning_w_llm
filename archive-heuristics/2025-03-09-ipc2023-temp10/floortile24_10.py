from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortile24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the distances the robot needs to move to reach the unpainted tiles.

    # Assumptions
    - The robot can only paint adjacent tiles.
    - The robot needs to have the correct color before painting.
    - The robot can change colors as needed.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles from static facts.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. For each tile that needs to be painted:
       - Check if the tile is already painted with the correct color in the current state. If so, skip it.
       - If the tile is not painted or painted with the wrong color:
         - Calculate the Manhattan distance from the robot's current location to the unpainted tile.
         - Estimate the number of moves required to reach the tile based on the Manhattan distance.
         - If the robot does not have the correct color, estimate the cost of changing the color.
         - Add the cost of painting the tile (1 action).
    3. Sum up the costs for all tiles to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal tile colors
        self.goal_tile_colors = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile = get_parts(goal)[1]
                color = get_parts(goal)[2]
                self.goal_tile_colors[tile] = color

        # Extract adjacency information
        self.adj = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)
            elif match(fact, "down", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)
            elif match(fact, "left", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)
            elif match(fact, "right", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)

        # Extract available colors
        self.available_colors = set()
        for fact in static_facts:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        total_cost = 0

        # Get robot location
        robot_location = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_location = get_parts(fact)[2]
                break

        # Get robot color
        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        # Check goal tiles
        for tile, goal_color in self.goal_tile_colors.items():
            painted_correctly = False
            for fact in state:
                if match(fact, "painted", tile, goal_color):
                    painted_correctly = True
                    break

            if not painted_correctly:
                # Estimate cost to paint this tile
                move_cost = self.estimate_move_cost(robot_location, tile)

                color_change_cost = 0
                if robot_color != goal_color:
                    color_change_cost = 1  # Assume 1 action to change color

                paint_cost = 1  # 1 action to paint

                total_cost += move_cost + color_change_cost + paint_cost

        return total_cost

    def estimate_move_cost(self, start_tile, goal_tile):
        """Estimate the number of moves required to reach the goal tile."""
        # Simple heuristic: Assume direct path (Manhattan distance)
        # This is a simplification and might not be accurate, but it's fast.
        # In a grid-like world, Manhattan distance can be a reasonable estimate.
        # More sophisticated pathfinding algorithms (e.g., A*) could be used for better accuracy,
        # but they would be more computationally expensive.

        # Find the shortest path using BFS
        queue = [(start_tile, 0)]
        visited = {start_tile}

        while queue:
            current_tile, distance = queue.pop(0)
            if current_tile == goal_tile:
                return distance

            if current_tile in self.adj:
                for neighbor in self.adj[current_tile]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
                        visited.add(neighbor)

        return float('inf')  # If no path is found
