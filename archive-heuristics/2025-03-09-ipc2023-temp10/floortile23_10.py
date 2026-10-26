from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortile23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors,
    considering the robot's current color, location, and the need to move and change colors.

    # Assumptions:
    - Each tile needs to be painted only once.
    - The robot can only paint adjacent tiles.
    - The robot may need to change colors multiple times.
    - Moving to an adjacent tile costs 1.
    - Painting a tile costs 1.
    - Changing color costs 1.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Determine the available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. For each tile, check if it is already painted with the correct color.
    3. If a tile needs painting:
       a. Check if the robot is at an adjacent tile. If not, estimate the number of moves required to reach an adjacent tile.
       b. Check if the robot has the correct color. If not, estimate the number of color changes required.
       c. Add 1 for the painting action itself.
    4. Sum up the costs for all tiles that need painting.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.adjacency = {}
        for fact in static_facts:
            parts = self._get_parts(fact)
            if parts[0] in ("up", "down", "left", "right"):
                self.adjacency.setdefault(parts[1], []).append((parts[0], parts[2]))

        self.available_colors = set()
        for fact in static_facts:
            parts = self._get_parts(fact)
            if parts[0] == "available-color":
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Estimate the minimum cost to paint all tiles to their goal colors."""
        state = node.state

        goal_tiles = {}
        for goal in self.goals:
            parts = self._get_parts(goal)
            if parts[0] == "painted":
                tile = parts[1]
                color = parts[2]
                goal_tiles[tile] = color

        robot_location = None
        robot_color = None
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == "robot-at":
                robot_location = parts[2]
            elif parts[0] == "robot-has":
                robot_color = parts[2]

        cost = 0
        for tile, goal_color in goal_tiles.items():
            painted_correctly = False
            for fact in state:
                parts = self._get_parts(fact)
                if parts[0] == "painted" and parts[1] == tile and parts[2] == goal_color:
                    painted_correctly = True
                    break

            if not painted_correctly:
                # Check if robot is adjacent to the tile
                adjacent = False
                if robot_location:
                    for direction, adjacent_tile in self.adjacency.get(robot_location, []):
                        if adjacent_tile == tile:
                            adjacent = True
                            break
                    for direction, adjacent_tile in self.adjacency.get(tile, []):
                        if adjacent_tile == robot_location:
                            adjacent = True
                            break

                if not adjacent:
                    cost += 1  # Estimate cost to move to an adjacent tile

                # Check if robot has the correct color
                if robot_color != goal_color:
                    cost += 1  # Estimate cost to change color

                cost += 1  # Cost to paint the tile

        return cost

    def _get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()

    def _match(self, fact, *args):
        """Check if a PDDL fact matches a given pattern."""
        parts = self._get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))
