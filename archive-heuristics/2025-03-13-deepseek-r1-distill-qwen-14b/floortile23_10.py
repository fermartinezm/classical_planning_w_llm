from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(robot-at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles with the correct colors.

    # Assumptions:
    - The robot can move up, down, left, or right to adjacent tiles.
    - The robot can change colors, which takes one action.
    - Each painting action requires the robot to be on the tile and have the correct color.

    # Heuristic Initialization
    - Extracts static facts to build the grid layout and adjacency information.
    - Maps each tile to its adjacent tiles for efficient distance calculations.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the current position of the robot.
    2. Determine the color the robot is currently holding.
    3. Identify all tiles that need to be painted and their required colors.
    4. For each tile that needs painting:
       a. If the robot doesn't have the required color, add the cost to change colors.
       b. Calculate the minimum distance from the robot's current position to the tile.
       c. Sum the movement cost and any color change cost.
    5. Return the total estimated cost to reach the goal state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and building grid information."""
        self.goals = task.goals
        static_facts = task.static

        # Build grid layout and adjacency information
        self.grid = {}
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}
        self.adjacent = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "up", "*", "*"):
                y, x = parts[1], parts[2]
                self.up[x] = y
                if x not in self.adjacent:
                    self.adjacent[x] = []
                self.adjacent[x].append(y)
            elif match(fact, "down", "*", "*"):
                y, x = parts[1], parts[2]
                self.down[x] = y
                if x not in self.adjacent:
                    self.adjacent[x] = []
                self.adjacent[x].append(y)
            elif match(fact, "left", "*", "*"):
                y, x = parts[1], parts[2]
                self.left[x] = y
                if x not in self.adjacent:
                    self.adjacent[x] = []
                self.adjacent[x].append(y)
            elif match(fact, "right", "*", "*"):
                y, x = parts[1], parts[2]
                self.right[x] = y
                if x not in self.adjacent:
                    self.adjacent[x] = []
                self.adjacent[x].append(y)

    def __call__(self, node):
        """Estimate the minimum cost to reach the goal state."""
        state = node.state

        # Extract current robot position and held color
        robot_pos = None
        held_color = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_pos = get_parts(fact)[2]
            if match(fact, "robot-has", "*", "*"):
                held_color = get_parts(fact)[2]

        # Extract goal tiles and their required colors
        goal_tiles = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile, color = get_parts(goal)[1], get_parts(goal)[2]
                goal_tiles[tile] = color

        # If no goals, return 0
        if not goal_tiles:
            return 0

        # Check if all goals are already achieved
        for tile, color in goal_tiles.items():
            if f"(painted {tile} {color})" not in state:
                break
        else:
            return 0

        total_cost = 0

        # For each goal tile, calculate required actions
        for tile, required_color in goal_tiles.items():
            current_fact = f"(painted {tile} {required_color})"
            if current_fact in state:
                continue

            # Check if tile is already painted with a different color
            if any(fact.startswith(f"(painted {tile} ") for fact in state):
                continue

            # Calculate distance from robot's current position to this tile
            if robot_pos not in self.adjacent or tile not in self.adjacent:
                continue  # Tile is unreachable

            visited = {}
            queue = deque()
            queue.append((robot_pos, 0))
            visited[robot_pos] = 0

            found = False
            while queue:
                current, dist = queue.popleft()
                if current == tile:
                    found = True
                    break
                for neighbor in self.adjacent.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

            if not found:
                continue  # Tile is unreachable, but in reality, static facts should ensure reachability

            distance = visited[tile]

            # Add movement cost
            total_cost += distance

            # Add cost to change color if needed
            if held_color != required_color:
                total_cost += 1  # Cost to change color

        return total_cost
