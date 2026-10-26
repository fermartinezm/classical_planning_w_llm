from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions to parse PDDL facts represented as strings
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

# BFS function to find shortest distances on the grid
def bfs_distance_map(start_tile, adjacency_map, clear_tiles):
    """
    Performs a BFS starting from start_tile, only traversing through clear_tiles.
    Returns a dictionary mapping reachable tiles to their shortest distance from start_tile.
    """
    q = deque([(start_tile, 0)])
    visited = {start_tile}
    distances = {start_tile: 0}

    while q:
        current_tile, dist = q.popleft()

        neighbors = adjacency_map.get(current_tile, [])

        for neighbor in neighbors:
            # A robot can move from current_tile to neighbor if neighbor is clear.
            # This is the condition for adding 'neighbor' to the queue.
            # The start_tile itself doesn't need to be clear to be the starting point.
            if neighbor in clear_tiles and neighbor not in visited:
                 visited.add(neighbor)
                 distances[neighbor] = dist + 1
                 q.append((neighbor, dist + 1))

    return distances

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the cost for each unpainted goal tile,
    considering the paint action itself, the cost for a robot to acquire the
    correct color, and the minimum movement cost for a robot to reach a clear
    adjacent tile from which to paint.

    # Assumptions
    - Tiles form a grid structure defined by up/down/left/right predicates.
    - A robot occupies a tile, making it not clear.
    - A robot can only move to a clear adjacent tile.
    - A robot can only paint a clear adjacent tile.
    - A robot can change color if it holds any color and the new color is available.
    - All action costs are 1.

    # Heuristic Initialization
    - Extracts goal conditions to know which tiles need which color.
    - Builds an adjacency map (graph) of the tile grid from static facts.

    # Step-by-Step Thinking for Computing Heuristic
    1.  Identify all goal tiles and their target colors.
    2.  Build the grid graph (adjacency list) from static up/down/left/right facts.
    3.  For a given state:
        a.  Identify current robot locations and colors.
        b.  Identify currently painted tiles and their colors.
        c.  Identify currently clear tiles.
        d.  Check if any goal tile is painted with the wrong color. If so, the goal is unreachable; return a large value (infinity).
        e.  Identify goal tiles that are currently unpainted (and thus must be clear, based on the check in 3d).
        f.  Initialize heuristic value `h = 0`.
        g.  For each unpainted goal tile `T` needing color `C`:
            i.  Add 1 to `h` for the paint action itself.
            ii. Calculate the minimum "preparation cost" for *any* robot to be ready to paint tile `T`. This cost is the sum of:
                - Color change cost: 0 if the robot already has color `C`, 1 otherwise (assuming one change_color action is sufficient).
                - Movement cost: The shortest path distance from the robot's current location to *any* clear tile adjacent to `T`. This distance is calculated using BFS on the grid, only traversing through currently clear tiles.
            iii. The minimum preparation cost for tile `T` is the minimum of these costs over all robots.
            iv. If no robot can reach a clear adjacent tile (BFS returns infinity), the goal is unreachable; return a large value.
            v. Add the minimum preparation cost for tile `T` to `h`.
        h.  Return the total `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the grid graph.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Store goal colors for each tile that needs to be painted
        self.goal_colors = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_colors[tile] = color

        # Build adjacency map (graph) of the tile grid
        self.adjacency_map = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # (relation tile1 tile2) means tile1 is relation-relative to tile2
                # e.g., (up t1 t2) means t1 is above t2. So t1 and t2 are adjacent.
                t1, t2 = parts[1], parts[2]
                self.adjacency_map.setdefault(t1, []).append(t2)
                self.adjacency_map.setdefault(t2, []).append(t1) # Adjacency is symmetric

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # Extract relevant information from the current state
        robot_locations = {}
        robot_colors = {}
        current_painted = {}
        current_clear = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == "painted":
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            elif parts[0] == "clear":
                tile = parts[1]
                current_clear.add(tile)

        h = 0
        large_value = 1000000 # Use a large number to represent infinity

        # Check for tiles painted with the wrong color or unpainted but not clear
        # If a goal tile is painted with the wrong color, the goal is unreachable.
        # If a goal tile is not painted (and thus needs painting) but is also not clear,
        # it's blocked and unreachable in this domain.
        for tile, goal_color in self.goal_colors.items():
            if tile in current_painted:
                if current_painted[tile] != goal_color:
                     return large_value
            elif tile not in current_clear:
                return large_value

        # Identify tiles that need painting (goal tiles that are not yet painted)
        tiles_to_paint = {} # {tile: color}
        for tile, goal_color in self.goal_colors.items():
            if tile not in current_painted: # If not painted, it must be clear (checked above)
                 tiles_to_paint[tile] = goal_color

        # Add cost for each paint action needed
        h += len(tiles_to_paint)

        # Add cost for getting robots into position with the right color for each tile
        for tile, needed_color in tiles_to_paint.items():
            min_robot_prep_cost = large_value # Minimum cost for any robot to paint this specific tile
            adj_tiles = self.adjacency_map.get(tile, []) # Tiles adjacent to the target tile

            # Find the minimum cost for any robot to be ready to paint this tile
            for robot, robot_loc in robot_locations.items():
                # Cost to get the correct color: 0 if robot has it, 1 for change_color otherwise
                color_cost = 0 if robot_colors.get(robot) == needed_color else 1

                min_move_cost_for_robot = large_value # Minimum move cost for *this* robot to get adjacent

                # Check if robot is already adjacent to the target tile
                if robot_loc in adj_tiles:
                     # Robot is already at a tile adjacent to the target tile. Move cost is 0.
                     min_move_cost_for_robot = 0
                else:
                    # Robot needs to move to a clear adjacent tile.
                    # Calculate distances from the robot's current location, only moving through clear tiles.
                    # The robot's current tile is the start, even if not clear.
                    distances_from_robot = bfs_distance_map(robot_loc, self.adjacency_map, current_clear)

                    # Find min distance from robot_loc to any clear adjacent tile of 'tile'
                    for adj_tile in adj_tiles:
                        if adj_tile in current_clear: # Robot must move *to* a clear adjacent tile
                            dist = distances_from_robot.get(adj_tile, large_value)
                            min_move_cost_for_robot = min(min_move_cost_for_robot, dist)

                # If the robot can reach a clear adjacent tile
                if min_move_cost_for_robot != large_value:
                     # Total cost for this robot to paint this tile = color_cost + move_cost
                     min_robot_prep_cost = min(min_robot_prep_cost, color_cost + min_move_cost_for_robot)

            # If no robot can reach a clear adjacent tile for this goal tile, the goal is unreachable
            if min_robot_prep_cost == large_value:
                return large_value

            # Add the minimum preparation cost (color + move) for this tile to the total heuristic
            h += min_robot_prep_cost

        return h
