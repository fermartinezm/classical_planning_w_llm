from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# New helper functions for Floortile
def build_grid_graph(static_facts):
    """Builds an adjacency map for tiles based on static connectivity facts."""
    adj_map = {}
    all_tiles = set()

    for fact in static_facts:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
            _, tile1, tile2 = parts
            all_tiles.add(tile1)
            all_tiles.add(tile2)
            adj_map.setdefault(tile1, set()).add(tile2)
            adj_map.setdefault(tile2, set()).add(tile1) # Connectivity is symmetric

    # Ensure all tiles from static facts are in the map, even if isolated (though unlikely in grid)
    for tile in all_tiles:
        adj_map.setdefault(tile, set())

    return adj_map, list(all_tiles) # Return list of all tiles for BFS

def compute_all_pairs_distances(adj_map, all_tiles):
    """Computes shortest path distances between all pairs of tiles using BFS."""
    dist_map = {}

    for start_tile in all_tiles:
        q = deque([(start_tile, 0)])
        visited = {start_tile}
        dist_map[(start_tile, start_tile)] = 0

        while q:
            current_tile, dist = q.popleft()

            for neighbor in adj_map.get(current_tile, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist_map[(start_tile, neighbor)] = dist + 1
                    q.append((neighbor, dist + 1))

    return dist_map

def get_adjacent_tiles(tile, adj_map):
    """Returns the set of tiles adjacent to the given tile."""
    return adj_map.get(tile, set())

def get_robot_info(state):
    """Extracts robot locations and colors from the current state."""
    robot_locations = {} # robot -> tile
    robot_colors = {}    # robot -> color

    for fact in state:
        parts = get_parts(fact)
        if match(fact, "robot-at", "*", "*"):
            _, robot, tile = parts
            robot_locations[robot] = tile
        elif match(fact, "robot-has", "*", "*"):
            _, robot, color = parts
            robot_colors[robot] = color

    return robot_locations, robot_colors

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost based on:
    1. The number of tiles that still need to be painted correctly.
    2. The number of color changes required across all robots.
    3. The minimum movement cost for robots to reach painting locations.

    # Heuristic Initialization
    - Extracts goal conditions (which tiles need which color).
    - Builds the grid graph from static connectivity facts.
    - Computes all-pairs shortest path distances on the grid.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all goal tiles and their required colors from the task's goals.
    2. Iterate through the goal tiles. For each goal tile, check the current state:
       - If the tile is painted with the correct color, it's satisfied.
       - If the tile is painted with a *wrong* color, the state is likely unsolvable for this goal tile. Return infinity.
       - If the tile is not painted correctly and not painted wrongly (implying it's clear based on domain structure), it needs painting. Add it to the set of unpainted goal tiles.
    3. If the set of unpainted goal tiles is empty, the goal is reached, and the heuristic is 0.
    4. Calculate the base cost: This is the number of unpainted goal tiles, as each requires at least one paint action.
    5. Determine the set of distinct colors needed for the unpainted tiles.
    6. Determine the colors currently held by each robot from the current state.
    7. Estimate color change cost: Calculate the number of needed colors that are *not* currently held by *any* robot. This is a lower bound on the number of `change_color` actions required across all robots. Add this to the total cost.
    8. Estimate movement cost: For each unpainted goal tile, the robot needs to reach an adjacent tile to paint it. Find the minimum distance from *any* robot's current location to *any* tile adjacent to the goal tile. Sum these minimum distances over all unpainted goal tiles. Add this sum to the total cost. If any unpainted goal tile is unreachable by any robot (e.g., no adjacent tiles, or all adjacent tiles are unreachable), return infinity.
    9. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal tiles and their required colors
        self.goal_tiles_colors = {}
        for goal in self.goals:
            # Assuming goals are always (painted tile color)
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_tiles_colors[tile] = color

        # 2. Build the grid graph and compute distances
        self.adj_map, all_tiles = build_grid_graph(static_facts)
        self.dist_map = compute_all_pairs_distances(self.adj_map, all_tiles)
        self.all_tiles = set(all_tiles) # Store set of all tiles for quick lookup

    def get_distance(self, tile1, tile2):
        """Retrieves the precomputed distance between two tiles."""
        # Returns infinity if tiles are not in the map or unreachable
        return self.dist_map.get((tile1, tile2), math.inf)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unpainted goal tiles
        unpainted_goal_tiles = {} # tile -> required_color
        for goal_tile, required_color in self.goal_tiles_colors.items():
            is_painted_correctly = False
            is_painted_wrongly = False

            for fact in state:
                parts = get_parts(fact)
                if len(parts) == 3 and parts[0] == "painted" and parts[1] == goal_tile:
                    painted_color = parts[2]
                    if painted_color == required_color:
                        is_painted_correctly = True
                        break # Found correct painting
                    else:
                         is_painted_wrongly = True # Found wrong painting

            if is_painted_wrongly:
                 # If a goal tile is painted with the wrong color, it's likely unsolvable.
                 # Return infinity.
                 return math.inf

            if not is_painted_correctly:
                 # If not painted correctly and not painted wrongly, it needs painting.
                 # We assume it's currently clear based on domain structure and problem instances.
                 unpainted_goal_tiles[goal_tile] = required_color


        # 2. If no tiles need painting, the goal is reached.
        if not unpainted_goal_tiles:
            return 0

        # 3. Get robot information
        robot_locations, robot_colors = get_robot_info(state)
        robots = list(robot_locations.keys()) # Get list of robot names

        # If there are unpainted tiles but no robots, it's unsolvable
        if not robots:
             return math.inf

        # 4. Calculate base cost (paint actions)
        total_cost = len(unpainted_goal_tiles) # Each unpainted tile needs one paint action

        # 5. Calculate color change cost
        needed_colors = set(unpainted_goal_tiles.values())
        held_needed_colors = {robot_colors[r] for r in robots if robot_colors.get(r) in needed_colors}
        # Minimum color changes needed is the number of needed colors not currently held by any robot
        total_cost += max(0, len(needed_colors) - len(held_needed_colors))

        # 6. Calculate movement cost
        movement_cost = 0
        for tile, required_color in unpainted_goal_tiles.items():
            # Find the minimum distance from any robot to any adjacent tile of the current tile
            min_dist_to_adj_from_any_robot = math.inf

            adjacent_tiles = get_adjacent_tiles(tile, self.adj_map)

            if not adjacent_tiles:
                 # Should not happen in a valid grid problem, but handle defensively
                 # If a goal tile has no adjacent tiles, it's unreachable for painting.
                 # This state is likely unsolvable. Return infinity.
                 return math.inf

            for robot in robots:
                robot_loc = robot_locations.get(robot)
                if robot_loc is None or robot_loc not in self.all_tiles:
                    # Robot location unknown or not on a valid tile? Should not happen in valid instances.
                    # If it does, this robot cannot contribute to painting this tile.
                    continue

                min_dist_from_this_robot = math.inf
                for adj_tile in adjacent_tiles:
                    dist = self.get_distance(robot_loc, adj_tile)
                    min_dist_from_this_robot = min(min_dist_from_this_robot, dist)

                min_dist_to_adj_from_any_robot = min(min_dist_to_adj_from_any_robot, min_dist_from_this_robot)

            # If min_dist_to_adj_from_any_robot is still infinity, it means the tile is unreachable by any robot
            if min_dist_to_adj_from_any_robot == math.inf:
                 return math.inf # Unreachable tile needing paint

            movement_cost += min_dist_to_adj_from_any_robot

        total_cost += movement_cost

        return total_cost
