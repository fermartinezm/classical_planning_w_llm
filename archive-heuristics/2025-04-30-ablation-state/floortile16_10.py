from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_R_C' into a tuple (R, C)."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            pass # Not a standard tile name format
    return None # Indicate parsing failed

def manhattan_distance(tile1_name, tile2_name, tile_coords):
    """Calculates the Manhattan distance between two tiles using their coordinates."""
    coords1 = tile_coords.get(tile1_name)
    coords2 = tile_coords.get(tile2_name)
    if coords1 is None or coords2 is None:
        # Should not happen if all tiles are parsed correctly
        # Return infinity as a large cost indicating inability to calculate distance
        return math.inf
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

# Heuristic class
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums the cost for each unpainted goal tile,
    considering the need for a paint action, a color change (if the required color
    is not held by any robot), and movement to bring a robot with the correct
    color adjacent to the tile.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates
      connecting tiles.
    - Tile names follow the format 'tile_R_C' allowing coordinate extraction.
    - Tiles painted with the wrong color in the initial state or reachable states
      are considered unsolvable (heuristic returns infinity).
    - Each unpainted goal tile's cost is calculated independently and summed,
      potentially overestimating but providing a useful gradient for greedy search.
    - A color change action is counted once per needed color if no robot has it.
    - Movement cost for a tile is the minimum Manhattan distance from a robot
      (with the right color, or any robot if color needs changing) to any adjacent tile.

    # Heuristic Initialization
    - Extract goal conditions to identify target tiles and colors.
    - Parse static facts ('up', 'down', 'left', 'right') to build the grid
      structure, mapping tile names to coordinates and creating an adjacency list.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles and their required colors from the task definition.
    2. Identify the current state of all tiles (painted color or clear) and the
       location and color of each robot from the current state facts.
    3. Identify the set of goal tiles that are not yet painted with the correct color.
       If any goal tile is painted with the *wrong* color, return infinity.
    4. If there are no unpainted goal tiles, the heuristic is 0 (goal state).
    5. Initialize the total heuristic cost.
    6. Add the number of unpainted goal tiles to the cost (representing the paint actions).
    7. Identify the set of colors required by the unpainted goal tiles.
    8. Identify the set of colors currently held by robots.
    9. For each required color that is not held by any robot, add 1 to the cost
       (representing a necessary color change action).
    10. Estimate movement cost: For each unpainted goal tile (T, C):
        a. Find the set of tiles adjacent to T using the pre-calculated adjacency information.
        b. Find the set of robots that currently hold color C.
        c. If no robot holds color C:
           - Calculate the minimum Manhattan distance from *any* robot's current location
             to *any* tile adjacent to T. Add this distance to the total cost.
        d. If at least one robot holds color C:
           - Calculate the minimum Manhattan distance from *any* robot holding color C
             to *any* tile adjacent to T. Add this distance to the total cost.
        e. If no adjacent tiles or no robots are found, return infinity.
    11. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the grid structure from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations and colors for each tile.
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Build grid structure: tile name -> (row, col) mapping and adjacency list.
        self.tile_coords = {}
        self.adj = {} # tile_name -> [adjacent_tile_names]

        # Collect all tile names from static facts (adjacency) and goal facts
        all_tiles = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if len(parts) > 1 and parts[0] in ['up', 'down', 'left', 'right']:
                 all_tiles.add(parts[1])
                 all_tiles.add(parts[2])
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'painted' and len(parts) > 1:
                 all_tiles.add(parts[1])


        # Parse coordinates and initialize adjacency
        for tile_name in all_tiles:
            coords = parse_tile_name(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords
                self.adj[tile_name] = [] # Initialize empty list

        # Populate adjacency list from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                pred, tile1, tile2 = parts
                # (up tile_y tile_x) means tile_y is up from tile_x
                # (down tile_y tile_x) means tile_y is down from tile_x
                # (left tile_y tile_x) means tile_y is left from tile_x
                # (right tile_y tile_x) means tile_y is right from tile_x
                # So, tile_y is adjacent to tile_x, and tile_x is adjacent to tile_y
                if tile1 in self.adj and tile2 in self.adj: # Ensure both tiles were parsed
                     if tile2 not in self.adj[tile1]:
                         self.adj[tile1].append(tile2)
                     if tile1 not in self.adj[tile2]:
                         self.adj[tile2].append(tile1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify current tile states (painted or clear)
        current_tile_state = {} # tile_name -> color
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted' and len(parts) == 3:
                current_tile_state[parts[1]] = parts[2]

        # Identify robot locations and colors
        robot_locations = {} # robot_name -> tile_name
        robot_colors = {} # robot_name -> color
        all_robots = set() # Collect all robot names
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                all_robots.add(robot)
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                all_robots.add(robot)

        unpainted_goal_tiles = {}
        for tile, required_color in self.goal_tiles.items():
            if tile in current_tile_state:
                if current_tile_state[tile] != required_color:
                    # Tile is painted with the wrong color - unsolvable from here
                    return math.inf
                # Else: Tile is painted correctly, continue
            else:
                # Tile is not painted (assumed clear if not painted)
                unpainted_goal_tiles[tile] = required_color

        # If all goal tiles are painted correctly, the goal is reached.
        if not unpainted_goal_tiles:
            return 0

        total_cost = 0

        # Cost component 1: Paint actions
        total_cost += len(unpainted_goal_tiles)

        # Cost component 2: Color changes
        needed_colors = set(unpainted_goal_tiles.values())
        robot_current_colors = set(robot_colors.values())
        colors_to_change = needed_colors - robot_current_colors
        total_cost += len(colors_to_change)

        # Cost component 3: Movement
        for tile_to_paint, required_color in unpainted_goal_tiles.items():
            min_dist_to_adjacent = math.inf

            adjacent_tiles = self.adj.get(tile_to_paint, [])
            if not adjacent_tiles:
                 # Goal tile has no adjacent tiles in the grid - unsolvable
                 return math.inf

            robots_with_required_color = {
                r for r, color in robot_colors.items() if color == required_color
            }

            robots_to_consider = []
            if not robots_with_required_color:
                # No robot has the color, any robot can change color and move
                robots_to_consider = list(all_robots)
            else:
                # At least one robot has the color, consider only those robots
                robots_to_consider = list(robots_with_required_color)

            if not robots_to_consider:
                 # No robots in the problem? Should not happen in valid problems.
                 return math.inf

            for robot in robots_to_consider:
                robot_tile = robot_locations.get(robot)
                if robot_tile is None:
                    # Robot location unknown - invalid state. Skip this robot.
                    continue

                for adj_tile in adjacent_tiles:
                    dist = manhattan_distance(robot_tile, adj_tile, self.tile_coords)
                    min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

            if min_dist_to_adjacent == math.inf:
                 # Cannot find a path from any considered robot to any adjacent tile
                 # This could happen if a robot is on a disconnected part of the grid
                 return math.inf
            else:
                 total_cost += min_dist_to_adjacent


        return total_cost
