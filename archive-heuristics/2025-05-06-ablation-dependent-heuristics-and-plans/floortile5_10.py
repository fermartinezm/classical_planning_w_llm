from fnmatch import fnmatch
from collections import defaultdict, deque

# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It sums the estimated cost for each
    unpainted goal tile independently, plus a cost for acquiring necessary colors.
    The cost for a single tile includes:
    1. A paint action.
    2. Movement cost for the closest robot to reach an adjacent tile from which painting is possible.
    The color cost is added separately for each required color that no robot currently holds.

    # Assumptions
    - The grid structure is defined by the 'up', 'down', 'left', 'right' predicates
      in the static facts.
    - All goal tiles that are not yet painted correctly are currently 'clear'.
    - The shortest path distance on the grid is a reasonable estimate for movement cost,
      ignoring 'clear' preconditions for intermediate tiles.
    - Robots can change color instantly if the target color is available (which is a static fact).
    - The cost of changing color is 1 action.
    - The cost of moving between adjacent tiles is 1 action.
    - The cost of painting a tile is 1 action.

    # Heuristic Initialization
    - Parses static facts to build the grid graph (adjacency list).
    - Identifies all tiles in the domain.
    - Computes all-pairs shortest paths between tiles using BFS.
    - Stores the required color for each goal tile.
    - Stores the mapping from a tile (to be painted) to the set of tiles where a robot must be located to paint it.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles that are not currently painted with the correct color. Store them as {tile: required_color}.
    2. If there are no such tiles, return 0 (goal state).
    3. Get the current location and color held by each robot.
    4. Initialize total_cost = 0.
    5. Identify the set of unique colors required by the unpainted goal tiles.
    6. Identify the set of unique colors currently held by the robots.
    7. For each unpainted goal tile T requiring color C:
       a. Add 1 to total_cost (for the paint action).
       b. Find the minimum shortest-path distance from any robot's current location
          to any tile X such that a robot at X can paint T (i.e., X is "adjacent" to T in the painting sense).
          Add this minimum distance to total_cost.
    8. For each color in the set of required colors:
       If that color is not present in the set of colors currently held by robots:
         Add 1 to total_cost (cost to acquire this color by *some* robot).
    9. Return total_cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure, computing distances,
        and storing goal information.
        """
        super().__init__(task)

        self.all_tiles = set()
        self.adj = defaultdict(set) # Adjacency list for the grid graph (for movement)

        # paint_adj[tile_to_paint] = {tile_robot_is_at_to_paint_it}
        # This maps a tile that needs to be painted to the locations a robot must be AT to paint it.
        self.paint_adj = defaultdict(set)

        # Process static facts to build the grid graph and find tiles
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                # (direction tile_y tile_x) means tile_y is in that direction from tile_x
                # Robot at tile_x can paint tile_y using paint_direction action
                tile_y, tile_x = parts[1], parts[2]

                # Add to adjacency list for movement (undirected)
                self.adj[tile_x].add(tile_y)
                self.adj[tile_y].add(tile_x)
                self.all_tiles.add(tile_x)
                self.all_tiles.add(tile_y)

                # Add to paint_adj mapping: robot at tile_x can paint tile_y
                self.paint_adj[tile_y].add(tile_x)


        # Process goal facts to find goal tiles and required colors
        self.goal_painted_tiles = {} # {tile: color}
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color
                self.all_tiles.add(tile) # Ensure goal tiles are included in all_tiles

        # Process initial state facts to find tiles (e.g., robot-at locations)
        # This ensures all tiles that can be occupied by a robot are included.
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == 'robot-at':
                  self.all_tiles.add(parts[2]) # Add robot's initial location as a tile


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = self._bfs(start_tile)


    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.all_tiles}
        if start_node not in self.all_tiles:
             # This start_node is not a known tile. Should not happen in valid instances/states.
             # Return distances dictionary with all infinities.
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            curr = queue.popleft()

            # Ensure curr is in adj before iterating (though it should be if in all_tiles)
            if curr in self.adj:
                for neighbor in self.adj[curr]:
                    # Ensure neighbor is a known tile
                    if neighbor in self.all_tiles and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[curr] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unpainted goal tiles
        unpainted_goals = {} # {tile: color}
        current_painted = {} # {tile: color}
        robot_info = {} # {robot_name: {'loc': tile, 'color': color}}

        # Extract current state information efficiently
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'painted':
                current_painted[parts[1]] = parts[2]
            elif len(parts) == 3 and parts[0] == 'robot-at':
                robot, loc = parts[1], parts[2]
                if robot not in robot_info: robot_info[robot] = {}
                robot_info[robot]['loc'] = loc
            elif len(parts) == 3 and parts[0] == 'robot-has':
                 robot, color = parts[1], parts[2]
                 if robot not in robot_info: robot_info[robot] = {}
                 robot_info[robot]['color'] = color

        # Populate unpainted_goals
        for tile, goal_color in self.goal_painted_tiles.items():
            if tile not in current_painted or current_painted[tile] != goal_color:
                 # Assuming if not painted correctly, it's clear and needs painting
                 unpainted_goals[tile] = goal_color

        # 2. If no unpainted goal tiles, we are in a goal state
        if not unpainted_goals:
            return 0

        total_cost = 0

        # 7. Calculate cost for each unpainted goal tile (paint + movement)
        for tile_to_paint, required_color in unpainted_goals.items():
            # a. Cost for paint action
            total_cost += 1

            # b. Minimum movement cost for any robot to reach a valid painting location
            min_dist_to_paint_loc = float('inf')

            # Find the tiles where a robot must be AT to paint tile_to_paint
            possible_robot_locations = self.paint_adj.get(tile_to_paint, set())

            if not possible_robot_locations:
                 # This tile cannot be painted by any robot from any adjacent tile defined in static facts.
                 # This implies an unsolvable problem or a malformed instance/domain.
                 # Return infinity or a very large number.
                 return float('inf')

            for robot_name, info in robot_info.items():
                robot_loc = info.get('loc')
                # Ensure robot_loc is known and is a tile in our graph
                if robot_loc is None or robot_loc not in self.distances:
                     # Robot location unknown or not a known tile? Should not happen in valid states.
                     # Treat as unreachable for this robot for this tile.
                     continue

                # Calculate distance from robot_loc to all possible_robot_locations
                for paint_loc in possible_robot_locations:
                    # Ensure paint_loc is a known tile and reachable from robot_loc
                    if paint_loc in self.distances[robot_loc]:
                         dist = self.distances[robot_loc][paint_loc]
                         min_dist_to_paint_loc = min(min_dist_to_paint_loc, dist)
                    # else: paint_loc is unreachable from robot_loc, distance remains inf

            # If min_dist_to_paint_loc is still infinity, it means no robot can reach any valid painting location.
            # This tile cannot be painted. Unsolvable.
            if min_dist_to_paint_loc == float('inf'):
                 return float('inf')

            total_cost += min_dist_to_paint_loc

        # 5. Identify required colors
        required_colors = set(unpainted_goals.values())

        # 6. Identify colors held by robots
        colors_held_by_robots = {info.get('color') for info in robot_info.values() if info.get('color') is not None}

        # 8. Add cost for acquiring needed colors
        for color in required_colors:
            if color not in colors_held_by_robots:
                total_cost += 1 # Cost for one change_color action for this color

        # 9. Return total heuristic value
        return total_cost
