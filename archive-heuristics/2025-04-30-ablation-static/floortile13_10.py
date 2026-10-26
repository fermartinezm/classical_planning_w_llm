from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


# Utility functions
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
    # Ensure the number of parts in the fact is at least the number of arguments in the pattern
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all goal tiles by summing the estimated cost
    for each unpainted goal tile. The cost for a single tile includes:
    1. A penalty if the tile is currently occupied by a robot (estimated cost for robot to move off).
    2. The minimum cost for any robot to get into a state where it can paint the tile
       (having the correct color and being adjacent), estimated by summing:
       - Cost to change color (1 if needed).
       - Minimum movement cost from the robot's current location to any adjacent tile.
    3. The cost of the paint action itself (1).

    # Assumptions
    - The grid structure is defined by `up`, `down`, `left`, `right` static facts.
    - Tiles needing painting are either `clear` or occupied by a robot.
    - If a tile is occupied by a robot, the robot must move off (cost 1).
    - Robots always have a color (no `free-color` state to handle).
    - The grid is connected, allowing movement between any two tiles (if not, unsolvable).
    - All necessary colors are available (`available-color` facts).
    - Tiles painted with the wrong color in the initial state are not handled explicitly;
      assuming solvable instances don't require repainting wrong colors.

    # Heuristic Initialization
    - Parses static facts to build the tile adjacency graph.
    - Computes all-pairs shortest paths (distances) on the tile grid using BFS.
    - Extracts the set of goal painted tiles.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile color)`.
    2. Filter these goals to find the set of tiles that are not currently painted correctly in the state.
    3. If no tiles need painting, the heuristic is 0.
    4. Identify the current location and color of each robot.
    5. For each unpainted goal tile `(t, c)`:
       a. Initialize the cost for this tile (`tile_h`) to 0.
       b. Check if any robot is currently located *on* tile `t`. If yes, add 1 to `tile_h` (estimated cost for the robot to move off).
       c. Find the set of tiles adjacent to `t` using the precomputed graph.
       d. For each robot `r` with location `r_loc` and color `r_color`:
          i. Calculate the cost for `r` to acquire color `c`: 0 if `r_color == c`, 1 otherwise.
          ii. Calculate the minimum distance from `r_loc` to any tile adjacent to `t` using the precomputed distances. If no adjacent tile is reachable, this distance is infinity.
          iii. The cost for robot `r` to be ready to paint `t` is (color cost) + (minimum distance).
       e. Find the minimum robot-ready cost over all robots. If no robot can reach an adjacent tile, the problem is likely unsolvable from this state (or requires complex pathfinding through occupied tiles, which this heuristic simplifies), return infinity.
       f. Add the minimum robot-ready cost and 1 (for the paint action) to `tile_h`.
    6. Sum `tile_h` for all unpainted goal tiles to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the tile graph and computing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Collect all tile objects
        self.all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Collect objects declared as 'tile' type
            if len(parts) >= 2 and parts[-1] == '- tile':
                 self.all_tiles.add(parts[0])
            # Collect tiles mentioned in adjacency facts
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                 self.all_tiles.add(parts[1])
                 self.all_tiles.add(parts[2])
            # Note: Tiles in initial state facts like (clear tile_x_y) or (robot-at robot tile_x_y)
            # are assumed to be covered by type declarations or adjacency facts in static.

        # 2. Build the adjacency graph
        self.graph = {tile: set() for tile in self.all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                # Ensure both are recognized tiles before adding edge
                if tile1 in self.graph and tile2 in self.graph:
                    self.graph[tile1].add(tile2)
                    self.graph[tile2].add(tile1) # Grid is undirected

        # 3. Compute all-pairs shortest paths (distances)
        self.distances = {}
        for start_node in self.all_tiles:
            self.distances[start_node] = self._bfs(start_node)

        # 4. Store goal painted tiles
        self.goal_painted_tiles = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_painted_tiles.add((tile, color))

    def _bfs(self, start_node):
        """
        Perform BFS from a start node to find distances to all reachable nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {node: float('inf') for node in self.all_tiles}
        
        # Handle cases where start_node might not be in the graph keys
        # (e.g., an isolated tile not mentioned in adjacency facts)
        if start_node not in self.graph:
             # If the start node is not in the graph keys, it means it has no neighbors
             # according to the static facts. Distances to all other nodes remain infinity.
             # However, if the start_node is in all_tiles, distance to itself is 0.
             if start_node in self.all_tiles:
                 distances[start_node] = 0
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is a valid key in the graph before accessing neighbors
            if current_node not in self.graph:
                 continue # Should not happen if queue only contains nodes from self.graph keys initially

            for neighbor in self.graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def get_adjacent_tiles(self, tile):
        """Get tiles directly adjacent to the given tile based on the graph."""
        # Return neighbors from the precomputed graph
        return list(self.graph.get(tile, []))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1 & 2. Identify unpainted goal tiles
        unpainted_goals = set()
        for tile, color in self.goal_painted_tiles:
            if f'(painted {tile} {color})' not in state:
                unpainted_goals.add((tile, color))

        # 3. If no tiles need painting, return 0
        if not unpainted_goals:
            return 0

        # 4. Identify robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            if match(fact, 'robot-at', '*', '*'):
                _, robot, loc = get_parts(fact)
                robot_locations[robot] = loc
            elif match(fact, 'robot-has', '*', '*'):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color

        occupied_tiles = set(robot_locations.values())

        total_h = 0

        # 5. For each unpainted goal tile
        for tile, color in unpainted_goals:
            tile_h = 0

            # a & b. Penalty if tile is occupied
            if tile in occupied_tiles:
                tile_h += 1 # Estimated cost for robot to move off

            # c. Find adjacent tiles
            adjacent_tiles_to_t = self.get_adjacent_tiles(tile)
            if not adjacent_tiles_to_t:
                 # This tile has no adjacent tiles in the graph - likely unsolvable
                 return float('inf')

            # d & e. Find the minimum cost for a robot to be ready to paint this tile
            min_robot_ready_cost = float('inf')

            any_robot_can_reach_adjacent = False

            for robot, r_loc in robot_locations.items():
                # Check if robot's location is in our precomputed distances
                if r_loc not in self.distances or r_loc not in self.all_tiles:
                     # Robot is on a tile not recognized or not in the graph. Problematic instance.
                     continue # Skip this robot

                color_cost = 0 if robot_colors.get(robot) == color else 1

                min_dist_to_adjacent = float('inf')
                for adj_t in adjacent_tiles_to_t:
                    # Check if the adjacent tile is in the distances computed from r_loc
                    if adj_t in self.distances[r_loc]:
                         dist = self.distances[r_loc][adj_t]
                         if dist != float('inf'):
                             min_dist_to_adjacent = min(min_dist_to_adjacent, dist)
                             any_robot_can_reach_adjacent = True # At least one path exists

                if min_dist_to_adjacent != float('inf'):
                    robot_ready_cost = color_cost + min_dist_to_adjacent
                    min_robot_ready_cost = min(min_robot_ready_cost, robot_ready_cost)

            if not any_robot_can_reach_adjacent:
                 # No robot can reach any tile adjacent to this goal tile. Unsolvable.
                 return float('inf')

            # f. Add costs for this tile
            tile_h += min_robot_ready_cost + 1 # +1 for the paint action

            total_h += tile_h

        # 6. Return total heuristic value
        return total_h
