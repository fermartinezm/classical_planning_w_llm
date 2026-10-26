from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Helper to parse a PDDL fact string into parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing the estimated minimum cost
    for each tile that needs to be painted according to the goal. The estimated minimum cost
    for painting a single tile with the required color is calculated as the sum of:
    1. The minimum number of moves required for any robot to reach any tile adjacent to the target tile.
    2. An additional cost of 1 if the robot chosen for the minimum movement cost does not currently
       hold the required color (representing a change_color action).
    3. The cost of the paint action itself (1).
    This sum is calculated independently for each unpainted goal tile and then summed up.

    Assumptions:
    - The problem instance is solvable.
    - All tiles form a connected grid structure based on adjacency facts.
    - All colors required by the goal are available colors in the domain.
    - Tiles that need painting according to the goal are initially clear.
    - The heuristic ignores potential conflicts between robots (e.g., multiple robots needing the same tile or path)
      and the dynamic nature of tile clearness during movement, using precomputed grid distances on the static grid graph.
    - The heuristic assumes any robot can change to any available color if it currently holds any color.

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic precomputes static information:
    - Extracts the set of goal paintings (tile and required color).
    - Extracts all tile objects from static adjacency facts.
    - Builds an adjacency list representation of the tile grid graph using the up/down/left/right facts.
    - Computes all-pairs shortest paths (distances) between all tiles using Breadth-First Search (BFS)
      on the grid graph. These distances represent the minimum number of move actions between tiles
      assuming all intermediate tiles are clear.

    Step-By-Step Thinking for Computing Heuristic:
    In the heuristic function (`__call__`), for a given state:
    1. Parse the current state to identify robot locations, robot held colors, and currently painted tiles.
    2. Determine the set of 'needed paintings' by comparing the goal paintings with the currently painted tiles.
    3. If there are no needed paintings, the state is a goal state, and the heuristic returns 0.
    4. Initialize the total heuristic value to 0.
    5. For each needed painting `(target_tile, required_color)`:
       a. Initialize the minimum cost for this specific painting task (`min_cost_for_this_tile`) to infinity.
       b. Get the set of tiles adjacent to `target_tile` from the precomputed adjacency information.
       c. If `target_tile` has no adjacent tiles in the graph, it's unreachable for painting, return infinity.
       d. For each robot `R` and its current location `robot_loc`:
          i. Calculate the color cost: 1 if `R`'s current color is not `required_color`, otherwise 0.
          ii. Find the minimum distance from `robot_loc` to any tile `X` in the set of adjacent tiles of `target_tile`, using the precomputed distances. Let this be `min_dist_from_robot_to_adj`.
          iii. If `min_dist_from_robot_to_adj` is not infinity (meaning the adjacent tile is reachable in the grid graph), calculate the total cost for robot `R` to paint `target_tile`: `min_dist_from_robot_to_adj + color_cost + 1` (where 1 is the paint action cost).
          iv. Update `min_cost_for_this_tile` with the minimum of its current value and the cost calculated in the previous step.
       e. If `min_cost_for_this_tile` is still infinity after checking all robots, it means the tile is unreachable by any robot, return infinity.
       f. Add `min_cost_for_this_tile` to the total heuristic value.
    6. Return the total heuristic value.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Extract goal paintings
        self.goal_paintings = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                self.goal_paintings.add((parts[1], parts[2]))

        # Extract all tile names
        all_tiles = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] in ["up", "down", "left", "right"]:
                 all_tiles.add(parts[1])
                 all_tiles.add(parts[2])

        # Build adjacency list
        self.adjacency = {tile: set() for tile in all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2] # Y, X in PDDL (dest, src)
                # Add bidirectional edges
                if tile2 in self.adjacency:
                    self.adjacency[tile2].add(tile1)
                if tile1 in self.adjacency:
                    self.adjacency[tile1].add(tile2)

        # Compute all-pairs shortest paths (BFS from each tile)
        self.distances = {}
        for start_tile in all_tiles:
            self.distances[start_tile] = {}
            queue = deque([(start_tile, 0)])
            visited = {start_tile}
            while queue:
                current_tile, dist = queue.popleft()
                self.distances[start_tile][current_tile] = dist
                for neighbor in self.adjacency.get(current_tile, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state

        # Parse current state
        robot_locations = {}
        robot_colors = {}
        painted_tiles = set()
        # clear_tiles = set() # Not needed for this heuristic calculation
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has":
                robot_colors[parts[1]] = parts[2]
            elif parts[0] == "painted":
                painted_tiles.add((parts[1], parts[2]))
            # elif parts[0] == "clear":
            #     clear_tiles.add(parts[1])

        # Identify needed paintings
        needed_paintings = {(t, c) for (t, c) in self.goal_paintings if (t, c) not in painted_tiles}

        # If goal is reached
        if not needed_paintings:
            return 0

        total_heuristic = 0

        # Calculate cost for each needed painting independently
        for target_tile, required_color in needed_paintings:
            min_cost_for_this_tile = float('inf')
            adjacent_tiles_T = self.adjacency.get(target_tile, set())

            # If the target tile has no adjacent tiles, it's unreachable for painting
            # This case should ideally not happen in valid grid problems, but handle defensively.
            if not adjacent_tiles_T:
                 # print(f"Warning: Tile {target_tile} has no adjacent tiles.")
                 return float('inf')

            for robot, robot_loc in robot_locations.items():
                # Cost to get the right color for this robot
                # Assumes robot_colors.get(robot) will return a color if robot_loc exists
                current_color = robot_colors.get(robot)
                if current_color is None:
                     # Robot exists but doesn't have a color? Check domain/instance.
                     # Domain says robots start with robot-has. change_color replaces.
                     # free-color predicate exists but isn't used in actions.
                     # Assume robots always have a color.
                     # print(f"Warning: Robot {robot} has no color.")
                     continue # Skip this robot if it somehow has no color

                color_cost = 1 if current_color != required_color else 0

                # Find minimum distance from robot_loc to any adjacent tile of target_tile
                min_dist_from_robot_to_adj = float('inf')
                if robot_loc in self.distances: # Ensure robot_loc is a known tile in the graph
                    for adj_tile in adjacent_tiles_T:
                        if adj_tile in self.distances[robot_loc]: # Ensure adj_tile is reachable from robot_loc
                            dist = self.distances[robot_loc][adj_tile]
                            min_dist_from_robot_to_adj = min(min_dist_from_robot_to_adj, dist)
                        # else:
                            # print(f"Warning: Adjacent tile {adj_tile} of {target_tile} not reachable from robot location {robot_loc}.")


                # If an adjacent tile is reachable from the robot's current location
                if min_dist_from_robot_to_adj != float('inf'):
                     # Cost for this robot to paint this tile = move_cost + color_cost + paint_cost
                     # paint_cost is 1
                     cost_this_robot = min_dist_from_robot_to_adj + color_cost + 1
                     min_cost_for_this_tile = min(min_cost_for_this_tile, cost_this_robot)

            # If the tile is unreachable by any robot (no robot can reach any adjacent tile)
            if min_cost_for_this_tile == float('inf'):
                 # print(f"Warning: Tile {target_tile} unreachable by any robot.")
                 return float('inf') # Problem likely unsolvable

            total_heuristic += min_cost_for_this_tile

        return total_heuristic
