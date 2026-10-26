from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a fact matches a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 city1-1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for BFS distance
def bfs_distance(start_node, target_nodes, graph):
    """
    Finds the shortest path distance from start_node to any node in target_nodes
    in an unweighted graph using BFS.
    Returns infinity if no path exists.
    """
    if not target_nodes:
        return float('inf')

    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft()

        if current_node in target_nodes:
            return dist

        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # No path found

# Inherit from Heuristic if available in the environment
# class floortileHeuristic(Heuristic):
class floortileHeuristic: # Using this if Heuristic base is not provided
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums the estimated cost for each unsatisfied goal
    tile independently. The cost for a single tile includes the paint action,
    the movement cost for the closest robot to get to an adjacent tile, and the
    cost for that robot to change color if needed.

    # Assumptions
    - Tiles painted with the wrong color cannot be fixed (unsolvable state for that tile).
    - Robot movement cost is the shortest path distance on the tile grid defined by
      adjacency facts (up, down, left, right), ignoring the 'clear' predicate for
      intermediate tiles during distance calculation. The target adjacent tile must
      be reachable via this path.
    - Changing color is always possible if the target color is available (static fact).
    - Multiple robots can work in parallel, but the heuristic sums costs assuming
      each tile's painting is achieved by the "best" robot for that tile in isolation.

    # Heuristic Initialization
    - Extracts goal conditions (which tiles need which color).
    - Builds the tile adjacency graph from static facts (up, down, left, right).
    - Stores available colors (for completeness, though not strictly used in cost logic).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify all goal facts of the form `(painted tile color)`.
    3. For each such goal fact `(painted tile_X color_Y)`:
        a. Check if `(painted tile_X color_Y)` is already true in the current state. If yes, continue to the next goal tile (cost is 0 for this tile).
        b. Check if `(painted tile_X color_Z)` is true for any `Z != color_Y`. If yes, this tile cannot be painted correctly according to the domain rules. Return a very large number (e.g., 1000) indicating likely unsolvable.
        c. If the tile is not painted with the correct color:
            i. Identify all tiles adjacent to `tile_X` using the pre-calculated adjacency graph. If there are no adjacent tiles, return a large number (unsolvable).
            ii. Initialize `min_robot_total_action_cost = infinity`.
            iii. For each robot `R_k`:
                - Get its current location `Loc(R_k)`.
                - Calculate the minimum movement distance from `Loc(R_k)` to any tile adjacent to `tile_X` using BFS on the tile graph. Let this be `move_dist`.
                - If `move_dist` is infinity (no path), this robot cannot reach any adjacent tile. Continue to the next robot.
                - If `move_dist` is finite:
                    - Calculate the cost for this robot to paint the tile: `robot_action_cost = move_dist`.
                    - Check if `R_k` has `color_Y` (`(robot-has R_k color_Y)`). If not, add 1 to `robot_action_cost` (for `change_color` action).
                    - Add 1 to `robot_action_cost` for the `paint` action itself.
                    - Update `min_robot_total_action_cost = min(min_robot_total_action_cost, robot_action_cost)`.
            iv. If `min_robot_total_action_cost` is still infinity after checking all robots, return a very large number (indicating likely unsolvable).
            v. Add `min_robot_total_action_cost` to the total heuristic cost.
    4. The total heuristic value is the sum of costs calculated for each unsatisfied goal tile.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions,
        building the tile adjacency graph, and storing available colors.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Extract goal paintings: {tile: color}
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # Build tile adjacency graph from static facts
        self.adjacency_list = {}
        # Collect all tiles mentioned in adjacency facts or goals
        all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Fact is (direction tile1 tile2) meaning tile1 is direction of tile2
                # So tile1 and tile2 are adjacent.
                tile1, tile2 = parts[1], parts[2]
                self.adjacency_list.setdefault(tile1, set()).add(tile2)
                self.adjacency_list.setdefault(tile2, set()).add(tile1) # Adjacency is symmetric
                all_tiles.add(tile1)
                all_tiles.add(tile2)

        # Add tiles from goals that might not be in adjacency facts (e.g., single tile problems)
        all_tiles.update(self.goal_paintings.keys())

        # Pre-calculate adjacent tiles set for each tile for quick lookup
        self.adjacent_tiles_map = {tile: self.adjacency_list.get(tile, set()) for tile in all_tiles}

        # Store available colors (for completeness, though not strictly used in cost logic)
        self.available_colors = {get_parts(fact)[1] for fact in static_facts if match(fact, "available-color", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract current state information
        current_paintings = {}
        robot_locations = {}
        robot_colors = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                current_paintings[parts[1]] = parts[2]
            elif parts[0] == "robot-at":
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has":
                robot_colors[parts[1]] = parts[2]

        total_heuristic_cost = 0

        # Iterate through each goal painting requirement
        for goal_tile, goal_color in self.goal_paintings.items():
            current_painted_color = current_paintings.get(goal_tile)

            # Check if the goal is already satisfied for this tile
            if current_painted_color == goal_color:
                continue

            # Check if the tile is painted with the wrong color (likely unsolvable)
            if current_painted_color is not None and current_painted_color != goal_color:
                # This state is likely unsolvable for this tile. Return a large value.
                return 1000 # Use a large constant

            # The tile needs to be painted with goal_color

            # Cost to get a robot with the correct color adjacent to the tile and paint it
            # This includes movement, color change, and the paint action itself.

            adjacent_tiles = self.adjacent_tiles_map.get(goal_tile, set())

            if not adjacent_tiles:
                 # Tile needs painting but has no adjacent tiles. Unsolvable.
                 # This should ideally not happen in valid problems requiring painting.
                 return 1000 # Use a large constant

            min_robot_total_action_cost = float('inf')

            # If there are no robots, this state is likely unsolvable
            if not robot_locations:
                 return 1000

            for robot, robot_location in robot_locations.items():
                robot_action_cost = 0

                # Calculate minimum distance from robot's current location to any adjacent tile
                # We ignore the 'clear' constraint for pathfinding simplicity in the heuristic.
                # A more complex heuristic could factor this in (e.g., BFS only through clear tiles).
                move_dist = bfs_distance(robot_location, adjacent_tiles, self.adjacency_list)

                if move_dist == float('inf'):
                    continue # This robot cannot reach any adjacent tile

                robot_action_cost += move_dist

                # Cost to change color if the robot doesn't have the goal color
                if robot_colors.get(robot) != goal_color:
                    robot_action_cost += 1 # Cost for change_color action

                # Cost for the paint action itself
                robot_action_cost += 1

                min_robot_total_action_cost = min(min_robot_total_action_cost, robot_action_cost)

            # If no robot can reach an adjacent tile, this state is likely unsolvable
            if min_robot_total_action_cost == float('inf'):
                 # This means for the current goal tile, no robot can reach an adjacent tile.
                 # This part of the goal is unreachable.
                 return 1000 # Use a large constant

            total_heuristic_cost += min_robot_total_action_cost

        return total_heuristic_cost
