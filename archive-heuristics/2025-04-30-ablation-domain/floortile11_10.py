from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse fact strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match fact strings
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts is at least the number of arguments in the pattern
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Helper function for BFS
def bfs_distance(adj_list, start_node, end_node):
    """
    Computes the shortest path distance between two nodes in a graph using BFS.
    Returns float('inf') if end_node is unreachable.
    """
    if start_node == end_node:
        return 0
    queue = deque([(start_node, 0)])
    visited = {start_node}
    while queue:
        current_node, dist = queue.popleft()
        if current_node == end_node:
            return dist
        if current_node in adj_list:
            for neighbor in adj_list[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # Should not happen in connected tile grids


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the estimated cost for each unpainted goal tile
    independently, considering the minimum cost for any robot to reach the tile
    and acquire the necessary color, plus the paint action.

    # Assumptions
    - Tiles are arranged in a grid connected by 'up', 'down', 'left', 'right' predicates.
    - Movement between adjacent tiles costs 1 action.
    - Getting a color costs 1 action if the robot doesn't have it (and it's available).
      It's assumed 'available-color' facts are static and colors are always available to pick up.
      It's assumed a robot can switch colors by picking up a new one (the old one is implicitly dropped or replaced).
    - Painting a tile costs 1 action.
    - A tile must be 'clear' to be painted. Tiles painted with the wrong color are not 'clear'.
    - Tiles painted with the wrong color cannot be repainted or cleared in a simple way
      (based on the provided predicates/actions). If a goal tile is painted with the wrong color,
      the heuristic returns a large value indicating a likely unsolvable state.
    - The heuristic sums costs for individual goal tiles, ignoring potential efficiencies
      from robots painting multiple tiles or conflicts between robots.

    # Heuristic Initialization
    - Extracts goal conditions to determine the target color for each tile.
    - Builds an adjacency list representation of the tile grid graph based on 'up', 'down', 'left', 'right' static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Parse the current state to find:
       - The current location of each robot.
       - The current color held by each robot.
       - Which tiles are currently painted and with what color.
    3. For each tile specified in the goal conditions:
       - Identify the required color for this tile.
       - Check the current state of this tile:
         - If the tile is painted with the *wrong* color:
           - The state is likely unsolvable. Return a large penalty value (e.g., 1000).
         - If the tile is already painted with the *correct* color:
           - This goal is satisfied. Add 0 cost for this tile and continue to the next goal tile.
         - If the tile is *not* painted (implicitly clear):
           - This tile needs to be painted with the required color.
           - Estimate the minimum cost for *any* robot to paint this single tile:
             - Find the minimum cost over all robots `R`:
               - Cost for robot `R` = (Cost to get color C if R doesn't have it) + (Cost to move R from its current location to the tile) + (Cost of paint action).
               - Cost to get color C: 1 if robot R does not have the required color C; 0 otherwise.
               - Cost to move: Calculate the shortest path distance (using BFS on the tile grid graph) from robot R's current location to the tile.
               - Cost of paint action: 1.
             - The minimum cost for this tile is the minimum of (color_cost + move_cost + 1) over all robots.
           - Add this minimum cost for the tile to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the tile graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal colors for each tile.
        self.goal_colors = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_colors[tile] = color

        # Build the adjacency list for the tile grid graph.
        self.adj_list = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                self.adj_list.setdefault(tile1, []).append(tile2)
                self.adj_list.setdefault(tile2, []).append(tile1) # Grid is undirected

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse state to find robot locations, colors, and painted tiles.
        robot_locations = {}
        robot_colors = {}
        painted_tiles = {}

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
                painted_tiles[tile] = color

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each goal tile and its required color.
        for tile, required_color in self.goal_colors.items():
            # Check if the tile is painted with the wrong color.
            if tile in painted_tiles and painted_tiles[tile] != required_color:
                # This state is likely unsolvable or requires complex actions not modeled.
                # Return a large penalty.
                return 1000 # Arbitrary large value

            # Check if the tile is already painted correctly.
            if tile in painted_tiles and painted_tiles[tile] == required_color:
                # Goal already satisfied for this tile.
                continue

            # If the tile is not painted (implicitly clear based on domain structure),
            # it needs to be painted.
            # Estimate the minimum cost for any robot to paint this tile.
            min_robot_cost_for_this_tile = float('inf')

            if not robot_locations: # Should not happen in valid problems, but handle defensively
                 return float('inf') # Cannot paint without robots

            for robot, r_loc in robot_locations.items():
                # Cost to move robot to the tile.
                move_cost = bfs_distance(self.adj_list, r_loc, tile)
                if move_cost == float('inf'):
                     # Tile is unreachable by this robot.
                     continue

                # Cost for the robot to get the required color.
                # Assumes get-color replaces current color and is always possible if available.
                color_cost = 1 if robot_colors.get(robot) != required_color else 0

                # Total cost for this robot to paint this tile (move + get_color + paint).
                robot_task_cost = move_cost + color_cost + 1 # +1 for the paint action

                min_robot_cost_for_this_tile = min(min_robot_cost_for_this_tile, robot_task_cost)

            # If the tile is unreachable by any robot, the state is likely unsolvable.
            if min_robot_cost_for_this_tile == float('inf'):
                 return float('inf') # Cannot reach a goal tile

            # Add the minimum estimated cost for this tile to the total.
            total_cost += min_robot_cost_for_this_tile

        return total_cost
