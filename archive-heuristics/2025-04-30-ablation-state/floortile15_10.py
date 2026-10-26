from fnmatch import fnmatch
from collections import deque

# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and ensure correct splitting
    return fact.strip()[1:-1].split()

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

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each unsatisfied
    goal tile independently. The cost for a single tile is estimated as the
    minimum cost for any robot to reach a position adjacent to the tile,
    acquire the correct color, and perform the paint action.

    # Assumptions
    - Goal tiles are initially either clear or already painted with the correct color.
      If a goal tile is painted with the wrong color, the state is considered a dead end.
    - The grid is connected, and robots can move between any two clear tiles.
      The heuristic ignores the 'clear' precondition for intermediate tiles along a path,
      only considering the distance on the static grid structure.
    - Robots always possess a color (either the required one or a different one).
    - The cost of changing color is 1 action.
    - The cost of moving between adjacent tiles is 1 action.
    - The cost of painting a tile is 1 action.
    - The heuristic sums costs independently for each goal tile, ignoring potential
      synergies like painting multiple nearby tiles or sharing color changes.

    # Heuristic Initialization
    - Parses static facts to build the grid graph (adjacency list for movement).
    - Parses static facts to build the paint adjacency list (which tile a robot must
      be on to paint a target tile).
    - Identifies all tile objects.
    - Computes all-pairs shortest path distances between all tiles on the grid using BFS.
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost `h` to 0.
    2. Parse the current state to find:
       - The current location of each robot.
       - The current color held by each robot.
       - Which tiles are currently painted and with what color.
       - Which tiles are currently clear.
    3. Iterate through each goal condition `(painted tile_i color_j)`:
       - Check if `tile_i` is already painted with `color_j` in the current state. If yes, this goal is satisfied for free; continue to the next goal.
       - Check if `tile_i` is painted with a *different* color in the current state. If yes, this state is likely a dead end in this domain; return a very large number (infinity).
       - If `tile_i` is not painted (i.e., it is `clear`):
         - This tile needs to be painted with `color_j`.
         - Find all possible tiles `tile_adj` where a robot could stand to paint `tile_i` (using the precomputed paint adjacency).
         - Calculate the minimum cost for *any* robot to paint `tile_i`:
           - Initialize `min_cost_for_tile` to infinity.
           - For each robot `r` with current location `tile_robot_r` and color `color_robot_r`:
             - Calculate the cost for robot `r` to acquire color `color_j`: 0 if `color_robot_r == color_j`, 1 otherwise (`change_color` action).
             - Calculate the minimum movement cost for robot `r` to reach *any* of the valid adjacent painting positions (`tile_adj`). This is the shortest path distance from `tile_robot_r` to the closest `tile_adj`. Use the precomputed distances. If no adjacent painting position is reachable, this robot cannot paint this tile.
             - The total cost for robot `r` to paint `tile_i` is the color cost + movement cost + 1 (for the paint action itself).
             - Update `min_cost_for_tile` with the minimum cost found across all robots.
         - If, after checking all robots, `min_cost_for_tile` is still infinity, the goal tile is unreachable; return a very large number (infinity).
         - Add `min_cost_for_tile` to the total heuristic cost `h`.
    4. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure, paint adjacency,
        and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        self.adj = {} # Adjacency list for movement graph
        self.paint_adj = {} # Map tile_painted -> set of tile_robot_pos
        self.tiles = set() # Set of all tile names

        # Parse static facts to build grid graph and paint adjacency
        for fact in static_facts:
            parts = get_parts(fact)
            # Check for directional predicates like (up tile_dest tile_src)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                pred, tile_dest, tile_src = parts
                self.tiles.add(tile_dest)
                self.tiles.add(tile_src)

                # Movement adjacency (bidirectional)
                # Robot at tile_src can move to tile_dest
                self.adj.setdefault(tile_src, set()).add(tile_dest)
                # Robot at tile_dest can move to tile_src (assuming grid is symmetric)
                self.adj.setdefault(tile_dest, set()).add(tile_src)

                # Paint adjacency (robot at tile_src paints tile_dest)
                self.paint_adj.setdefault(tile_dest, set()).add(tile_src)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.tiles:
            self.distances[start_tile] = self._bfs(start_tile)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.tiles}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the adjacency list
            if current_node in self.adj:
                for neighbor in self.adj[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse current state facts
        robot_locations = {}
        robot_colors = {}
        painted_tiles = {} # tile -> color
        # clear_tiles = set() # Not strictly needed if we assume goal tiles are either painted or clear

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
            # elif parts[0] == "clear":
            #      clear_tiles.add(parts[1]) # Not used in current logic

        total_cost = 0  # Initialize action cost counter.

        # Iterate through goal conditions
        for goal in self.goals:
            goal_parts = get_parts(goal)
            if goal_parts[0] == "painted":
                tile_i, color_j = goal_parts[1], goal_parts[2]

                # Check if goal is already satisfied
                if tile_i in painted_tiles and painted_tiles[tile_i] == color_j:
                    continue # Goal already met

                # Check if tile is painted with the wrong color (dead end)
                if tile_i in painted_tiles and painted_tiles[tile_i] != color_j:
                    return float('inf') # Cannot repaint in this domain

                # If tile is not painted (i.e., it is clear)
                # This tile needs to be painted with color_j
                min_cost_for_tile = float('inf')

                # Find all positions a robot can be to paint tile_i
                paint_pos_i = self.paint_adj.get(tile_i, set())

                if not paint_pos_i:
                    # Should not happen in a valid grid problem, but handle defensively
                    # If a tile cannot be painted from anywhere, it's unreachable
                    return float('inf')

                # Consider each robot
                for robot_r, tile_robot_r in robot_locations.items():
                    color_robot_r = robot_colors.get(robot_r) # Get robot's current color

                    # Cost to get the correct color
                    color_cost = 0
                    # Check if robot has a color and it's the wrong one
                    # Assuming robots always have a color based on domain actions/examples
                    if color_robot_r != color_j:
                         color_cost = 1 # Change color action

                    # Minimum movement cost to reach any valid painting position
                    move_cost = float('inf')
                    if tile_robot_r in self.distances:
                        for pos in paint_pos_i:
                            if pos in self.distances[tile_robot_r]:
                                move_cost = min(move_cost, self.distances[tile_robot_r][pos])

                    # If a valid painting position is reachable by this robot
                    if move_cost != float('inf'):
                        # Total cost for this robot to paint this tile = color change + move + paint
                        robot_cost = color_cost + move_cost + 1
                        min_cost_for_tile = min(min_cost_for_tile, robot_cost)

                # If no robot can reach a painting position for this tile
                if min_cost_for_tile == float('inf'):
                    return float('inf') # Goal is unreachable

                total_cost += min_cost_for_tile

        return total_cost
