from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return [] # Return empty list for invalid format
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all goal tiles that are currently
    unpainted or painted incorrectly. It sums the number of paint actions needed,
    the estimated color changes required, and the estimated movement cost for
    robots to reach painting positions.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - Tiles are named in the format 'tile_R_C'.
    - Wrongly painted goal tiles indicate a potentially unsolvable state (or very high cost).
    - Movement cost is estimated using shortest path on the tile graph, ignoring 'clear' status as a relaxation for distance calculation (but 'clear' is required for actual movement and painting).
    - Color change cost is estimated by counting how many needed colors are not currently held by any robot.

    # Heuristic Initialization
    - Parses goal conditions to identify target colors for tiles.
    - Builds the tile adjacency graph from static 'up', 'down', 'left', 'right' facts.
    - Computes all-pairs shortest paths (distances) on the tile graph.
    - Stores adjacent tiles required for painting for each tile.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that need to be painted (i.e., their goal color is not currently present on the tile).
    2. Check for goal tiles that are painted with the wrong color; if any exist, return a very high heuristic value (indicating a likely dead end).
    3. Count the number of tiles that need painting. This is a base cost (each needs one paint action).
    4. Determine the set of unique colors required for the tiles identified in step 1.
    5. Calculate the color change cost: Count how many of the required colors are not currently held by any robot. Each such color adds 1 to the cost, representing the need for a robot to change to that color. This is a lower bound assuming one robot can acquire each missing color.
    6. Calculate the movement cost: For each tile that needs painting, find the minimum distance from any robot's current location to *any* tile adjacent to the tile that needs painting (a tile from which the paint action can be performed). Sum these minimum distances over all tiles that need painting. The distance calculation uses the precomputed shortest paths on the tile graph.
    7. The total heuristic value is the sum of the paint action count, the color change cost, and the movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the tile graph for distance calculations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations and colors for tiles
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color
            # Ignore other types of goal predicates if any exist

        # Build tile adjacency graph and store paint-adjacent tiles
        self.adj_list = {} # For general movement distance
        self.paint_adjacent_tiles = {} # Tiles from which a specific tile can be painted
        self.all_tiles = set()

        direction_predicates = ["up", "down", "left", "right"]

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in direction_predicates and len(parts) == 3:
                # (direction tile_y tile_x) means tile_y is in that direction from tile_x
                # Robot at tile_x can paint tile_y
                # Movement is between tile_x and tile_y
                dir_pred, tile_y, tile_x = parts

                self.all_tiles.add(tile_x)
                self.all_tiles.add(tile_y)

                # Add edge for movement graph (undirected)
                self.adj_list.setdefault(tile_x, []).append(tile_y)
                self.adj_list.setdefault(tile_y, []).append(tile_x)

                # Store which tiles are paint-adjacent for tile_y
                self.paint_adjacent_tiles.setdefault(tile_y, []).append(tile_x)

        # Compute all-pairs shortest paths (distances) on the tile graph
        self.distances = {}
        for start_node in self.all_tiles:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {start_node: 0}
        q = deque([start_node])
        visited = {start_node}

        while q:
            current_node = q.popleft()
            current_dist = distances[current_node]

            # Ensure current_node is in adj_list (might be an isolated tile if problem is malformed)
            if current_node in self.adj_list:
                for neighbor in self.adj_list[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = current_dist + 1
                        q.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current state information
        robot_locs = {}
        robot_colors = {}
        current_painted = {} # {tile: color}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot, location = parts[1], parts[2]
                robot_locs[robot] = location
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            # We don't strictly need 'clear' facts explicitly listed if we assume
            # a tile is either painted or clear (or occupied by robot, which makes it not clear).
            # The goal check handles if it's painted correctly or not.

        # Identify tiles that need painting according to the goal
        tiles_that_need_goal_color = [] # List of (tile, goal_color)
        needed_colors = set()
        wrongly_painted = False

        for goal_tile, goal_color in self.goal_painted_tiles.items():
             # Check if the goal predicate (painted goal_tile goal_color) is NOT in the state
             goal_predicate_str = f"(painted {goal_tile} {goal_color})"
             if goal_predicate_str not in state:
                 # This tile needs to be painted with goal_color
                 tiles_that_need_goal_color.append((goal_tile, goal_color))
                 needed_colors.add(goal_color)

                 # Check if it's wrongly painted
                 # Iterate through current_painted dictionary for efficiency
                 if goal_tile in current_painted and current_painted[goal_tile] != goal_color:
                     wrongly_painted = True
                     break # Found a wrongly painted tile, state is bad

        if wrongly_painted:
             # A goal tile is painted with the wrong color. Likely unsolvable.
             return float('inf')

        # --- Heuristic Calculation ---

        # 1. Base cost: one paint action for each tile that needs painting
        num_paint_actions = len(tiles_that_need_goal_color)

        # 2. Color change cost: For each needed color, if no robot has it, add 1
        current_robot_colors_set = set(robot_colors.values())
        color_change_cost = sum(1 for color in needed_colors if color not in current_robot_colors_set)

        # 3. Movement cost: Sum of minimum distances for robots to reach painting positions
        movement_cost = 0
        for goal_tile, goal_color in tiles_that_need_goal_color:
            min_dist_to_paint_adj = float('inf')

            # Find the minimum distance from any robot to any tile adjacent to goal_tile
            # from which painting is possible.
            # A robot at tile_x can paint tile_y if (dir tile_y tile_x) is true.
            # So, if goal_tile is tile_y, the robot needs to be at tile_x.
            # The tiles tile_x are stored in self.paint_adjacent_tiles[goal_tile].

            paint_adj_candidates = self.paint_adjacent_tiles.get(goal_tile, [])

            if not paint_adj_candidates:
                 # This goal tile has no adjacent tiles defined in static facts from which it can be painted.
                 # This might indicate a malformed problem or an unreachable tile.
                 # Treat as unsolvable or unreachable.
                 return float('inf') # Should not happen in valid problems

            # If there are no robots, no painting is possible
            if not robot_locs:
                 return float('inf') # Should not happen in valid problems

            for robot, loc_r in robot_locs.items():
                # Ensure robot location is in our distance graph (should be if graph built correctly from static facts)
                if loc_r not in self.distances:
                    # This should not happen in valid problems, but defensive check
                     continue # Skip this robot if its location is unknown/invalid

                for paint_adj_tile in paint_adj_candidates:
                    if paint_adj_tile in self.distances[loc_r]:
                         dist = self.distances[loc_r][paint_adj_tile]
                         min_dist_to_paint_adj = min(min_dist_to_paint_adj, dist)

            if min_dist_to_paint_adj == float('inf'):
                 # No robot can reach any tile adjacent to this goal tile. Unreachable.
                 return float('inf')

            movement_cost += min_dist_to_paint_adj

        # Total heuristic is the sum of the components
        total_heuristic = num_paint_actions + color_change_cost + movement_cost

        return total_heuristic
