from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
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

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the minimum number of actions required to paint
    all goal tiles with their target colors. It sums the estimated cost for
    each individual tile that is not yet painted correctly. The estimated cost
    for a single tile is the minimum cost for any robot to reach an adjacent
    tile, acquire the correct color if needed, and perform the paint action.

    # Assumptions
    - The grid topology is defined by the up/down/left/right predicates.
    - All tiles mentioned in adjacency predicates form a connected component.
    - If a tile is painted with the wrong color, the state is considered unsolvable
      and the heuristic returns infinity.
    - The cost of each action (move, change_color, paint) is 1.

    # Heuristic Initialization
    - Parses static facts to build the grid graph (adjacency list).
    - Identifies all tiles in the grid.
    - Precomputes shortest path distances between all pairs of tiles using BFS.
    - Stores the goal state, specifically which tiles need to be painted with which colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted according to the goal state.
    2. For each goal tile, check its current state:
       - If it is already painted with the correct color, it contributes 0 to the heuristic.
       - If it is painted with a *wrong* color, the state is likely unsolvable; return infinity.
       - If it is `clear` (or any other state not matching the goal painted state), it needs painting.
    3. For each tile that needs painting with a specific goal color:
       - Calculate the minimum cost for *any* robot to paint this tile.
       - The cost for a robot R to paint tile T with color C is:
         - Minimum moves for R from its current location to *any* tile adjacent to T.
         - + 1 if R does not currently have color C (cost to change color).
         - + 1 (cost of the paint action itself).
       - The minimum cost for tile T is the minimum of the above cost over all robots.
    4. The total heuristic value is the sum of the minimum costs calculated for each tile that needs painting.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        self.neighbors = {}
        self.all_tiles = set()

        # Build the adjacency list (graph) from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Predicate is (direction tile1 tile2) meaning tile1 is in that direction from tile2
                # So tile1 and tile2 are neighbors.
                tile1, tile2 = parts[1], parts[2]
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)
                self.neighbors.setdefault(tile1, set()).add(tile2)
                self.neighbors.setdefault(tile2, set()).add(tile1)

        self.all_tiles = list(self.all_tiles) # Convert to list

        # Precompute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.all_tiles:
            self.dist[start_node] = self._bfs(start_node)

        # Store goal painted states
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if match(goal, "painted", "*", "*"):
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all reachable nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {node: float('inf') for node in self.all_tiles}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Get neighbors, handling nodes that might not have any connections listed
            for neighbor in self.neighbors.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # Parse current state
        robot_loc = {}
        robot_color = {}
        tile_state = {} # {tile: 'clear' or ('painted', color)}
        robots = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if match(fact, "robot-at", "*", "*"):
                robot, tile = parts[1], parts[2]
                robot_loc[robot] = tile
                robots.add(robot)
            elif match(fact, "robot-has", "*", "*"):
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
                robots.add(robot)
            elif match(fact, "clear", "*"):
                tile = parts[1]
                tile_state[tile] = 'clear'
            elif match(fact, "painted", "*", "*"):
                tile, color = parts[1], parts[2]
                tile_state[tile] = ('painted', color)

        # Identify tiles that need painting and check for wrong colors
        needs_painting = {} # {tile: goal_color}
        for tile, goal_color in self.goal_painted.items():
            current_state = tile_state.get(tile)

            if current_state != ('painted', goal_color):
                # Tile is not painted correctly
                needs_painting[tile] = goal_color

                # Check if it's painted with the wrong color (unsolvable state)
                if current_state is not None and current_state != 'clear':
                     if current_state[0] == 'painted' and current_state[1] != goal_color:
                         # Found a tile painted with the wrong color
                         return float('inf') # Indicate unsolvable state

        # If all goal tiles are painted correctly, the goal is reached
        if not needs_painting:
            return 0

        total_heuristic = 0

        # Calculate cost for each tile that needs painting
        for tile, goal_color in needs_painting.items():
            min_cost_for_this_tile = float('inf')

            # Find tiles adjacent to the current tile
            adjacent_tiles = self.neighbors.get(tile, set())
            if not adjacent_tiles:
                 # A goal tile has no neighbors - likely an invalid problem or unsolvable
                 return float('inf')

            # Find the minimum cost for any robot to paint this tile
            for robot in robots:
                current_robot_loc = robot_loc.get(robot)
                current_robot_color = robot_color.get(robot)

                # If robot state is incomplete or robot is not on a known tile, skip
                if current_robot_loc is None or current_robot_color is None or current_robot_loc not in self.dist:
                    continue

                # Cost to change color if needed
                color_cost = 1 if current_robot_color != goal_color else 0

                # Minimum moves to get adjacent to the tile
                min_move_cost = float('inf')
                for adj_tile in adjacent_tiles:
                    # Ensure the adjacent tile is reachable from robot's location
                    if adj_tile in self.dist[current_robot_loc]:
                         min_move_cost = min(min_move_cost, self.dist[current_robot_loc][adj_tile])

                # If min_move_cost is still inf, this robot cannot reach the tile
                if min_move_cost == float('inf'):
                    cost_R_paints_T = float('inf')
                else:
                    # Total cost for this robot to paint this tile
                    cost_R_paints_T = min_move_cost + color_cost + 1 # +1 for the paint action

                min_cost_for_this_tile = min(min_cost_for_this_tile, cost_R_paints_T)

            # If no robot can reach this tile, the state is unsolvable
            if min_cost_for_this_tile == float('inf'):
                 return float('inf')

            total_heuristic += min_cost_for_this_tile

        return total_heuristic
