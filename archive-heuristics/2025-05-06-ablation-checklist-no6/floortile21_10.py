import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    # Basic check for number of parts if no wildcard is used in args
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to paint all goal tiles that are not yet correctly painted.
    For each such tile, it calculates the minimum cost for any robot to reach
    an adjacent tile, change color if needed, and paint the tile. The total
    heuristic is the sum of these minimum costs for all unsatisfied goal tiles.

    Assumptions:
    - Goal tiles that are not correctly painted are assumed to be clear
      or can be made clear (ignoring the domain restriction that only clear
      tiles can be painted, which might make some states unsolvable if a goal
      tile is painted with the wrong color). Based on examples, goal tiles
      are initially clear if not correctly painted.
    - Movement cost is based on shortest path on the static grid graph,
      ignoring the 'clear' precondition for movement target tiles.
    - Color change costs 1 action. Painting costs 1 action.
    - The heuristic sums the minimum cost for each unsatisfied goal tile
      independently, ignoring potential synergies or conflicts between goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static grid information and goals.
        """
        super().__init__(task)

        # Store goal tiles and their required colors
        self.goal_tiles = {} # {tile_name: required_color}
        for goal in self.goals:
            # Goal facts are like (painted tile_X_Y color_Z)
            if match(goal, "painted", "*", "*"):
                tile_name, color_name = get_parts(goal)[1:]
                self.goal_tiles[tile_name] = color_name

        # Build the static grid graph (adjacency list) based on connectivity predicates
        self.adjacency_list = collections.defaultdict(list)
        all_tiles = set()

        # Predicates are (direction tile_from tile_to) e.g. (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
        # A robot at tile_0_1 can move up to tile_1_1. A robot at tile_1_1 can move down to tile_0_1.
        # A robot at tile_0_1 can paint tile_1_1 using paint_up if (up tile_1_1 tile_0_1) is true.
        # So, if (up Y X) is true, X is adjacent to Y, and a robot at X can paint Y.
        # The adjacency list should represent which tiles are adjacent for movement/painting.
        # If (up Y X) is true, Y and X are adjacent.
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2] # e.g., tile_1_1, tile_0_1 for (up tile_1_1 tile_0_1)
                # Add bidirectional edges between tile1 and tile2
                self.adjacency_list[tile1].append(tile2)
                self.adjacency_list[tile2].append(tile1)
                all_tiles.add(tile1)
                all_tiles.add(tile2)

        self.all_tiles = list(all_tiles) # Store all tile names

        # Precompute all-pairs shortest paths using BFS on the static grid
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = self._bfs(start_tile)

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find distances to all reachable nodes
        in the static grid graph.
        """
        distances = {node: float('inf') for node in self.all_tiles}
        distances[start_node] = 0
        queue = collections.deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            # Get neighbors from the adjacency list. Use .get for safety.
            for neighbor in self.adjacency_list.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # If the current state is a goal state, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Identify current robot locations and colors
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "robot-at":
                robot_name, tile_name = parts[1], parts[2]
                robot_locations[robot_name] = tile_name
            elif len(parts) == 3 and parts[0] == "robot-has":
                 robot_name, color_name = parts[1], parts[2]
                 robot_colors[robot_name] = color_name

        # Identify unsatisfied goal tiles
        unsatisfied_goals = {} # {tile_name: required_color}
        for tile_name, required_color in self.goal_tiles.items():
            # Check if the tile is currently painted with the required color
            is_correctly_painted = False
            for fact in state:
                parts = get_parts(fact)
                if len(parts) == 3 and parts[0] == "painted" and parts[1] == tile_name and parts[2] == required_color:
                    is_correctly_painted = True
                    break

            # If not correctly painted, it's an unsatisfied goal tile.
            # We add it to the list of goals to work on.
            if not is_correctly_painted:
                 unsatisfied_goals[tile_name] = required_color

        total_cost = 0

        # Calculate the minimum cost for each unsatisfied goal tile independently
        for tile_name, required_color in unsatisfied_goals.items():
            # Find adjacent tiles to the goal tile where a robot could paint from.
            # A robot needs to be AT a tile X such that Y is adjacent to X.
            # The adjacency list already captures these relationships.
            adjacent_tiles = self.adjacency_list.get(tile_name, [])

            # If a goal tile has no adjacent tiles in the graph, it's unreachable/unpaintable.
            # This indicates an unsolvable state or a problem definition issue.
            if not adjacent_tiles:
                 return float('inf') # Return infinity as it's likely impossible

            min_cost_for_tile = float('inf')

            # Consider each robot to find the one that can paint this tile fastest
            for robot_name, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot_name) # Get color robot is holding

                # Find minimum distance from robot's current location to any adjacent tile
                min_dist_to_adj = float('inf')
                if robot_loc in self.distances: # Ensure robot's current location is a known tile
                    for adj_tile in adjacent_tiles:
                        if adj_tile in self.distances[robot_loc]: # Ensure adjacent tile is a known tile
                             dist = self.distances[robot_loc][adj_tile]
                             min_dist_to_adj = min(min_dist_to_adj, dist)

                # If the robot cannot reach any adjacent tile, it cannot paint this tile.
                if min_dist_to_adj == float('inf'):
                    continue # Try the next robot

                # Cost to change color if the robot doesn't have the required color.
                # We assume the required color is available (checked in domain init).
                color_change_cost = 1 if robot_color != required_color else 0

                # Total estimated cost for this robot to paint this specific tile:
                # moves to adjacent tile + color change (if needed) + paint action
                cost_by_this_robot = min_dist_to_adj + color_change_cost + 1

                min_cost_for_tile = min(min_cost_for_tile, cost_by_this_robot)

            # If no robot can reach an adjacent tile to paint this goal tile,
            # this goal is likely impossible to satisfy from this state.
            if min_cost_for_tile == float('inf'):
                 return float('inf') # Return infinity

            # Add the minimum cost found for this specific goal tile to the total
            total_cost += min_cost_for_tile

        return total_cost
