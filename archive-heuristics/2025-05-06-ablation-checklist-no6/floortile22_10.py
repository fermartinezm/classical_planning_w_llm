# from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available

from fnmatch import fnmatch
import collections # For deque

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
    # Check if the number of parts is sufficient to match the pattern args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Inherit from Heuristic if available in the environment
# class floortileHeuristic(Heuristic):
class floortileHeuristic: # Use this if Heuristic base class is not provided in the same file
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet correctly painted. It sums the minimum estimated cost for each
    unpainted goal tile, where the minimum is taken over all robots. The cost for
    a single robot to paint a tile includes changing color (if necessary), moving
    to a clear adjacent tile, and the paint action itself. Movement cost is estimated
    using BFS on the graph of clear tiles.

    # Heuristic Initialization
    - Extracts the goal conditions.
    - Builds the tile adjacency graph from static facts.
    - Stores available colors.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify all goal tiles that are currently not painted with the required color.
    2. For each such unpainted goal tile:
       a. Determine the required color.
       b. Find all tiles adjacent to the goal tile.
       c. For each robot:
          i. Determine the robot's current location and color.
          ii. Calculate the cost for the robot to change to the required color (1 if different, 0 if same).
          iii. Calculate the minimum number of moves required for the robot to reach *any* clear tile adjacent to the goal tile. This is done using BFS on the graph of tiles, considering only clear tiles as traversable nodes.
          iv. The estimated cost for this robot to paint this tile is (color change cost) + (minimum move cost) + 1 (for the paint action).
       d. Find the minimum estimated cost among all robots to paint this specific goal tile.
       e. If no robot can paint this tile (e.g., no reachable clear adjacent tile), the state is likely unsolvable (return infinity).
    3. The total heuristic value is the sum of the minimum costs calculated for each unpainted goal tile.
    4. If all goal tiles are painted correctly, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build tile graph (adjacency list) from static facts
        self.tile_graph = self._build_tile_graph(self.static_facts)

        # Store available colors (optional, but good practice)
        self.available_colors = {
            get_parts(fact)[1]
            for fact in self.static_facts
            if match(fact, "available-color", "*")
        }

    def _build_tile_graph(self, static_facts):
        """Builds an undirected graph of tiles based on adjacency facts."""
        graph = collections.defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Adjacency facts are like (direction tile1 tile2)
                # This means tile1 and tile2 are adjacent.
                tile1 = parts[1]
                tile2 = parts[2]
                graph[tile1].add(tile2)
                graph[tile2].add(tile1) # Graph is undirected for movement
        return dict(graph) # Convert defaultdict to dict

    def _get_clear_tiles(self, state):
        """Extracts the set of clear tiles from the current state."""
        return {get_parts(fact)[1] for fact in state if match(fact, "clear", "*")}

    def _get_robot_info(self, state):
        """Extracts robot locations and colors from the current state."""
        robots = {}
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "robot-at", "*", "*"):
                robot_name, location = parts[1], parts[2]
                if robot_name not in robots:
                    robots[robot_name] = {}
                robots[robot_name]['location'] = location
            elif match(fact, "robot-has", "*", "*"):
                robot_name, color = parts[1], parts[2]
                if robot_name not in robots:
                    robots[robot_name] = {}
                robots[robot_name]['color'] = color
        return robots

    def _get_adjacent_tiles(self, tile):
        """Gets adjacent tiles for a given tile from the pre-built graph."""
        return self.tile_graph.get(tile, set())

    def _bfs(self, start_node, graph, traversable_nodes):
        """
        Perform BFS from start_node on graph, only traversing through traversable_nodes.
        Returns distances to all reachable traversable_nodes.
        The start_node itself does not need to be in traversable_nodes to start,
        but subsequent nodes on the path must be.
        """
        distances = {}
        queue = collections.deque([(start_node, 0)])
        visited = {start_node}

        while queue:
            (current_tile, dist) = queue.popleft()
            distances[current_tile] = dist # Store distance to current_tile

            if current_tile in graph: # Ensure current_tile is in the graph
                for neighbor in graph[current_tile]:
                    # Can move to neighbor if neighbor is clear AND not visited
                    if neighbor in traversable_nodes and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return distances # Returns distances to reachable *clear* tiles from start_node

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state
        clear_tiles = self._get_clear_tiles(state)
        robots_info = self._get_robot_info(state)

        unpainted_goals = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # We only care about (painted tile color) goals
            if match(goal, "painted", "*", "*"):
                tile, color = parts[1], parts[2]
                # Check if the goal fact is NOT in the current state
                if goal not in state:
                     unpainted_goals.add((tile, color))

        if not unpainted_goals:
            return 0 # Goal reached

        total_heuristic = 0
        infinity = float('inf')

        # Pre-calculate BFS distances for all robots from their current locations
        # This avoids re-running BFS for every unpainted goal tile.
        robot_distances = {}
        for robot_name, info in robots_info.items():
             start_node = info['location']
             # BFS from robot_location to all reachable *clear* tiles
             robot_distances[robot_name] = self._bfs(start_node, self.tile_graph, clear_tiles)


        for goal_tile, goal_color in unpainted_goals:
            min_cost_for_tile = infinity
            adj_tiles = self._get_adjacent_tiles(goal_tile)

            # The goal tile itself must be clear to be painted.
            # If it's not clear and not painted with the goal color, it's unsolvable.
            # Assuming valid problems, unpainted goal tiles are clear.
            # Add check for robustness.
            if goal_tile not in clear_tiles:
                 # This goal tile cannot be painted because it's not clear.
                 # Since it's in unpainted_goals, it's not painted with the correct color.
                 # This state is likely unsolvable.
                 return infinity


            for robot_name, info in robots_info.items():
                robot_location = info['location']
                # Ensure robot_color exists in info (handle cases where robot-has might be missing, though unlikely in valid PDDL)
                robot_color = info.get('color')

                # A robot needs a color to paint. If it has no color, it cannot paint or change color.
                # The PDDL implies a robot always has *a* color or is 'free-color' (unused).
                # Let's assume robot_color is always present if robot-has fact exists.
                # If robot_color is None, this robot cannot paint.
                if robot_color is None:
                    # This robot cannot paint. Skip it for this tile.
                    continue

                color_change_cost = 0
                if robot_color != goal_color:
                    # Cost to change color is 1. Assumes goal_color is available.
                    # The PDDL requires (available-color ?c2) for change_color.
                    # Goal colors must be available.
                    color_change_cost = 1

                min_move_cost = infinity

                # Find min distance from robot_location to any clear adjacent tile
                distances_from_robot = robot_distances.get(robot_name, {})

                # Need to reach a tile adjacent to goal_tile AND that adjacent tile must be clear.
                reachable_clear_adj_tiles = [
                    adj_tile for adj_tile in adj_tiles
                    if adj_tile in clear_tiles and adj_tile in distances_from_robot
                ]

                if reachable_clear_adj_tiles:
                    # Find the minimum distance among reachable clear adjacent tiles
                    min_move_cost = min(distances_from_robot[adj_tile] for adj_tile in reachable_clear_adj_tiles)


                if min_move_cost == infinity:
                    # This robot cannot reach any clear adjacent tile.
                    cost_for_R = infinity
                else:
                    # Cost is color change + move to clear adjacent tile + paint action
                    cost_for_R = color_change_cost + min_move_cost + 1

                min_cost_for_tile = min(min_cost_for_tile, cost_for_R)

            if min_cost_for_tile == infinity:
                 # No robot can paint this tile.
                 return infinity

            total_heuristic += min_cost_for_tile

        return total_heuristic
