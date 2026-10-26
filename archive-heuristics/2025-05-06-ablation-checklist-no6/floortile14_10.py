from collections import defaultdict, deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic

# Helper functions
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
# class floortileHeuristic(Heuristic): # Replace with Heuristic when integrating the base class
class floortileHeuristic:
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to paint all goal tiles that are currently unpainted.
    For each unpainted goal tile, it calculates the minimum cost for any robot
    to paint it, considering color change and movement costs, and sums these minimums.

    Cost for a single unpainted goal tile T needing color C_goal:
    min_{R, X adjacent to T} [ (1 if robot_color(R) != C_goal else 0) + distance(robot_location(R), X) + 1 ]
    where distance is the shortest path on the grid.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the grid graph for distance calculations.
        """
        self.goals = task.goals

        # Build adjacency list and collect all tile names from static facts
        self.adj = defaultdict(set)
        all_tiles = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                t1, t2 = parts[1], parts[2]
                self.adj[t1].add(t2)
                self.adj[t2].add(t1) # Grid movement is bidirectional
                all_tiles.add(t1)
                all_tiles.add(t2)
        self.all_tiles = list(all_tiles) # Convert to list for consistent iteration order

        # Pre-compute all-pairs shortest paths (distances) on the grid graph
        self.distances = {}
        for start_tile in self.all_tiles:
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            self.distances[(start_tile, start_tile)] = 0

            while q:
                current_tile, dist = q.popleft()

                for neighbor in self.adj.get(current_tile, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_tile, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # Extract goal colors for each tile that needs to be painted
        self.goal_colors = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_colors[tile] = color

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Identify goal tiles that are not yet painted correctly
        unpainted_goal_tiles = []
        for tile, goal_color in self.goal_colors.items():
            # Check if the tile is painted with the goal color
            if f"(painted {tile} {goal_color})" not in state:
                 # Also check if it's painted with the wrong color.
                 # If it is, the state is likely unsolvable in this domain.
                 # The domain only allows painting clear tiles.
                 # If a tile is painted with the wrong color, it can never be cleared or repainted.
                 # We can detect this and return infinity.
                 is_wrongly_painted = False
                 for fact in state:
                     parts = get_parts(fact)
                     if parts[0] == 'painted' and parts[1] == tile and parts[2] != goal_color:
                         is_wrongly_painted = True
                         break
                 if is_wrongly_painted:
                     # This tile is painted with the wrong color, goal is unreachable
                     return float('inf')

                 # If not painted with the goal color and not wrongly painted,
                 # it must be clear (or not mentioned as painted/clear, assume clear if goal).
                 # We only need to paint tiles that are clear and need painting.
                 # The paint action requires (clear ?y). So, if it's not painted correctly,
                 # it's either clear or wrongly painted.
                 # We already handled wrongly painted, so assume it's clear if it's a goal
                 # tile not painted correctly.
                 unpainted_goal_tiles.append(tile)


        # If all goal tiles are painted correctly, the heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        # Find current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # If there are unpainted goal tiles but no robots, the goal is unreachable
        if not robot_locations:
             return float('inf')

        total_cost = 0

        # For each unpainted goal tile, find the minimum cost for any robot to paint it
        for tile in unpainted_goal_tiles:
            goal_color = self.goal_colors[tile]
            min_tile_cost = float('inf')

            # Find tiles adjacent to the target tile (where a robot can stand to paint it)
            adjacent_tiles_for_painting = self.adj.get(tile, set())

            # If a goal tile has no adjacent tiles, it cannot be painted. Unsolvable.
            # This check might be redundant if the grid is always connected and goal tiles are part of it.
            # But it's safer to include.
            if not adjacent_tiles_for_painting:
                 return float('inf')

            # Calculate the minimum cost for each robot to paint this tile
            for robot, robot_location in robot_locations.items():
                robot_color = robot_colors.get(robot) # Use .get in case robot has no color initially (unlikely in this domain)

                # Cost to get the correct color
                color_cost = 1 if robot_color != goal_color else 0

                # Minimum cost to move from robot_location to any adjacent_tile_for_painting
                min_move_cost_for_robot = float('inf')
                for adjacent_tile in adjacent_tiles_for_painting:
                    # Ensure the robot's current location and the adjacent tile are in the precomputed distances
                    # If a tile is not in self.all_tiles, it means it wasn't mentioned in static facts defining the grid.
                    # This could happen if the robot is on a tile outside the main grid, or if the grid is disconnected.
                    # If robot_location is not in self.all_tiles, distances from it are not computed.
                    # If adjacent_tile is not in self.all_tiles, distances to it are not computed.
                    # We assume valid problems where robots are on grid tiles and goal tiles are on grid tiles.
                    if (robot_location, adjacent_tile) in self.distances:
                         move_cost = self.distances[(robot_location, adjacent_tile)]
                         min_move_cost_for_robot = min(min_move_cost_for_robot, move_cost)
                    # else: adjacent_tile is unreachable from robot_location in the grid graph, or one of the tiles is not in the graph.
                    # This robot cannot reach this adjacent tile via the defined grid paths.

                # If the robot cannot reach any adjacent tile, it cannot paint this tile.
                if min_move_cost_for_robot == float('inf'):
                    continue # Skip this robot for this tile

                # Total cost for this robot to paint this tile: color change + move + paint action
                robot_tile_cost = color_cost + min_move_cost_for_robot + 1 # +1 for the paint action

                # Update the minimum cost for this tile across all robots
                min_tile_cost = min(min_tile_cost, robot_tile_cost)

            # If after checking all robots, no robot can paint this tile, the state is likely unsolvable.
            if min_tile_cost == float('inf'):
                 return float('inf')

            # Add the minimum cost for this tile to the total heuristic cost
            total_cost += min_tile_cost

        return total_cost
