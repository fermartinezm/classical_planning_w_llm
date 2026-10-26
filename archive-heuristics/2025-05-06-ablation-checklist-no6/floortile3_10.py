from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic
import math
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to handle cases where args might be shorter than parts due to trailing wildcards
    # or where parts might be shorter than args (should return False in that case)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to reach the goal by summing the minimum cost
    to paint each unsatisfied goal tile independently. The cost for a
    single tile includes movement to an adjacent tile, changing color
    if necessary, and the paint action itself.

    This heuristic is non-admissible as it calculates the minimum cost
    for each tile independently, potentially double-counting robot
    movements or color changes that could satisfy multiple goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the tile grid graph and
        precomputing all-pairs shortest paths.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build tile map and adjacency list from static facts
        self.tile_names = set()
        self.adj = {} # tile_name -> [neighbor_tile_name, ...]

        # Collect all tile names and build adjacency list
        for fact in self.static_facts:
            parts = get_parts(fact)
            # Adjacency facts are (direction tile_y tile_x) meaning tile_y is in that direction from tile_x
            # This implies an edge between tile_x and tile_y
            if parts[0] in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                tile_y, tile_x = parts[1], parts[2]
                self.tile_names.add(tile_x)
                self.tile_names.add(tile_y)
                # Add undirected edges
                self.adj.setdefault(tile_x, []).append(tile_y)
                self.adj.setdefault(tile_y, []).append(tile_x)

        # Precompute all-pairs shortest paths using BFS
        self.dist = {} # dist[start_tile][end_tile] = distance
        for start_tile in self.tile_names:
            self.dist[start_tile] = {}
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            self.dist[start_tile][start_tile] = 0

            while q:
                current_tile, current_d = q.popleft()

                # Ensure current_tile is in adj before accessing
                for neighbor in self.adj.get(current_tile, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[start_tile][neighbor] = current_d + 1
                        q.append((neighbor, current_d + 1))

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Estimates the cost by summing the minimum cost for each unpainted
        goal tile to be painted.
        """
        state = node.state

        # Identify robot locations and colors in the current state
        robot_loc = {}
        robot_color = {}
        robots = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_loc[robot] = tile
                robots.add(robot)
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
                robots.add(robot)

        # If there are no robots and the goal is not met, it's impossible
        if not robots and not self.goals <= state:
             return math.inf

        total_heuristic = 0

        # Iterate through all goal facts to find unsatisfied painted goals
        for goal_fact in self.goals:
            # We only care about (painted tile color) goals
            if not match(goal_fact, "painted", "*", "*"):
                 continue # Ignore other types of goals if any

            goal_parts = get_parts(goal_fact)
            # Ensure goal fact has expected number of parts
            if len(goal_parts) != 3:
                continue

            goal_tile = goal_parts[1]
            goal_color = goal_parts[2]

            # Check if this specific goal fact is already true in the state
            if goal_fact in state:
                continue # This goal is satisfied

            # Check if the tile is painted with a different color (impossible state)
            is_wrongly_painted = False
            for fact in state:
                if match(fact, "painted", goal_tile, "*"):
                    painted_color_parts = get_parts(fact)
                    if len(painted_color_parts) == 3: # Ensure fact is well-formed
                        painted_color = painted_color_parts[2]
                        if painted_color != goal_color:
                            is_wrongly_painted = True
                            break
            if is_wrongly_painted:
                return math.inf # Goal is impossible to achieve from this state

            # If the goal is not satisfied and not wrongly painted, the tile needs painting.
            # It must be clear (or implicitly clear if not mentioned as painted).
            # Calculate the minimum cost to paint this specific tile.

            min_cost_to_paint_tile = math.inf

            # Find neighbors of the goal tile from the precomputed graph
            neighbors_of_goal_tile = self.adj.get(goal_tile, [])

            # If a goal tile has no neighbors, it cannot be painted by adjacent actions.
            # This state is likely impossible unless the goal was already met (handled above).
            if not neighbors_of_goal_tile:
                 return math.inf

            # Find the minimum cost among all robots to paint this tile
            for robot in robots:
                r_loc = robot_loc.get(robot)
                r_color = robot_color.get(robot)

                # Should always have location and color in a valid state, but check defensively
                if r_loc is None or r_color is None:
                    continue

                # Calculate the minimum distance from the robot's current location
                # to any tile adjacent to the goal tile.
                min_dist_to_neighbor = math.inf
                for neighbor in neighbors_of_goal_tile:
                    # Check if the robot's location and the neighbor are in the distance map
                    # (handles cases where a tile might be disconnected, though unlikely in grid)
                    if r_loc in self.dist and neighbor in self.dist[r_loc]:
                         min_dist_to_neighbor = min(min_dist_to_neighbor, self.dist[r_loc][neighbor])

                # If the robot cannot reach any neighbor, it cannot paint this tile
                if min_dist_to_neighbor == math.inf:
                    continue # Try the next robot

                # Cost for this robot to paint the tile:
                # 1. Movement cost to reach an adjacent tile: min_dist_to_neighbor
                # 2. Paint action cost: 1
                # 3. Color change cost: 1 if the robot's current color is not the goal color

                cost_this_robot = min_dist_to_neighbor # moves
                cost_this_robot += 1 # paint action

                # Add cost for changing color if the robot doesn't have the required color
                if r_color != goal_color:
                    cost_this_robot += 1 # change_color action

                # Update the minimum cost to paint this tile across all robots
                min_cost_to_paint_tile = min(min_cost_to_paint_tile, cost_this_robot)

            # If after checking all robots, no robot can paint this tile, the goal is impossible
            if min_cost_to_paint_tile == math.inf:
                 return math.inf

            # Add the minimum cost required for this tile to the total heuristic
            total_heuristic += min_cost_to_paint_tile

        # The total heuristic is the sum of minimum costs for each unpainted goal tile
        return total_heuristic
