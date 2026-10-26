from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact string defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
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

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their required colors. It sums the estimated cost for each unpainted goal
    tile independently, considering the closest robot, the distance to an adjacent
    clear tile, and the need to change color.

    # Assumptions
    - Goal tiles are initially clear or painted with the correct color. If a goal
      tile is painted with the wrong color in an intermediate state, the problem
      is considered unsolvable from that state (heuristic returns infinity).
    - Movement cost between adjacent tiles is 1 action.
    - Changing color costs 1 action.
    - Painting a tile costs 1 action.
    - The heuristic ignores potential negative interactions between robots (e.g.,
      blocking paths, competing for the same tile).
    - BFS for distance calculation only considers 'clear' tiles as traversable.

    # Heuristic Initialization
    - Extracts goal requirements (tile -> required_color).
    - Builds an adjacency map of tiles based on 'up', 'down', 'left', 'right'
      static facts. This map represents the grid structure.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles that are not yet painted with the correct color.
    2. For each unpainted goal tile (Y) requiring color (C):
        a. Find all tiles (X) that are adjacent to Y using the pre-computed
           adjacency map.
        b. For each robot (R):
            i. Determine the robot's current location (L) and current color (C_R).
            ii. Calculate the shortest path distance (in terms of move actions)
                from the robot's current location (L) to *any* tile (X) adjacent
                to Y. This is done using a Breadth-First Search (BFS) on the
                tile grid graph. The BFS only traverses through tiles that are
                currently marked as 'clear' in the state.
            iii. If no path exists to an adjacent clear tile, this robot cannot
                 paint Y from its current position without external help (e.g.,
                 another robot clearing a path). Consider this cost infinite for this robot.
            iv. If a path exists with distance 'dist':
                - The base cost for this robot to paint Y is `dist` (moves) + 1 (paint action).
                - If the robot's current color (C_R) is not the required color (C),
                  add 1 to the cost for a 'change_color' action.
            v. Keep track of the minimum cost among all robots to paint tile Y.
        c. If the goal tile Y is currently painted with a color different from C,
           the state is unsolvable, return infinity.
        d. Add the minimum cost found in step 2.b.v to the total heuristic value.
    3. The total heuristic value is the sum of the minimum costs for all unpainted
       goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal requirements and the tile grid structure.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal requirements: map tile_name -> required_color
        self.goal_requirements = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                # Fact is (painted tile_name color)
                if len(parts) == 3:
                    tile_name, color = parts[1], parts[2]
                    self.goal_requirements[tile_name] = color
                # Assuming task.goals contains ground facts like "(painted tile_1_1 white)"

        # Build adjacency map: tile_name -> set of adjacent tile_names
        self.adjacency_map = {}
        self.all_tiles = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Fact is (direction tile1 tile2) meaning tile1 is direction of tile2
                # So tile1 and tile2 are adjacent.
                tile1, tile2 = parts[1], parts[2]
                self.adjacency_map.setdefault(tile1, set()).add(tile2)
                self.adjacency_map.setdefault(tile2, set()).add(tile1)
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)

        # Ensure all goal tiles are included even if they have no adjacency facts (unlikely)
        self.all_tiles.update(self.goal_requirements.keys())


    def _bfs_distance(self, start_tile, target_tiles_set, state, adjacency_map):
        """
        Performs BFS to find the shortest path distance from start_tile to any tile
        in target_tiles_set, only traversing through 'clear' tiles.
        Returns the distance or float('inf') if no path exists.
        """
        if start_tile in target_tiles_set:
            return 0 # Already at a target tile

        queue = deque([(start_tile, 0)])
        visited = {start_tile}

        # Convert state set to a set for faster lookups of clear facts
        state_set = set(state)

        while queue:
            current_tile, dist = queue.popleft()

            # Check if we reached a target tile
            if current_tile in target_tiles_set:
                return dist

            # Explore neighbors
            for neighbor in adjacency_map.get(current_tile, set()):
                # Check if neighbor is clear and not visited
                if f"(clear {neighbor})" in state_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # No path found

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
             return 0

        # Check for unsolvable states: goal tile painted wrong color
        # This check is crucial for correctness.
        for goal_tile, required_color in self.goal_requirements.items():
            # Check if the tile is painted with *any* color
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "painted" and len(parts) == 3:
                    painted_tile, painted_color = parts[1], parts[2]
                    if painted_tile == goal_tile and painted_color != required_color:
                        # Goal tile is painted with the wrong color - unsolvable
                        return float('inf')

        # Find robot locations and colors
        robot_info = {} # robot_name -> {'location': tile_name, 'color': color}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "robot-at" and len(parts) == 3:
                    robot, location = parts[1], parts[2]
                    robot_info.setdefault(robot, {})['location'] = location
                elif parts[0] == "robot-has" and len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_info.setdefault(robot, {})['color'] = color

        total_cost = 0

        # Iterate through unpainted goal tiles
        for goal_tile, required_color in self.goal_requirements.items():
            # Check if this goal tile is already painted correctly
            if f"(painted {goal_tile} {required_color})" in state:
                continue # This goal is already satisfied

            # Find tiles adjacent to the goal tile (these are potential painting spots)
            adjacent_to_goal_tile = self.adjacency_map.get(goal_tile, set())

            # If a goal tile has no adjacent tiles (shouldn't happen in valid problems),
            # it's impossible to paint.
            if not adjacent_to_goal_tile:
                 # This indicates a problem with the domain/instance definition if a goal tile has no neighbors
                 # In a real scenario, this might be an unsolvable state.
                 return float('inf')

            min_cost_for_tile = float('inf')

            # Find the best robot to paint this tile
            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if not robot_location or not robot_color:
                    # Robot info is incomplete, skip (should not happen in valid states)
                    continue

                # Calculate distance from robot to any adjacent clear tile
                dist_to_adjacent_clear_tile = self._bfs_distance(
                    robot_location,
                    adjacent_to_goal_tile,
                    state,
                    self.adjacency_map
                )

                # If no path exists to an adjacent clear tile, this robot can't do it alone
                if dist_to_adjacent_clear_tile == float('inf'):
                    continue

                # Cost includes movement and paint action
                cost_R = dist_to_adjacent_clear_tile + 1 # +1 for the paint action

                # Add cost for changing color if needed
                if robot_color != required_color:
                    cost_R += 1 # +1 for the change_color action

                min_cost_for_tile = min(min_cost_for_tile, cost_R)

            # If no robot could reach an adjacent clear tile for this goal tile,
            # the state is likely unsolvable or requires complex coordination.
            if min_cost_for_tile == float('inf'):
                 return float('inf')

            total_cost += min_cost_for_tile

        return total_cost
