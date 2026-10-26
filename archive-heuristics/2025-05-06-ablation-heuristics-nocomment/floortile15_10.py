from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Parses a fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if args is longer than parts
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing up
    three components:
    1. The number of goal tiles that are not yet painted correctly (each needs a paint action).
    2. The number of distinct colors required by the unpainted goal tiles that no robot currently possesses (each requires a change_color action by at least one robot).
    3. An estimate of the movement cost, calculated as the sum of the minimum Manhattan distances from each robot to the closest unpainted goal tile.

    Assumptions:
    - The grid structure can be fully deduced from static adjacency facts.
    - The grid is connected.
    - Solvable problems do not start with goal tiles painted with the wrong color.
    - The heuristic ignores the 'clear' predicate constraint for movement path calculation (uses Manhattan distance), which is an approximation.
    - The heuristic assumes any robot can perform any task (paint, change color, move).

    Heuristic Initialization:
    The constructor processes static facts to build data structures needed for the heuristic calculation:
    - `tile_coords_map`: Maps each tile name to its (row, col) coordinate. This is deduced by performing a BFS traversal starting from an arbitrary tile, using the 'up', 'down', 'left', 'right' static facts to determine relative positions.
    - `goal_tiles_info`: Maps each goal tile name to the color it needs to be painted.
    - `all_tiles`: A set of all tile names in the problem.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state from the search node.
    2. Parse the state to find:
       - Current location of each robot.
       - Current color held by each robot.
       - Which tiles are currently painted and with what color.
    3. Identify unsatisfied goal tiles: Iterate through the `goal_tiles_info`. A goal tile `t` needing color `c` is unsatisfied if the state does not contain `(painted t c)`.
    4. Check for dead ends: If any goal tile `t` is found in the state as `(painted t c')` where `c'` is not the required color from `goal_tiles_info`, the state is a dead end. Return `float('inf')`.
    5. If there are no unsatisfied goal tiles, the state is a goal state. Return 0.
    6. Calculate the base heuristic value: Initialize `h = 0`. Add the number of unsatisfied goal tiles to `h`. This accounts for the minimum number of paint actions required.
    7. Calculate color change cost: Identify the set of colors required by the unsatisfied goal tiles. Identify the set of colors currently held by robots. For each required color that is not held by any robot, add 1 to `h`. This estimates the minimum number of `change_color` actions needed to make all required colors available. The heuristic assumes one robot changing color is sufficient for the purpose of this calculation.
    8. Calculate movement cost: For each robot, find the minimum Manhattan distance from its current location to the closest unsatisfied goal tile. Sum these minimum distances over all robots. Add this sum to `h`. This estimates the total movement effort required for robots to get close to their painting tasks, simplifying the multi-robot assignment and ignoring the 'clear' constraint for movement paths.
    9. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static

        self.goal_tiles_info = {}
        for goal in self.goals:
            # Assuming goals are always (painted tile color)
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_tiles_info[tile] = color

        # Deduce tile coordinates from static facts
        self.tile_coords_map = {}
        self.all_tiles = set()

        # Collect adjacency relations
        adj_relations = defaultdict(list) # tile -> list of (relation, neighbor_tile)
        for fact in self.static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                rel, tile1, tile2 = parts
                # In PDDL: (relation tile_dest tile_src)
                # So, relation from tile2 to tile1
                adj_relations[tile2].append((rel, tile1))
                # Also add the reverse relation for building the graph
                reverse_rel = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}[rel]
                adj_relations[tile1].append((reverse_rel, tile2))

                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)

        # BFS to deduce coordinates
        start_tile = None
        # Try to find a tile mentioned in init or goals first for robustness
        # Find any tile mentioned in init state
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Check if the second argument is a tile (e.g., robot-has robot1 black)
             # or the third argument is a tile (e.g., robot-at robot1 tile_0_1)
             if len(parts) > 1 and parts[1] in self.all_tiles:
                 start_tile = parts[1]
                 break
             if len(parts) > 2 and parts[2] in self.all_tiles:
                 start_tile = parts[2]
                 break

        # If no tile found in init, try goals
        if start_tile is None:
             for goal_tile in self.goal_tiles_info.keys():
                 if goal_tile in self.all_tiles:
                     start_tile = goal_tile
                     break

        # Fallback to any tile if still not found
        if start_tile is None and self.all_tiles:
             start_tile = next(iter(self.all_tiles))

        if start_tile:
            queue = deque([(start_tile, (0, 0))])
            visited = {start_tile}
            self.tile_coords_map[start_tile] = (0, 0)

            while queue:
                (current_tile, (r, c)) = queue.popleft()

                # Iterate through neighbors using the collected relations
                for relation, neighbor_tile in adj_relations.get(current_tile, []): # Use .get for safety
                     if neighbor_tile not in visited:
                        visited.add(neighbor_tile)
                        nr, nc = r, c
                        if relation == 'up':
                            nr = r - 1
                        elif relation == 'down':
                            nr = r + 1
                        elif relation == 'left':
                            nc = c - 1
                        elif relation == 'right':
                            nc = c + 1

                        self.tile_coords_map[neighbor_tile] = (nr, nc)
                        queue.append((neighbor_tile, (nr, nc)))

            # Adjust coordinates so minimum row/col is 0
            if self.tile_coords_map:
                min_r = min(r for r, c in self.tile_coords_map.values())
                min_c = min(c for r, c in self.tile_coords_map.values())
                self.tile_coords_map = {
                    tile: (r - min_r, c - min_c)
                    for tile, (r, c) in self.tile_coords_map.items()
                }


    def manhattan_distance(self, tile1, tile2):
        """Calculates Manhattan distance between two tiles using precomputed coordinates."""
        coords1 = self.tile_coords_map.get(tile1)
        coords2 = self.tile_coords_map.get(tile2)
        if coords1 is None or coords2 is None:
            # This indicates an issue if tiles exist but weren't mapped
            # Should not happen in valid problems if BFS logic is correct
            # Returning infinity makes this tile unreachable for distance calculation
            return float('inf')
        r1, c1 = coords1
        r2, c2 = coords2
        return abs(r1 - r2) + abs(c1 - c2)


    def __call__(self, node):
        state = node.state

        robot_locations = {}
        robot_colors = {}
        painted_tiles = {}

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color

        unsatisfied_goal_tiles = {} # tile -> required_color
        for goal_tile, required_color in self.goal_tiles_info.items():
            current_painted_color = painted_tiles.get(goal_tile)

            if current_painted_color is None:
                # Tile is not painted at all, needs painting
                unsatisfied_goal_tiles[goal_tile] = required_color
            elif current_painted_color != required_color:
                # Tile is painted with the wrong color - dead end
                return float('inf')
            # Else: Tile is painted with the correct color, goal satisfied for this tile

        # If all goal tiles are satisfied
        if not unsatisfied_goal_tiles:
            return 0

        h = 0

        # Component 1: Cost for paint actions
        h += len(unsatisfied_goal_tiles)

        # Component 2: Cost for color changes
        needed_colors = set(unsatisfied_goal_tiles.values())
        available_robot_colors = set(robot_colors.values())
        color_change_cost = 0
        # Simulate availability for other needed colors check within this loop
        simulated_available_colors = set(available_robot_colors)
        for color in needed_colors:
            if color not in simulated_available_colors:
                color_change_cost += 1
                simulated_available_colors.add(color) # Assume one robot can get this color

        h += color_change_cost

        # Component 3: Movement cost
        # Sum of minimum distances from each robot to the closest unsatisfied goal tile
        movement_cost = 0
        unsatisfied_tiles_list = list(unsatisfied_goal_tiles.keys())

        # Handle case where no robots exist (shouldn't happen in valid problems)
        if not robot_locations:
             # If there are unsatisfied goals but no robots, it's unsolvable
             # This heuristic returns infinity for dead ends, so this fits.
             return float('inf')

        for robot, loc in robot_locations.items():
            min_dist_for_robot = float('inf')
            for goal_tile in unsatisfied_tiles_list:
                 dist = self.manhattan_distance(loc, goal_tile)
                 min_dist_for_robot = min(min_dist_for_robot, dist)

            # Only add movement cost if there are reachable unsatisfied tiles
            # If min_dist_for_robot is inf, it means the robot cannot reach any
            # unsatisfied tile based on the mapped grid.
            if min_dist_for_robot != float('inf'):
                 movement_cost += min_dist_for_robot
            # If min_dist_for_robot remains inf, it means this specific robot
            # cannot reach any of the currently unsatisfied goal tiles based on
            # the precomputed grid map. This might indicate an unsolvable problem
            # or just that this robot is currently in an isolated part of the grid
            # or the goal tiles are in an unmapped part. Assuming connected grid
            # and all tiles are mapped, this case implies the robot is far away.
            # We just don't add infinity to the sum if a single robot can't reach.
            # The total heuristic will be infinity only if |U| > 0 and no robot
            # can reach any tile in U (all min_dist_for_robot are inf).
            # This happens if the BFS in __init__ didn't map all tiles, or if
            # the grid is truly disconnected in a way that separates robots from goals.


        h += movement_cost

        return h
