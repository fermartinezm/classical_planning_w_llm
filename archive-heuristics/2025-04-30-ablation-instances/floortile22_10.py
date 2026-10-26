from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available here

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_2 black)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_tile, target_tiles, clear_tiles_set, move_adj_map):
    """
    Performs BFS to find the shortest path from start_tile to any tile in target_tiles,
    only traversing through tiles present in clear_tiles_set.
    Returns the minimum distance or float('inf') if no path exists.
    """
    q = deque([(start_tile, 0)])
    visited = {start_tile}

    while q:
        current_tile, dist = q.popleft()

        if current_tile in target_tiles:
            return dist

        # Get neighbors from the movement adjacency map
        neighbors = move_adj_map.get(current_tile, [])

        for neighbor in neighbors:
            # Check if the neighbor is clear and not visited
            # The clear_tiles_set contains facts like '(clear tile_X_Y)'
            if f'(clear {neighbor})' in clear_tiles_set and neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, dist + 1))

    # If loop finishes, no path found to any target tile
    return float('inf')


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by focusing on the
    unpainted goal tiles. It calculates the minimum cost required to paint the
    "easiest" unpainted goal tile, considering the robot's current location,
    color, and the grid layout. The cost for a single tile includes changing
    color (if needed), moving to a clear tile adjacent to the target tile
    (specifically, above or below it, as paint actions are vertical), and the
    paint action itself.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' static facts.
    - Paint actions ('paint_up', 'paint_down') require the robot to be
      immediately below ('paint_up') or above ('paint_down') the target tile.
    - Movement actions require the destination tile to be 'clear'.
    - The heuristic assumes solvable problems do not require clearing tiles
      that are wrongly painted in the initial state (as there's no explicit clear action
      for painted tiles in the domain). If a goal tile is wrongly painted,
      the heuristic returns infinity.
    - The heuristic calculates movement cost using BFS on the grid, only
      traversing through 'clear' tiles.

    # Heuristic Initialization
    - Parses the goal conditions to identify which tiles need to be painted
      and with which color.
    - Parses static facts ('up', 'down', 'left', 'right') to build a movement
      adjacency map for the grid.
    - Parses static facts ('up', 'down') to build a map identifying which tiles
      can be painted from which adjacent tiles (the 'paintable_from' map).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles that are currently not painted with their target color.
       While doing this, check if any goal tile is painted with the *wrong* color.
       If so, the state is likely unsolvable in this domain, return infinity.
    2. If there are no unpainted goal tiles, the state is a goal state, return 0.
    3. Get the robot's current location and the color it is holding from the state.
    4. Get the set of all 'clear' tiles from the state. These are the traversable nodes for movement.
    5. Initialize the minimum estimated cost to paint any single unpainted goal tile to infinity.
    6. For each unpainted goal tile (target_tile, target_color):
        a. Calculate the 'color cost': 1 if the robot's current color is different from target_color, otherwise 0.
        b. Identify the set of tiles from which the robot can paint the target_tile (i.e., the tile immediately above or below it, based on static 'up'/'down' facts). This is looked up in the precomputed `paintable_from_map`.
        c. Filter this set to find only those paintable-from tiles that are currently 'clear' in the state. These are the potential destinations for the robot before painting.
        d. If there are no clear paintable-from tiles, this target tile is currently unreachable for painting via standard moves. Its cost is effectively infinite in this state.
        e. If there are clear paintable-from tiles, perform a Breadth-First Search (BFS) on the grid graph (using the movement adjacency map and only traversing clear tiles) to find the shortest distance from the robot's current location to *any* tile in the set of clear paintable-from tiles. This is the 'movement cost'.
        f. If the BFS finds a path (movement cost is finite), calculate the total estimated cost to paint this specific target tile: color_cost + movement_cost + 1 (for the paint action itself).
        g. Update the minimum estimated cost found so far with the cost calculated for this target tile.
    7. Return the overall minimum estimated cost among all unpainted goal tiles. If no unpainted goal tile is reachable/paintable, this will be infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations and colors for each tile.
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                _, tile, color = parts
                self.goal_painted_tiles[tile] = color

        # Build adjacency maps from static facts.
        self.move_adj_map = {}
        self.paintable_from_map = {}
        all_tiles = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3:
                pred, tile1, tile2 = parts
                if pred in ["up", "down", "left", "right"]:
                    all_tiles.add(tile1)
                    all_tiles.add(tile2)

                    # Movement adjacency is symmetric
                    self.move_adj_map.setdefault(tile1, []).append(tile2)
                    self.move_adj_map.setdefault(tile2, []).append(tile1)

                # Paintable_from is based on up/down relationship
                # (up y x) means robot at x paints y (above x)
                if pred == "up":
                    target_tile = tile1 # y
                    robot_pos_to_paint = tile2 # x
                    self.paintable_from_map.setdefault(target_tile, []).append(robot_pos_to_paint)
                # (down y x) means robot at x paints y (below x)
                elif pred == "down":
                    target_tile = tile1 # y
                    robot_pos_to_paint = tile2 # x
                    self.paintable_from_map.setdefault(target_tile, []).append(robot_pos_to_paint)

        # Ensure all tiles found are keys in maps, even if they have no adj/paintable_from
        for tile in all_tiles:
             self.move_adj_map.setdefault(tile, [])
             self.paintable_from_map.setdefault(tile, [])

        # Remove duplicates from adjacency lists
        for tile in self.move_adj_map:
            self.move_adj_map[tile] = list(set(self.move_adj_map[tile]))
        for tile in self.paintable_from_map:
            self.paintable_from_map[tile] = list(set(self.paintable_from_map[tile]))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unpainted goal tiles and check for wrongly painted ones
        unpainted_goal_tiles = {}
        for tile, color in self.goal_painted_tiles.items():
            current_painted_fact = None
            is_clear = False
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'painted' and parts[1] == tile:
                    current_painted_fact = fact
                    break
                if parts[0] == 'clear' and parts[1] == tile:
                    is_clear = True

            if current_painted_fact == f'(painted {tile} {color})':
                # Tile is already painted correctly, goal satisfied for this tile
                pass
            elif current_painted_fact is not None:
                 # Tile is painted with the wrong color
                 # Assuming this makes the problem unsolvable in this domain
                 return float('inf')
            elif is_clear:
                 # Tile is clear and needs painting
                 unpainted_goal_tiles[tile] = color
            # else: Tile is neither clear nor painted (shouldn't happen in valid states?)
            # or it's painted with the correct color (handled above).

        # 2. If no unpainted goal tiles, return 0
        if not unpainted_goal_tiles:
            return 0

        # 3. Get robot location and color
        robot_loc = None
        robot_color = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot_loc = parts[2] # (robot-at robot1 tile_X_Y)
            elif parts[0] == 'robot-has':
                robot_color = parts[2] # (robot-has robot1 color)
        # Assuming only one robot and it always has a color based on domain

        # 4. Get set of clear tiles (as strings like '(clear tile_X_Y)')
        clear_tiles_set = {fact for fact in state if match(fact, "clear", "*")}


        # 5. Initialize minimum estimated cost
        min_cost_overall = float('inf')

        # 6. For each unpainted goal tile
        for target_tile, target_color in unpainted_goal_tiles.items():
            # a. Calculate color cost
            color_cost = 1 if robot_color != target_color else 0

            # b. Identify paintable-from tiles
            # These are tiles x such that robot at x can paint target_tile
            potential_paintable_from_tiles = self.paintable_from_map.get(target_tile, [])

            # c. Filter for clear paintable-from tiles
            clear_paintable_from_tiles = [
                t for t in potential_paintable_from_tiles
                if f'(clear {t})' in clear_tiles_set
            ]

            # d. Calculate movement cost using BFS
            min_dist_to_paintable = float('inf')
            if clear_paintable_from_tiles:
                 # The BFS needs to start from the robot's current location.
                 # The BFS can only traverse tiles that are currently clear.
                 # The BFS target is any tile in clear_paintable_from_tiles.
                 min_dist_to_paintable = bfs(robot_loc, clear_paintable_from_tiles, clear_tiles_set, self.move_adj_map)


            # e. Calculate total cost for this tile and update minimum
            if min_dist_to_paintable != float('inf'):
                # Cost is color change (if needed) + moves to paintable spot + paint action
                cost_for_tile = color_cost + min_dist_to_paintable + 1
                min_cost_overall = min(min_cost_overall, cost_for_tile)
            # If min_dist_to_paintable is inf, this tile is unreachable/unpaintable now,
            # its high cost won't lower min_cost_overall unless all are unreachable.


        # 7. Return the overall minimum estimated cost
        # If min_cost_overall is still infinity, it means no unpainted goal tile
        # is currently reachable/paintable. This state might be a dead end.
        # Returning infinity guides the search away from such states.
        return min_cost_overall
