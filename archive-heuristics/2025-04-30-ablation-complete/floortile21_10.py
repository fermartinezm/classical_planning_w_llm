from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity

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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It considers the number of tiles that need painting,
    the number of colors that need to be acquired by robots, and the estimated
    movement cost for robots to reach the vicinity of the tiles.

    # Assumptions
    - Tiles are arranged in a grid structure, and tile names follow the format 'tile_row_col'.
    - Movement is restricted to adjacent clear tiles (up, down, left, right).
    - Painting a tile requires a robot to be at an adjacent tile and the target tile to be clear.
    - If a goal tile is painted with the wrong color or is not clear (and not painted correctly),
      it is considered a state with a very high heuristic cost, pushing the search away.

    # Heuristic Initialization
    - Parses static facts to build a grid representation:
        - Maps tile names ('tile_row_col') to (row, col) coordinates.
        - Maps tile names to their adjacent tile names.
    - Extracts the goal color for each goal tile.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles and their required colors from the task definition.
    2. Identify the current state of all tiles: whether they are clear, painted (and with which color).
    3. Identify the current location and held color for each robot.
    4. Filter the goal tiles to find those that are not yet painted correctly.
    5. For each unpainted goal tile:
        - If it is painted with the wrong color, return a large heuristic value (dead end).
        - If it is not clear (and not painted correctly), return a large heuristic value (blocked, high cost state).
        - Otherwise (the tile is clear and needs painting):
            - Add 1 to the heuristic for the paint action required for this tile.
            - Calculate the minimum Manhattan distance from any robot's current location to any tile adjacent to this goal tile. Add this minimum distance to the heuristic (representing movement cost).
    6. Identify the set of distinct colors required by the unpainted goal tiles.
    7. Count how many of these required colors are not currently held by any robot. Add this count to the heuristic (representing the minimum number of change_color actions needed).
    8. The total heuristic value is the sum of the paint costs, movement costs, and color change costs calculated in the previous steps.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid information and goal states.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Build grid structure: tile name -> (row, col) and tile name -> set of adjacent tile names
        self.tile_coords = {}
        self.tile_adj = {}
        tile_relations = {} # Store relation type for coordinate deduction

        all_tiles = set()
        # Collect all tile names from static and goal facts
        for fact in static_facts | self.goals:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith("tile_"):
                     all_tiles.add(part)

        # Build tile_relations and initial tile_adj
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                rel_type, t1, t2 = parts # t1 is related to t2 by rel_type (e.g., t1 is up from t2)
                if t2 not in tile_relations: # Store relation from t2 to t1
                    tile_relations[t2] = {}
                tile_relations[t2][t1] = rel_type # t1 is rel_type from t2

                # Build symmetric adjacency for BFS
                if t1 not in self.tile_adj:
                    self.tile_adj[t1] = set()
                if t2 not in self.tile_adj:
                    self.tile_adj[t2] = set()
                self.tile_adj[t1].add(t2)
                self.tile_adj[t2].add(t1)


        # Deduce coordinates using BFS
        if all_tiles:
            start_tile = next(iter(all_tiles)) # Pick an arbitrary start tile
            self.tile_coords[start_tile] = (0, 0)
            queue = [(start_tile, (0, 0))]
            visited = {start_tile}

            while queue:
                current_tile, (r, c) = queue.pop(0)

                # Explore neighbors and deduce coordinates
                if current_tile in tile_relations:
                    for neighbor, rel_type in tile_relations[current_tile].items():
                        if neighbor not in visited:
                            visited.add(neighbor)
                            nr, nc = r, c
                            # current_tile is the reference point (r, c)
                            # neighbor is rel_type from current_tile
                            # Assuming row 0 is bottom (row index increases upwards)
                            if rel_type == 'up': # neighbor is up from current
                                nr = r + 1
                            elif rel_type == 'down': # neighbor is down from current
                                nr = r - 1
                            elif rel_type == 'left': # neighbor is left from current
                                nc = c - 1
                            elif rel_type == 'right': # neighbor is right from current
                                nc = c + 1
                            self.tile_coords[neighbor] = (nr, nc)
                            queue.append((neighbor, (nr, nc)))

    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculate Manhattan distance between two tiles."""
        if tile1_name not in self.tile_coords or tile2_name not in self.tile_coords:
            # This can happen if the grid is disconnected and a tile wasn't reached by BFS
            # or if a tile name in state/goal wasn't in static facts defining relations.
            # In a well-formed problem, all tiles should be connected.
            # Treat as infinite distance if coordinates are unknown.
            return math.inf
        r1, c1 = self.tile_coords[tile1_name]
        r2, c2 = self.tile_coords[tile2_name]
        return abs(r1 - r2) + abs(c1 - c2)

    def get_adjacent_tiles(self, tile_name):
        """Get the set of tiles adjacent to the given tile."""
        return self.tile_adj.get(tile_name, set())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "robot-at", "*", "*"):
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif match(fact, "robot-has", "*", "*"):
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # Identify clear and painted tiles
        clear_tiles = set()
        painted_tiles = {} # tile -> color
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "clear", "*"):
                clear_tiles.add(parts[1])
            elif match(fact, "painted", "*", "*"):
                painted_tiles[parts[1]] = parts[2]

        # Identify unpainted goal tiles that are currently clear
        unpainted_clear_goals = {} # tile -> color
        for tile, goal_color in self.goal_tiles.items():
            if tile in painted_tiles:
                # If painted with wrong color, it's a state we want to avoid
                if painted_tiles[tile] != goal_color:
                    return 1000000 # Large value for states with wrong paint
                # If painted with correct color, goal is satisfied for this tile
            else:
                # Tile is not painted. Check if it's clear.
                if tile not in clear_tiles:
                     # If not clear and not painted correctly, it's blocked.
                     # Cannot paint until it's clear. High cost state.
                     return 1000000 # Large value for blocked states

                # Tile is clear and needs painting
                unpainted_clear_goals[tile] = goal_color

        # If all goal tiles are painted correctly, heuristic is 0
        if not unpainted_clear_goals:
            return 0

        h = 0

        # Cost for painting each tile (minimum 1 action per tile)
        h += len(unpainted_clear_goals)

        # Cost for acquiring necessary colors
        needed_colors = set(unpainted_clear_goals.values())
        colors_held_by_robots = set(robot_colors.values())
        colors_to_acquire = needed_colors - colors_held_by_robots
        h += len(colors_to_acquire) # Minimum change_color actions needed

        # Cost for movement
        move_cost = 0
        for tile, color in unpainted_clear_goals.items():
            min_dist_to_adj = math.inf
            adj_tiles = self.get_adjacent_tiles(tile)

            if not adj_tiles:
                 # Tile has no adjacent tiles, cannot be painted. This implies unsolvability.
                 return 1000000

            for robot, robot_loc in robot_locations.items():
                for adj_tile in adj_tiles:
                    dist = self.manhattan_distance(robot_loc, adj_tile)
                    min_dist_to_adj = min(min_dist_to_adj, dist)

            # If min_dist_to_adj is still infinity, it means no robot can reach
            # any adjacent tile (e.g., disconnected grid). Unsolvable.
            if min_dist_to_adj == math.inf:
                 return 1000000

            # Add the minimum distance for *some* robot to reach *an* adjacent tile
            # This overestimates if one robot serves multiple tiles, but is greedy.
            move_cost += min_dist_to_adj

        h += move_cost

        return h
