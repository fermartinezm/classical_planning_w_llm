import re
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def parse_tile_name(tile_name_str):
    """Parses a tile name like 'tile_R_C' into (R, C) coordinates."""
    match = re.match(r"tile_(\d+)_(\d+)", tile_name_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Return None or raise error if format is unexpected
    return None

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        Estimates the cost to reach the goal by summing the estimated cost
        for each individual goal tile that is not yet painted correctly.
        For each unpainted goal tile, the estimated cost is the minimum cost
        for any robot to reach an adjacent tile, acquire the correct color,
        and perform the paint action. If the goal tile is currently occupied
        by a robot, an additional cost is added for that robot to move off.

    Assumptions:
        - Tile names follow the format 'tile_R_C' where R and C are integers
          representing row and column.
        - The grid defined by adjacency predicates is connected.
        - Incorrectly painted tiles (painted with a color different from the goal)
          indicate an unsolvable state, and the heuristic returns a large value.
        - Manhattan distance is a reasonable estimate for movement cost on the grid,
          ignoring dynamic obstacles (other robots, painted tiles) for path calculation,
          but accounting for a robot occupying the target tile for painting.
        - The cost of changing color is 1, regardless of the current color or
          the target color (as long as the target color is available).
        - The cost of moving a robot off an occupied goal tile is 1.

    Heuristic Initialization:
        - Parses goal facts to identify target tiles and their required colors.
        - Parses static adjacency facts ('up', 'down', 'left', 'right') to
          build a mapping from tile names to (row, column) coordinates and
          an adjacency list for each tile. This allows efficient calculation
          of Manhattan distances and finding adjacent tiles.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value to 0.
        2. Check the current state for any goal tile 't' that is painted with
           a color 'c'' different from its required goal color 'c'. If such a
           tile exists, the state is likely unsolvable; return a large heuristic value.
        3. Identify the current location and color of each robot.
        4. Identify the state of each goal tile: is it painted correctly,
           painted incorrectly (handled in step 2), clear, or occupied by a robot?
        5. Create two lists of unpainted goal tiles:
           - 'unpainted_clear_goals': Tiles that need painting and are currently clear.
           - 'unpainted_occupied_goals': Tiles that need painting and are currently
             occupied by a robot (store the robot name as well).
        6. For each tile 't' in 'unpainted_clear_goals' (needs color 'c'):
           a. Calculate the minimum cost for *any* robot 'r' to paint 't' with 'c'.
              This cost for robot 'r' is:
              - Minimum Manhattan distance from 'r''s current location to any tile
                adjacent to 't'.
              - Plus 1 if 'r' does not currently have color 'c' (cost to change color).
              - Plus 1 for the paint action itself.
           b. Add this minimum cost (over all robots) to the total heuristic.
        7. For each tile 't' in 'unpainted_occupied_goals' (needs color 'c',
           occupied by 'r_on_t'):
           a. Add 1 to the total heuristic. This represents the estimated cost
              for robot 'r_on_t' to move off the tile 't'.
           b. Calculate the minimum cost for *any* robot 'r'' (which could be
              'r_on_t' after moving, or another robot) to paint 't' with 'c',
              *assuming* 't' is now clear. This cost for robot 'r'' is:
              - Minimum Manhattan distance from 'r''s current location to any tile
                adjacent to 't'.
              - Plus 1 if 'r'' does not currently have color 'c'.
              - Plus 1 for the paint action itself.
           c. Add this minimum cost (over all robots) to the total heuristic.
        8. If there are no unpainted goal tiles (either clear or occupied),
           the state is a goal state, and the heuristic is 0. This check is
           implicitly handled if the lists in step 5 are empty.
        9. Return the total heuristic value.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        self.tile_coords = {}
        self.tile_adj = {}

        # Collect all tile names and parse coordinates from adjacency facts
        tile_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Adjacency facts define the grid structure and involve tiles
            if parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                tile_names.add(tile1)
                tile_names.add(tile2)

        for tile_name in tile_names:
            coords = parse_tile_name(tile_name)
            if coords:
                self.tile_coords[tile_name] = coords
                self.tile_adj[tile_name] = [] # Initialize adjacency list

        # Build adjacency list (undirected graph)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                # Add symmetric adjacency if both tiles were parsed
                if tile1 in self.tile_adj and tile2 in self.tile_adj:
                     self.tile_adj[tile1].append(tile2)
                     self.tile_adj[tile2].append(tile1)

        # Remove duplicates from adjacency lists (optional but clean)
        for tile in self.tile_adj:
             self.tile_adj[tile] = list(set(self.tile_adj[tile]))


    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles using their coordinates."""
        if tile1_name not in self.tile_coords or tile2_name not in self.tile_coords:
            # Should not happen with valid instances and correct parsing
            return float('inf')

        r1, c1 = self.tile_coords[tile1_name]
        r2, c2 = self.tile_coords[tile2_name]
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, node):
        state = node.state

        unpainted_clear_goals = []
        unpainted_occupied_goals = []
        robot_loc = {}
        robot_color = {}

        # Identify robot states and current tile states (painted/occupied)
        is_painted = {}
        is_occupied = {} # Maps tile -> robot name
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, loc = parts[1], parts[2]
                robot_loc[robot] = loc
                is_occupied[loc] = robot
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
            elif parts[0] == "painted":
                 tile, color = parts[1], parts[2]
                 is_painted[tile] = color

        # Check for incorrectly painted tiles first
        for tile, goal_color in self.goal_tiles.items():
             if tile in is_painted and is_painted[tile] != goal_color:
                 # Tile is painted with the wrong color - likely unsolvable
                 return 1000000 # Large heuristic value

        # Categorize unpainted goal tiles
        for tile, goal_color in self.goal_tiles.items():
            if tile not in is_painted or is_painted[tile] != goal_color:
                 # Tile is not painted with the correct color
                 if tile in is_occupied:
                     # Tile is occupied by a robot
                     unpainted_occupied_goals.append((tile, goal_color, is_occupied[tile]))
                 else:
                     # Tile is clear (not painted, not occupied implies clear)
                     unpainted_clear_goals.append((tile, goal_color))

        # If all goal tiles are painted correctly, heuristic is 0
        if not unpainted_clear_goals and not unpainted_occupied_goals:
            return 0

        h = 0

        # Cost for unpainted and clear goal tiles
        for tile, goal_color in unpainted_clear_goals:
            min_cost_for_tile = float('inf')
            # Find the minimum cost for any robot to paint this tile
            for robot, r_loc in robot_loc.items():
                r_color = robot_color.get(robot) # Get robot color

                move_cost = float('inf')
                # Find minimum distance to any adjacent tile
                if tile in self.tile_adj:
                    for adj_t in self.tile_adj[tile]:
                        if adj_t in self.tile_coords and r_loc in self.tile_coords:
                             move_cost = min(move_cost, self.manhattan_distance(r_loc, adj_t))

                # Cost to change color if needed
                color_cost = 1 if r_color != goal_color else 0

                # Total cost for this robot to paint this tile
                robot_cost = move_cost + color_cost + 1 # move + color change + paint

                # Update minimum cost over all robots for this tile
                min_cost_for_tile = min(min_cost_for_tile, robot_cost)

            # Add the minimum cost for this tile to the total heuristic
            h += min_cost_for_tile if min_cost_for_tile != float('inf') else 1000000 # Add large penalty if tile is unreachable


        # Cost for unpainted but occupied goal tiles
        for tile, goal_color, r_on_t in unpainted_occupied_goals:
            # Add cost for the robot to move off the tile
            h += 1

            # Now calculate the cost for any robot to paint this tile, assuming it's clear
            min_cost_to_paint_after_move_off = float('inf')
            # Find the minimum cost for any robot to paint this tile *after* it becomes clear
            for robot, r_loc in robot_loc.items(): # Use current robot locations
                r_color = robot_color.get(robot)

                move_cost = float('inf')
                # Find minimum distance to any adjacent tile
                if tile in self.tile_adj:
                    for adj_t in self.tile_adj[tile]:
                         if adj_t in self.tile_coords and r_loc in self.tile_coords:
                            move_cost = min(move_cost, self.manhattan_distance(r_loc, adj_t))

                # Cost to change color if needed
                color_cost = 1 if r_color != goal_color else 0

                # Total cost for this robot to paint this tile (after move-off)
                robot_cost = move_cost + color_cost + 1 # move + color change + paint

                # Update minimum cost over all robots for this tile
                min_cost_to_paint_after_move_off = min(min_cost_to_paint_after_move_off, robot_cost)

            # Add the minimum cost for this tile (after move-off) to the total heuristic
            h += min_cost_to_paint_after_move_off if min_cost_to_paint_after_move_off != float('inf') else 1000000 # Add large penalty if tile is unreachable


        return h
