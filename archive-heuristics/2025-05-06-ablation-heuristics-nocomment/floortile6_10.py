from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

def get_parts(fact):
    """Helper function to parse PDDL fact strings."""
    # Removes leading/trailing parens and splits by space
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        estimated costs for each individual unpainted goal tile. For each
        unpainted goal tile, the estimated cost includes:
        1. A cost of 1 for the paint action itself.
        2. An estimated cost for a robot to acquire the correct color (1 if the
           robot needs to change color, 0 otherwise).
        3. An estimated movement cost for the robot to reach a tile adjacent
           to the goal tile. This is estimated using the Manhattan distance
           from the robot's current location to the closest adjacent tile
           of the goal tile.
        The heuristic considers the minimum such cost over all available robots
        for each unpainted goal tile and sums these minimums.

    Assumptions:
        - Solvable instances represent a grid structure where tiles are named
          as 'tile_row_col' allowing coordinate parsing.
        - The grid defined by 'up', 'down', 'left', 'right' predicates is
          connected and goal tiles have at least one adjacent tile.
        - Robots always have a color assigned in the initial state.
        - Unpainted goal tiles are assumed to be 'clear' in solvable states.
        - Tiles painted with a color different from the required goal color
          are not goal tiles in solvable instances (as there's no action to
          unpaint or repaint).
        - Manhattan distance is used as a proxy for movement cost, ignoring
          potential obstacles (non-clear tiles other than the target).

    Heuristic Initialization:
        - Parses the goal facts to store a mapping from goal tile names to
          their required colors (`self.goal_tiles`).
        - Parses static adjacency facts ('up', 'down', 'left', 'right') to
          build an adjacency list (`self.adj`) representing the grid graph.
        - Parses tile names ('tile_row_col') encountered in static facts to
          store their (row, col) coordinates (`self.tile_coords`) for
          Manhattan distance calculations.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the heuristic value `h` to 0.
        2. Extract the current locations and colors of all robots from the
           current state.
        3. Identify the set of goal tiles that are not currently painted with
           their required color based on the state and the precomputed goal
           tiles.
        4. If there are no unpainted goal tiles, the state is a goal state,
           so return 0.
        5. For each identified unpainted goal tile `tile` requiring `required_color`:
           a. Add 1 to `h`. This accounts for the final paint action needed for this tile.
           b. Calculate the minimum cost for *any* robot to get into a position
              to paint this specific tile. This involves iterating through all
              robots:
              i. Determine the color change cost for the current robot: 1 if the
                 robot's current color is different from `required_color`, otherwise 0.
              ii. Determine the movement cost for the current robot: Calculate the
                  minimum Manhattan distance from the robot's current location
                  to any tile adjacent to the goal tile `tile`.
              iii. The total cost for this robot for this tile is the sum of the
                   color change cost and the movement cost.
              iv. Keep track of the minimum total cost found across all robots
                  for this specific goal tile.
           c. If, after checking all robots, the minimum cost for this tile is
              still infinite (which implies no robot can reach an adjacent tile,
              suggesting the tile is unreachable), return a large integer value
              (e.g., 1,000,000) to indicate a likely unsolvable state.
           d. Add the calculated minimum robot cost for this tile to `h`.
        6. Return the final calculated value of `h`.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.goal_tiles = {} # {tile: color}
        self.adj = {} # {tile: [adj_tiles]}
        self.tile_coords = {} # {tile: (row, col)}

        # Parse goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                # Goal is (painted tile color)
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Parse static facts to build grid graph and get coordinates
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ['up', 'down', 'left', 'right']:
                # Fact is (pred tile1 tile2) meaning tile1 is pred of tile2
                # Add bidirectional edge between tile1 and tile2
                tile1, tile2 = parts[1], parts[2]
                self.adj.setdefault(tile1, []).append(tile2)
                self.adj.setdefault(tile2, []).append(tile1)

                # Parse and store coordinates if not already seen
                if tile1 not in self.tile_coords:
                    self.tile_coords[tile1] = self._parse_tile_name(tile1)
                if tile2 not in self.tile_coords:
                    self.tile_coords[tile2] = self._parse_tile_name(tile2)

    def _parse_tile_name(self, tile_name):
        """Parses 'tile_row_col' string into (row, col) tuple."""
        try:
            parts = tile_name.split('_')
            # Assuming tile names are always in the format tile_row_col
            return int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            # This should not happen in valid problem instances
            print(f"Error: Could not parse tile name {tile_name}.")
            # Return a default or raise error, depending on desired behavior for malformed input
            # Returning (0,0) might allow heuristic to continue but could be misleading
            # Raising an error might be better for debugging malformed instances
            raise ValueError(f"Invalid tile name format: {tile_name}")


    def _manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles using stored coordinates."""
        # Ensure coordinates are available (should be from __init__)
        if tile1_name not in self.tile_coords or tile2_name not in self.tile_coords:
             # This indicates a problem with parsing in __init__ or an unexpected tile name
             print(f"Error: Coordinates not found for {tile1_name} or {tile2_name}.")
             # Return a large distance to penalize paths involving unknown tiles
             return 1_000_000 # Use a large number instead of inf

        c1 = self.tile_coords[tile1_name]
        c2 = self.tile_coords[tile2_name]
        return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

    def _get_adjacent_tiles(self, tile_name):
        """Returns a list of tiles adjacent to the given tile."""
        return self.adj.get(tile_name, [])

    def __call__(self, node):
        state = node.state
        h = 0

        # Extract dynamic state information
        robot_loc = {} # {robot: tile}
        robot_color = {} # {robot: color}
        painted_facts = set() # Store painted facts for quick lookup

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_loc[robot] = tile
            elif pred == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
            elif pred == 'painted':
                painted_facts.add(fact)

        # Identify unpainted goal tiles
        unpainted_goal_tiles = [] # [(tile, required_color), ...]
        for tile, required_color in self.goal_tiles.items():
            goal_fact = f'(painted {tile} {required_color})'
            if goal_fact not in painted_facts:
                unpainted_goal_tiles.append((tile, required_color))

        # If all goal tiles are painted correctly, heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        # Calculate heuristic for each unpainted goal tile
        for tile, required_color in unpainted_goal_tiles:
            # Cost for the paint action itself
            h += 1

            # Find the minimum cost for any robot to get ready to paint this tile
            min_robot_cost_for_tile = float('inf')

            # Check if there are any robots
            if not robot_loc:
                 # No robots, likely unsolvable
                 return 1_000_000 # Return a large number

            # Get adjacent tiles for the goal tile
            adjacent_tiles = self._get_adjacent_tiles(tile)
            if not adjacent_tiles:
                 # Goal tile is isolated or not properly connected, likely unsolvable
                 return 1_000_000 # Return a large number

            for robot, r_loc in robot_loc.items():
                current_color = robot_color.get(robot) # Get robot's current color

                # Cost to change color if needed
                # Assumes robot-has is always true for some color in solvable states
                color_change_cost = 1 if current_color != required_color else 0

                # Minimum movement cost from robot's location to any adjacent tile of the goal tile
                min_dist_r_to_adj_of_tile = float('inf')
                for adj_tile in adjacent_tiles:
                    dist = self._manhattan_distance(r_loc, adj_tile)
                    min_dist_r_to_adj_of_tile = min(min_dist_r_to_adj_of_tile, dist)

                # Total cost for this robot to paint this tile (movement + color change)
                robot_cost_for_tile = color_change_cost + min_dist_r_to_adj_of_tile

                # Update minimum cost over all robots for this tile
                min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost_for_tile)

            # If min_robot_cost_for_tile is still inf, it means no robot could reach an adjacent tile
            if min_robot_cost_for_tile == float('inf'):
                 # This goal tile is unreachable by any robot. State is likely unsolvable.
                 return 1_000_000 # Return a large number

            # Add the minimum cost to get a robot ready for this tile to the total heuristic
            h += min_robot_cost_for_tile

        return h
