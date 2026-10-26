from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in a module named heuristics
# from heuristics.heuristic_base import Heuristic

# Mock Heuristic base class if not running in the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError

        def __repr__(self):
            return "<MockHeuristic>"


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format - maybe return empty list or raise error
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Requires the number of parts in the fact to exactly match the number of args.

    - `fact`: The complete fact as a string, e.g., "(painted tile1 white)".
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
    This heuristic estimates the minimum number of actions required to paint
    all goal tiles with their required colors. It sums the estimated cost
    for each goal tile that is not yet painted correctly. The estimated cost
    for a single tile includes:
    1. Moving the robot to a tile adjacent (above or below) to the target tile.
    2. Changing the robot's color if it doesn't have the required color.
    3. Painting the tile.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - Robots can only paint tiles directly above or below their current position
      using 'paint_up' and 'paint_down' actions.
    - The cost of each action (move, change_color, paint) is 1.
    - The heuristic uses shortest path distance on the grid graph (computed via BFS)
      as a lower bound for movement cost.
    - The heuristic ignores negative interactions like a tile becoming not 'clear'
      when painted or moved onto, except for the requirement that the paint target
      must be clear (which is usually true if it's not painted).
    - The goal consists only of (painted tile color) facts.
    - There is at least one robot in the problem instance, and its state (position, color)
      is consistently represented in the state facts.
    - All tiles mentioned in the problem (initial state, goals, static) are part of a
      single connected grid structure defined by 'up', 'down', 'left', 'right' facts.

    # Heuristic Initialization
    - Extract goal painting requirements: a map from tile to the required color.
    - Parse static facts ('up', 'down', 'left', 'right') to build a graph
      representation of the grid and identify potential paint positions (above/below).
    - Identify all unique tiles mentioned in the problem instance (static, initial state, goals).
    - Precompute shortest path distances between all pairs of identified tiles
      on the grid using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location and the color it is holding. If robot state is incomplete or refers to unknown tiles, return infinity.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each goal tile and its required color:
       - Check if the tile is already painted with the correct color in the current state.
       - If the tile is *not* painted correctly:
         - Determine the tiles directly above and below the target tile (potential painting positions).
         - If no paint positions are found for this goal tile (based on 'up_of'/'down_of' and known tiles), this goal tile might be unreachable or unpaintable by this method. Assign infinity for this tile's cost.
         - Calculate the minimum distance from the robot's current location to any of these potential painting positions using the precomputed distances. This is the estimated movement cost. If any paint position is unreachable from the robot's current position, assign infinity for this tile's cost.
         - Add this minimum distance to the total cost.
         - Check if the robot currently holds the required color. If not, add 1 to the total cost (for the 'change_color' action).
         - Add 1 to the total cost (for the 'paint_up' or 'paint_down' action).
    4. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the grid graph for distance calculations.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_paintings = {}
        for goal in self.goals:
            # Goal format: (painted tile_name color_name)
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Build the grid graph and identify tile relationships
        self.adj = {} # Adjacency list for BFS
        self.up_of = {} # Maps tile_below -> tile_above
        self.down_of = {} # Maps tile_above -> tile_below
        self.all_tiles = set()

        # Collect all potential tile objects from static facts, initial state, and goals
        # This is a heuristic way to find tiles without the full PDDL object list
        # Look for objects used in predicates that typically involve tiles
        tile_predicates = {"up", "down", "left", "right", "robot-at", "clear", "painted"}

        for fact_set in [self.static, self.goals, task.initial_state]: # Include initial_state
             for fact in fact_set:
                 parts = get_parts(fact)
                 if parts and parts[0] in tile_predicates:
                     if parts[0] in ["up", "down", "left", "right"] and len(parts) == 3:
                         self.all_tiles.add(parts[1])
                         self.all_tiles.add(parts[2])
                     elif parts[0] == "robot-at" and len(parts) == 3:
                         self.all_tiles.add(parts[2]) # The tile object
                     elif parts[0] == "clear" and len(parts) == 2:
                         self.all_tiles.add(parts[1]) # The tile object
                     elif parts[0] == "painted" and len(parts) == 3:
                         self.all_tiles.add(parts[1]) # The tile object

        # Build adjacency and directional maps using only identified tiles
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3:
                pred, tile1, tile2 = parts
                if pred in ["up", "down", "left", "right"]:
                    # Ensure both are identified as tiles before adding to adj/directional maps
                    if tile1 in self.all_tiles and tile2 in self.all_tiles:
                        self.adj.setdefault(tile1, []).append(tile2)
                        self.adj.setdefault(tile2, []).append(tile1)

                        # Store directional relationships for painting positions
                        if pred == "up": # (up ?y ?x) means y is up from x
                            self.up_of[tile2] = tile1 # tile1 is above tile2
                            self.down_of[tile1] = tile2 # tile2 is below tile1
                        elif pred == "down": # (down ?y ?x) means y is down from x
                            self.down_of[tile2] = tile1 # tile1 is below tile2
                            self.up_of[tile1] = tile2 # tile2 is above tile1
                    # left/right are not used for paint positions in this domain

        # Remove duplicate neighbors in adj lists
        for tile in self.adj:
             self.adj[tile] = list(set(self.adj[tile]))

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = self._bfs(start_tile)

    def _bfs(self, start_tile):
        """Performs BFS from a start tile to find distances to all other tiles."""
        distances_from_start = {tile: float('inf') for tile in self.all_tiles}
        if start_tile not in self.all_tiles:
             # Should not happen if start_tile comes from self.all_tiles, but defensive check
             return distances_from_start

        distances_from_start[start_tile] = 0
        queue = deque([start_tile])

        while queue:
            current_tile = queue.popleft()
            current_dist = distances_from_start[current_tile]

            # Ensure current_tile has entries in adj (tiles might exist but have no connections if parsing is partial)
            if current_tile in self.adj:
                for neighbor in self.adj[current_tile]:
                    # Ensure neighbor is a known tile
                    if neighbor in self.all_tiles and distances_from_start[neighbor] == float('inf'):
                        distances_from_start[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances_from_start


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot's current position and color
        robot_name = None
        robot_pos = None
        robot_color = None

        # Find the robot name and position first
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot_name, robot_pos = parts[1], parts[2]
                break # Assuming one robot relevant to heuristic

        # Then find the color for that robot
        if robot_name:
             for fact in state:
                 parts = get_parts(fact)
                 if parts and parts[0] == "robot-has" and len(parts) == 3 and parts[1] == robot_name:
                     robot_color = parts[2]
                     break # Found robot color

        # If robot state is incomplete or refers to an unknown tile, goals are likely unreachable
        if robot_pos is None or robot_color is None or robot_pos not in self.all_tiles:
             return float('inf')


        total_estimated_cost = 0

        for tile, required_color in self.goal_paintings.items():
            # Check if the tile is already painted correctly
            is_painted_correctly = False
            # Check if it's painted with the required color
            if f"(painted {tile} {required_color})" in state:
                 is_painted_correctly = True

            if not is_painted_correctly:
                # This tile needs to be painted with required_color

                # Find potential painting positions (tiles above/below)
                paint_positions = []
                if tile in self.up_of and self.up_of[tile] in self.all_tiles:
                    paint_positions.append(self.up_of[tile])
                if tile in self.down_of and self.down_of[tile] in self.all_tiles:
                    paint_positions.append(self.down_of[tile])

                # If no paint positions are found for a goal tile, it's likely unreachable
                if not paint_positions:
                     return float('inf') # Cannot paint this goal tile

                # Calculate minimum distance from robot to a paint position
                min_dist = float('inf')
                # Ensure robot_pos is in the precomputed distances (should be if robot_pos in self.all_tiles)
                if robot_pos in self.distances:
                    for paint_pos in paint_positions:
                        # Ensure paint_pos is in the precomputed distances (should be if paint_pos in self.all_tiles)
                        if paint_pos in self.distances.get(robot_pos, {}): # Use .get for safety
                             min_dist = min(min_dist, self.distances[robot_pos][paint_pos])

                # If no path exists to any paint position, this goal tile is unreachable
                if min_dist == float('inf'):
                     return float('inf') # Cannot reach paint position for this goal tile

                # Add movement cost
                total_estimated_cost += min_dist

                # Add color change cost if needed
                if robot_color != required_color:
                    total_estimated_cost += 1 # Cost of change_color action

                # Add paint action cost
                total_estimated_cost += 1 # Cost of paint_up or paint_down action

        return total_estimated_cost
