from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-fact strings
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_2 black)".
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
    This heuristic estimates the number of actions required to paint all
    goal tiles correctly. It sums three components: the number of tiles
    that need painting, the estimated cost for color changes, and the
    estimated movement cost to reach the vicinity of the first tile to paint.

    # Assumptions
    - Tiles are named using a 'tile_R_C' format where R and C are integers
      representing row and column.
    - The grid connectivity (up, down, left, right) is defined in static facts.
    - The robot always possesses exactly one color.
    - Movement cost is estimated using Manhattan distance, ignoring obstacles
      (clear tiles). This makes the heuristic non-admissible but faster to compute.

    # Heuristic Initialization
    - Extracts the target color for each tile specified in the goal conditions.
    - Parses tile names to map them to (row, column) coordinates.
    - Builds an adjacency list for tiles based on 'up', 'down', 'left', 'right'
      static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location (tile) and the color it is holding.
    2. Identify the set of "unsatisfied" goal tiles. An unsatisfied tile is one
       that is specified in the goal as `(painted T C)` but the state does
       not contain this exact fact. This includes tiles that are `clear` or
       painted with the wrong color.
    3. If there are no unsatisfied tiles, the heuristic value is 0 (goal state).
    4. If there are unsatisfied tiles, the heuristic value is calculated as the sum of:
       a. **Paint Cost:** The number of unsatisfied tiles. Each needs one paint action.
       b. **Color Change Cost:** Estimate the cost to acquire the necessary colors.
          - Identify all distinct colors required by the unsatisfied tiles.
          - If the robot's current color is not among the needed colors, add 1
            (cost to change from a useless color).
          - For each needed color that the robot does *not* currently possess,
            add 1 (cost to change to that color later). This is a simplification
            assuming sequential color changes.
       c. **Movement Cost:** Estimate the cost to reach the first tile to paint.
          - Calculate the minimum Manhattan distance from the robot's current
            location to *any* tile that is adjacent to *any* of the unsatisfied
            goal tiles. This estimates the cost to get to the "work area".
    5. Sum the costs from steps 4a, 4b, and 4c.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts
        to build necessary data structures.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Include initial state tiles

        # Extract goal painted tiles: map tile name to goal color
        self.goal_painted_tiles = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3:
                    _, tile, color = parts
                    self.goal_painted_tiles[tile] = color

        # Build tile maps (name <-> coords) and adjacency list
        self.tile_coords = {} # Map tile name to (row, col)
        self.coords_tile = {} # Map (row, col) to tile name
        self.adj = {} # Adjacency list: tile -> list of adjacent tiles

        # Collect all unique tile names from initial, goal, and static facts
        all_tiles = set()
        for fact in self.initial_state | self.goals | self.static:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith("tile_"):
                     all_tiles.add(part)

        # Parse tile names and store coordinates
        for tile in all_tiles:
            try:
                # Expecting format 'tile_R_C'
                parts = tile.split('_')
                if len(parts) == 3 and parts[0] == 'tile':
                    r, c = int(parts[1]), int(parts[2])
                    self.tile_coords[tile] = (r, c)
                    self.coords_tile[(r, c)] = tile
                    self.adj[tile] = [] # Initialize adjacency list
            except ValueError:
                 # Ignore tiles with unparseable names
                 pass

        # Build adjacency list from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                if t1 in self.adj and t2 in self.adj: # Ensure both tiles were parsed
                    self.adj[t1].append(t2)
                    self.adj[t2].append(t1)

        # Remove duplicates from adjacency lists
        for tile in self.adj:
            self.adj[tile] = list(set(self.adj[tile]))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify robot's current location and color
        robot_tile = None
        robot_color = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    # Assuming one robot
                    robot_tile = parts[2]
            elif match(fact, "robot-has", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                    # Assuming one robot
                    robot_color = parts[2]

        # Handle case where robot state is not found (shouldn't happen in valid states)
        if robot_tile is None:
             return float('inf') # Cannot solve if robot location is unknown

        robot_coords = self.tile_coords.get(robot_tile)
        if robot_coords is None:
             return float('inf') # Cannot solve if robot tile coordinates are unknown

        # 2. Identify unsatisfied goal tiles
        unsatisfied_tiles = set()
        for tile, goal_color in self.goal_painted_tiles.items():
            # A tile is unsatisfied if the goal fact "(painted tile goal_color)" is NOT in the state
            if "(painted {} {})".format(tile, goal_color) not in state:
                unsatisfied_tiles.add(tile)

        # 3. If no unsatisfied tiles, goal is reached
        if not unsatisfied_tiles:
            return 0

        # 4. Calculate heuristic components
        h = 0

        # a. Paint Cost: 1 action per unsatisfied tile
        h += len(unsatisfied_tiles)

        # b. Color Change Cost
        needed_colors = {self.goal_painted_tiles[tile] for tile in unsatisfied_tiles}
        color_cost = 0
        if needed_colors:
            if robot_color and robot_color not in needed_colors:
                color_cost += 1 # Cost to change from a useless color
            colors_to_acquire = set(needed_colors)
            if robot_color in colors_to_acquire:
                colors_to_acquire.remove(robot_color)
            color_cost += len(colors_to_acquire) # Cost to acquire other needed colors
        h += color_cost

        # c. Movement Cost: Minimum Manhattan distance to any tile adjacent to any unsatisfied tile
        min_dist_to_any_adjacent_of_any_unsatisfied = float('inf')

        for tile in unsatisfied_tiles:
            adjacent_tiles = self.adj.get(tile, [])
            if not adjacent_tiles:
                 # If a goal tile has no adjacent tiles, it cannot be painted. Problem is unsolvable.
                 return float('inf')

            for adj_tile in adjacent_tiles:
                adj_coords = self.tile_coords.get(adj_tile)
                if adj_coords is not None:
                    # Calculate Manhattan distance from robot to this adjacent tile
                    dist = abs(robot_coords[0] - adj_coords[0]) + abs(robot_coords[1] - adj_coords[1])
                    min_dist_to_any_adjacent_of_any_unsatisfied = min(min_dist_to_any_adjacent_of_any_unsatisfied, dist)

        # Add movement cost if a reachable adjacent tile was found
        if min_dist_to_any_adjacent_of_any_unsatisfied != float('inf'):
             h += min_dist_to_any_adjacent_of_any_unsatisfied
        # If min_dist remains inf, it means no adjacent tile coords were found for any unsatisfied tile,
        # which implies an issue with the grid definition or parsing, leading to an unsolvable state.
        # Returning infinity is appropriate.

        return h
