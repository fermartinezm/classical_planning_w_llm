# Need to import necessary modules
from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

# Helper function to match a fact against a pattern
def match(fact, *args):
    """Checks if a fact string matches a pattern of parts."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for Breadth-First Search
def bfs(start_tile, target_tiles_set, adj_list):
    """
    Performs BFS to find the shortest distance from start_tile to any tile
    in target_tiles_set using the given adjacency list.
    Returns float('inf') if no target tile is reachable.
    """
    if start_tile in target_tiles_set:
        return 0

    queue = deque([(start_tile, 0)])
    visited = {start_tile}

    while queue:
        curr_tile, dist = queue.popleft()

        # Check neighbors
        for next_tile in adj_list.get(curr_tile, []):
            if next_tile in target_tiles_set:
                return dist + 1
            if next_tile not in visited:
                visited.add(next_tile)
                queue.append((next_tile, dist + 1))

    # Target not reachable from start_tile
    return float('inf')

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    three components: the number of tiles that still need to be painted
    correctly, the cost to acquire the necessary colors, and the minimum
    movement cost for a robot to get into a position to paint each
    unpainted goal tile. It is designed for greedy best-first search and
    is not admissible.

    Assumptions:
    - Tiles form a connected grid structure defined by the up, down, left,
      and right static predicates.
    - Goal tiles are initially clear if they are not already painted
      correctly. No tiles are initially painted with the wrong color
      required by the goal.
    - Robots always hold a color initially (the 'free-color' predicate
      is not used in actions and assumed not relevant for initial state).

    Heuristic Initialization:
    In the constructor (__init__), the heuristic precomputes static
    information from the task definition:
    - `self.goal_painted_tiles`: A set of (tile, color) tuples representing
      the desired painted state of goal tiles.
    - `self.available_colors`: A set of available colors in the domain.
    - `self.adj`: An adjacency list representing the tile movement graph.
      An edge exists from tile X to tile Y if a robot can move from X to Y
      using a move action (based on up/down/left/right static facts).
    - `self.paintable_from`: A dictionary mapping each tile T to a set of
      tiles X from which a robot can paint T. This is derived from the
      up/down/left/right static facts (e.g., if (up T X) is true, a robot
      at X can paint T using paint_up).

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic value for a given state is computed as follows:
    1. Identify the set of unsatisfied goal tiles `U`. These are the
       (tile, color) pairs from `self.goal_painted_tiles` that are not
       present as `(painted tile color)` facts in the current state.
    2. If `U` is empty, the state is a goal state, and the heuristic is 0.
    3. Initialize the total heuristic value `h` to 0.
    4. Add the number of unsatisfied goal tiles to `h`. This represents
       a lower bound on the number of paint actions required. `h += len(U)`.
    5. Identify the set of colors `NeededColors` that are required by the
       unsatisfied goal tiles in `U`.
    6. Identify the set of colors `RobotColors` currently held by the robots
       in the current state.
    7. For each color `C` in `NeededColors`, if `C` is not present in
       `RobotColors`, add 1 to `h`. This estimates the cost of one robot
       changing color to acquire a needed color that no robot currently has.
    8. For each unsatisfied goal tile `(T, C)` in `U`:
        a. Determine the set of tiles `PaintableFrom(T)` from which a robot
           can paint tile `T`. This set was precomputed in `__init__`.
        b. Find the minimum number of moves required for *any* robot to reach *any* tile in `PaintableFrom(T)`. This is done by running BFS from each robot's current location to the set `PaintableFrom(T)` and taking the minimum distance.
        c. Add this minimum movement cost to `h`.
    9. Return the final heuristic value `h`.
    """
    def __init__(self, task):
        # Store goal facts as (tile, color) pairs
        self.goal_painted_tiles = set()
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "painted":
                self.goal_painted_tiles.add((parts[1], parts[2]))

        # Store available colors
        self.available_colors = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "available-color":
                self.available_colors.add(parts[1])

        # Build adjacency list for movement graph (from -> to)
        self.adj = defaultdict(list)
        # Build paintable_from mapping (tile_to_paint -> set of tiles robot can be at)
        self.paintable_from = defaultdict(set)

        # Collect all tiles mentioned in static facts to ensure they are in adj/paintable_from dicts
        all_tiles = set()

        for fact in task.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Static fact is (direction tile_y tile_x)
                # Robot moves from tile_x to tile_y
                # Robot at tile_x can paint tile_y
                dir, tile_y, tile_x = parts
                self.adj[tile_x].append(tile_y)
                self.paintable_from[tile_y].add(tile_x)
                all_tiles.add(tile_x)
                all_tiles.add(tile_y)
            elif len(parts) == 2 and parts[0] in ["clear", "painted"]:
                 all_tiles.add(parts[1])
            # Add tiles from goal if not already included (though static should cover connectivity)
            for tile, color in self.goal_painted_tiles:
                 all_tiles.add(tile)


        # Ensure all relevant tiles are keys in adj and paintable_from, even if they have no neighbors/painters
        # This prevents errors in BFS or paintable_from lookup
        for tile in all_tiles:
             self.adj.setdefault(tile, [])
             self.paintable_from.setdefault(tile, set())


    def __call__(self, node):
        state = node.state

        # Get current robot locations and colors
        current_robot_locations = {}
        current_robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                current_robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has":
                current_robot_colors[parts[1]] = parts[2]

        # Get currently painted tiles
        current_painted_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                current_painted_tiles.add((parts[1], parts[2]))

        # Identify unsatisfied goal tiles
        unsatisfied_goals = self.goal_painted_tiles - current_painted_tiles

        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        h = 0

        # Component 1: Number of paint actions needed
        h += len(unsatisfied_goals)

        # Component 2: Color changes needed
        needed_colors = {color for tile, color in unsatisfied_goals}
        robot_colors = set(current_robot_colors.values())
        for color in needed_colors:
            if color not in robot_colors:
                h += 1 # Assume one robot needs to change color

        # Component 3: Movement cost for robots to get into painting position
        for tile_to_paint, color_needed in unsatisfied_goals:
            # Find tiles from which this tile can be painted
            paintable_from_tiles = self.paintable_from.get(tile_to_paint, set())

            if not paintable_from_tiles:
                 # This tile cannot be painted from anywhere based on static facts.
                 # This indicates an unsolvable problem or an issue with the domain/instance.
                 # Assign a very high heuristic value.
                 return float('inf')

            min_movement_cost_for_tile = float('inf')

            # Find the minimum distance from any robot to any paintable_from tile
            for robot, robot_location in current_robot_locations.items():
                 dist = bfs(robot_location, paintable_from_tiles, self.adj)
                 min_movement_cost_for_tile = min(min_movement_cost_for_tile, dist)

            # Add the minimum movement cost for this tile to the total heuristic
            if min_movement_cost_for_tile != float('inf'):
                h += min_movement_cost_for_tile
            else:
                 # Target paintable tile is unreachable from all robots
                 # Problem is likely unsolvable from this state
                 return float('inf')


        return h
