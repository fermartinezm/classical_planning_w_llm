# Import necessary modules
from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a minimal Heuristic base class if not provided by the environment
# If running in an environment that provides heuristics.heuristic_base.Heuristic,
# you can remove this definition and uncomment the inheritance in floortileHeuristic class.
class Heuristic:
    """
    Minimal base class definition for a heuristic.
    Expected to be initialized with a task object and have a __call__ method
    that takes a node object (which has a state attribute).
    """
    def __init__(self, task):
        self.task = task
        pass # Initialize heuristic-specific data

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        node.state is expected to be the current state representation.
        """
        raise NotImplementedError("Heuristic __call__ method must be implemented by subclasses.")


# Define helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(painted tile_1_1 white)" -> ["painted", "tile_1_1", "white"]
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parse 'tile_X_Y' into (X, Y) integer coordinates."""
    # Example: "tile_1_1" -> (1, 1)
    parts = tile_name.split('_')
    # Assuming format is always tile_row_col
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            # Handle cases where X or Y are not integers
            return None # Indicate parsing failed
    return None # Indicate parsing failed or name format is wrong

def manhattan_distance(loc1, loc2):
    """Calculate Manhattan distance between two (row, col) coordinates."""
    r1, c1 = loc1
    r2, c2 = loc2
    return abs(r1 - r2) + abs(c1 - c2)

def dist_adj(loc_robot, loc_tile):
    """
    Calculate minimum moves for robot at loc_robot to reach a tile adjacent to loc_tile.
    Robot must be AT an adjacent tile to paint.
    loc_robot and loc_tile are (row, col) tuples.
    """
    d = manhattan_distance(loc_robot, loc_tile)
    if d == 0:
        # Robot is at the tile, needs 1 move to get to an adjacent tile
        return 1
    else:
        # Robot is d > 0 away, needs d-1 moves to get 1 unit away (adjacent)
        return d - 1

# Define the heuristic class
class floortileHeuristic(Heuristic): # Inherit from the base class
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost by summing, for each unsatisfied goal tile,
    the minimum cost for any robot to reach an adjacent tile with the
    correct color, plus the cost of the paint action.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and tile coordinates."""
        super().__init__(task) # Call the base class constructor

        # Store goal paintings: {tile_name: color_name}
        self.goal_paintings = {}
        for goal in self.goals:
            # Ensure the goal fact has enough parts before accessing indices
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "painted":
                tile_name, color_name = parts[1], parts[2]
                self.goal_paintings[tile_name] = color_name
            # Ignore other types of goal predicates if any

        # Parse tile names to get coordinates: {tile_name: (row, col)}
        # Collect all tile names mentioned in goals, initial state, and static facts
        all_tile_names = set()
        # Collect from goals
        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tile_names.add(part)
        # Collect from initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tile_names.add(part)
        # Collect from static facts (especially adjacency predicates)
        for fact in task.static:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tile_names.add(part)


        self.tile_coords = {}
        for tile_name in all_tile_names:
             coords = parse_tile_name(tile_name)
             if coords is not None:
                 self.tile_coords[tile_name] = coords
             # else: This tile name doesn't fit the expected pattern. Ignore it.


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Check for goal state
        # This check is crucial for correctness, especially if goals include non-'painted' predicates.
        # If goals <= state is True, all goal conditions are met.
        if self.goals <= state:
            return 0

        # Identify current robot states: {robot_name: {'location': tile_name, 'color': color_name}}
        robot_info = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) >= 3: # Ensure enough parts
                if parts[0] == "robot-at":
                    robot_name, tile_name = parts[1], parts[2]
                    if robot_name not in robot_info:
                        robot_info[robot_name] = {}
                    robot_info[robot_name]['location'] = tile_name
                elif parts[0] == "robot-has":
                     robot_name, color_name = parts[1], parts[2]
                     if robot_name not in robot_info:
                         robot_info[robot_name] = {}
                     robot_info[robot_name]['color'] = color_name

        # Identify currently painted tiles: {tile_name: color_name}
        current_painted = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) >= 3 and parts[0] == "painted":
                tile_name, color_name = parts[1], parts[2]
                current_painted[tile_name] = color_name

        # Identify needed paintings and check for dead ends
        needed_paintings = [] # List of (tile_name, color_name) tuples
        for goal_tile, goal_color in self.goal_paintings.items():
            if goal_tile not in current_painted:
                # Tile is not painted, needs painting
                needed_paintings.append((goal_tile, goal_color))
            elif current_painted[goal_tile] != goal_color:
                # Tile is painted with the wrong color - dead end
                return float('inf')
            # If tile is in current_painted and color matches goal_color, it's satisfied.

        # If needed_paintings is empty, it means all 'painted' goals are met.
        # Since the initial check `self.goals <= state` failed, it implies there are
        # other goal predicates that are not met. This heuristic doesn't account
        # for them, so it would return 0. This is acceptable for a non-admissible
        # domain-dependent heuristic focused on the main task (painting).

        total_heuristic_cost = 0

        # Calculate cost for each needed painting
        for tile_name, needed_color in needed_paintings:
            tile_loc = self.tile_coords.get(tile_name)
            if tile_loc is None:
                 # This goal tile was not found in the tile_coords map.
                 # This indicates an inconsistency in the problem definition
                 # or an issue with tile name parsing.
                 # If we can't locate a goal tile, it's likely unsolvable.
                 return float('inf')

            min_cost_to_paint_this_tile = float('inf')

            # Find the minimum cost among all robots to paint this tile
            # Ensure there is at least one robot
            if not robot_info:
                 # No robots exist, cannot paint. Unsolvable.
                 return float('inf')

            for robot_name, info in robot_info.items():
                # Ensure robot info is complete (has location and color)
                if 'location' in info and 'color' in info:
                    robot_loc_name = info['location']
                    robot_color = info['color']

                    robot_loc = self.tile_coords.get(robot_loc_name)
                    if robot_loc is None:
                        # Robot is on a tile not found in the coords map. Unsolvable?
                        # Or maybe just skip this robot if its location is unknown.
                        # Let's assume all robot locations are valid tiles in the grid.
                        continue # Skip this robot if its location is not mapped

                    # Cost to get robot adjacent to the tile
                    move_cost = dist_adj(robot_loc, tile_loc)

                    # Cost to get the correct color
                    color_cost = 0 if robot_color == needed_color else 1

                    # Total cost for this robot to be ready to paint this tile
                    cost_for_this_robot = move_cost + color_cost

                    # Update minimum cost for this tile
                    min_cost_to_paint_this_tile = min(min_cost_to_paint_this_tile, cost_for_this_robot)

            # Add the minimum cost to get ready + the paint action cost (1)
            # If min_cost_to_paint_this_tile is still inf, it means no robot
            # could be assigned to paint this tile (e.g., no robots exist,
            # or all robots are on unmapped tiles). This makes the state unsolvable.
            if min_cost_to_paint_this_tile != float('inf'):
                 total_heuristic_cost += min_cost_to_paint_this_tile + 1
            else:
                 # No robot can paint this needed tile. Unsolvable state.
                 return float('inf')


        return total_heuristic_cost
