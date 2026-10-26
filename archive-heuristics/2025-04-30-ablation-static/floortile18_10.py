import math

# Assuming heuristic_base is available in the specified path
# If running as a standalone file or in a different structure, this import might need adjustment.
# For the purpose of providing the code as requested, we keep the specified import path.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Provide a dummy base class if heuristic_base is not found,
    # allowing the code structure to be checked, but it won't run
    # in isolation without the planner framework.
    print("Warning: heuristics.heuristic_base not found. Using dummy Heuristic class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Dummy heuristic called")


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # print(f"Warning: Malformed fact string: {fact}") # Optional: add logging/warnings
        return []
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses 'tile_R_C' into (R, C) tuple."""
    # Assumes tile_name is like 'tile_1_1', 'tile_0_5', etc.
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            # PDDL object names are typically case-insensitive, but string matching here is sensitive.
            # Assuming standard lowercase tile names from problem files.
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # print(f"Warning: Could not parse tile coordinates from {tile_name}") # Optional
            return None
    else:
        # print(f"Warning: Unexpected tile name format: {tile_name}") # Optional
        return None

def manhattan_distance(coord1, coord2):
    """Calculates Manhattan distance between two (R, C) coordinates."""
    if coord1 is None or coord2 is None:
        return float('inf') # Cannot calculate distance if coordinates are missing
    r1, c1 = coord1
    r2, c2 = coord2
    return abs(r1 - r2) + abs(c1 - c2)

# Heuristic class
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each goal tile
    that is not yet painted correctly. The cost for a single tile is estimated
    as the minimum cost for any robot to reach an adjacent tile with the correct
    color and perform the paint action. Movement cost is estimated using Manhattan
    distance, ignoring the 'clear' precondition for intermediate tiles but
    assuming the target adjacent tile is clear (which it must be for the paint
    action's precondition).

    # Assumptions
    - Goal tiles are initially clear or correctly painted. If a goal tile is
      painted with the wrong color, the problem might be unsolvable or the
      heuristic might be inaccurate (assuming such states are not reachable
      in valid problem instances).
    - All colors required by the goal are available.
    - There is at least one robot.
    - The grid structure is regular enough for Manhattan distance to be a
      reasonable estimate of movement cost.
    - Tile names are in the format 'tile_R_C' where R and C are integers.
    - All tiles relevant to goals or robot initial positions are connected
      and appear in static adjacency facts or clear facts.

    # Heuristic Initialization
    - Extract the goal conditions, specifically the required color for each goal tile.
    - Parse static facts to build a mapping from tile names to grid coordinates (R, C).
    - Parse static facts to build an adjacency map for tiles based on 'up', 'down',
      'left', and 'right' predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the current state of each robot: its location and the color it holds.
    3. Identify which goal tiles are not currently painted with their required color.
       A goal tile needs painting if it is listed in the goal conditions and is
       not currently asserted as `(painted tile required_color)` in the state.
       (It is assumed such tiles are `clear`).
    4. For each goal tile `T` that needs to be painted with color `C` and is not
       currently painted `C`:
        a. Initialize a minimum cost for this tile `min_cost_for_tile` to infinity.
        b. For each robot `R`:
            i. Get the robot's current location `L_R` and color `C_R`.
            ii. Calculate the cost to get the correct color: `color_cost = 1` if `C_R != C`, otherwise `0`.
            iii. Calculate the minimum movement cost for robot `R` to reach *any* tile `L_adj` that is adjacent to `T`. This is done by finding all tiles adjacent to `T` (using the precomputed adjacency map) and calculating the minimum Manhattan distance from `L_R` to each `L_adj`. Let this be `min_move_cost`. If `T` has no known adjacent tiles or `L_R` has no known coordinates, `min_move_cost` remains infinity.
            iv. The estimated cost for robot `R` to paint tile `T` is `robot_cost = color_cost + min_move_cost`.
            v. Update `min_cost_for_tile = min(min_cost_for_tile, robot_cost)`.
        c. If `min_cost_for_tile` is still infinity (meaning no robot can reach an adjacent tile), this tile cannot be painted by any robot in a way captured by this heuristic. For a non-admissible heuristic, we could potentially return infinity for the whole state, but summing up reachable costs is more informative. We only add cost if `min_cost_for_tile` is finite.
        d. If `min_cost_for_tile` is finite, add the minimum cost found for any robot to paint tile `T`, plus the cost of the paint action itself (which is 1), to the total heuristic: `total_heuristic += min_cost_for_tile + 1`.
    5. Return the `total_heuristic`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal requirements: tile -> color
        self.goal_painted_status = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_painted_status[tile] = color


        # Map tile names to coordinates (R, C)
        self.tile_coords = {}
        # Build adjacency map: tile -> set of adjacent tiles
        self.tile_adjacencies = {}

        # Collect all tile names from static facts first
        all_tile_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] in ["up", "down", "left", "right"]:
                 if len(parts) == 3:
                    all_tile_names.add(parts[1])
                    all_tile_names.add(parts[2])
            elif parts[0] == "clear":
                 if len(parts) == 2:
                    all_tile_names.add(parts[1])


        # Parse coordinates for all collected tile names
        for tile_name in all_tile_names:
             coords = parse_tile_name(tile_name)
             if coords is not None:
                self.tile_coords[tile_name] = coords


        # Build adjacency map using the parsed coordinates
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ["up", "down", "left", "right"]:
                if len(parts) == 3:
                    tile1, tile2 = parts[1], parts[2]
                    # Add adjacency (bidirectional)
                    self.tile_adjacencies.setdefault(tile1, set()).add(tile2)
                    self.tile_adjacencies.setdefault(tile2, set()).add(tile1)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Initialize total heuristic cost
        total_cost = 0

        # 2. Identify current robot states (location and color)
        robot_states = {} # robot_name -> {'location': tile_name, 'color': color_name}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "robot-at":
                if len(parts) == 3:
                    robot, location = parts[1], parts[2]
                    robot_states.setdefault(robot, {})['location'] = location
            elif parts[0] == "robot-has":
                if len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_states.setdefault(robot, {})['color'] = color

        # 3. Identify unpainted goal tiles
        unpainted_goal_tiles = {} # tile_name -> required_color
        current_painted_status = {} # tile_name -> color_it_is_painted_with

        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "painted":
                 if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    current_painted_status[tile] = color

        for goal_tile, required_color in self.goal_painted_status.items():
            # Check if the goal tile is painted with the required color
            is_painted_correctly = (goal_tile in current_painted_status and
                                    current_painted_status[goal_tile] == required_color)

            # If not painted correctly, it's an unpainted goal tile
            if not is_painted_correctly:
                 unpainted_goal_tiles[goal_tile] = required_color


        # 4. For each unpainted goal tile, calculate minimum cost
        for target_tile, required_color in unpainted_goal_tiles.items():
            min_cost_for_tile = float('inf')

            # Get coordinates of the target tile
            target_coords = self.tile_coords.get(target_tile)
            if target_coords is None:
                 continue # Skip this tile

            # Find adjacent tiles of the target tile
            adjacent_tiles = self.tile_adjacencies.get(target_tile, set())
            if not adjacent_tiles:
                 continue # Skip this tile

            # Calculate minimum cost for any robot to paint this tile
            for robot_name, robot_info in robot_states.items():
                robot_location = robot_info.get('location')
                robot_color = robot_info.get('color')

                if robot_location is None or robot_color is None:
                    continue # Robot state incomplete, skip this robot for this tile

                # Cost to get the correct color
                color_cost = 1 if robot_color != required_color else 0

                # Cost to move to an adjacent tile
                min_move_cost = float('inf')
                robot_coords = self.tile_coords.get(robot_location)

                if robot_coords is None:
                    continue # Robot location not in coords map, skip this robot for this tile

                for adj_tile in adjacent_tiles:
                    adj_coords = self.tile_coords.get(adj_tile)
                    if adj_coords is None:
                         continue # Adjacent tile not in coords map, skip this adjacent tile
                    move_cost = manhattan_distance(robot_coords, adj_coords)
                    min_move_cost = min(min_move_cost, move_cost)

                # If no reachable adjacent tiles were found, min_move_cost is inf
                if min_move_cost == float('inf'):
                     continue # Cannot reach an adjacent tile with this robot

                # Total cost for this robot for this tile = color_cost + move_cost
                robot_cost = color_cost + min_move_cost

                # Update minimum cost for this tile across all robots
                min_cost_for_tile = min(min_cost_for_tile, robot_cost)

            # Add cost for this tile to total heuristic (if reachable by at least one robot)
            if min_cost_for_tile != float('inf'):
                 # Add 1 for the paint action itself
                 total_cost += min_cost_for_tile + 1

        # 5. Return the total heuristic
        return total_cost
