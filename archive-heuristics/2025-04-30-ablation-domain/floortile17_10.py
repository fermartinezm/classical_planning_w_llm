from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have the same number of parts as args for a strict match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with the correct colors. It sums the estimated costs for painting each unpainted goal tile, considering the need for paint actions, color acquisition (pickup/drop), and robot movement.

    # Assumptions
    - Tiles are arranged in a grid, and movement cost between tile_r1_c1 and tile_r2_c2 is the Manhattan distance abs(r1-r2) + abs(c1-c2).
    - Colors are always available in the system (either held by a robot or in the available pool). Painting a tile makes the color available again.
    - Tiles that need painting are initially clear or become clear implicitly (the heuristic doesn't explicitly model clearing actions).
    - The heuristic sums costs independently for each unpainted tile's movement, potentially overestimating total movement but providing a gradient.
    - The heuristic estimates color acquisition costs based on global color availability, not specific robot assignments.

    # Heuristic Initialization
    - The heuristic stores the goal conditions to identify which tiles need to be painted and with which colors.
    - Static facts are available in task.static but not explicitly used in this version, as grid structure is inferred from tile names and color availability is assumed based on domain mechanics.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic value for a given state:

    1.  **Identify Unpainted Goal Tiles:** Iterate through the goal facts and the current state facts to find all `(painted T C)` goals that are not yet true in the state. Store these as a set of `(tile, color)` tuples.
    2.  **Base Cost (Paint Actions):** Initialize the heuristic value `h` with the number of unpainted goal tiles. Each unpainted tile requires at least one `paint` action.
    3.  **Color Acquisition Cost (Pickup Actions):**
        -   Determine the set of colors required by the unpainted goal tiles.
        -   Determine the set of colors currently held by robots in the state.
        -   Identify colors that are needed but not currently held by any robot.
        -   Add the count of these colors to `h`. This estimates the number of `pickup` actions needed for colors not currently in possession of any robot.
    4.  **Color Acquisition Cost (Drop Actions):**
        -   Identify colors currently held by robots that are *not* needed for any unpainted goal tile.
        -   Count the number of robots holding such unneeded colors.
        -   Add this count to `h`. This estimates the number of `drop` actions needed for robots to free up grippers holding irrelevant colors.
    5.  **Movement Cost:**
        -   For each unpainted goal tile `T`:
            -   Find the current location of every robot.
            -   Calculate the Manhattan distance from each robot's location to tile `T` (by parsing tile names like `tile_r_c` into coordinates `(r, c)`).
            -   Find the minimum distance among all robots to tile `T`.
            -   Add this minimum distance to `h`. This estimates the minimum movement required for *some* robot to reach this specific unpainted tile.
    6.  **Return Total Cost:** The final value of `h` is the estimated number of actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are available in task.static but not explicitly used
        # in this heuristic's calculation logic based on tile names and state.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Unpainted Goal Tiles
        unpainted_goals = set()
        for goal_fact in self.goals:
            # Ensure the goal fact is a painted predicate
            if match(goal_fact, "painted", "*", "*"):
                 if goal_fact not in state:
                    # Extract tile and color from the goal fact
                    parts = get_parts(goal_fact)
                    if len(parts) == 3: # Should be (painted tile color)
                        tile, color = parts[1], parts[2]
                        unpainted_goals.add((tile, color))

        # If all goals are met, heuristic is 0
        if not unpainted_goals:
            return 0

        # Initialize heuristic value
        h = 0

        # 2. Base Cost (Paint Actions)
        h += len(unpainted_goals)

        # 3. & 4. Color Acquisition Cost (Pickup and Drop Actions)
        needed_colors = {color for tile, color in unpainted_goals}
        robot_colors = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "robot-has", "*", "*")}
        colors_held_by_robots = set(robot_colors.values())

        # Cost for colors needed but not held by any robot
        colors_to_pickup = needed_colors - colors_held_by_robots
        h += len(colors_to_pickup)

        # Cost for robots holding colors that are not needed for any unpainted tile
        unneeded_held_colors = colors_held_by_robots - needed_colors
        robots_holding_unneeded_colors = {R for R, C in robot_colors.items() if C in unneeded_held_colors}
        h += len(robots_holding_unneeded_colors)


        # 5. Movement Cost
        robot_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "robot-at", "*", "*")}

        for tile, color in unpainted_goals:
            min_dist = float('inf')
            tile_coords = None
            try:
                # Parse tile name tile_r_c into (r, c)
                parts = tile.split('_')
                if len(parts) == 3 and parts[0] == 'tile':
                    tile_coords = (int(parts[1]), int(parts[2]))
                # else: tile name format is unexpected, skip movement cost for this tile

            except (ValueError, IndexError):
                # Handle cases where tile name is not in expected format
                tile_coords = None # Ensure tile_coords is None on error

            if tile_coords is not None:
                for robot, location in robot_locations.items():
                    loc_coords = None
                    try:
                        # Parse location name tile_r_c into (r, c)
                        loc_parts = location.split('_')
                        if len(loc_parts) == 3 and loc_parts[0] == 'tile':
                            loc_coords = (int(loc_parts[1]), int(loc_parts[2]))
                        # else: location name format is unexpected, skip distance calculation for this robot

                    except (ValueError, IndexError):
                        # Handle cases where location name is not in expected format
                        loc_coords = None # Ensure loc_coords is None on error

                    if loc_coords is not None:
                        dist = abs(tile_coords[0] - loc_coords[0]) + abs(tile_coords[1] - loc_coords[1])
                        min_dist = min(min_dist, dist)

                # Add the minimum distance found for this tile
                if min_dist != float('inf'): # Should always find at least one robot location
                    h += min_dist
                # else: No valid robot locations found, cannot estimate movement, h remains unchanged for this tile's movement cost

        # 6. Return Total Cost
        return h
