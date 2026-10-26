import re
import math
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential whitespace issues and ensure correct splitting
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Not a valid PDDL fact string format we expect
        return []
    # Remove outer parentheses and split by whitespace
    parts = fact[1:-1].split()
    return parts

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` are allowed in the pattern arguments.
    """
    fact_parts = get_parts(fact)
    if len(fact_parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(fact_parts, args))

def parse_tile_coords(tile_name):
    """Parses tile name 'tile_X_Y' into integer coordinates (X, Y)."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Return None for invalid format
    return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles."""
    coords1 = parse_tile_coords(tile1_name)
    coords2 = parse_tile_coords(tile2_name)
    if coords1 is None or coords2 is None:
        # This indicates an issue with tile naming convention or invalid tile names.
        # For heuristic, returning a large value is safer than erroring out,
        # implying these locations are effectively infinitely far apart.
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all goal tiles by summing the number
    of paint actions required and the estimated minimum movement cost for robots
    to reach and connect the locations of tiles needing the same color.
    It uses Manhattan distance as a proxy for grid distance and calculates
    a Minimum Spanning Tree (MST) weight on the target tiles for each color,
    plus the minimum distance from an available robot to any of those tiles.

    # Assumptions
    - Tiles are named in the format 'tile_X_Y' where X and Y are integers representing grid coordinates.
    - Movement is restricted to adjacent tiles (up, down, left, right), making Manhattan distance a valid shortest path metric.
    - Robots are pre-assigned a fixed color they can paint with and cannot change it.
    - Goal tiles are either 'clear' or already painted with the correct color in the initial state. The task is to paint the 'clear' goal tiles or tiles painted with the wrong color (though examples suggest only 'clear' tiles need painting). The heuristic assumes any goal (painted T C) not in the state needs painting.
    - Painting a tile requires a robot with the correct color to be at the tile.

    # Heuristic Initialization
    - Extracts goal conditions to identify which tiles need to be painted and with which color.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify Unsatisfied Goals: Determine which `(painted tile color)` goal facts are not present in the current state. These are the tiles that still need to be painted. Group these unsatisfied goals by the required color.
    2.  Identify Robot Status: Find the current location `(robot-at robot tile)` and color `(robot-has robot color)` for each robot in the state.
    3.  Check Solvability: For each color required by an unsatisfied goal, check if there is at least one robot that has that color. If not, the problem is likely unsolvable from this state, and the heuristic should return infinity.
    4.  Calculate Cost Per Color: Iterate through each color C that is required by at least one unsatisfied goal tile.
        a.  Get the list of tiles `Tiles_C` that need to be painted with color C.
        b.  Get the list of robots `Robots_C` that have color C, and their current locations `Robot_Locations_C`.
        c.  Add the number of tiles `|Tiles_C|` to the total heuristic. This accounts for the paint actions (1 action per tile).
        d.  Estimate Movement Cost for Color C:
            i.  Calculate the Minimum Spanning Tree (MST) weight on the set of target tiles `Tiles_C` using Manhattan distance as edge weights. This MST weight estimates the minimum travel cost to connect all tiles needing the same color. Add this MST weight to the total heuristic. If `|Tiles_C| <= 1`, the MST weight is 0.
            ii. Calculate the minimum Manhattan distance from any robot in `Robots_C` to any tile in `Tiles_C`. This estimates the cost to get the "first" robot of the correct color to the cluster of tiles needing that color. Add this minimum distance to the total heuristic. If `Robots_C` is empty or `Tiles_C` is empty, this cost is 0 (though the solvability check should handle the empty robot case).
    5.  Sum Costs: The total heuristic value is the sum of the paint action costs and the estimated movement costs calculated for each color.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # We only care about (painted tile color) goals
        self.painted_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.painted_goals[tile] = color
            # Ignore other types of goals if any

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Unsatisfied Goals and group by color
        unsatisfied_goals_by_color = {}
        for tile, goal_color in self.painted_goals.items():
            # Check if the tile is already painted with the goal color
            # Efficiently check if the specific goal fact exists in the state frozenset
            goal_fact_str = f"(painted {tile} {goal_color})"
            if goal_fact_str not in state:
                # This tile needs to be painted with goal_color
                if goal_color not in unsatisfied_goals_by_color:
                    unsatisfied_goals_by_color[goal_color] = []
                unsatisfied_goals_by_color[goal_color].append(tile)

        # If all painted goals are satisfied, heuristic is 0
        if not unsatisfied_goals_by_color:
            return 0

        # 2. Identify Robot Status
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "robot-at" and len(parts) == 3:
                    robot, location = parts[1], parts[2]
                    robot_locations[robot] = location
                elif parts[0] == "robot-has" and len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color

        # 3. Check Solvability (based on available robot colors)
        available_colors = set(robot_colors.values())
        for required_color in unsatisfied_goals_by_color.keys():
             if required_color not in available_colors:
                 # There is at least one color needed for a goal tile that no robot has
                 # This state is likely unsolvable. Return infinity.
                 return float('inf')


        total_heuristic = 0

        # 4. Calculate Cost Per Color
        for color, tiles_to_paint in unsatisfied_goals_by_color.items():
            # Paint actions cost for this color
            total_heuristic += len(tiles_to_paint)

            # Find robots that have this color and their locations
            robots_with_color = [r for r, c in robot_colors.items() if c == color]
            # Assuming every robot with a color also has a location in a valid state
            robot_start_locations = [robot_locations[r] for r in robots_with_color]

            # Estimate Movement Cost for Color C

            # a. MST weight on target tiles
            if len(tiles_to_paint) > 1:
                locations_list = tiles_to_paint # MST nodes are the target tiles
                num_locations = len(locations_list)

                min_dist = [float('inf')] * num_locations
                in_mst = [False] * num_locations
                mst_weight = 0

                # Start MST from the first target tile (arbitrary choice)
                min_dist[0] = 0

                for _ in range(num_locations):
                    # Pick the location with the minimum distance value from the set
                    # of locations not yet included in MST
                    min_val = float('inf')
                    min_index = -1

                    for v in range(num_locations):
                        if not in_mst[v] and min_dist[v] < min_val:
                            min_val = min_dist[v]
                            min_index = v

                    if min_index == -1:
                         # This indicates a disconnected graph of target tiles,
                         # which shouldn't happen in a valid grid problem.
                         # If it occurs, it might suggest an unsolvable subproblem.
                         # We add the remaining min_dist values as a penalty.
                         mst_weight += sum(md for i, md in enumerate(min_dist) if not in_mst[i])
                         break

                    in_mst[min_index] = True
                    mst_weight += min_val # Add the edge weight used to connect it

                    # Update min_dist values of the adjacent locations (all other locations)
                    # Only consider locations not yet in MST
                    for v in range(num_locations):
                        if not in_mst[v]:
                            dist = manhattan_distance(locations_list[min_index], locations_list[v])
                            if dist < min_dist[v]:
                                min_dist[v] = dist

                total_heuristic += mst_weight

            # b. Minimum distance from any robot to any target tile
            min_start_to_target_dist = float('inf')
            if robot_start_locations and tiles_to_paint:
                for robot_loc in robot_start_locations:
                    for target_tile in tiles_to_paint:
                        dist = manhattan_distance(robot_loc, target_tile)
                        min_start_to_target_dist = min(min_start_to_target_dist, dist)

            # Add the minimum distance to get a robot to the target cluster
            if min_start_to_target_dist != float('inf'):
                 total_heuristic += min_start_to_target_dist
            # Note: If robot_start_locations is empty for this color, min_start_to_target_dist remains inf,
            # but this case is already handled by the solvability check returning inf earlier.

        return total_heuristic
