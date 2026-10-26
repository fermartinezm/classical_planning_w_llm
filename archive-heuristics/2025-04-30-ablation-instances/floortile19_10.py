from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

# Helper functions (can be defined outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(painted tile_1_2 black)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """Extract row and column integers from a tile name like 'tile_r_c'."""
    try:
        # Tile names are expected to be in the format 'tile_row_col'
        _, r_str, c_str = tile_name.split('_')
        return int(r_str), int(c_str)
    except ValueError:
        # This should not happen with valid PDDL instances for this domain
        print(f"Warning: Unexpected tile name format: {tile_name}")
        return None, None # Indicate failure to parse

def distance(tile1_name, tile2_name):
    """Calculate Manhattan distance between two tiles."""
    r1, c1 = get_coords(tile1_name)
    r2, c2 = get_coords(tile2_name)
    if r1 is None or r2 is None:
        return float('inf') # Cannot calculate distance for invalid tile names
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions (move, change_color, paint)
    required to paint all goal tiles that are currently unpainted. It uses a
    greedy approach to select the next tile to paint, prioritizing the closest
    unpainted goal tile.

    # Assumptions
    - The goal specifies a set of tiles that must be painted with specific colors.
    - Tiles not specified in the goal as painted are assumed to be irrelevant
      to the goal state (their painted/clear status doesn't prevent goal achievement),
      or are implicitly required to be clear (though the domain doesn't support unpainting).
      This heuristic assumes valid instances where only tiles listed in the goal
      as `painted` need attention if they are currently `clear`.
    - The robot always has a color.
    - The grid structure is defined by `up`, `down`, `left`, `right` facts, and
      Manhattan distance is a reasonable approximation for movement cost on this grid.
    - Action costs are 1.
    - The heuristic does not check for or handle unsolvable states arising from
      tiles being painted with the wrong color or blocking paths, assuming valid instances.

    # Heuristic Initialization
    - Extracts the goal conditions from the task to create a map from each goal tile
      to its required color (`goal_colors`).
    - Extracts static adjacency facts (`up`, `down`, `left`, `right`) from the task
      to build an adjacency map (`adjacency_map`). This map allows quickly finding
      the neighbors of any given tile, which is necessary to determine locations
      adjacent to a target tile for painting.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the cost by simulating a greedy process of painting
    the unpainted goal tiles one by one. The process accounts for the costs of
    changing color, moving to the painting location, and the paint action itself.

    1.  **Identify Unpainted Goal Tiles:** Determine the set of tiles that are
        specified in the goal as needing to be painted but are currently `clear`
        in the given state. Let this set be `unpainted_goal_tiles`.
    2.  **Goal Check:** If `unpainted_goal_tiles` is empty, it means all goal
        painting requirements are met in this state, so the heuristic returns 0.
    3.  **Initialization:** Initialize the total heuristic cost `h` to 0. Get the
        robot's current location (`current_loc`) and color (`current_color`) from the state.
        Create a working set `remaining_tiles` initialized with `unpainted_goal_tiles`.
    4.  **Greedy Painting Loop:** While there are tiles left in `remaining_tiles`:
        a.  **Select Next Tile:** Find the tile `target_tile` within `remaining_tiles`
            that is "closest" to the robot's `current_loc`. Closeness is defined
            as the minimum Manhattan distance from `current_loc` to *any* tile
            that is adjacent to `target_tile`. Identify this specific adjacent
            location (`target_adj_loc`) that minimizes the distance.
        b.  **Determine Target Color:** Get the required color (`target_color`) for
            `target_tile` from the precomputed `goal_colors` map.
        c.  **Account for Color Change:** If the robot's `current_color` is different
            from `target_color`, add 1 to `h` (representing the `change_color` action)
            and update `current_color` to `target_color`.
        d.  **Account for Movement:** Calculate the movement cost (`move_cost`) as the
            Manhattan distance from the robot's `current_loc` to the chosen
            `target_adj_loc`. Add `move_cost` to `h`. Update `current_loc` to
            `target_adj_loc` (as the robot moves to this location).
        e.  **Account for Painting:** Add 1 to `h` (representing the `paint` action).
        f.  **Update Remaining:** Remove `target_tile` from the `remaining_tiles` set.
    5.  **Return Heuristic Value:** After the loop finishes (all unpainted goal tiles
        have been processed), return the total accumulated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Map goal tiles to their required colors
        self.goal_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                # Fact is like (painted tile_1_2 black)
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_colors[tile] = color
                # else: ignore malformed goal facts

        # Build adjacency map from static facts
        self.adjacency_map = {}
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Fact is like (up tile_y tile_x) meaning tile_y is adjacent to tile_x
                # The predicate name indicates the direction from tile2 to tile1.
                # Adjacency is symmetric.
                pred, tile1, tile2 = parts
                self.adjacency_map.setdefault(tile1, []).append(tile2)
                self.adjacency_map.setdefault(tile2, []).append(tile1)

        # Remove duplicates from adjacency lists (a tile might be adjacent via multiple directions if grid wraps, though unlikely here)
        for tile in self.adjacency_map:
            self.adjacency_map[tile] = list(set(self.adjacency_map[tile]))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Get robot location and color
        robot_loc = None
        robot_color = None
        # Assuming only one robot and it always has a color
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot_loc = parts[2] # (robot-at robot1 tile_0_4)
            elif parts and parts[0] == "robot-has" and len(parts) == 3:
                 robot_color = parts[2] # (robot-has robot1 white)

        if robot_loc is None or robot_color is None:
             # This state is likely invalid or represents an unsolvable situation
             # where robot info is missing. Return infinity.
             return float('inf')

        # 2. Identify unpainted goal tiles
        unpainted_goal_tiles = set()
        # Get all tiles that are currently clear
        clear_tiles = {get_parts(fact)[1] for fact in state if match(fact, "clear", "*")}

        # Check which goal tiles are currently clear
        for goal_tile in self.goal_colors:
            if goal_tile in clear_tiles:
                 unpainted_goal_tiles.add(goal_tile)
            # Note: This heuristic assumes tiles painted with the wrong color
            # or non-goal tiles that are painted do not occur in solvable instances
            # or do not block progress in a way that needs explicit heuristic handling.


        # 3. If no unpainted goal tiles, goal is met for painting
        if not unpainted_goal_tiles:
            return 0

        # 4. Greedy calculation
        h = 0
        current_loc = robot_loc
        current_color = robot_color
        remaining_tiles = set(unpainted_goal_tiles)
        goal_colors_map = self.goal_colors
        adj_map = self.adjacency_map

        while remaining_tiles:
            # Find the closest unpainted goal tile (measured by distance to an adjacent cell)
            closest_tile = None
            min_dist_to_adj = float('inf')
            best_adj_loc = None # The specific adjacent location we will move to

            for t in remaining_tiles:
                # Find the adjacent location to 't' that is closest to 'current_loc'
                adjacent_locations = adj_map.get(t, [])
                if not adjacent_locations:
                    # This tile is isolated or adjacency info is missing - problem likely unsolvable
                    # Or it's a tile on the edge/corner with fewer neighbors.
                    # If it's a goal tile needing paint and has no neighbors, it's unreachable.
                    print(f"Warning: Goal tile {t} has no adjacent tiles in static facts.")
                    return float('inf') # Cannot reach/paint this tile

                for adj_loc in adjacent_locations:
                    dist = distance(current_loc, adj_loc)
                    if dist < min_dist_to_adj:
                        min_dist_to_adj = dist
                        closest_tile = t
                        best_adj_loc = adj_loc # Store the adjacent location itself

            if closest_tile is None:
                 # This should not happen if remaining_tiles is not empty and adj_map is correctly built
                 # and all goal tiles have at least one neighbor.
                 print("Error: Could not find closest tile to paint.")
                 return float('inf') # Should not be reachable in solvable problems

            target_tile = closest_tile
            target_color = goal_colors_map[target_tile]
            target_adj_loc = best_adj_loc

            # Cost to get color
            if current_color != target_color:
                h += 1 # change_color action cost
                current_color = target_color # Robot now has the target color

            # Cost to move to adjacent location
            # The distance function calculates grid steps (Manhattan distance)
            move_cost = distance(current_loc, target_adj_loc)
            h += move_cost # move action costs
            current_loc = target_adj_loc # Robot is now at the adjacent location, ready to paint

            # Cost to paint
            h += 1 # paint action cost

            # Mark the tile as painted for the heuristic calculation's remaining steps
            remaining_tiles.remove(target_tile)

        return h
