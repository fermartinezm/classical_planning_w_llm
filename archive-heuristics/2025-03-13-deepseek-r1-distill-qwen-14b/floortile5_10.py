from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from itertools import product
import re

def extract_number(s):
    return int(re.findall(r'\d+', s)[0])

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The distance the robot needs to move to each tile.
    - The necessity of changing colors and the associated actions.

    # Assumptions:
    - The robot can move up, down, left, or right on the grid.
    - The robot can change color at any clear tile.
    - Each tile requires exactly one painting action.

    # Heuristic Initialization
    - Extracts goal conditions and static facts from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted and their required colors.
    2. Group these tiles by their required color.
    3. For each color group:
       a. If the robot's current color does not match the group's color, add the cost of changing color.
       b. Calculate the Manhattan distance to the farthest tile in the group.
       c. Add the number of tiles in the group (each requiring one paint action).
    4. Sum the costs for all color groups to get the total heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Extract grid layout from static facts
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.up[x] = y
                self.down[y] = x
            elif match(fact, "left", "*", "*"):
                x, y = get_parts(fact)[1], get_parts(fact)[2]
                self.left[x] = y
                self.right[y] = x

        # Precompute adjacency for distance calculations
        self.tiles = set()
        for fact in static_facts:
            if match(fact, "tile"):
                self.tiles.add(get_parts(fact)[0])

        # Precompute coordinates for each tile
        self.tile_coords = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*"):
                x, y = get_parts(fact)[2], get_parts(fact)[1]
                if x not in self.tile_coords:
                    self.tile_coords[x] = (0, 0)
                if y not in self.tile_coords:
                    self.tile_coords[y] = (0, 1)
            elif match(fact, "left", "*", "*"):
                x, y = get_parts(fact)[1], get_parts(fact)[2]
                if x not in self.tile_coords:
                    self.tile_coords[x] = (1, 0)
                if y not in self.tile_coords:
                    self.tile_coords[y] = (1, -1)

    def __call__(self, node):
        state = node.state
        current_tile = None
        current_color = None
        painted = {}
        goals = {}

        # Extract current state information
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                current_tile = get_parts(fact)[1]
            if match(fact, "robot-has", "*", "*"):
                current_color = get_parts(fact)[1]
            if match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)[1], get_parts(fact)[2]
                painted[tile] = color

        # Extract goal information
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile, color = get_parts(goal)[1], get_parts(goal)[2]
                goals[tile] = color

        # Check if all goals are already achieved
        if all(tile in painted and painted[tile] == color for tile, color in goals.items()):
            return 0

        # Group tiles by their required color
        color_groups = {}
        for tile, color in goals.items():
            if tile not in painted or painted[tile] != color:
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(tile)

        total_cost = 0

        # For each color group, calculate the cost
        for color, tiles in color_groups.items():
            if not tiles:
                continue

            # Check if color change is needed
            if current_color != color:
                # Check if current tile is clear
                if any(fact == f"(clear {current_tile})" for fact in state):
                    # Can change color at current tile
                    total_cost += 2  # change color and move to farthest tile
                else:
                    # Need to move to a clear tile first
                    # Find nearest clear tile
                    clear_tiles = [fact[7:-1] for fact in state if fact.startswith("(clear "))
                    if not clear_tiles:
                        continue  # no clear tile available, assume infinite cost
                    # Calculate distances from current_tile to all clear tiles
                    min_distance = min(
                        abs(extract_number(x1) - extract_number(x2)) +
                        abs(extract_number(y1) - extract_number(y2))
                        for x1, y1 in self.tile_coords.values()
                        for x2 in clear_tiles
                        if x2 in self.tile_coords
                    )
                    total_cost += min_distance + 2  # move to clear tile and change color

                current_color = color

            # Calculate distances to each tile in the group
            distances = []
            for tile in tiles:
                if tile in self.tile_coords:
                    x1, y1 = self.tile_coords[current_tile]
                    x2, y2 = self.tile_coords[tile]
                    distance = abs(x2 - x1) + abs(y2 - y1)
                    distances.append(distance)

            if not distances:
                continue

            max_distance = max(distances)
            num_tiles = len(tiles)
            total_cost += max_distance + num_tiles

        return total_cost
