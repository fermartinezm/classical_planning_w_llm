from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque
import math # For float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It considers the cost of painting each required tile,
    changing robot colors, and moving robots to adjacent tiles. The heuristic
    returns infinity for detected unsolvable states.

    # Assumptions
    - The problem instance represents a grid of tiles.
    - Tiles are connected by up, down, left, right predicates, defining the grid graph.
    - Robots can only paint tiles adjacent to their current location.
    - A tile must be clear to be painted.
    - If a goal tile is not painted with the target color, it is assumed to be clear
      unless it is painted with a different color (which indicates an unsolvable state).
    - All required colors are available in the domain.

    # Heuristic Initialization
    - Parses static facts to build the adjacency graph of tiles based on connectivity predicates.
    - Computes all-pairs shortest paths between tiles on the grid graph using BFS.
    - Stores the goal conditions, specifically which tiles need to be painted with which colors.
    - Stores available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unmet Goals and Check for Unsolvable States:
       - Iterate through each goal condition `(painted T C)`.
       - If `(painted T C)` is not true in the current state:
         - Check if tile `T` is currently painted with *any* color `C'`.
         - If `T` is painted with a color `C'` where `C' != C`, the state is unsolvable. Return infinity.
         - If `T` is not painted with `C` and not painted with any other color (meaning it's clear), add `(T, C)` to the set of unmet goals.
       - If any required color `C` for an unmet goal `(T, C)` is not listed as `available-color` in the static facts, the state is unsolvable. Return infinity.

    2. Check for Goal State:
       - If the set of unmet goals is empty, the current state is a goal state. Return 0.

    3. Calculate Color Change Cost:
       - Identify the set of distinct colors required by the unmet goals.
       - Identify the set of colors currently held by robots.
       - Count how many required colors are *not* currently held by any robot. Each such color requires at least one `change_color` action globally. This count is the color change cost.

    4. Calculate Paint Cost:
       - Each unmet goal tile requires one `paint` action. The paint cost is the total number of unmet goals.

    5. Calculate Movement Cost:
       - For each tile `T` in the unmet goals:
         - Find all tiles `X` adjacent to `T` using the precomputed adjacency graph.
         - Determine the minimum shortest path distance from *any* robot's current location to *any* of the tiles `X` adjacent to `T`.
         - If a tile `T` has no adjacent tiles, or if no robot can reach any adjacent tile, the state is unsolvable. Return infinity.
         - Sum these minimum distances for all unmet goal tiles. This sum is the movement cost.

    6. Calculate Total Heuristic:
       - The total heuristic value is the sum of the color change cost, the paint cost, and the movement cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing grid information."""
        self.goals = task.goals
        self.static_facts = task.static # Store static facts for later checks

        # Store goal requirements: {tile_name: color_name}
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # Build adjacency graph from static facts
        self.adj = defaultdict(list)
        self.all_tiles = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Predicate format is (direction tile_y tile_x) where tile_y is in that direction from tile_x
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # We want adjacency: tile_0_1 is adjacent to tile_1_1
                tile_y, tile_x = parts[1], parts[2]
                self.adj[tile_x].append(tile_y)
                self.adj[tile_y].append(tile_x) # Assume connections are bidirectional
                self.all_tiles.add(tile_x)
                self.all_tiles.add(tile_y)

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_tile in self.all_tiles:
            self.dist[start_tile] = {}
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            while q:
                curr_tile, d = q.popleft()
                self.dist[start_tile][curr_tile] = d
                for neighbor in self.adj.get(curr_tile, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1))

        # Store available colors
        self.available_colors = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "available-color", "*")}


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Identify unmet goals and check for unsolvable states
        unmet_goals = {} # {tile: color}
        for tile, color in self.goal_paintings.items():
            is_painted_correctly = f"(painted {tile} {color})" in state
            if not is_painted_correctly:
                 # Check if it's painted with the wrong color
                 # A tile is either clear or painted with *one* color.
                 is_painted_wrongly = False
                 for fact in state:
                     if match(fact, "painted", tile, "*"):
                         # Found a painted fact for this tile
                         if fact != f"(painted {tile} {color})":
                             is_painted_wrongly = True
                             break # Found wrong color
                         # else: it's painted with the correct color, but we already checked !is_painted_correctly, so this branch is not taken.

                 if is_painted_wrongly:
                     # If a goal tile is painted with the wrong color, it's unsolvable.
                     return math.inf
                 else:
                     # If not painted correctly and not painted wrongly, it must be clear
                     # and needs painting.
                     unmet_goals[tile] = color

        # Check if required colors are available at all. If not, unsolvable.
        needed_colors_for_unmet = set(unmet_goals.values())
        for color in needed_colors_for_unmet:
            if color not in self.available_colors:
                 # Required color is not available in the domain. Unsolvable.
                 return math.inf

        # 2. If no unmet goals, heuristic is 0
        if not unmet_goals:
            return 0

        # 3. Calculate color change cost
        robot_colors = {get_parts(fact)[2] for fact in state if match(fact, "robot-has", "*", "*")}
        change_color_cost = sum(1 for color in needed_colors_for_unmet if color not in robot_colors)

        # 4. Calculate paint cost
        paint_cost = len(unmet_goals)

        # 5. Calculate movement cost
        robot_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "robot-at", "*", "*")}

        movement_cost = 0
        for tile_to_paint, required_color in unmet_goals.items():
            min_dist_to_adjacent = math.inf
            adjacent_tiles = self.adj.get(tile_to_paint, [])

            # If a tile has no adjacent tiles, it cannot be painted. Unsolvable.
            if not adjacent_tiles:
                 return math.inf

            reachable_adjacent_found = False
            for robot, r_loc in robot_locations.items():
                # Check if robot location is a valid tile in our graph
                if r_loc not in self.dist:
                    # Robot location not in precomputed distances? Malformed problem? Unsolvable.
                    return math.inf

                for adj_tile in adjacent_tiles:
                    # Check if adjacent tile is a valid tile in our graph
                    if adj_tile in self.dist[r_loc]:
                         min_dist_to_adjacent = min(min_dist_to_adjacent, self.dist[r_loc][adj_tile])
                         reachable_adjacent_found = True

            # If no robot can reach any adjacent tile, it's unsolvable.
            if not reachable_adjacent_found:
                 return math.inf

            movement_cost += min_dist_to_adjacent

        # 6. Total heuristic
        total_cost = change_color_cost + paint_cost + movement_cost

        return total_cost
