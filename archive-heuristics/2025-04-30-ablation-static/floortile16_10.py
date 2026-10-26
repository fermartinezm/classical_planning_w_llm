# Required imports
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or raise an error
        # For robustness, return empty list or handle appropriately
        return []
    return fact[1:-1].split()

# Helper function to check if a fact matches a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the cost to reach the goal state by summing up
    several components: the number of tiles that still need to be painted
    correctly, the number of colors that need to be acquired by robots,
    the number of goal tiles that need to be cleared, and an estimate
    of the movement cost for robots to reach painting positions.

    # Assumptions
    - The goal is to paint a specific set of tiles with specific colors.
    - Tiles are arranged in a grid-like structure defined by up/down/left/right predicates.
    - A robot must be at an adjacent tile in the correct direction to paint a tile.
    - A tile must be clear to be painted or moved onto.
    - All colors required by the goal are available.
    - Robots can change color if an available color is specified.

    # Heuristic Initialization
    The heuristic precomputes the following from the task definition:
    - `goal_paintings`: A dictionary mapping goal tiles to their required colors.
    - `all_tiles`: A set of all tile names involved in the grid.
    - `adjacencies`: An adjacency list representing the grid graph.
    - `required_paint_positions`: A dictionary mapping a tile to be painted to the set of tiles where a robot must be located to paint it.
    - `distances`: A dictionary storing shortest path distances between all pairs of tiles in the grid, computed using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as the sum of four components:

    1.  **Unpainted Goal Tiles Cost:** Count the number of goal tiles that are not currently painted with their required color. Each such tile requires at least one `paint_...` action. Add this count to the heuristic.

    2.  **Color Acquisition Cost:** Identify the set of colors required by the unpainted goal tiles. Count how many of these required colors are *not* currently held by any robot. Each such color requires at least one `change_color` action by some robot. Add this count to the heuristic.

    3.  **Tile Clearing Cost:** Count the number of unpainted goal tiles that are *not* currently `clear`. A tile is not clear if it is occupied by a robot or painted (but we only care about unpainted goal tiles here, so it must be occupied by a robot). The robot occupying such a tile must move off, requiring at least one `move_...` action. Add this count to the heuristic.

    4.  **Robot Movement Cost:** For each robot, find the minimum shortest path distance from its current location to *any* tile that is a required painting position for *any* unpainted goal tile. Sum these minimum distances over all robots. This estimates the total movement effort needed for robots to get into positions where they can start painting. Add this sum to the heuristic.

    The total heuristic value is the sum of these four components.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing grid distances.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        self.all_tiles = set()
        self.adjacencies = {}
        self.required_paint_positions = {}

        # Build adjacency list and required paint positions from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            pred = parts[0]
            if pred in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)

                # tile1 is adjacent to tile2, and tile2 is adjacent to tile1
                self.adjacencies.setdefault(tile1, set()).add(tile2)
                self.adjacencies.setdefault(tile2, set()).add(tile1)

                # To paint tile1 (e.g., tile1 is up from tile2), robot must be at tile2
                self.required_paint_positions.setdefault(tile1, set()).add(tile2)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = {}
            queue = deque([(start_tile, 0)])
            visited = {start_tile}
            self.distances[start_tile][start_tile] = 0

            while queue:
                current_tile, dist = queue.popleft()

                if current_tile in self.adjacencies:
                    for neighbor in self.adjacencies[current_tile]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_tile][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Parse dynamic facts from the current state
        robot_locations = {}
        robot_colors = set()
        clear_status = set()
        painted_status = {} # {tile: color}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif pred == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors.add(color)
            elif pred == "clear":
                tile = parts[1]
                clear_status.add(tile)
            elif pred == "painted":
                tile, color = parts[1], parts[2]
                painted_status[tile] = color

        h = 0 # Initialize heuristic value

        # Component 1: Unpainted Goal Tiles
        unpainted_goals = set() # Store (tile, color) tuples
        for goal_tile, goal_color in self.goal_paintings.items():
            # Check if the tile is painted with the correct color in the current state
            if painted_status.get(goal_tile) != goal_color:
                 unpainted_goals.add((goal_tile, goal_color))

        h += len(unpainted_goals) # Cost for paint actions

        # Component 2: Color Acquisition
        needed_colors = {color for (tile, color) in unpainted_goals}
        colors_to_acquire = needed_colors - robot_colors
        h += len(colors_to_acquire) # Cost for change_color actions

        # Component 3: Tile Clearing
        tiles_to_clear = set()
        for goal_tile, goal_color in unpainted_goals:
            # If the goal tile is not clear, a robot must be on it and needs to move off
            # Note: A painted tile is also not clear, but we only consider unpainted goal tiles here.
            # If an unpainted goal tile is not clear, it must be occupied by a robot.
            if goal_tile not in clear_status:
                 # Verify it's occupied by a robot (should be if not clear and not painted correctly)
                 is_occupied = False
                 for robot, loc in robot_locations.items():
                     if loc == goal_tile:
                         is_occupied = True
                         break
                 if is_occupied:
                     tiles_to_clear.add(goal_tile)

        h += len(tiles_to_clear) # Cost for move_... actions to clear tiles

        # Component 4: Robot Movement to Painting Position
        movement_cost = 0
        robot_current_locations = list(robot_locations.values())

        # Find all required painting positions for all unpainted goal tiles
        all_required_paint_positions = set()
        for goal_tile, goal_color in unpainted_goals:
             if goal_tile in self.required_paint_positions:
                 all_required_paint_positions.update(self.required_paint_positions[goal_tile])

        if all_required_paint_positions:
            # For each robot, find the minimum distance to any required painting position
            for robot_loc in robot_current_locations:
                min_dist_for_robot = float('inf')
                if robot_loc in self.distances: # Ensure robot location is in the grid graph
                    for req_pos_tile in all_required_paint_positions:
                        if req_pos_tile in self.distances[robot_loc]:
                            min_dist_for_robot = min(min_dist_for_robot, self.distances[robot_loc][req_pos_tile])

                if min_dist_for_robot != float('inf'):
                    movement_cost += min_dist_for_robot
                # else: Robot is in a location not in the precomputed grid? (Shouldn't happen in valid problems)
                #  or no required paint positions are reachable (problem might be unsolvable or disconnected)
                #  In a solvable problem, all tiles should be connected.

        h += movement_cost

        return h
