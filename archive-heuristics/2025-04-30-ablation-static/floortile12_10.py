import collections
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    fact = fact.strip()
    if fact.startswith('(') and fact.endswith(')'):
        fact = fact[1:-1]
    return fact.split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently clear. It sums the estimated costs for each unpainted goal tile,
    considering the paint action, necessary color changes, and movement.

    # Assumptions
    - Tiles are arranged in a grid structure defined by up/down/left/right predicates.
    - The grid is connected.
    - Tiles are initially either clear or painted (correctly or incorrectly).
    - The goal requires specific tiles to be painted with specific colors.
    - There is no action to unpaint a tile or change its color directly once painted.
    - If a tile is painted with the wrong color according to the goal, the problem is unsolvable.
    - Robots always have a color (not free-color), and can change to any available color.

    # Heuristic Initialization
    - Parses goal facts to store the required color for each goal tile (`self.goal_paintings`).
    - Parses static facts (up, down, left, right) and initial state facts to build the grid adjacency graph and collect all tile names.
    - Computes all-pairs shortest paths (BFS distances) between all tiles on the grid.
    - Stores available colors from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all tiles that are goal tiles but are not currently painted with the correct color.
    2. Check if any goal tile is currently painted with a *wrong* color. If so, the state is likely unsolvable, return infinity.
    3. Filter the remaining unpainted goal tiles to find those that are currently `clear`. These are the tiles that need to be painted.
    4. Initialize the heuristic value `h` to 0.
    5. Add 1 to `h` for each `clear` goal tile. This accounts for the paint action required for each such tile.
    6. Determine which colors are needed for the `clear` goal tiles.
    7. Determine which colors are currently held by the robots in the current state.
    8. For each needed color: if it's an available color (from initial state) but is not currently held by *any* robot, add 1 to `h`. This accounts for the minimum number of `change_color` actions required (assuming one robot can acquire the color). This is a lower bound on color changes. If a needed color is not an available color, the problem is unsolvable, return infinity.
    9. Estimate the movement cost: For each `clear` goal tile `T`, find the minimum BFS distance from *any* robot's current location to *any* tile adjacent to `T`. Sum these minimum distances and add the total to `h`. This approximates the movement effort needed to get robots into position to paint the required tiles, ignoring robot coordination and conflicts. If any goal tile is unreachable by any robot, return infinity.
    10. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to get all tiles and available colors

        # 1. Parse goal facts to get required paintings
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # 2. Build the grid adjacency graph and collect all tile names
        self.adjacency = collections.defaultdict(set)
        all_tiles = set()

        # Collect tiles and build adjacency from static facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Direction predicates are (direction tile1 tile2) meaning tile1 is direction of tile2
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # This implies an edge between tile1 and tile2
                tile1, tile2 = parts[1], parts[2]
                self.adjacency[tile1].add(tile2)
                self.adjacency[tile2].add(tile1) # Grid connections are typically bidirectional
                all_tiles.add(tile1)
                all_tiles.add(tile2)

        # Also collect tiles mentioned in goals and initial state
        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts[1:]: # Check arguments
                 if part.startswith('tile_'): # Simple check for tile names
                     all_tiles.add(part)

        for fact in self.initial_state:
             parts = get_parts(fact)
             for part in parts[1:]:
                 if part.startswith('tile_'):
                     all_tiles.add(part)
             if parts[0] == 'robot-at': # Robot location is a tile
                  if len(parts) > 2 and parts[2].startswith('tile_'):
                    all_tiles.add(parts[2])
             elif parts[0] == 'painted' or parts[0] == 'clear': # Painted/clear tiles
                  if len(parts) > 1 and parts[1].startswith('tile_'):
                    all_tiles.add(parts[1])

        self.all_tiles = list(all_tiles) # Store as list

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.all_tiles:
            queue = collections.deque([(start_tile, 0)])
            visited = {start_tile}
            self.distances[(start_tile, start_tile)] = 0

            while queue:
                current_tile, dist = queue.popleft()

                # If a tile has no adjacency facts, it's an isolated node in the graph
                # BFS from it will only find itself. This is handled correctly.
                if current_tile in self.adjacency:
                    for neighbor in self.adjacency[current_tile]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_tile, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))

        # Store available colors
        self.available_colors = set()
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'available-color' and len(parts) > 1:
                self.available_colors.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        robot_locations = {}
        robot_colors = {}
        current_paintings = {}
        current_clear_tiles = set()
        all_robots = set() # Collect robot names

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                all_robots.add(robot)
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                all_robots.add(robot)
            elif parts[0] == "painted":
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color
            elif parts[0] == "clear":
                tile = parts[1]
                current_clear_tiles.add(tile)

        # 1. Identify unpainted goal tiles and check for wrongly painted tiles
        clear_goal_tiles = set() # Store (tile, color) tuples
        wrongly_painted_tiles_exist = False

        for goal_tile, goal_color in self.goal_paintings.items():
            if goal_tile in current_paintings:
                if current_paintings[goal_tile] != goal_color:
                    # Tile is painted, but with the wrong color. Unsolvable.
                    wrongly_painted_tiles_exist = True
                    break # No need to check further
            elif goal_tile in current_clear_tiles:
                 # Tile needs to be painted and is currently clear
                 clear_goal_tiles.add((goal_tile, goal_color))
            # else: tile is not clear and not painted (shouldn't happen based on domain structure)
            # or tile is not a goal tile (ignore)

        if wrongly_painted_tiles_exist:
            return float('inf') # Return infinity for unsolvable states

        # If all goal tiles are painted correctly, the goal is reached.
        # The heuristic should be 0 only at the goal.
        # Check if all goal paintings are satisfied.
        goal_satisfied = True
        for goal_tile, goal_color in self.goal_paintings.items():
             if goal_tile not in current_paintings or current_paintings[goal_tile] != goal_color:
                 goal_satisfied = False
                 break

        if goal_satisfied:
             return 0

        # 4. Initialize heuristic value
        h = 0

        # 5. Add cost for paint actions
        h += len(clear_goal_tiles)

        # 6. & 7. & 8. Add cost for color changes
        needed_colors = {color for tile, color in clear_goal_tiles}
        # Consider colors held by any robot
        available_robot_colors = {robot_colors.get(robot) for robot in all_robots if robot in robot_colors}

        colors_to_acquire_count = 0
        for needed_color in needed_colors:
            # Check if this needed color is actually available in the domain
            if needed_color not in self.available_colors:
                 # Needed color is not available, problem is likely unsolvable
                 return float('inf')

            # If the needed color is available but no robot currently has it
            if needed_color not in available_robot_colors:
                 colors_to_acquire_count += 1

        h += colors_to_acquire_count

        # 9. Add cost for movement
        movement_cost = 0
        for tile, color in clear_goal_tiles:
            min_dist_to_any_robot_to_any_adjacent_tile = float('inf')

            # Find adjacent tiles for the current goal tile
            adjacent_tiles = self.adjacency.get(tile, set())

            # If a goal tile has no adjacent tiles defined in the graph, it's unreachable for painting.
            # This indicates an unsolvable problem or a malformed instance.
            if not adjacent_tiles:
                 return float('inf')

            # Find the minimum distance from any robot to any adjacent tile
            found_reachable_adjacent = False
            for robot in all_robots:
                if robot in robot_locations: # Ensure robot location is known
                    robot_loc = robot_locations[robot]
                    for adj_tile in adjacent_tiles:
                        # Distance from robot_loc to adj_tile
                        if (robot_loc, adj_tile) in self.distances:
                             dist = self.distances[(robot_loc, adj_tile)]
                             min_dist_to_any_robot_to_any_adjacent_tile = min(min_dist_to_any_robot_to_any_adjacent_tile, dist)
                             found_reachable_adjacent = True

            # If no robot can reach any adjacent tile of this goal tile, it's unreachable.
            if not found_reachable_adjacent or min_dist_to_any_robot_to_any_adjacent_tile == float('inf'):
                 return float('inf') # Should not happen in solvable problems with connected grid and robots placed

            movement_cost += min_dist_to_any_robot_to_any_adjacent_tile

        h += movement_cost

        # 10. Return total heuristic value
        return h
