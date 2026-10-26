from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact_string):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    fact_string = fact_string.strip()
    if not fact_string or not fact_string.startswith('(') or not fact_string.endswith(')'):
        return [] # Return empty list for invalid format
    # Remove outer parentheses and split by whitespace
    return fact_string[1:-1].split()

# Assuming Heuristic base class is imported as 'Heuristic'
# from heuristics.heuristic_base import Heuristic


class floortileHeuristic: # Inherit from Heuristic if available in the environment
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal
    tiles with their target colors. It calculates the minimum cost for each
    unpainted goal tile independently, considering the closest robot, the
    distance to an adjacent tile, and the need to change color, and sums
    these minimum costs. If any goal tile is unreachable by any robot,
    the heuristic returns infinity.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates
      between tiles.
    - The actions 'move_up', 'move_down', 'move_left', 'move_right' allow
      movement between adjacent tiles defined by the static predicates.
    - The 'paint_...' actions require the robot to be at a tile adjacent
      to the target tile, have the correct color, and the target tile must be clear.
    - Unpainted goal tiles are assumed to be clear. If a goal tile is not clear
      (e.g., painted with the wrong color), the problem is likely unsolvable
      within the defined actions, and the heuristic will return infinity.
    - All available colors are listed in the static facts.

    # Heuristic Initialization
    - Parses static facts to build the grid graph (adjacency list) based on
      'up', 'down', 'left', 'right' predicates.
    - Identifies all unique tile names involved in connectivity.
    - Precomputes all-pairs shortest path distances between tiles using BFS.
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts of the form `(painted T C)`.
    2. Identify which of these goal facts are *not* satisfied in the current state. These are the unpainted goal tiles.
    3. For each unpainted goal tile `T` that needs color `C`:
       a. Find the set of tiles adjacent to `T` using the precomputed graph.
       b. Calculate the minimum cost for *any* robot to paint tile `T` with color `C`.
          - Initialize `min_cost_T` to infinity.
          - For each robot `r`:
            - Get the robot's current location `robot_loc` and color `robot_color` from the state.
            - Calculate the minimum grid distance from `robot_loc` to *any* tile adjacent to `T`. Let this be `min_dist_to_adj`. This uses the precomputed distances.
            - If `min_dist_to_adj` is infinity (robot cannot reach any adjacent tile), this robot cannot paint `T`.
            - Otherwise, the cost for robot `r` to reach an adjacent tile is `min_dist_to_adj`.
            - The cost to paint is 1 action.
            - If the robot's current color `robot_color` is None (robot has no color) or is not `C`, an additional `change_color` action (cost 1) is needed.
            - The total cost for robot `r` to paint `T` is `min_dist_to_adj + 1 + (1 if robot_color is None or robot_color != C else 0)`.
            - Update `min_cost_T = min(min_cost_T, cost_for_robot)`.
       c. If after checking all robots, `min_cost_T` is still infinity, it means this goal tile is unreachable. The state is likely unsolvable, so return infinity immediately.
       d. Otherwise, add `min_cost_T` to the total heuristic value.
    4. The total heuristic value is the sum of `min_cost_T` for all unpainted goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        self.adj_list = {} # Graph: tile_name -> list of adjacent tile_names
        all_tiles = set()

        # Build adjacency list
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                # Fact is (direction tile_A tile_B) meaning tile_A is direction of tile_B
                # Movement action move_direction moves from ?x to ?y where (direction ?y ?x)
                # So, if (up tile_A tile_B) is true, moving up from tile_B goes to tile_A.
                # This means tile_A is adjacent to tile_B, and vice-versa.
                dir_pred, tile_y, tile_x = parts # ?y is direction of ?x
                
                # Add tiles to the set of all tiles
                all_tiles.add(tile_x)
                all_tiles.add(tile_y)

                # Initialize adjacency list entries if they don't exist
                if tile_x not in self.adj_list:
                    self.adj_list[tile_x] = []
                if tile_y not in self.adj_list:
                    self.adj_list[tile_y] = []
                
                # Add adjacency in both directions, avoiding duplicates
                if tile_y not in self.adj_list[tile_x]:
                     self.adj_list[tile_x].append(tile_y)
                if tile_x not in self.adj_list[tile_y]:
                     self.adj_list[tile_y].append(tile_x)

        # Ensure all tiles found have an entry in adj_list, even if empty (isolated)
        for tile in all_tiles:
             if tile not in self.adj_list:
                 self.adj_list[tile] = []

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        # Iterate through tiles that are part of the graph connectivity
        for start_tile in self.adj_list:
            self.distances[start_tile] = self._bfs(start_tile)

    def _bfs(self, start_tile):
        """Performs BFS from start_tile to find distances to all other tiles."""
        # Initialize distances for all tiles known from static facts (part of the graph)
        distances = {tile: float('inf') for tile in self.adj_list}
        
        # If the start_tile is not in the graph (e.g., an object exists but is not connected)
        # distances will remain infinity for all nodes. This is correct.
        if start_tile not in distances:
             return distances

        distances[start_tile] = 0
        queue = deque([start_tile])

        while queue:
            current_tile = queue.popleft()

            # Check if current_tile has neighbors in the graph
            if current_tile in self.adj_list:
                for neighbor in self.adj_list[current_tile]:
                    # Ensure neighbor is also a known tile in the graph
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_tile] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Extract robot information
        robot_locations = {}
        robot_colors = {}
        robots = set()
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                robots.add(robot)
            elif len(parts) == 3 and parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                robots.add(robot) # Ensure robot is added even if only color is known initially

        # Extract currently painted tiles
        current_painted = {} # (tile -> color) mapping for painted tiles
        for fact in state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == 'painted':
                 tile, color = parts[1], parts[2]
                 current_painted[tile] = color

        total_heuristic = 0

        # Iterate through goal conditions to find unpainted goal tiles
        for goal_fact in self.goals:
            goal_parts = get_parts(goal_fact)
            if len(goal_parts) == 3 and goal_parts[0] == 'painted':
                goal_tile, goal_color = goal_parts[1], goal_parts[2]

                # Check if the goal tile is already painted with the correct color
                if current_painted.get(goal_tile) == goal_color:
                    continue # This goal is satisfied

                # This tile needs to be painted with goal_color
                # Calculate minimum cost for any robot to paint this tile

                min_cost_for_tile = float('inf')

                # Find tiles adjacent to the goal tile
                # Ensure goal_tile is in the graph before looking up neighbors
                adjacent_tiles = self.adj_list.get(goal_tile, [])

                # If the goal tile itself is not in the graph, it's unreachable
                if goal_tile not in self.adj_list:
                     return float('inf') # Goal tile is isolated/unreachable

                # If the goal tile has no adjacent tiles, it cannot be painted
                if not adjacent_tiles:
                     return float('inf') # Goal tile has no paintable locations

                for robot in robots:
                    robot_loc = robot_locations.get(robot) # Get current location
                    robot_color = robot_colors.get(robot) # Get current color

                    # If robot location is unknown or not in the graph, it cannot paint
                    if robot_loc is None or robot_loc not in self.distances:
                         continue # Cannot use this robot

                    # Find minimum distance from robot_loc to any adjacent tile
                    min_dist_to_adj = float('inf')
                    
                    # Iterate through adjacent tiles of the goal tile
                    for adj_tile in adjacent_tiles:
                         # Ensure adj_tile is a known tile and reachable from robot_loc
                         if adj_tile in self.distances[robot_loc]:
                            min_dist_to_adj = min(min_dist_to_adj, self.distances[robot_loc][adj_tile])

                    # If robot can reach an adjacent tile
                    if min_dist_to_adj != float('inf'):
                        # Cost is moves + paint action (cost 1)
                        cost_for_robot = min_dist_to_adj + 1

                        # Add cost for changing color if needed
                        # Check if robot_color is known and different from goal_color
                        if robot_color is None or robot_color != goal_color:
                            # Assume changing to goal_color is possible if it's an available color
                            # (available-color is static, so goal colors are always available)
                            cost_for_robot += 1 # change_color action cost

                        min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

                # If min_cost_for_tile is still infinity after checking all robots,
                # it means no robot can reach an adjacent tile to paint this goal tile.
                # This state is likely unsolvable.
                if min_cost_for_tile == float('inf'):
                     return float('inf') # Found an unreachable goal tile

                total_heuristic += min_cost_for_tile

        return total_heuristic
