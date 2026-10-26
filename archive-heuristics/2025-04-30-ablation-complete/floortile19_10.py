# Assume this import works in the target environment
from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Utility function for BFS distance calculation
from collections import deque

def bfs_distance_to_targets(start_node, target_nodes, graph):
    """
    Find the shortest distance from start_node to any node in target_nodes using BFS.
    Returns float('inf') if no target is reachable.
    """
    # If start_node is one of the targets, distance is 0 moves.
    if start_node in target_nodes:
        return 0

    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft()

        # Check neighbors
        for neighbor in graph.get(current_node, []):
            # If a neighbor is a target node, we found the shortest path to a target.
            if neighbor in target_nodes:
                return dist + 1 # Distance is current distance + 1 for the move to neighbor
            # If neighbor hasn't been visited, add it to the queue
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    # If the queue is empty and no target was reached, targets are unreachable
    return float('inf')


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles
    with their required colors. It sums the estimated cost for each individual
    unsatisfied goal tile, considering the cost of painting, clearing the tile
    if occupied, and getting the closest robot with the correct color adjacent
    to the tile.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' facts.
    - Robots can move between adjacent tiles. The 'clear' precondition for
      movement is relaxed for distance calculation in the heuristic.
    - Painting a tile requires a robot with the correct color to be adjacent
      to the tile, and the tile must be 'clear'.
    - Painting makes a tile not 'clear'.
    - Changing color costs 1 action and requires the new color to be 'available'.
    - Solvable problems imply that tiles needing painting are either 'clear'
      or occupied by a robot. Tiles painted with the wrong color are not
      expected in solvable initial states for tiles that are goals.
    - The grid is connected, allowing movement between any two tiles (ignoring 'clear' for distance calculation relaxation).

    # Heuristic Initialization
    - Extracts goal conditions.
    - Builds the grid adjacency graph from static 'up', 'down', 'left', 'right' facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:

    1. Identify all goal facts of the form `(painted tile color)` that are not
       true in the current state. These are the unsatisfied goal tiles.
    2. If there are no unsatisfied goal tiles, the heuristic is 0 (goal state).
    3. Identify the current location and color of each robot from the state.
    4. For each unsatisfied goal tile `T` requiring color `C`:
       a. Initialize the cost for this tile's painting task to 0.
       b. Add 1 to the cost for the `paint` action itself.
       c. Check if tile `T` is currently `clear` in the state by looking for
          the fact `(clear T)`. If this fact is not present, the tile is not
          `clear`. Assuming solvable problems, if a tile needs painting and
          is not clear, it must be occupied by a robot. Add 1 to the cost for
          this tile, representing the action needed for the occupying robot
          to move off and make the tile clear.
       d. Find the minimum cost among all robots to get one ready to paint tile `T`
          with color `C`. This involves:
          i. For each robot `R`:
             - Calculate the cost to change `R`'s color to `C`. This is 1 if `R`
               currently has a different color than `C`, and 0 otherwise.
             - Calculate the shortest distance from `R`'s current location to
               any tile adjacent to `T` using BFS on the grid graph. This distance
               represents the minimum number of move actions required. The 'clear'
               precondition for movement is relaxed in this distance calculation
               to provide a simple distance estimate.
             - The cost for robot `R` to be ready is the color change cost plus
               the movement distance.
          ii. The minimum of these costs over all robots is the minimum cost to
              get *a* robot ready to paint tile `T`.
       e. Add this minimum robot readiness cost to the cost for tile `T`.
       f. If no robot can reach an adjacent tile (BFS returns infinity), add a
          large penalty (e.g., 1000) to the heuristic, indicating this tile
          is likely unreachable in a reasonable plan.
    5. The total heuristic value is the sum of the costs calculated for each
       unsatisfied goal tile.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the grid adjacency graph from static facts.
        """
        super().__init__(task)

        # Build the grid adjacency graph from static facts
        self.grid_adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            # Look for adjacency facts: (direction tile1 tile2)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                t1, t2 = parts[1], parts[2]
                # Add bidirectional edges as movement is possible in both directions
                self.grid_adj.setdefault(t1, set()).add(t2)
                self.grid_adj.setdefault(t2, set()).add(t1)

        # Store goal facts for easy access and checking
        self.goal_facts = set(self.goals)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # 1. Identify unsatisfied goal tiles and their required colors
        unpainted_goals = {} # {tile: color}
        # Iterate through goal facts to find painted goals
        for goal_fact in self.goal_facts:
             g_parts = get_parts(goal_fact)
             # Check if it's a painted goal fact and if it's not in the current state
             if g_parts and g_parts[0] == 'painted' and len(g_parts) == 3:
                 goal_tile, goal_color = g_parts[1], g_parts[2]
                 if goal_fact not in state:
                     unpainted_goals[goal_tile] = goal_color

        # 2. If no unsatisfied goals, we are in a goal state
        if not unpainted_goals:
            return 0

        # 3. Identify robot locations and colors
        robot_info = {} # {robot: {'location': tile, 'color': color}}
        # We need to find all robots first to ensure we have entries for all
        robots_in_state = set()
        for fact in state:
             f_parts = get_parts(fact)
             if f_parts:
                 if f_parts[0] == 'robot-at' and len(f_parts) == 3:
                     robots_in_state.add(f_parts[1])
                 elif f_parts[0] == 'robot-has' and len(f_parts) == 3:
                      robots_in_state.add(f_parts[1])

        # Initialize robot info structure
        for robot in robots_in_state:
             robot_info[robot] = {'location': None, 'color': None}

        # Populate robot info from state facts
        for fact in state:
            f_parts = get_parts(fact)
            if f_parts:
                if f_parts[0] == 'robot-at' and len(f_parts) == 3:
                    robot, location = f_parts[1], f_parts[2]
                    if robot in robot_info: # Ensure robot was identified
                        robot_info[robot]['location'] = location
                elif f_parts[0] == 'robot-has' and len(f_parts) == 3:
                    robot, color = f_parts[1], f_parts[2]
                    if robot in robot_info: # Ensure robot was identified
                        robot_info[robot]['color'] = color

        # 4. Calculate heuristic contribution for each unpainted goal tile
        for tile, required_color in unpainted_goals.items():
            # Cost of the paint action itself
            cost_to_paint_action = 1

            # Cost to clear the tile if occupied by a robot
            # Check if '(clear tile)' is NOT in the state
            cost_to_clear_tile = 0
            if '(clear ' + tile + ')' not in state:
                 # Assuming solvable problems, if not clear and needs painting,
                 # it must be occupied by a robot. That robot needs to move off.
                 cost_to_clear_tile = 1

            # Find tiles adjacent to the target tile (where robot must be to paint)
            adjacent_tiles_for_painting = self.grid_adj.get(tile, [])

            # If the tile has no adjacent tiles in the graph, it's isolated.
            # This should not happen in valid grid problems, but handle defensively.
            if not adjacent_tiles_for_painting:
                 # This tile cannot be painted. Problem likely unsolvable.
                 # Add a large penalty.
                 h += 1000 # Arbitrary large penalty
                 continue # Move to the next unpainted goal tile

            # Minimum cost to get *any* robot ready to paint this tile
            min_cost_get_robot_ready = float('inf')

            # Iterate through all robots to find the best one for this tile
            for robot, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                # A robot must have a location and color to be considered
                if robot_location is None or robot_color is None:
                    continue

                # Cost to change color if needed
                cost_change_color = 1 if robot_color != required_color else 0

                # Find shortest distance from robot_location to any tile in adjacent_tiles_for_painting
                # This is a multi-target BFS.
                dist_robot_to_adj_tile = bfs_distance_to_targets(robot_location, adjacent_tiles_for_painting, self.grid_adj)

                # If the robot can reach an adjacent tile
                if dist_robot_to_adj_tile != float('inf'):
                    cost_for_this_robot = cost_change_color + dist_robot_to_adj_tile
                    min_cost_get_robot_ready = min(min_cost_get_robot_ready, cost_for_this_robot)

            # If no robot can reach an adjacent tile for this goal tile
            if min_cost_get_robot_ready == float('inf'):
                 # This tile is unreachable by any robot. Add a large penalty.
                 h += 1000 # Arbitrary large penalty
            else:
                 # Add the total estimated cost for this tile
                 h += cost_to_clear_tile + cost_to_paint_action + min_cost_get_robot_ready

        return h
