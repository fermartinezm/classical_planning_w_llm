from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


# Helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # For fixed-arity predicates, the number of parts must match the number of args
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper BFS function to get distances to all reachable clear tiles
def bfs_all_distances(start_tile, clear_tiles, adjacency_list):
    """
    Performs BFS from start_tile to find distances to all reachable tiles,
    moving only through tiles that are in the clear_tiles set.
    Returns a dictionary {tile: distance}.
    """
    distances = {start_tile: 0}
    queue = deque([start_tile])

    # A robot can start *from* a tile that isn't clear (because the robot is on it)
    # but can only move *to* a clear tile. The BFS explores neighbors that are clear.

    while queue:
        current_tile = queue.popleft()
        dist = distances[current_tile]

        neighbors = adjacency_list.get(current_tile, set())
        for neighbor in neighbors:
            # Can only move to a neighbor if it is clear AND not visited yet
            if neighbor in clear_tiles and neighbor not in distances:
                distances[neighbor] = dist + 1
                queue.append(neighbor)

    return distances


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the cost for each unpainted goal tile,
    where the cost for a single tile includes the paint action, the minimum
    movement cost for any robot to reach an adjacent clear tile, and the cost
    for that robot to have the correct color.

    # Assumptions
    - All goal tiles that are not yet painted correctly are currently clear.
    - The grid is connected, allowing movement between tiles unless blocked by occupancy.
    - Robots always have a color (represented by robot-has) if color changes are needed.
    - The cost of any action (move, paint, change_color) is 1.
    - All robots capable of painting are listed with their initial location in the problem's initial state.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted and with which color.
    - Extracts static facts to build the adjacency list representing the grid structure.
    - Identifies all available colors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to find the location and color of each robot, and which tiles are painted or clear.
    2. Identify all goal tiles that are not currently painted with the required color. Let this set of unsatisfied goals be U.
    3. If U is empty, the heuristic is 0 (goal state).
    4. Initialize the total heuristic cost to 0.
    5. For each robot, perform a Breadth-First Search (BFS) starting from its current location, traversing only through tiles that are currently clear. This calculates the minimum movement distance from the robot to every reachable clear tile. Store these distances.
    6. For each unsatisfied goal (tile T, required_color C) in U:
       a. Add 1 to the total cost (for the paint action).
       b. Find all tiles X that are adjacent to T based on the precomputed adjacency list. These are the potential locations a robot must reach to paint T.
       c. For this specific painting task (T, C), calculate the minimum cost for *any* robot R to be able to perform the paint action. This cost for robot R is:
          - The minimum number of moves required for R to get from its current location to *any* tile X adjacent to T, using the precomputed BFS distances.
          - Plus 1 if robot R currently holds a color different from C, and 0 otherwise.
       d. Find the minimum of this cost over all available robots R.
       e. Add this minimum robot cost to the total heuristic cost. If no robot can reach any adjacent tile, the state is likely unsolvable, and the heuristic should return infinity.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal painted tiles
        self.goal_painted_tiles = {}
        for goal in self.goals:
            # Goal facts are typically (painted tile_name color_name)
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_painted_tiles[tile] = color

        # Extract static facts to build adjacency list and get available colors
        self.adjacency_list = {}
        self.available_colors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Adjacency is bidirectional
                t1, t2 = parts[1], parts[2]
                self.adjacency_list.setdefault(t1, set()).add(t2)
                self.adjacency_list.setdefault(t2, set()).add(t1)
            elif match(fact, "available-color", "*"):
                 _, color = parts
                 self.available_colors.add(color)

        # We will collect robot names from the state in __call__

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Parse current state
        robot_locations = {}
        robot_colors = {}
        painted_tiles_state = {}
        current_clear_tiles = set()
        all_robots_in_state = set() # Collect robots present in the current state

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = parts
                robot_locations[robot] = tile
                all_robots_in_state.add(robot)
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = parts
                robot_colors[robot] = color
                all_robots_in_state.add(robot) # Also add robots with colors
            elif match(fact, "painted", "*", "*"):
                _, tile, color = parts
                painted_tiles_state[tile] = color
            elif match(fact, "clear", "*"):
                 _, tile = parts
                 current_clear_tiles.add(tile)

        # Identify unpainted goal tiles
        unpainted_goal_tiles = []
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            is_painted_correctly = painted_tiles_state.get(goal_tile) == goal_color
            if not is_painted_correctly:
                unpainted_goal_tiles.append((goal_tile, goal_color))

        if not unpainted_goal_tiles:
            return 0 # Goal state

        if not all_robots_in_state:
             # Cannot paint if no robots exist in the state
             return float('inf') # Unsolvable

        total_heuristic = 0
        unsolvable = False

        # Calculate distances from each robot to all reachable clear tiles
        robot_distances = {}
        for robot in all_robots_in_state:
            robot_loc = robot_locations.get(robot)
            if robot_loc is None:
                 # Robot exists but its location is unknown? Should not happen in valid state.
                 # Treat as unreachable for now.
                 robot_distances[robot] = {} # Empty distances means unreachable
                 continue

            # BFS starts from robot_loc, moving through clear tiles.
            # The robot_loc itself might not be in current_clear_tiles if the robot is on it.
            # The bfs_all_distances function handles this by adding start_tile to visited initially.
            robot_distances[robot] = bfs_all_distances(robot_loc, current_clear_tiles, self.adjacency_list)


        # Calculate cost for each unpainted goal tile
        for goal_tile, goal_color in unpainted_goal_tiles:
            # Cost for the paint action
            total_heuristic += 1

            # Find adjacent tiles that can serve as paint locations
            adjacent_tiles = self.adjacency_list.get(goal_tile, set())

            min_robot_task_cost = float('inf')

            for robot in all_robots_in_state:
                # Find minimum distance from robot_loc to any adjacent tile of goal_tile
                min_dist_to_adjacent = float('inf')
                distances_from_robot = robot_distances.get(robot, {}) # Get precomputed distances

                for adj_tile in adjacent_tiles:
                    # Check if the adjacent tile is reachable by this robot
                    if adj_tile in distances_from_robot:
                        min_dist_to_adjacent = min(min_dist_to_adjacent, distances_from_robot[adj_tile])

                # If no adjacent tile is reachable by this robot, this robot cannot paint this tile
                if min_dist_to_adjacent == float('inf'):
                    continue

                # Calculate color change cost
                robot_color = robot_colors.get(robot)
                color_cost = 0
                # If robot_color is None, it might have free-color or initial state issue.
                # Assuming solvable instances mean robots needing to paint have a color or can get one.
                # If robot has the wrong color, needs 1 change_color action.
                # Check if robot_color is not None before comparing
                if robot_color is not None and robot_color != goal_color:
                     # Check if the goal_color is even available to change to
                     if goal_color not in self.available_colors:
                          unsolvable = True
                          break # Break robot loop
                     color_cost = 1 # Cost to change color
                # If robot_color is None (e.g., free-color), we assume it cannot change color
                # using the change_color action, making it unable to paint this tile
                # unless goal_color is also None (which is not possible for painted predicate).
                # So if robot_color is None and goal_color is not None, this robot cannot paint.
                elif robot_color is None and goal_color is not None:
                     continue # This robot cannot paint this color


                robot_task_cost = min_dist_to_adjacent + color_cost
                min_robot_task_cost = min(min_robot_task_cost, robot_task_cost)

            if unsolvable:
                 break # Break goal tile loop

            if min_robot_task_cost == float('inf'):
                # No robot can reach any adjacent tile for this goal tile
                unsolvable = True
                break # Break goal tile loop

            total_heuristic += min_robot_task_cost

        if unsolvable:
             return float('inf') # Or a large number indicating unsolvable

        return total_heuristic
