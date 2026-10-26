from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Remove leading/trailing parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.
    Args can include wildcards '*'.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles
    with the correct colors. It calculates the cost for each unpainted goal tile
    independently and sums these costs. The cost for a single tile is the minimum
    cost for any robot to reach a tile from which the target tile can be painted,
    change color if necessary, and paint it.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - All tiles mentioned in goal conditions or adjacency facts are part of the grid.
    - The 'clear' predicate must be true to paint a tile. The heuristic assumes
      that if a tile needs painting according to the goal, it is either 'clear'
      or painted with the wrong color. If painted with the wrong color, the
      heuristic still counts it as needing painting, implicitly assuming a way
      to clear it or ignoring the dead-end possibility for heuristic calculation.
      (Note: The domain as written makes wrong-colored tiles dead ends for painting.
       A simple sum heuristic might still guide search towards states with fewer
       wrongly painted tiles or closer robots, even if it can't guarantee solvability).
    - Robot movement cost is the shortest path distance on the grid.
    - Color change costs 1 action. Painting costs 1 action.
    - Multiple robots can exist and work in parallel, but the heuristic sums
      the minimum cost per tile across robots, which is a simplification of
      true multi-agent coordination.

    # Heuristic Initialization
    - Build the tile adjacency graph from static 'up', 'down', 'left', 'right' facts.
    - Build a mapping from a tile to the set of tiles from which a robot can paint it,
      based on the directional adjacency facts.
    - Precompute shortest path distances between all pairs of tiles using BFS on the
      adjacency graph.
    - Extract the goal conditions (which tiles need which colors).
    - Extract available colors (not strictly used in calculation but good practice).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that are not currently painted with the correct color.
       A tile needs painting if the goal requires it to be painted with color C,
       and it is currently clear or painted with a different color C'.
    2. If all goal tiles are correctly painted, the heuristic is 0.
    3. For each goal tile `T` that needs to be painted with color `C`:
       a. Determine the minimum cost for *any* robot to paint this tile.
       b. The cost for a single robot `R` to paint tile `T` with color `C` is:
          - Movement cost: Shortest path distance from robot `R`'s current location
            to any tile `X` from which tile `T` can be painted (i.e., where `(dir T X)`
            is true for some direction `dir`).
          - Color change cost: 1 if robot `R` does not currently have color `C`, otherwise 0.
          - Painting cost: 1 action.
          - Total robot cost for tile `T` = Movement cost + Color change cost + Painting cost.
       c. The minimum robot cost for tile `T` is the minimum of the total robot costs
          calculated over all robots.
    4. The total heuristic value is the sum of the minimum robot costs for each
       unsatisfied goal tile. If any goal tile is unreachable by any robot, the
       heuristic returns a large value indicating a likely unsolvable state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Map tile adjacency for movement
        self.adj = {}
        self.all_tiles = set()
        # Map tile_to_paint -> set of tiles_robot_must_be_at
        self.paintable_from = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                pred, tile1, tile2 = parts
                # For movement: tile1 and tile2 are adjacent
                self.adj.setdefault(tile1, set()).add(tile2)
                self.adj.setdefault(tile2, set()).add(tile1)
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)

                # For painting: (dir tile_y tile_x) means robot at tile_x can paint tile_y
                tile_y, tile_x = tile1, tile2 # Assuming (dir Y X) means Y is dir from X
                self.paintable_from.setdefault(tile_y, set()).add(tile_x)


        # Precompute all-pairs shortest paths
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = self._bfs(start_tile)

        # Store goal locations for each tile
        self.goal_painted = {} # tile -> color
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

        # Store available colors (not strictly needed for this heuristic calculation, but good practice)
        self.available_colors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "available-color":
                self.available_colors.add(parts[1])

    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in self.all_tiles}
        if start_node not in self.all_tiles:
             # Should not happen in valid instances, but handle defensively
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            # Handle cases where a tile might be in self.all_tiles but not have entries in self.adj
            for neighbor in self.adj.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify current robot states (location and color)
        robot_states = {} # robot_name -> {'loc': tile, 'color': color}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                pred, robot, obj = parts[0], parts[1], parts[2]
                if pred == "robot-at":
                    robot_states.setdefault(robot, {})['loc'] = obj
                elif pred == "robot-has":
                    robot_states.setdefault(robot, {})['color'] = obj

        # Identify unsatisfied goal tiles
        unsatisfied_goals = {} # tile -> goal_color
        current_painted = {} # tile -> color (if painted)
        # current_clear = set() # Not strictly needed for identifying unsatisfied goals this way

        for fact in state:
            parts = get_parts(fact)
            # if len(parts) == 2 and parts[0] == "clear":
            #     current_clear.add(parts[1])
            if len(parts) == 3 and parts[0] == "painted":
                 current_painted[parts[1]] = parts[2]

        for tile, goal_color in self.goal_painted.items():
            # Check if the tile is painted with the correct color
            if tile not in current_painted or current_painted[tile] != goal_color:
                 # It's either clear, or painted with the wrong color
                 # In either case, the goal for this tile is not met
                 unsatisfied_goals[tile] = goal_color

        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        total_heuristic_cost = 0

        # Calculate cost for each unsatisfied goal tile
        for tile, goal_color in unsatisfied_goals.items():
            min_robot_cost_for_tile = float('inf')

            # Find the minimum cost for any robot to paint this tile
            for robot_name, robot_state in robot_states.items():
                robot_loc = robot_state.get('loc')
                robot_color = robot_state.get('color')

                if robot_loc is None or robot_color is None:
                    # Robot state is incomplete, skip this robot for now
                    continue

                # Movement cost: distance from robot_loc to any tile from which 'tile' can be painted
                min_dist_to_paint_loc = float('inf')
                # Ensure the tile can be painted from somewhere and robot_loc is in our distance map
                if tile in self.paintable_from and robot_loc in self.distances:
                    for paint_loc in self.paintable_from[tile]:
                        if paint_loc in self.distances[robot_loc]: # Ensure paint_loc is in the distance map from robot_loc
                             min_dist_to_paint_loc = min(min_dist_to_paint_loc, self.distances[robot_loc][paint_loc])

                # If no valid paint location is reachable by this robot
                if min_dist_to_paint_loc == float('inf'):
                    continue # This robot cannot paint this tile

                # Color change cost
                color_cost = 1 if robot_color != goal_color else 0

                # Painting cost
                paint_cost = 1

                # Total cost for this robot for this tile
                robot_cost_for_tile = min_dist_to_paint_loc + color_cost + paint_cost

                min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost_for_tile)

            # Add the minimum cost for this tile to the total heuristic
            # If no robot can paint this tile, min_robot_cost_for_tile remains inf.
            # Summing inf will result in inf.
            total_heuristic_cost += min_robot_cost_for_tile

        # If total_heuristic_cost is still inf (e.g., no robots, or no reachable paint locations for any goal),
        # return a large number or inf.
        if total_heuristic_cost == float('inf'):
             # This state is likely unsolvable or very far from goal
             return 1000000 # Return a large finite number instead of inf

        return total_heuristic_cost
