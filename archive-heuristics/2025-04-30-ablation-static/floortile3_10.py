# from heuristics.heuristic_base import Heuristic # Assuming this is available
from fnmatch import fnmatch
from collections import deque, defaultdict
import math # for infinity

# Define a dummy Heuristic base class if not running in the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently clear. It sums the number of unpainted goal tiles,
    the estimated number of color changes needed, and the minimum distance
    the robot needs to travel to reach a position from which it can paint
    at least one of the unpainted goal tiles.

    # Assumptions
    - The problem involves a grid of tiles connected by up/down/left/right relations.
    - Goal tiles are either clear or painted with the correct color in solvable states.
      Tiles painted with the wrong color are considered dead ends and are ignored
      by the heuristic (they don't contribute to pending tasks).
    - The heuristic is designed for a single robot, or calculates cost based on robot1's state.
      (Based on the provided domain file actions which use a single robot parameter ?r).
    - Movement cost between adjacent tiles is 1.
    - Color change cost is 1.
    - Paint action cost is 1.

    # Heuristic Initialization
    - Build the grid graph based on static up/down/left/right predicates.
    - Precompute all-pairs shortest paths on the grid graph using BFS.
    - Identify for each tile, the set of tiles from which it can be painted
      (its adjacent tiles according to the grid structure, based on static facts
       like (up target_tile robot_location)).
    - Store goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location and the color it is holding.
    2. Identify all tiles that are currently clear.
    3. Determine the set of "pending tasks": goal tiles that need to be painted
       with a specific color and are currently clear.
    4. If there are no pending tasks, check if all goals are met. If yes, return 0.
       If no, return infinity (unsolvable from this state, likely due to wrongly painted tiles).
    5. The heuristic value starts with the number of pending tasks (representing
       the minimum number of paint actions required).
    6. Calculate the minimum number of color changes needed: Count the distinct
       colors required by the pending tasks. If the robot's current color is
       one of these required colors, subtract 1 (as the first set of tasks
       can potentially use the current color). Add this number to the heuristic.
    7. Calculate the minimum movement cost: Find the set of tiles from which
       any of the pending goal tiles can be painted. Calculate the shortest
       distance from the robot's current location to *any* tile in this set.
       Add this minimum distance to the heuristic.
    8. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph, computing distances,
        and identifying paint locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph and map tiles to nodes (integers 0..N-1)
        self.tile_to_id = {}
        self.id_to_tile = []
        self.adj = defaultdict(list) # Adjacency list using tile names

        # Collect all tiles and build initial adjacency based on static facts
        all_tiles = set()
        # Also collect paint locations based on static facts
        self.paint_locations = defaultdict(set) # paint_locations[target_tile] = {robot_location1, ...}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                all_tiles.add(tile1)
                all_tiles.add(tile2)
                # The predicates define the relationship (e.g., tile1 is up from tile2)
                # We need bidirectional edges for movement
                self.adj[tile1].append(tile2)
                self.adj[tile2].append(tile1) # Grid is undirected for movement

                # If (Direction Y X) is true, robot at X can paint Y using paint_Direction
                # So X is a paint location for Y
                # parts[1] is the target tile (?y), parts[2] is the robot location (?x)
                self.paint_locations[tile1].add(tile2)


        # Assign IDs to tiles
        for i, tile in enumerate(sorted(list(all_tiles))): # Sort for consistent ID assignment
            self.tile_to_id[tile] = i
            self.id_to_tile.append(tile)

        num_tiles = len(self.id_to_tile)
        self.dist = {} # Store distances: dist[tile1][tile2] = distance

        # Compute all-pairs shortest paths using BFS from each tile
        for start_tile in self.id_to_tile:
            self.dist[start_tile] = {}
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            self.dist[start_tile][start_tile] = 0

            while q:
                current_tile, d = q.popleft()

                # Check if current_tile exists in adj (should always for nodes in id_to_tile)
                # Use .get for safety, though direct access should be fine here
                for neighbor_tile in self.adj.get(current_tile, []):
                    if neighbor_tile not in visited:
                        visited.add(neighbor_tile)
                        self.dist[start_tile][neighbor_tile] = d + 1
                        q.append((neighbor_tile, d + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify robot's current location and color
        robot_location = None
        robot_color = None
        # Assuming robot1 based on instance examples, but domain uses ?r.
        # Let's find *a* robot and its state. If multiple, this picks one arbitrarily.
        # For simplicity, assume robot1 if present, otherwise the first one found.
        robot_name = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                r, loc = get_parts(fact)[1:]
                if robot_name is None or r == 'robot1': # Prioritize robot1 or take first
                    robot_name = r
                    robot_location = loc
            if match(fact, "robot-has", "*", "*"):
                 r, color = get_parts(fact)[1:]
                 if robot_name is None or r == 'robot1': # Prioritize robot1 or take first
                    robot_name = r # Ensure we track the robot whose color we found
                    robot_color = color

        # If robot state is incomplete or no robot found, and paint goals exist, it's unsolvable.
        has_paint_goals = any(match(goal, "painted", "*", "*") for goal in self.goals)
        if (robot_location is None or robot_color is None) and has_paint_goals:
             return float('inf')
        # If no robot and no paint goals, check if other goals are met.
        if robot_location is None and robot_color is None and not has_paint_goals:
             all_goals_met = True
             for goal in self.goals:
                 if goal not in state:
                     all_goals_met = False
                     break
             if all_goals_met:
                 return 0
             else:
                  # Goals not met, no paint goals, no robot. Unsolvable.
                  return float('inf')


        # 2. Identify clear tiles
        clear_tiles = {get_parts(fact)[1] for fact in state if match(fact, "clear", "*")}

        # 3. Determine pending tasks: goal tiles that need painting and are clear
        pending_tasks = set() # Store as (tile, color) tuples
        required_colors = set()
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, target_tile, target_color = get_parts(goal)
                # A task is pending if the goal is not met AND the tile is clear
                # (Assuming wrongly painted tiles are dead ends or don't occur in solvable instances)
                if f"(painted {target_tile} {target_color})" not in state and target_tile in clear_tiles:
                     pending_tasks.add((target_tile, target_color))
                     required_colors.add(target_color)

        # 4. If no pending tasks, check if all goals are met.
        if not pending_tasks:
            all_goals_met = True
            for goal in self.goals:
                if goal not in state:
                    all_goals_met = False
                    break
            if all_goals_met:
                return 0
            else:
                # Some goals are not met (must be non-paint goals, or wrongly painted tiles)
                # If non-paint goals, this heuristic doesn't cover them. Assume paint goals are the main task.
                # If pending_tasks is empty, but goals are not met, it implies wrongly painted tiles.
                return float('inf')


        # 5. Heuristic starts with the number of paint actions
        h = len(pending_tasks)

        # 6. Add cost for color changes
        num_color_changes = len(required_colors) - (1 if robot_color in required_colors else 0)
        h += max(0, num_color_changes) # Ensure non-negative

        # 7. Add cost for movement
        min_dist_to_paint_loc_to_any_target = float('inf')
        found_reachable_paint_loc = False

        for target_tile, _ in pending_tasks:
            # Find paint locations for this target tile
            paint_locs = self.paint_locations.get(target_tile, set())

            for paint_loc in paint_locs:
                # Check if robot_location and paint_loc are valid nodes in our graph
                if robot_location in self.dist and paint_loc in self.dist[robot_location]:
                    dist = self.dist[robot_location][paint_loc]
                    min_dist_to_paint_loc_to_any_target = min(min_dist_to_paint_loc_to_any_target, dist)
                    found_reachable_paint_loc = True
                # else: paint_loc is not reachable from robot_location, ignore this paint_loc

        # If no reachable paint location was found for *any* pending task, it's unsolvable
        if not found_reachable_paint_loc:
            return float('inf')

        h += min_dist_to_paint_loc_to_any_target

        return h
