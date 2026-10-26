# Imports needed
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Utility functions
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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all goal tiles with their correct colors. It sums the estimated minimum cost for each unpainted goal tile independently. The cost for a single tile is estimated as the minimum cost for any robot to reach an adjacent tile, change color if needed, and paint the tile.

    # Assumptions
    - The grid structure defined by up/down/left/right predicates is static.
    - Robots can move between adjacent clear tiles.
    - Robots can change color if they currently hold a color and the target color is available.
    - Tiles needing painting are initially clear unless already painted with the goal color.
    - Tiles painted with the wrong color make the problem unsolvable.
    - Robots must hold a color (`robot-has`) to paint or change color. Robots without a `robot-has` fact cannot paint.
    - The cost of move, paint, and change_color actions is 1.

    # Heuristic Initialization
    - Parses static facts to build the grid graph (adjacency list) based on up/down/left/right predicates.
    - Computes all-pairs shortest paths on the grid graph using BFS.
    - Extracts the set of goal tiles and their required colors from the goal conditions.
    - (Optional but good practice) Stores available colors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location and color of each robot.
    2. Initialize the total heuristic cost to 0.
    3. For each tile that needs to be painted according to the goal:
        a. Check if the tile is already painted with the correct color in the current state. If yes, this tile contributes 0 to the heuristic.
        b. Check if the tile is painted with a *wrong* color. If yes, the problem is unsolvable from this state; return infinity.
        c. If the tile needs painting (is currently clear or not painted with the goal color):
            i. Find all tiles adjacent to the goal tile using the precomputed grid graph.
            ii. Initialize the minimum cost to paint this tile to infinity.
            iii. For each robot:
                - Get the robot's current location and color.
                - If the robot does not currently hold a color (`robot-has` is false, i.e., `robot_color` is None), it cannot paint this tile. Skip this robot.
                - Calculate the minimum distance from the robot's current location to *any* tile adjacent to the goal tile, using the precomputed shortest paths.
                - If no adjacent tile is reachable by this robot, this robot cannot paint the tile. Skip this robot.
                - If reachable, calculate the cost for this robot:
                    - Cost = (minimum distance to adjacent tile) + 1 (for the paint action).
                    - If the robot's current color is not the required goal color, add 1 (for the change_color action).
                - Update the minimum cost to paint this tile with the minimum cost found so far across all robots.
            iv. If the minimum cost for this tile is still infinity (no capable robot can reach an adjacent tile), the problem is likely unsolvable from this state; return infinity.
            v. Add the calculated minimum cost for this tile to the total heuristic.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build grid graph (adjacency list) from static facts
        self.adj = {}
        self.all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Ensure fact has enough parts before accessing indices 1 and 2
                if len(parts) >= 3:
                    t1, t2 = parts[1], parts[2]
                    self.adj.setdefault(t1, set()).add(t2)
                    self.adj.setdefault(t2, set()).add(t1)
                    self.all_tiles.add(t1)
                    self.all_tiles.add(t2)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.all_tiles:
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            self.distances[(start_tile, start_tile)] = 0 # Distance to self is 0

            while q:
                current_tile, dist = q.popleft()

                for neighbor in self.adj.get(current_tile, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_tile, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # Store goal tiles and required colors
        self.goal_tiles = {} # tile -> color
        for g in self.goals:
            # Ensure goal fact has enough parts before accessing indices
            g_parts = get_parts(g)
            if len(g_parts) >= 3 and g_parts[0] == "painted":
                t, c = g_parts[1], g_parts[2]
                self.goal_tiles[t] = c

        # Store available colors (optional, mainly for debugging/validation)
        self.available_colors = {get_parts(fact)[1] for fact in static_facts if match(fact, "available-color", "*")}
        # Could add a check here if any goal color is not in available_colors, but assuming valid problems.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check for wrongly painted goal tiles (unsolvable state)
        for goal_tile, goal_color in self.goal_tiles.items():
            # Check if the tile is painted with *any* color
            for fact in state:
                fact_parts = get_parts(fact)
                if len(fact_parts) >= 3 and fact_parts[0] == "painted" and fact_parts[1] == goal_tile:
                    c_painted = fact_parts[2]
                    if c_painted != goal_color:
                        # Tile is painted with the wrong color, problem is unsolvable from here.
                        return math.inf # Use infinity for unsolvable states

        # Identify current robot locations and colors
        current_robot_locations = {} # robot -> tile
        current_robot_colors = {} # robot -> color (None if robot-has is false)
        robots = set() # Collect all robot names
        for fact in state:
            fact_parts = get_parts(fact)
            if len(fact_parts) >= 3:
                if fact_parts[0] == "robot-at":
                    r, loc = fact_parts[1], fact_parts[2]
                    current_robot_locations[r] = loc
                    robots.add(r)
                elif fact_parts[0] == "robot-has":
                    r, c = fact_parts[1], fact_parts[2]
                    current_robot_colors[r] = c # Store the color
                    robots.add(r)
                # Note: We don't explicitly track 'free-color', we infer it if 'robot-has' is missing.


        total_heuristic = 0

        # Calculate cost for each unpainted goal tile
        for goal_tile, goal_color in self.goal_tiles.items():
            # Check if the goal is already satisfied
            goal_satisfied = f"(painted {goal_tile} {goal_color})" in state

            if not goal_satisfied:
                # This tile needs to be painted
                min_cost_for_tile = math.inf

                # Find tiles adjacent to the goal tile
                adjacent_tiles = self.adj.get(goal_tile, set())

                if not adjacent_tiles:
                     # Goal tile is isolated or not in the grid graph - likely unsolvable
                     # This case should ideally be caught in __init__ if the goal tile isn't in self.all_tiles
                     # but checking here provides robustness.
                     return math.inf

                # Consider each robot
                for robot in robots:
                    robot_loc = current_robot_locations.get(robot)
                    robot_color = current_robot_colors.get(robot) # None if robot-has is false

                    # A robot must be located somewhere and must have a color to paint
                    if robot_loc is None or robot_color is None:
                         continue # This robot cannot paint, skip it for this tile

                    # Find minimum distance from robot_loc to any adjacent tile of goal_tile
                    cost_to_reach_adjacent = math.inf
                    for adj_t in adjacent_tiles:
                        if (robot_loc, adj_t) in self.distances:
                             cost_to_reach_adjacent = min(cost_to_reach_adjacent, self.distances[(robot_loc, adj_t)])

                    if cost_to_reach_adjacent == math.inf:
                        # This robot cannot reach any tile adjacent to the goal tile
                        continue # Skip this robot for this tile

                    # Calculate cost for this robot to paint the tile
                    cost_for_this_robot = cost_to_reach_adjacent + 1 # +1 for paint action

                    # Add cost for changing color if needed
                    if robot_color != goal_color:
                         # We assume the goal_color is available if the problem is solvable.
                         # The change_color action requires (available-color ?c2).
                         # This is a static fact checked implicitly by assuming solvable problems.
                         cost_for_this_robot += 1 # +1 for change_color action

                    # Update minimum cost for this tile
                    min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)

                # If after checking all robots, no capable robot can paint this tile, it's unsolvable.
                if min_cost_for_tile == math.inf:
                     return math.inf

                total_heuristic += min_cost_for_tile

        return total_heuristic
