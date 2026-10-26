from heuristics.heuristic_base import Heuristic
from collections import deque
import math

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing, for each
    robot, the minimum estimated cost for that robot to paint *any* of the
    currently unsatisfied goal tiles it is capable of painting. The estimated
    cost for a robot to paint a specific tile includes the movement cost to
    reach a tile adjacent to the target tile, the cost to change color if
    needed, and the paint action cost.

    Assumptions:
    - The problem involves a grid of tiles connected by up/down/left/right
      relations defined in the static facts.
    - Tile names are consistent identifiers used in connectivity, state, and
      goal facts.
    - Robots start with a color, and all colors required by the goal are
      available (as per domain definition and examples).
    - Tiles needing painting are initially clear. The heuristic does not
      explicitly model the 'clear' predicate dependencies beyond the initial
      state assumption and the requirement for the paint action.
    - The cost of each action is 1.

    Heuristic Initialization:
    In the constructor, the heuristic preprocesses the static information:
    1. It identifies all unique tile names present in the static connectivity
       predicates (up, down, left, right). It also includes tiles mentioned
       in the goal facts to ensure the graph covers all relevant locations.
    2. It builds an adjacency list representation of the tile grid graph based
       on the static connectivity facts. The graph is treated as undirected.
    3. It computes the shortest path distance between all pairs of tiles in the
       grid graph using Breadth-First Search (BFS) starting from each tile.
       These distances are stored in a dictionary for quick lookup during
       heuristic computation.
    4. It stores the set of goal facts and available colors from the task's
       static information.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify all goal facts (painted tiles) that are not satisfied in the
       current state. If no unsatisfied goals, the heuristic is 0.
    2. Parse the current locations and colors of all robots from the state.
    3. Initialize the total heuristic value to 0.
    4. For each robot:
       a. Determine the robot's current location and color.
       b. Initialize the minimum estimated cost for this robot to paint *any*
          single unpainted goal tile to infinity.
       c. Iterate through each unsatisfied goal fact (representing a target
          tile and the required color):
          i. Check if the robot is capable of painting with the required color.
             This is true if the robot currently has the color, or if it has
             any color and the required color is an available color in the
             domain.
          ii. If the robot can paint with the required color:
              - Calculate the cost to change color: 0 if the robot already has
                the required color, 1 otherwise.
              - Find all tiles adjacent to the target tile using the
                precomputed graph.
              - Find the minimum shortest path distance from the robot's current
                location to any of the tiles adjacent to the target tile, using
                the precomputed distances. If the target tile has no adjacent
                tiles or the robot's location is not in the graph, this distance
                will be infinity.
              - Calculate the estimated cost for this robot to paint this
                specific target tile:
                (minimum distance to adjacent tile) + (color change cost) + 1
                (for the paint action).
              - Update the minimum cost for this robot if this calculated cost
                is lower than the current minimum for this robot.
       d. If the minimum cost found for this robot is not infinity (meaning it
          can reach and paint at least one unpainted goal tile), add this
          minimum cost to the total heuristic value.
    5. Return the total heuristic value. If the total heuristic is 0 but there
       are still unsatisfied goals, it implies no robot can reach/paint any
       remaining goal, indicating an unsolvable state from here, so return
       infinity.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static

        self.available_colors = set()
        self.tiles = set()
        self.adj = {}

        # 1. & 2. Build graph and collect tiles/colors from static facts
        for fact_str in self.static_facts:
            parsed_fact = self._parse_fact(fact_str)
            predicate = parsed_fact[0]
            args = parsed_fact[1:]

            if predicate in ['up', 'down', 'left', 'right'] and len(args) == 2:
                tile1, tile2 = args
                self.tiles.add(tile1)
                self.tiles.add(tile2)
                # Add bidirectional edges
                self.adj.setdefault(tile1, []).append(tile2)
                self.adj.setdefault(tile2, []).append(tile1)

            elif predicate == 'available-color' and len(args) == 1:
                self.available_colors.add(args[0])

        # Ensure all tiles mentioned in goals are included, even if isolated (unlikely)
        for goal_fact_str in self.goals:
             if goal_fact_str.startswith('(painted '):
                 parsed_goal = self._parse_fact(goal_fact_str)
                 if len(parsed_goal) == 3:
                     self.tiles.add(parsed_goal[1]) # Add the tile name

        # Ensure all tiles in adj list are in tiles set (should be true by setdefault)
        self.tiles.update(self.adj.keys())
        for neighbors in self.adj.values():
             self.tiles.update(neighbors)

        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for start_tile in self.tiles:
            self.distances[start_tile] = self._bfs(start_tile)

    def _parse_fact(self, fact_string):
        """Parses a fact string into a list of strings."""
        # Remove outer parentheses and split by space
        return fact_string[1:-1].split()

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: math.inf for node in self.tiles}
        if start_node not in self.tiles:
             # Should not happen if self.tiles is populated correctly
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Get neighbors, handling potential missing nodes in adj (shouldn't happen if tiles are consistent)
            neighbors = self.adj.get(current_node, [])

            for neighbor in neighbors:
                # Ensure neighbor is a valid tile in our set
                if neighbor in self.tiles and distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def _get_distance(self, tile1, tile2):
        """Looks up precomputed distance between two tiles."""
        # Handle cases where tile might not be in the graph (e.g., error in PDDL)
        if tile1 not in self.distances or tile2 not in self.distances.get(tile1, {}):
             # This should ideally not happen in valid problems
             # print(f"Warning: Distance requested for unknown tile pair: {tile1}, {tile2}")
             return math.inf
        return self.distances[tile1][tile2]

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.
        """
        state = node.state

        # 1. Identify unsatisfied goal facts
        unsatisfied_goals = []
        for goal_fact_str in self.goals:
            if goal_fact_str not in state:
                # Ensure it's a painted goal fact and parse it
                if goal_fact_str.startswith('(painted '):
                     parsed_goal = self._parse_fact(goal_fact_str)
                     if len(parsed_goal) == 3:
                         unsatisfied_goals.append((parsed_goal[1], parsed_goal[2])) # Store as (tile, color) tuple

        # If no unsatisfied goals, we are in a goal state
        if not unsatisfied_goals:
            return 0

        # 2. Parse robot locations and colors
        robot_locations = {}
        robot_colors = {}

        for fact_str in state:
            parsed_fact = self._parse_fact(fact_str)
            predicate = parsed_fact[0]
            args = parsed_fact[1:]

            if predicate == 'robot-at' and len(args) == 2:
                robot, location = args
                robot_locations[robot] = location
            elif predicate == 'robot-has' and len(args) == 2:
                robot, color = args
                robot_colors[robot] = color

        # If no robots are present or located (shouldn't happen in valid problems),
        # and there are unsatisfied goals, it's likely unsolvable from here.
        # Return infinity.
        if not robot_locations:
             return math.inf


        # 3. Initialize total heuristic
        total_heuristic = 0

        # 4. For each robot, find its minimum cost to paint any unsatisfied tile
        for robot, current_location in robot_locations.items():
            current_color = robot_colors.get(robot) # Can be None if robot-has fact is missing (unlikely)

            min_robot_cost = math.inf

            # Iterate through each unsatisfied goal (tile, color)
            for target_tile, target_color in unsatisfied_goals:

                # Check if the robot can paint with the target color
                # Assumes robot always has *a* color if robot-has fact exists,
                # and available_colors covers all colors that can be acquired.
                # If robot has no color fact, assume it cannot change color.
                can_paint = (current_color == target_color) or (current_color is not None and target_color in self.available_colors)

                if can_paint:
                    # Cost to get the target color
                    color_change_cost = 0 if current_color == target_color else 1

                    # Find adjacent tiles to the target tile
                    adjacent_tiles = self.adj.get(target_tile, [])

                    min_dist_to_adj = math.inf
                    # Find minimum distance from robot's location to any adjacent tile
                    for adj_tile in adjacent_tiles:
                        dist = self._get_distance(current_location, adj_tile)
                        min_dist_to_adj = min(min_dist_to_adj, dist)

                    # If target tile has adjacent tiles and robot location is valid
                    if min_dist_to_adj != math.inf:
                         # Estimated cost for this robot to paint this specific tile
                         # Cost = Movement + Color Change + Paint Action
                         robot_tile_cost = min_dist_to_adj + color_change_cost + 1

                         # Update the minimum cost for this robot
                         min_robot_cost = min(min_robot_cost, robot_tile_cost)

            # 4d. Add the minimum cost for this robot to the total heuristic
            # Only add if the robot can actually contribute (i.e., can reach/paint something)
            if min_robot_cost != math.inf:
                total_heuristic += min_robot_cost

        # 5. Return the total heuristic value
        # If total_heuristic is still 0 but there are unsatisfied goals,
        # it means no robot can reach/paint any remaining goal. This implies
        # an unsolvable state from here. Return infinity.
        # Note: total_heuristic can be 0 if there are robots but they can only
        # paint tiles that are already painted correctly. This case is covered
        # by the unsatisfied_goals check at the beginning.
        # If total_heuristic is 0 here, it means either no robots could paint
        # any *unsatisfied* goal (e.g., wrong colors, disconnected graph) or
        # there were no robots (handled above).
        if total_heuristic == 0 and unsatisfied_goals:
             return math.inf

        return total_heuristic
