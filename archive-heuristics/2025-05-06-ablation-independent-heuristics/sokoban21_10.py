from heuristics.heuristic_base import Heuristic
from task import Task
import re
from collections import deque
import math

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        Estimates the cost to reach the goal by summing the shortest path
        distances for each box to its goal location on the full grid, and
        adding the shortest path distance for the robot to reach any location
        adjacent to any box that needs moving, calculated on the currently
        clear grid.

    Assumptions:
        - Location names are strings following 'loc_X_Y' pattern (though parsing doesn't rely on this pattern, just the string name).
        - Adjacent facts define an undirected graph connectivity.
        - Goal is defined by (at boxX locY) facts for all boxes.
        - State includes (at-robot loc), (at box loc), and (clear loc) facts.
        - The grid defined by adjacent facts is consistent.

    Heuristic Initialization:
        - Parses adjacent facts from task.static to build the full grid graph (`self.adj_graph`).
        - Parses goal facts from task.goals to store target locations for boxes (`self.box_goals`).
        - Stores all location names (`self.all_locations`).

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state from the node.
        2. Extract the robot's current location, current location for each box, and the set of clear locations from the state facts.
        3. Identify the set of boxes that are not yet at their goal locations by comparing current box locations with the stored goal locations.
        4. If this set is empty, the current state is a goal state, and the heuristic value is 0.
        5. Calculate the sum of shortest path distances for each box in the set (from step 3) to its respective goal location. This distance is computed using BFS on the full grid graph (`self.adj_graph`) built during initialization. If any box's goal is unreachable on the full grid, the heuristic is infinity (dead end).
        6. Build the 'clear' grid graph for robot movement based on the current state. The nodes in this graph are locations that are currently marked as clear or the robot's current location. The edges are adjacent relations between these traversable nodes, derived from the full grid graph.
        7. Calculate the minimum shortest path distance for the robot from its current location to *any* location that is adjacent to *any* box that needs moving (from step 3). This distance is computed using BFS on the 'clear' grid graph built in step 6. If the robot cannot reach any such adjacent location, the heuristic is infinity (dead end).
        8. The final heuristic value is the sum of the total box distances (step 5) and the minimum robot distance to a box (step 7).
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.box_goals = {}
        self.adj_graph = {} # Full grid graph: location -> list of adjacent locations
        self.all_locations = set()

        # Parse goal facts
        for goal_fact in self.task.goals:
            predicate, args = self._parse_fact(goal_fact)
            if predicate == 'at' and len(args) == 2:
                box_name, goal_loc = args
                self.box_goals[box_name] = goal_loc

        # Parse static adjacent facts to build the full grid graph
        for static_fact in self.task.static:
            predicate, args = self._parse_fact(static_fact)
            if predicate == 'adjacent' and len(args) == 3:
                loc1, loc2, direction = args
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
                if loc1 not in self.adj_graph:
                    self.adj_graph[loc1] = []
                if loc2 not in self.adj_graph:
                    self.adj_graph[loc2] = []
                # Add edges in both directions
                self.adj_graph[loc1].append(loc2)
                self.adj_graph[loc2].append(loc1)

        # Remove duplicates from adj_graph values (shouldn't be necessary with careful parsing, but safe)
        for loc in self.adj_graph:
             self.adj_graph[loc] = list(set(self.adj_graph[loc]))


    def _parse_fact(self, fact_string):
        """Helper to parse PDDL fact strings."""
        # Remove surrounding parentheses and split by spaces
        fact_string = fact_string.strip('()')
        parts = fact_string.split()
        if not parts: # Handle empty string case
            return None, []
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _shortest_path_bfs(self, start, end, graph):
        """BFS to find shortest path distance."""
        if start == end:
            return 0
        # Check if start or end are valid nodes in the graph
        if start not in graph or end not in graph:
             # This can happen if the location is not part of the connected grid
             # defined by adjacent facts, which might indicate an issue with the PDDL
             # or a location that is unreachable from the main grid.
             # For robustness, treat as unreachable.
             return math.inf

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # graph[current_loc] is a list of neighbor locations
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return math.inf # Not reachable

    def __call__(self, node):
        state = node.state

        # 2. Extract robot and box locations, and clear locations
        loc_robot = None
        box_locations = {}
        clear_locations = set()

        for fact in state:
            predicate, args = self._parse_fact(fact)
            if predicate == 'at-robot' and len(args) == 1:
                loc_robot = args[0]
            elif predicate == 'at' and len(args) == 2:
                box_name, loc = args
                box_locations[box_name] = loc
            elif predicate == 'clear' and len(args) == 1:
                clear_locations.add(args[0])

        # Ensure robot location is known (should always be in a valid state)
        if loc_robot is None:
             return math.inf # Indicate invalid state

        # 3. Identify boxes not at goal
        boxes_to_move = [b for b, loc in box_locations.items() if self.box_goals.get(b) != loc]

        # 4. Goal check
        if not boxes_to_move:
            return 0 # All boxes are at their goals

        # 5. Calculate sum of box distances to goals (on full grid)
        sum_box_distances = 0
        for box_name in boxes_to_move:
            current_loc = box_locations[box_name]
            goal_loc = self.box_goals.get(box_name)
            # Ensure box has a goal and is a known location in the graph
            if goal_loc is None or current_loc not in self.adj_graph or goal_loc not in self.adj_graph:
                 # This box or its goal is not part of the connected grid
                 return math.inf # Indicate dead end or invalid problem

            dist = self._shortest_path_bfs(current_loc, goal_loc, self.adj_graph)
            if dist == math.inf:
                # Box cannot reach its goal even on the full grid -> dead end
                return math.inf
            sum_box_distances += dist

        # 6. Build the 'clear' grid graph for robot movement
        # Robot can move to clear locations or its current location
        robot_traversable_locations = clear_locations.copy()
        robot_traversable_locations.add(loc_robot)

        clear_adj_graph = {}
        # Initialize clear_adj_graph with traversable nodes
        for loc in robot_traversable_locations:
             clear_adj_graph[loc] = []

        # Add edges between traversable nodes based on the full graph
        for loc1 in robot_traversable_locations:
             # Get neighbors from the full graph
             if loc1 in self.adj_graph:
                 for loc2 in self.adj_graph[loc1]:
                     # Add edge if neighbor is also traversable
                     if loc2 in robot_traversable_locations:
                         clear_adj_graph[loc1].append(loc2)

        # 7. Calculate minimum robot distance to a location adjacent to any box_to_move
        min_robot_distance = math.inf
        robot_can_reach_any_push_pos = False

        for box_name in boxes_to_move:
            loc_b = box_locations[box_name]
            # Find locations adjacent to loc_b from the full graph
            adjacent_to_box = self.adj_graph.get(loc_b, [])

            for adj_loc in adjacent_to_box:
                 # The robot needs to reach adj_loc to push the box at loc_b
                 # Check if adj_loc is traversable by the robot in the current state
                 if adj_loc in robot_traversable_locations:
                     # Check if loc_robot and adj_loc are valid nodes in the clear graph
                     # This check is redundant if robot_traversable_locations was used to build clear_adj_graph keys
                     # but doesn't hurt. The BFS itself handles nodes not in the graph.
                     dist = self._shortest_path_bfs(loc_robot, adj_loc, clear_adj_graph)
                     min_robot_distance = min(min_robot_distance, dist)
                     if dist != math.inf:
                         robot_can_reach_any_push_pos = True

        # If robot cannot reach any push position for any box that needs moving, it's a dead end
        if not robot_can_reach_any_push_pos and boxes_to_move:
             return math.inf

        # 8. Combine distances
        # The heuristic is the sum of the total box movement cost (estimated by sum_box_distances)
        # and the robot's initial positioning cost (estimated by min_robot_distance).
        h_value = sum_box_distances + min_robot_distance

        # Heuristic should be 0 only at goal.
        # If boxes_to_move is not empty, sum_box_distances > 0 (since goal_loc != current_loc implies dist > 0).
        # min_robot_distance >= 0.
        # So h_value > 0 if not goal.

        return h_value
