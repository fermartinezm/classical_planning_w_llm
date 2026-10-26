# Required imports
from collections import deque
import math # For float('inf')

class sokobanHeuristic:
    """
    Summary:
    This heuristic estimates the cost to reach a goal state in the Sokoban domain
    by summing the estimated costs for each box that is not yet at its goal.
    The estimated cost for a single box is the sum of two components:
    1. The minimum number of pushes required to move the box from its current
       location to its goal location, calculated as the shortest path distance
       on the location graph, avoiding locations occupied by other boxes.
    2. The minimum number of robot moves required to position the robot
       correctly to make the *first* push towards the box's goal, calculated
       as the shortest path distance for the robot, avoiding locations
       occupied by *any* box.
    If any required path (for box or robot) is unreachable, a large penalty
    is added to the heuristic, guiding the search away from dead ends.
    This heuristic is non-admissible and designed for greedy best-first search.

    Assumptions:
    - The locations are nodes in a graph defined by the 'adjacent' predicates.
    - Every box in the initial state has a corresponding goal location specified
      in the task's goal condition.
    - The graph defined by 'adjacent' predicates is traversable for both
      the robot and boxes (when pushed), considering dynamic obstacles.
    - The heuristic uses a large finite value (1000) to penalize states
      where a box or the robot cannot reach a required location.

    Heuristic Initialization:
    The constructor `__init__` parses the static facts from the task definition.
    It builds two graph representations of the locations:
    - `self.graph`: Maps a location and a direction to the adjacent location
      (e.g., `graph[loc_A]['right'] = loc_B` if `(adjacent loc_A loc_B right)` is true).
    - `self.reverse_graph`: Maps a location and a direction to the location
      behind it in that direction (e.g., `reverse_graph[loc_B]['right'] = loc_A`
      if `(adjacent loc_A loc_B right)` is true). This is used to find the
      required robot position for a push.
    It also identifies the goal location for each box by parsing the goal facts.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Initialize the total heuristic value `total_h` to 0.
    2.  Parse the current state to find the robot's current location (`robot_loc`)
        and the current location of each box (`box_locations`).
    3.  Create a set of all locations currently occupied by boxes (`all_box_locations`).
    4.  Iterate through each box and its current location (`box`, `loc_b`) in
        `box_locations`.
    5.  Retrieve the goal location (`goal_b`) for the current box from the
        pre-parsed `self.box_goals`.
    6.  If the box does not have a goal or is already at its goal location,
        continue to the next box.
    7.  **Box Path Calculation:**
        a.  Perform a Breadth-First Search (BFS) on the location graph (`self.graph`)
            starting from `loc_b` to find the shortest path to `goal_b`.
        b.  The BFS considers locations in `all_box_locations` *excluding* `loc_b`
            as obstacles (`box_obstacles`). A box cannot be pushed into a location
            occupied by another box.
        c.  The BFS returns the shortest distance (`dist_box`) and a predecessor map
            (`box_predecessors`) to reconstruct the path.
        d.  If `goal_b` is unreachable for the box (`dist_box` is infinity), add a
            large penalty to `total_h` and continue to the next box.
    8.  **Robot Positioning Calculation:**
        a.  If `dist_box` is greater than 0, the box needs to move. Determine the
            first location (`next_loc`) the box must move to along a shortest path
            from `loc_b` to `goal_b` using the `box_predecessors` map.
        b.  Find the direction (`push_dir`) from `loc_b` to `next_loc` using
            `self.graph`.
        c.  Determine the required robot push location (`robot_push_loc`) which is
            the location behind `loc_b` in the `push_dir` using `self.reverse_graph`.
        d.  If `robot_push_loc` cannot be determined (e.g., no location behind `loc_b`
            in that direction), add a large penalty to `total_h` and continue.
        e.  Perform a BFS on the location graph (`self.graph`) starting from
            `robot_loc` to find the shortest path to `robot_push_loc`.
        f.  This BFS considers locations in `all_box_locations` as obstacles. The
            robot cannot move into any location occupied by a box.
        g.  The BFS returns the shortest distance (`dist_robot`).
        h.  If `robot_push_loc` is unreachable for the robot (`dist_robot` is infinity),
            add a large penalty to `total_h` and continue.
    9.  Add the sum `dist_box + dist_robot` to `total_h`.
    10. After iterating through all boxes, return the final `total_h`.
    """
    def __init__(self, task):
        self.graph = {}
        self.reverse_graph = {}
        self.box_goals = {}

        # Parse static facts
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.graph.setdefault(l1, {})[direction] = l2
                self.reverse_graph.setdefault(l2, {})[direction] = l1

        # Parse goal facts
        for goal in task.goals:
            parts = goal.strip('()').split()
            if parts[0] == 'at':
                box, loc = parts[1], parts[2]
                self.box_goals[box] = loc

    def shortest_path_bfs(self, start, goal, graph, obstacles):
        """
        Performs BFS to find the shortest path distance and predecessors.
        Nodes are locations, edges are defined by the graph.
        Avoids locations in the obstacles set.
        Returns (distance, predecessor_map) or (float('inf'), None) if unreachable.
        """
        queue = deque([(start, 0)])
        visited = {start}
        predecessor = {start: None}

        # If start is an obstacle and not the goal, it's unreachable
        if start in obstacles and start != goal:
             return math.inf, None

        while queue:
            curr_loc, dist = queue.popleft()

            if curr_loc == goal:
                return dist, predecessor

            # Check if curr_loc is in graph (might not be if it's an isolated location)
            if curr_loc not in graph:
                 continue

            for direction, next_loc in graph[curr_loc].items():
                if next_loc not in obstacles and next_loc not in visited:
                    visited.add(next_loc)
                    predecessor[next_loc] = curr_loc
                    queue.append((next_loc, dist + 1))

        return math.inf, None # Goal not reachable

    def __call__(self, state):
        robot_loc = None
        box_locations = {}

        # Parse state facts
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1] != 'robot': # Assuming 'at' only applies to boxes
                 box, loc = parts[1], parts[2]
                 box_locations[box] = loc

        total_h = 0
        large_penalty = 1000 # Penalty for unreachable goals/push locations

        # Build set of all box locations once
        all_box_locations = set(box_locations.values())

        for box, loc_b in box_locations.items():
            goal_b = self.box_goals.get(box)

            # If box has no goal or is already at goal, skip
            if goal_b is None or loc_b == goal_b:
                continue

            # Obstacles for box pathfinding: locations occupied by other boxes
            box_obstacles = all_box_locations - {loc_b}

            # 1. Calculate shortest path distance for the box
            dist_box, box_predecessors = self.shortest_path_bfs(loc_b, goal_b, self.graph, box_obstacles)

            if dist_box == math.inf:
                total_h += large_penalty
                continue

            # 2. Find the first step of the box path and required robot push location
            next_loc = goal_b
            if dist_box > 0:
                 # Walk back from goal until we find the node whose predecessor is loc_b
                 # This node is the second node in the path (the one after loc_b)
                 # If dist_box is 1, the predecessor of goal_b is loc_b, so next_loc remains goal_b
                 current_node_in_path = goal_b
                 # Ensure current_node_in_path is in predecessors map before accessing it
                 while current_node_in_path in box_predecessors and box_predecessors[current_node_in_path] != loc_b and current_node_in_path != loc_b:
                     current_node_in_path = box_predecessors[current_node_in_path]
                 next_loc = current_node_in_path # The node after loc_b is the one whose predecessor is loc_b

            if next_loc is None or next_loc == loc_b: # Path reconstruction failed or dist_box was 0 (already handled)
                 # This case should ideally not be reached if dist_box > 0 and finite
                 total_h += large_penalty
                 continue

            # Find direction of the first step: loc_b -> next_loc
            push_dir = None
            # Need to iterate through directions from loc_b to find which one leads to next_loc
            for direction, adj_loc in self.graph.get(loc_b, {}).items():
                if adj_loc == next_loc:
                    push_dir = direction
                    break

            if push_dir is None: # Should not happen if next_loc is reachable from loc_b
                 total_h += large_penalty
                 continue

            # Find required robot push location: the location behind loc_b in push_dir
            # Need to check if loc_b exists in reverse_graph and if push_dir exists for loc_b
            robot_push_loc = self.reverse_graph.get(loc_b, {}).get(push_dir)

            if robot_push_loc is None: # Cannot push box in that direction (no space behind)
                 total_h += large_penalty
                 continue

            # 3. Calculate shortest path distance for the robot
            # Obstacles for robot pathfinding: locations occupied by *any* box
            robot_obstacles = all_box_locations

            dist_robot, _ = self.shortest_path_bfs(robot_loc, robot_push_loc, self.graph, robot_obstacles)

            if dist_robot == math.inf:
                total_h += large_penalty
                continue

            # Add costs for this box
            total_h += dist_box + dist_robot

        return total_h
