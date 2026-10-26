import collections
import re
import math

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        two components:
        1. The sum of the minimum shortest path distances for each box
           that is not yet at a goal location, to reach any of its designated
           goal locations.
        2. The minimum shortest path distance from the robot's current location
           to any box that is not yet at a goal location.

        The shortest path distances are computed using Breadth-First Search (BFS)
        on the grid graph derived from the 'adjacent' static facts.

    Assumptions:
        - The PDDL domain uses 'location', 'direction', 'box' types.
        - Locations are connected via 'adjacent' facts, forming a graph.
        - Robot location is given by '(at-robot ?l)'.
        - Box locations are given by '(at ?b ?l)'.
        - Goal is specified by '(at ?b ?l)' facts for boxes.
        - The grid defined by 'adjacent' facts is traversable by the robot
          (ignoring 'clear' for robot movement distance calculation, which is a simplification).
        - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A).
          The BFS assumes this for simplicity in building the undirected graph.
        - The heuristic does not explicitly check for complex deadlocks (e.g., boxes
          in non-goal corners or against walls blocking other boxes). Unsolvable
          states where a box cannot reach *any* goal or the robot cannot reach
          *any* box are assigned an infinite heuristic value.

    Heuristic Initialization:
        - Parses the static facts ('adjacent') to build an undirected graph
          representation of the grid locations using an adjacency list.
        - Parses the goal facts ('at' for boxes) to create a mapping from
          each box to its set of target goal locations.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state using task.goal_reached(). If yes, the heuristic is 0.
        2. Extract the robot's current location from the state facts. If not found, return infinity.
        3. Extract the current location of each box from the state facts.
        4. Initialize the total heuristic value (sum of box-to-goal distances) to 0.
        5. Initialize the minimum distance from the robot to any box that needs moving to infinity.
        6. Get the set of all boxes that have goal locations specified in the task.
        7. For each box in this set:
           a. Get the box's current location from the state. If the box is not in the state, return infinity (unsolvable).
           b. Get the set of goal locations for this box.
           c. Check if the box is currently at any of its goal locations.
           d. If the box is *not* at a goal:
              i. Compute the shortest distance from the box's current location to *any* of its goal locations using BFS on the grid graph.
              ii. If the distance is infinite (box cannot reach any goal), the state is likely unsolvable; return infinity immediately.
              iii. Add this distance to the total box-to-goal distance.
              iv. Compute the shortest distance from the robot's current location to the box's current location using BFS.
              v. Update the minimum robot-to-box distance found so far among boxes not at goal.
        8. If there were any boxes that needed moving (i.e., the count of boxes not at goal is greater than 0):
           a. If the minimum robot-to-box distance is infinite (robot cannot reach any box that needs moving), the state is likely unsolvable; return infinity.
           b. Add the minimum robot-to-box distance to the total heuristic value.
        9. Return the calculated total heuristic value.
    """
    def __init__(self, task):
        self.task = task
        self.goal_facts = task.goals
        self.static_facts = task.static

        # Graph: location -> list of adjacent locations (for BFS distance)
        self.adj_list = collections.defaultdict(list)
        # box_name -> list of goal locations
        self.box_goals = collections.defaultdict(list)

        # Build the adjacency graph from static facts
        self._build_adjacency_graph()

        # Parse goal facts to map boxes to their goal locations
        self._parse_goal_facts()

    def _build_adjacency_graph(self):
        """
        Builds the adjacency list representation of the grid from static facts.
        Assumes adjacency is symmetric for BFS distance calculation.
        """
        # Regex to extract locations from adjacent facts like '(adjacent loc_1_1 loc_1_2 right)'
        adj_pattern = re.compile(r"\(adjacent (\S+) (\S+) \S+\)")
        for fact in self.static_facts:
            match = adj_pattern.match(fact)
            if match:
                loc1, loc2 = match.groups()
                # Add bidirectional edges for undirected distance graph
                self.adj_list[loc1].append(loc2)
                self.adj_list[loc2].append(loc1)

    def _parse_goal_facts(self):
        """
        Parses goal facts to map boxes to their target locations.
        Only considers '(at ?b ?l)' goals.
        """
        # Regex to extract box and location from goal facts like '(at box1 loc_2_4)'
        at_pattern = re.compile(r"\(at (\S+) (\S+)\)")
        for goal_fact in self.goal_facts:
            match = at_pattern.match(goal_fact)
            if match:
                box, loc = match.groups()
                self.box_goals[box].append(loc)
            # Ignore other types of goal facts if any exist in the domain (none in sokoban.pddl)

    def _bfs_distance(self, start_loc, target_locs):
        """
        Computes the shortest distance from start_loc to any location in target_locs
        using BFS on the adjacency graph (self.adj_list).

        Args:
            start_loc: The starting location string.
            target_locs: A list or set of target location strings.

        Returns:
            The shortest distance (integer), or math.inf if no path exists
            or target_locs is empty or start_loc is None.
        """
        if start_loc is None or not target_locs:
            return math.inf

        # Handle case where start is already a target
        if start_loc in target_locs:
            return 0

        q = collections.deque([(start_loc, 0)])
        visited = {start_loc}

        while q:
            current_loc, dist = q.popleft()

            # Check neighbors
            for neighbor in self.adj_list.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in target_locs:
                        return dist + 1 # Found a target
                    q.append((neighbor, dist + 1))

        return math.inf # No path found to any target

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The heuristic value (integer or math.inf).
        """
        # Check if goal is reached (heuristic is 0 at goal)
        if self.task.goal_reached(state):
            return 0

        # Extract robot and box locations from the current state
        robot_loc = None
        box_locations = {} # box_name -> location

        # Regex to extract info from state facts
        at_robot_pattern = re.compile(r"\(at-robot (\S+)\)")
        at_box_pattern = re.compile(r"\(at (\S+) (\S+)\)")

        for fact in state:
            match_robot = at_robot_pattern.match(fact)
            if match_robot:
                robot_loc = match_robot.group(1)
                # No 'continue' here, need to process all facts

            match_box = at_box_pattern.match(fact)
            if match_box:
                box, loc = match_box.groups()
                box_locations[box] = loc
                # No 'continue' here, need to process all facts

        # If robot location is not found (shouldn't happen in valid states), return inf
        if robot_loc is None:
             return math.inf

        # Heuristic calculation
        total_box_goal_dist = 0
        boxes_not_at_goal_count = 0 # Count boxes that need moving
        min_robot_dist_to_any_box_not_at_goal = math.inf

        # Iterate through all boxes that have goal locations specified in the task.
        # This ensures we consider all boxes relevant to the goal.
        all_boxes_in_goals = set(self.box_goals.keys())

        for box in all_boxes_in_goals:
            current_box_loc = box_locations.get(box) # Get box location from current state
            goal_locs = self.box_goals.get(box, [])

            # If a box required by the goal is not in the current state, it's unsolvable.
            if current_box_loc is None:
                 return math.inf

            is_at_goal = current_box_loc in goal_locs

            if not is_at_goal:
                boxes_not_at_goal_count += 1

                # Component 1: Box-to-goal distance
                dist_box_to_goal = self._bfs_distance(current_box_loc, goal_locs)
                # If a box cannot reach any goal, the state is unsolvable.
                if dist_box_to_goal == math.inf:
                    return math.inf
                total_box_goal_dist += dist_box_to_goal

                # Component 2 part: Calculate robot distance to this specific box
                dist_robot_to_this_box = self._bfs_distance(robot_loc, [current_box_loc])
                # Update minimum robot distance to any box that needs moving
                min_robot_dist_to_any_box_not_at_goal = min(min_robot_dist_to_any_box_not_at_goal, dist_robot_to_this_box)

        # Add the minimum robot distance to any box that needs moving, only if there are boxes to move
        if boxes_not_at_goal_count > 0:
             # If the robot cannot reach any box that needs moving, the state is unsolvable.
             if min_robot_dist_to_any_box_not_at_goal == math.inf:
                 return math.inf
             total_h = total_box_goal_dist + min_robot_dist_to_any_box_not_at_goal
        else:
             # If boxes_not_at_goal_count is 0, it means all boxes required by the goal are at their goal locations.
             # Given the PDDL examples, goals are only box locations. Therefore, if count is 0, the goal is reached.
             # The initial check `if self.task.goal_reached(state): return 0` handles this case.
             # This else block should logically not be reached if the goal is not met.
             # If it is reached, it implies an inconsistency or a problem type not covered.
             # We return 0 here, relying on the initial check to ensure correctness for valid problems.
             total_h = 0

        return total_h
