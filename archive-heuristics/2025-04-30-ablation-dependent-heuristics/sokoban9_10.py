import collections

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing two components:
    1. The sum of shortest path distances for each box from its current location
       to its designated goal location, calculated on the grid graph ignoring
       dynamic obstacles (other boxes, robot). This represents the minimum
       number of push actions required for the boxes.
    2. The minimum shortest path distance for the robot from its current location
       to any location adjacent to any box that is not yet at its goal. This
       represents the cost for the robot to get into a position to push a box.

    Assumptions:
    - The PDDL domain follows the structure provided (predicates at-robot, at, clear, adjacent).
    - Location names are strings that can be used as graph nodes.
    - The goal state specifies the target location for each box using `(at boxX locY)` facts.
    - The grid graph defined by `adjacent` facts is connected such that goal locations are reachable from initial box locations and robot locations can reach positions adjacent to boxes (unless the instance is unsolvable).
    - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A). The PDDL examples show this pattern.

    Heuristic Initialization:
    The heuristic is initialized with the planning task. It parses the static
    `adjacent` facts to build an adjacency list representation of the grid graph.
    It also parses the goal facts to determine the target location for each box.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Extract the current location of the robot and all boxes from the state.
    2.  Identify which boxes are not yet at their goal locations. If all boxes
        are at their goals, the heuristic is 0.
    3.  For each box not at its goal:
        a.  Find its designated goal location (parsed during initialization).
        b.  Calculate the shortest path distance from the box's current location
            to its goal location using BFS on the pre-built graph. This BFS
            treats only walls (implicit in the graph structure) as obstacles,
            ignoring other boxes and the robot. This distance is the minimum
            number of pushes required for this box *if the path were clear*.
    4.  Sum the minimum push distances calculated for all boxes not at their goals.
        This is the 'box_cost'.
    5.  Calculate the 'robot_cost':
        a.  Identify the current robot location.
        b.  Determine the set of locations occupied by boxes. These locations
            act as obstacles for the robot's movement BFS.
        c.  Initialize a minimum robot distance `min_robot_dist` to infinity.
        d.  For each box that is not yet at its goal:
            i.  Get the box's current location `L_box`.
            ii. Find all locations `L_adj` that are adjacent to `L_box` in the graph.
            iii. For each `L_adj`:
                 - Calculate the shortest path distance from the robot's current
                   location to `L_adj` using BFS. This BFS considers the
                   `robot_obstacles` (all box locations) determined in step 5b.
                 - Update `min_robot_dist` with the minimum distance found so far.
        e.  If `min_robot_dist` is still infinity (robot cannot reach any adjacent
            cell of any box that needs moving), the state might be a dead end.
            Return infinity.
    6.  The total heuristic value is the sum of `box_cost` and `min_robot_dist`.
    """

    def __init__(self, task):
        # Build the graph from adjacent facts
        self.graph = collections.defaultdict(list)
        self.locations = set()

        for fact in task.static:
            fact_parts = fact.strip("()").split()
            if fact_parts[0] == 'adjacent':
                l1, l2 = fact_parts[1], fact_parts[2]
                # Add bidirectional edges for BFS
                self.graph[l1].append(l2)
                self.graph[l2].append(l1)
                self.locations.add(l1)
                self.locations.add(l2)

        # Parse goal facts to find box-goal mapping
        self.box_goals = {}
        for goal in task.goals:
            goal_parts = goal.strip("()").split()
            if goal_parts[0] == 'at' and len(goal_parts) == 3:
                box_name, goal_loc = goal_parts[1], goal_parts[2]
                self.box_goals[box_name] = goal_loc

    def bfs_distance(self, start, end, graph, obstacles=None):
        """
        Calculates the shortest path distance between start and end locations
        in the graph, avoiding obstacles.
        Returns float('inf') if end is unreachable from start.
        """
        if obstacles is None:
            obstacles = set()

        if start == end:
            return 0

        # If start or end is explicitly listed as an obstacle, it's unreachable
        # unless start == end (handled above).
        if start in obstacles or end in obstacles:
             return float('inf')

        queue = collections.deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Check neighbors from the graph
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    # A neighbor is reachable if it's not an obstacle and not visited
                    if neighbor not in obstacles and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # If the queue is empty and the end was not reached
        return float('inf')

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        robot_loc = None
        box_locations = {} # {box_name: location}

        # Parse state facts
        for fact in state:
            fact_parts = fact.strip("()").split()
            if fact_parts[0] == 'at-robot':
                robot_loc = fact_parts[1]
            elif fact_parts[0] == 'at' and len(fact_parts) == 3:
                box_name, loc = fact_parts[1], fact_parts[2]
                box_locations[box_name] = loc
            # 'clear' facts are not needed for this heuristic calculation

        # Check if goal is reached
        goal_reached = True
        for box, goal_loc in self.box_goals.items():
            if box not in box_locations or box_locations[box] != goal_loc:
                goal_reached = False
                break
        if goal_reached:
            return 0

        # Calculate box cost (sum of distances to goals)
        box_cost = 0
        boxes_to_move = []
        for box, current_loc in box_locations.items():
            # Only consider boxes that have a goal location defined and are not there yet
            if box in self.box_goals and current_loc != self.box_goals[box]:
                boxes_to_move.append(box)
                goal_loc = self.box_goals[box]
                # BFS for box ignores dynamic obstacles (other boxes, robot)
                dist = self.bfs_distance(current_loc, goal_loc, self.graph, obstacles=set())
                if dist == float('inf'):
                    # Box cannot reach its goal - unsolvable state
                    return float('inf')
                box_cost += dist

        # If no boxes need moving but goal wasn't reached, something is wrong
        # (e.g., robot not at goal in a domain where robot must be at goal).
        # In Sokoban, only box locations matter for the goal.
        # This case should ideally not happen if goal_reached check is correct.
        if not boxes_to_move:
             return 0 # Should have been caught by goal_reached check

        # Calculate robot cost (min distance to a push position)
        min_robot_dist = float('inf')
        # Obstacles for robot BFS are locations occupied by ANY box
        robot_obstacles = set(box_locations.values())

        if robot_loc is None:
             # Robot location is unknown - invalid state
             return float('inf')

        for box in boxes_to_move:
            l_box = box_locations[box]
            # Find locations adjacent to the box
            if l_box in self.graph:
                for l_adj in self.graph[l_box]:
                    # Calculate distance from robot to this adjacent location
                    # Robot BFS considers all box locations as obstacles
                    dist = self.bfs_distance(robot_loc, l_adj, self.graph, obstacles=robot_obstacles)
                    min_robot_dist = min(min_robot_dist, dist)

        # If robot cannot reach any adjacent cell of any box that needs moving
        if min_robot_dist == float('inf'):
             # This state might be a dead end for the robot
             return float('inf')

        # Total heuristic is sum of box costs and minimum robot cost
        return box_cost + min_robot_dist

