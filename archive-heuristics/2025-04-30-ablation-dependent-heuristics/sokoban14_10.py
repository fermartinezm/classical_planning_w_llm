import math
from collections import deque

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing two components:
    1. The sum of shortest path distances for each box from its current location
       to its assigned goal location. This estimates the minimum number of pushes
       required for all boxes, assuming the robot is always in position.
    2. The shortest path distance from the robot's current location to the nearest
       box that is not yet at its goal location. This estimates the cost for the
       robot to reach a box it needs to push.
    The heuristic returns 0 if the goal is reached. It returns infinity if any
    box cannot reach its goal or if the robot cannot reach any box that needs
    pushing, indicating a likely unsolvable state or a state from which the goal
    is unreachable through standard moves/pushes on the static graph.

    Assumptions:
    - The PDDL domain follows the structure provided (predicates at-robot, at, adjacent, clear; actions move, push).
    - Locations are represented as strings (e.g., 'loc_1_1').
    - The goal is specified as a conjunction of (at box goal_location) facts for all boxes.
    - The adjacency information in the static facts defines a graph of connected locations.
    - Solvable instances have finite heuristic values.
    - The heuristic calculates distances on the static graph defined by 'adjacent' predicates, ignoring dynamic obstacles (other boxes, robot) for distance calculations.

    Heuristic Initialization:
    The constructor parses the static facts to build an adjacency graph representing
    the connections between locations. It also parses the goal facts to determine
    the target location for each box. The set of all locations mentioned in
    adjacent facts and goal facts is also identified. These structures (adjacency
    graph, goal locations, all locations) are stored for use in the heuristic
    computation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state satisfies the goal conditions using `self.task.goal_reached(state)`. If yes, the heuristic value is 0.
    2.  Parse the current state (`state`) to extract the robot's location (`robot_loc`) and the location of each box (`box_locations` dictionary).
    3.  Build a dictionary mapping each location to its shortest path distance from the robot's current location using Breadth-First Search (BFS) on the precomputed static adjacency graph (`self.adj_graph`). Store this in `robot_distances`.
    4.  Initialize the total box-goal distance component (`h_boxes`) to 0.
    5.  Create a list `boxes_not_at_goal` containing the names of all boxes that are not currently at their assigned goal locations according to `self.goal_locations`.
    6.  Iterate through each box name in `self.goal_locations`:
        a.  Get the box's current location (`current_loc`) from `box_locations` and its goal location (`goal_loc`) from `self.goal_locations`.
        b.  If `current_loc` is not equal to `goal_loc`:
            i.  Calculate the shortest path distance from `current_loc` to `goal_loc` using BFS on `self.adj_graph`. Store this in `box_distances`.
            ii. Retrieve the distance `dist_box_goal = box_distances.get(goal_loc, float('inf'))`.
            iii. If `dist_box_goal` is infinity, it means the goal location is unreachable from the box's current location on the static graph. This state is likely unsolvable or requires moving other objects first in a way not captured by simple distance. Return `float('inf')`.
            iv. Add `dist_box_goal` to `h_boxes`.
    7.  Initialize the robot-to-nearest-box distance component (`h_robot`) to 0.
    8.  If the `boxes_not_at_goal` list is not empty:
        a.  Initialize `min_robot_dist_to_box` to infinity.
        b.  Iterate through each box name in `boxes_not_at_goal`:
            i.  Get the box's current location (`box_loc`) from `box_locations`.
            ii. Retrieve the distance from the robot to this box location `dist_robot_to_box = robot_distances.get(box_loc, float('inf'))`.
            iii. Update `min_robot_dist_to_box = min(min_robot_dist_to_box, dist_robot_to_box)`.
        c.  If `min_robot_dist_to_box` is infinity, it means the robot cannot reach any box that needs pushing on the static graph. Return `float('inf')`.
        d.  Set `h_robot = min_robot_dist_to_box`.
    9.  The final heuristic value is the sum `h_boxes + h_robot`.
    """

    def __init__(self, task):
        self.task = task
        self.adj_graph = {}
        self.goal_locations = {}
        self.all_locations = set()
        self._parse_static_and_goals()

    def _parse_static_and_goals(self):
        """Parses static facts and goal facts to build graph and goal locations."""
        # Standard directions and their opposites
        opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # Build adjacency graph and collect all locations
        for fact_str in self.task.static:
            # Robust parsing: strip parentheses and split by space
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'adjacent' and len(args) == 3:
                l1, l2, direction = args[0], args[1], args[2]
                self.all_locations.add(l1)
                self.all_locations.add(l2)

                if l1 not in self.adj_graph:
                    self.adj_graph[l1] = {}
                self.adj_graph[l1][l2] = direction

                # Add the reverse direction edge if the direction is standard
                rev_direction = opposite_dir.get(direction)
                if rev_direction:
                    if l2 not in self.adj_graph:
                        self.adj_graph[l2] = {}
                    self.adj_graph[l2][l1] = rev_direction
                # Note: If direction is not one of the standard ones, no reverse edge is added.
                # This is fine as the graph only represents explicitly stated adjacencies.

        # Parse goal locations
        for fact_str in self.task.goals:
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                box, loc = args[0], args[1]
                self.goal_locations[box] = loc
                # Add goal locations to all_locations if they weren't in adjacent facts
                self.all_locations.add(loc)


    def _bfs(self, start_loc, adj_graph, all_locations):
        """Computes shortest path distances from start_loc to all other locations."""
        distances = {loc: float('inf') for loc in all_locations}

        # Ensure start_loc is a valid location in our known set
        if start_loc not in all_locations:
             # This start location is not part of the graph derived from adjacent facts
             # or goal facts. It's an isolated location. No path exists to any other
             # location in the graph.
             # Returning the distances dictionary initialized with infinity for all
             # known locations is a safe default.
             return distances # All distances remain infinity

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            curr_loc = queue.popleft()

            # Check if curr_loc has neighbors in the graph
            # It should if it's in all_locations and the graph is built correctly,
            # unless it's a goal location that was isolated or a start location
            # that was isolated.
            if curr_loc in adj_graph:
                for next_loc in adj_graph[curr_loc]:
                    if distances[next_loc] == float('inf'):
                        distances[next_loc] = distances[curr_loc] + 1
                        queue.append(next_loc)

        return distances

    def __call__(self, state):
        """Computes the heuristic value for the given state."""
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        robot_loc = None
        box_locations = {} # box_name -> location

        # Parse current state
        for fact_str in state:
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at-robot' and len(args) == 1:
                robot_loc = args[0]
            elif predicate == 'at' and len(args) == 2:
                box, loc = args[0], args[1]
                box_locations[box] = loc

        # Should always have a robot location in a valid state
        if robot_loc is None:
             # This indicates an invalid state representation or problem.
             # Return infinity as it's likely unsolvable from here.
             return float('inf')

        # Compute distances from robot to all locations
        robot_distances = self._bfs(robot_loc, self.adj_graph, self.all_locations)

        h_boxes = 0
        boxes_not_at_goal = []

        # Calculate box-goal distances
        # Iterate through goal locations to ensure we consider all required boxes
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box) # Get box location from current state

            # If a box required by the goal is not present in the current state,
            # something is fundamentally wrong with the state representation.
            if current_loc is None:
                 return float('inf') # Indicate unsolvable

            if current_loc != goal_loc:
                boxes_not_at_goal.append(box)

                # Calculate distance from current box location to its goal
                # Need BFS from the box's current location
                box_distances = self._bfs(current_loc, self.adj_graph, self.all_locations)
                dist_box_goal = box_distances.get(goal_loc, float('inf'))

                if dist_box_goal == float('inf'):
                    # Box cannot reach its goal location on the static graph
                    return float('inf')

                h_boxes += dist_box_goal

        # Calculate robot-to-nearest-box distance
        h_robot = 0
        if boxes_not_at_goal:
            min_robot_dist_to_box = float('inf')
            for box in boxes_not_at_goal:
                box_loc = box_locations[box]
                # Use the precomputed robot_distances
                dist_robot_to_box = robot_distances.get(box_loc, float('inf'))
                min_robot_dist_to_box = min(min_robot_dist_to_box, dist_robot_to_box)

            if min_robot_dist_to_box == float('inf'):
                 # Robot cannot reach any box that needs pushing on the static graph
                 return float('inf')
            h_robot = min_robot_dist_to_box
        # If boxes_not_at_goal is empty, h_robot remains 0, which is correct
        # as the robot doesn't need to reach any box if they are all at goals.

        return h_boxes + h_robot
