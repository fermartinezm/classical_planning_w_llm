# Import necessary modules
import collections
import math # For float('inf')

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Remove surrounding parentheses and split by spaces
    parts = fact_string[1:-1].split()
    # Basic cleaning for potential nested parentheses or trailing ones
    parts[0] = parts[0].lstrip('(')
    parts[-1] = parts[-1].rstrip(')')
    return tuple(parts)

# Helper function to get the opposite direction
def get_opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen for valid directions

# Define the heuristic class
class SokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated costs for each box to reach its goal location. The
        estimated cost for a single box is the sum of:
        1. The shortest path distance (number of pushes) for the box from its
           current location to its goal location, considering other boxes as
           obstacles.
        2. The shortest path distance (number of robot moves) for the robot
           from its current location to a position adjacent to the box, from
           which it can push the box towards its goal, considering all boxes
           as obstacles.

    Assumptions:
        - The locations form a graph defined by the 'adjacent' predicates.
        - The goal state specifies a unique target location for each box.
        - Location names are unique identifiers.
        - The graph is connected for relevant locations.
        - The heuristic assumes a greedy strategy where the robot moves to
          position a box for its next push towards the goal.
        - The PDDL structure follows the provided examples.

    Heuristic Initialization:
        The constructor parses the static facts from the task definition.
        It builds an adjacency list representation of the location graph
        based on the 'adjacent' predicates. It also stores the list of
        adjacent relations including directions, which is needed to determine
        pushing positions.
        It parses the goal facts to determine the target location for each box.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value to 0.
        2. Parse the current state to find the robot's location and the
           current location of each box.
        3. Collect the set of locations currently occupied by all boxes.
        4. For each box specified in the goal state (and its corresponding goal location):
            a. Get the box's current location. If the box is not found in the state,
               return infinity as the state is invalid or unsolvable.
            b. If the box is already at its goal location, continue to the next box.
            c. Determine the set of obstacles for the box's movement: these are the
               locations of all *other* boxes.
            d. Compute the shortest path distance (`box_dist`) for the box from its
               current location to its goal location using BFS on the location graph,
               avoiding the box obstacles. Store the predecessor map (`pred_box`)
               from the BFS for path reconstruction.
            e. If `box_dist` is infinity, the box cannot reach its goal (e.g., trapped).
               The state is likely unsolvable, so return infinity immediately.
            f. If `box_dist` is 0 (box is at goal), this case was handled in step 4b.
            g. If `box_dist` > 0, determine the location `first_step_loc` adjacent to the
               box's current location that is the first step on a shortest path
               towards the goal. This is found by backtracking one step from
               the goal using the `pred_box` map.
            h. Determine the direction (`direction_to_goal`) from the box's current
               location (`box_loc`) to `first_step_loc`.
            i. Determine the required robot pushing location (`robot_push_loc`).
               This is the location adjacent to `box_loc` in the opposite direction
               of `direction_to_goal`. Find this location by searching the stored
               adjacent relations. If no such location exists (e.g., box against a wall),
               return infinity.
            j. Determine the set of obstacles for the robot's movement: these are the
               locations of *all* boxes.
            k. Compute the shortest path distance (`robot_dist`) for the robot from its
               current location to `robot_push_loc` using BFS on the location graph,
               avoiding the robot obstacles.
            l. If `robot_dist` is infinity, the robot cannot reach the required pushing
               position. The state is likely unsolvable, so return infinity immediately.
            m. Add `box_dist` (pushes) + `robot_dist` (robot moves) to the total heuristic.
        5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information.

        @param task: The planning task object.
        """
        self.graph = collections.defaultdict(list)
        self.adjacencies = [] # Stores (loc1, loc2, dir)
        self.locations = set()

        # Parse static facts to build the graph and adjacencies
        for fact_string in task.static:
            parsed = parse_fact(fact_string)
            if parsed[0] == 'adjacent':
                l1, l2, direction = parsed[1], parsed[2], parsed[3]
                self.graph[l1].append(l2)
                self.graph[l2].append(l1) # Adjacency is symmetric
                self.adjacencies.append((l1, l2, direction))
                self.locations.add(l1)
                self.locations.add(l2)

        # Parse goal facts to find box target locations
        self.box_goals = {}
        for goal_fact_string in task.goals:
            parsed = parse_fact(goal_fact_string)
            if parsed[0] == 'at':
                box, loc = parsed[1], parsed[2]
                self.box_goals[box] = loc

        # Store all box names for easy access
        self.all_boxes = set(self.box_goals.keys())


    def shortest_path_distance(self, start, end, graph, obstacles):
        """
        Computes the shortest path distance between start and end locations
        in the given graph, avoiding obstacle locations. Returns distance
        and predecessor map for path reconstruction.

        @param start: The starting location.
        @param end: The target location.
        @param graph: The adjacency list representation of the location graph.
        @param obstacles: A set of locations that cannot be visited.
        @return: A tuple (distance, predecessor_map) or (math.inf, None)
                 if the end is unreachable or start/end/obstacles are invalid locations.
        """
        if start not in graph or end not in graph:
             return (math.inf, None) # Invalid locations

        if start == end:
            return (0, {start: None})

        queue = collections.deque([(start, 0)])
        visited = {start}
        pred = {start: None}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return (dist, pred)

            # Check if current_loc is still valid in graph (should be if from queue)
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if neighbor not in visited and neighbor not in obstacles:
                        visited.add(neighbor)
                        pred[neighbor] = current_loc
                        queue.append((neighbor, dist + 1))

        return (math.inf, None) # End is unreachable


    def get_box_location_map(self, state):
        """Helper to get a map from box name to its current location."""
        box_locations = {}
        for fact_string in state:
            parsed = parse_fact(fact_string)
            if parsed[0] == 'at' and parsed[1] in self.all_boxes:
                 box_locations[parsed[1]] = parsed[2]
        return box_locations

    def get_robot_location(self, state):
        """Helper to get the robot's current location."""
        for fact_string in state:
            parsed = parse_fact(fact_string)
            if parsed[0] == 'at-robot':
                return parsed[1]
        return None # Should not happen in a valid state

    def get_first_step_and_direction(self, start, end, pred_map):
        """
        Given a predecessor map from BFS from start to end, finds the location
        adjacent to start that is the first step towards end, and the direction.
        Assumes start != end and path exists (dist > 0).
        """
        if pred_map is None or end not in pred_map or start == end:
             return (None, None) # Should not happen if dist is finite and > 0

        # Backtrack from end until we reach the location whose predecessor is start
        curr = end
        try:
            while pred_map.get(curr) != start:
                if pred_map.get(curr) is None:
                     # Path doesn't lead back to start - error in pred_map or logic
                     return (None, None)
                curr = pred_map[curr]
        except KeyError: # Handle cases where pred_map lookup fails unexpectedly
             return (None, None)

        # curr is now the location such that pred_map[curr] == start
        first_step_loc = curr

        # Find the direction from start to first_step_loc
        for l1, l2, direction in self.adjacencies:
            if l1 == start and l2 == first_step_loc:
                return (first_step_loc, direction)
            # Check the reverse adjacency as well
            if l2 == start and l1 == first_step_loc:
                 # Find the direction from l2 (start) to l1 (first_step_loc)
                 # We need the direction of the edge (start, first_step_loc)
                 # The stored adjacency is (l1, l2, dir) or (l2, l1, opp_dir)
                 # We are looking for the entry where one end is start and the other is first_step_loc
                 for l_a, l_b, dir_ab in self.adjacencies:
                      if (l_a == start and l_b == first_step_loc):
                           return (first_step_loc, dir_ab)
                      if (l_b == start and l_a == first_step_loc):
                           return (first_step_loc, get_opposite_direction(dir_ab)) # Direction from start to first_step_loc

        return (None, None) # Should not happen if first_step_loc is adjacent to start


    def get_robot_push_location(self, box_loc, direction_of_box_movement):
        """
        Finds the location adjacent to box_loc from which the robot can push
        the box in the given direction (direction_of_box_movement).
        This is the location adjacent in the opposite direction.
        """
        opposite_dir = get_opposite_direction(direction_of_box_movement)
        if opposite_dir is None:
             return None

        # Find l1 such that adjacent(l1, box_loc, opposite_dir)
        for l1, l2, direction in self.adjacencies:
            if l2 == box_loc and direction == opposite_dir:
                return l1

        return None # Cannot find a valid pushing location


    def sokobanHeuristic(self, state):
        """
        Computes the domain-dependent heuristic for a given state.

        @param state: The current state (frozenset of facts).
        @return: The heuristic value (estimated cost to reach the goal).
        """
        total_heuristic = 0
        robot_loc = self.get_robot_location(state)
        box_locations = self.get_box_location_map(state)

        # If robot location is not found, state is invalid
        if robot_loc is None:
             return math.inf

        # Collect locations of all boxes
        locations_of_all_boxes = set(box_locations.values())

        # Check if all boxes required by goals are present in the state
        # This check assumes the set of boxes in the initial state is the same
        # as the set of boxes in the goal state.
        if set(self.box_goals.keys()) != set(box_locations.keys()):
             # Some goal boxes are missing from the state or extra boxes are present
             return math.inf


        for box_name, goal_loc in self.box_goals.items():
            box_loc = box_locations.get(box_name)

            # This check is technically redundant due to the check above,
            # but kept for clarity/safety within the loop.
            if box_loc is None:
                 return math.inf

            if box_loc == goal_loc:
                continue # Box is already at its goal

            # Obstacles for box movement: other boxes
            locations_of_other_boxes = locations_of_all_boxes - {box_loc}

            # 1. Box distance to goal
            # The box cannot move into a location occupied by another box.
            # It also cannot move into a location that is not in the graph (wall).
            # The graph implicitly handles walls.
            box_dist, pred_box = self.shortest_path_distance(
                box_loc, goal_loc, self.graph, locations_of_other_boxes
            )

            if box_dist == math.inf:
                # Box cannot reach goal (e.g., trapped by other boxes or walls)
                return math.inf # State is likely unsolvable

            # 2. Robot distance to pushing position
            # Find the first step location and direction for the box towards the goal
            # This is only needed if the box is not already at the goal (box_dist > 0)
            # If box_dist is 0, the box is at the goal, and we continue the loop.
            # So we only need to consider box_dist > 0 here.
            first_step_loc, direction_to_goal = self.get_first_step_and_direction(
                box_loc, goal_loc, pred_box
            )

            if first_step_loc is None or direction_to_goal is None:
                 # Error case, finite dist > 0 but cannot find first step/direction
                 # This might happen if the graph is weird or pred_map is inconsistent
                 return math.inf

            # Find the required robot location to push the box towards the goal
            robot_push_loc = self.get_robot_push_location(box_loc, direction_to_goal)

            if robot_push_loc is None:
                 # Cannot find a valid pushing location (e.g., box against wall/corner
                 # in a way that blocks pushing towards goal).
                 return math.inf # State is likely unsolvable

            # Obstacles for robot movement: all boxes
            # The robot cannot move into a location occupied by any box.
            robot_dist, _ = self.shortest_path_distance(
                robot_loc, robot_push_loc, self.graph, locations_of_all_boxes
            )

            if robot_dist == math.inf:
                # Robot cannot reach the required pushing position
                return math.inf # State is likely unsolvable

            # Add costs for this box: pushes + robot moves to get ready for first push
            total_heuristic += box_dist + robot_dist

        # Heuristic is 0 only if all boxes are at their goals.
        # If total_heuristic is 0, it means all boxes were already at goals.
        # If total_heuristic > 0, it's a non-goal state.
        # The heuristic value is finite if all boxes can reach their goals
        # and the robot can reach the required pushing spots.

        return total_heuristic
