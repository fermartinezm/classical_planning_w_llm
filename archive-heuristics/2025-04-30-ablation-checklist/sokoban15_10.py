import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing:
    1. The minimum number of pushes required for each box to reach its goal location
       (approximated by the shortest path distance on the grid).
    2. The minimum number of robot moves required to reach a position from which
       it can make the first push for any box that is not yet at its goal.

    # Assumptions
    - The problem specifies a unique goal location for each box.
    - The grid structure is defined by `adjacent` predicates.
    - Assumes 1:1 mapping between boxes and goal locations as specified in the goal.
    - Ignores potential deadlocks where a box is pushed into a corner it cannot leave.
    - Ignores interactions between boxes (e.g., one box blocking another).
    - The cost of a 'move' action is 1, and the cost of a 'push' action is 1.
      A push action effectively moves both the robot and the box one step.
      The heuristic counts pushes and robot moves separately.

    # Heuristic Initialization
    - Parses the static `adjacent` facts to build a graph representation of the grid.
    - Extracts the goal location for each box from the task goals.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Build the grid graph from `adjacent` facts during initialization.
    2. Extract box goal locations from the task goals during initialization.
    3. In the `__call__` method:
       - Find the robot's current location.
       - Find the current location of each box.
       - Initialize the total heuristic value `h = 0`.
       - Identify all boxes that are not currently at their goal locations.
       - For each non-goal box:
         - Calculate the shortest path distance from the box's current location to its goal location using BFS on the grid graph. This distance represents the minimum number of pushes required for this box. Add this distance to `h`.
       - Calculate the robot's contribution to the heuristic:
         - Find the minimum distance the robot needs to travel to get into a position to push *any* of the non-goal boxes for their *first* step towards their respective goals.
         - For each non-goal box:
           - Determine the target location for the box's first push (the location adjacent to the box's current location on a shortest path towards its goal).
           - Determine the required robot location to make that push (the location adjacent to the box's current location in the *same* direction as the push).
           - Calculate the shortest path distance from the robot's current location to this required robot location.
         - Take the minimum of these robot distances over all non-goal boxes. Add this minimum distance to `h`.
       - If there are no non-goal boxes, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and extracting box goals.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build the adjacency graph from static facts
        self.graph = self._build_graph(self.static_facts)

        # Store goal locations for each box
        self.box_goals = self._extract_box_goals(self.goals)

        # Define opposite directions for finding robot push position
        self.opposite_direction = {
            'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'
        }

    def _build_graph(self, static_facts):
        """
        Builds an adjacency list representation of the grid from 'adjacent' facts.
        Graph: {location: {direction: adjacent_location}}
        """
        graph = collections.defaultdict(dict)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                graph[loc1][direction] = loc2
                # We don't explicitly add the reverse direction here,
                # but the PDDL typically provides both (l1 l2 dir) and (l2 l1 opp_dir).
                # If not, we would need to infer it or rely on the PDDL being complete.
                # The example PDDL *does* provide both directions.
        return dict(graph) # Convert defaultdict back to dict

    def _extract_box_goals(self, goals):
        """
        Extracts the goal location for each box from the goal conditions.
        Assumes goals are of the form (at boxX loc_Y_Z).
        """
        box_goals = {}
        for goal in goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                box_goals[box] = location
        return box_goals

    def _shortest_path_distance(self, start, end, graph):
        """
        Performs BFS to find the shortest path distance between two locations.
        Returns distance or float('inf') if no path exists.
        Also returns the predecessor map to reconstruct paths.
        """
        if start == end:
            return 0, {}

        queue = collections.deque([(start, 0)])
        visited = {start}
        predecessors = {}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist, predecessors

            for direction, neighbor_loc in graph.get(current_loc, {}).items():
                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    predecessors[neighbor_loc] = current_loc
                    queue.append((neighbor_loc, dist + 1))

        return float('inf'), {} # No path found

    def _get_first_step_and_direction(self, start, end, graph):
        """
        Finds the location and direction of the first step on a shortest path
        from start to end. Returns (next_loc, direction) or (None, None) if no path.
        """
        dist, predecessors = self._shortest_path_distance(start, end, graph)

        if dist == 0 or dist == float('inf'):
            return None, None # Already at goal or unreachable

        # Reconstruct the path backwards from end to start
        path = []
        current = end
        while current != start:
            path.append(current)
            current = predecessors.get(current)
            if current is None: # Should not happen if dist is not inf and start != end
                 return None, None
        path.reverse() # Path from start to end (excluding start)

        next_loc = path[0] # The location after start on the shortest path

        # Find the direction from start to next_loc
        for direction, neighbor in graph.get(start, {}).items():
            if neighbor == next_loc:
                return next_loc, direction

        return None, None # Should not happen if next_loc was found via graph traversal

    def _find_required_robot_pos(self, box_loc, push_direction, graph):
        """
        Finds the location the robot must be at to push a box at box_loc
        in the given push_direction.
        This is the location L such that (adjacent L box_loc push_direction).
        Returns the location string or None if no such location exists.
        """
        # Iterate through all locations and their neighbors to find the one
        # that is adjacent to box_loc in the specified direction.
        for loc in graph:
             if push_direction in graph[loc] and graph[loc][push_direction] == box_loc:
                 return loc
        return None # No location found from which to push the box in that direction

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
                break
        if robot_loc is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf')

        # 2. Find current box locations
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                current_box_locations[parts[1]] = parts[2]

        total_heuristic = 0
        non_goal_boxes = []

        # 3. Calculate minimum pushes for each box
        for box, goal_loc in self.box_goals.items():
            current_loc = current_box_locations.get(box)
            if current_loc is None:
                 # Box is not 'at' any location, maybe 'in' something?
                 # According to domain, boxes are only 'at' locations.
                 # This state might be invalid or a dead end.
                 return float('inf')

            if current_loc != goal_loc:
                non_goal_boxes.append(box)
                # Distance for the box to reach its goal
                box_dist, _ = self._shortest_path_distance(current_loc, goal_loc, self.graph)
                if box_dist == float('inf'):
                    # Box cannot reach its goal
                    return float('inf')
                total_heuristic += box_dist # Each push moves the box one step

        # If all boxes are at their goals, the heuristic is 0
        if not non_goal_boxes:
            return 0

        # 4. Calculate robot cost to get into position for the first push
        min_robot_dist_to_push_pos = float('inf')

        for box in non_goal_boxes:
            current_loc = current_box_locations[box]
            goal_loc = self.box_goals[box]

            # Find the first step location and direction towards the goal
            next_box_loc, push_direction = self._get_first_step_and_direction(current_loc, goal_loc, self.graph)

            if next_box_loc is not None: # Path exists
                # Find the required robot position to make this push
                required_robot_pos = self._find_required_robot_pos(current_loc, push_direction, self.graph)

                if required_robot_pos is not None: # Required position exists
                    # Calculate distance from current robot location to required position
                    robot_dist, _ = self._shortest_path_distance(robot_loc, required_robot_pos, self.graph)
                    min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_dist)

        # Add the minimum robot distance needed to start pushing any non-goal box
        if min_robot_dist_to_push_pos == float('inf'):
             # No pushable non-goal box found (e.g., all stuck in corners)
             return float('inf')

        total_heuristic += min_robot_dist_to_push_pos

        return total_heuristic

