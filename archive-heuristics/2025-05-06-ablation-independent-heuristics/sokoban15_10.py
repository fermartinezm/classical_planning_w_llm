import collections
import re
import heapq # Not directly used in this heuristic, but often useful for pathfinding
import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task # Assuming Task and Operator are available from the planner framework


# Helper function to parse location names like 'loc_R_C'
def parse_loc(loc_name):
    """Parses a location string 'loc_R_C' into a (row, col) tuple."""
    match = re.match(r'loc_(\d+)_(\d+)', loc_name)
    if match:
        try:
            row = int(match.group(1))
            col = int(match.group(2))
            return (row, col)
        except ValueError:
            # Should not happen if regex matches digits, but good practice
            return None
    # Handle cases where location name doesn't match expected format
    # Could log a warning or raise an error depending on desired strictness
    # logging.warning(f"Could not parse location name: {loc_name}")
    return None

# Helper function for Breadth-First Search
def bfs(start, end, obstacles, adj_list):
    """
    Finds the shortest path distance between start and end locations
    using BFS on the adjacency list, avoiding obstacles.
    Returns float('inf') if end is unreachable.
    """
    if start == end:
        return 0
    if start in obstacles: # Cannot start inside an obstacle
        return float('inf')

    queue = collections.deque([(start, 0)])
    visited = {start}
    # Obstacles are locations the path cannot go through
    # Walls are implicitly handled by adj_list (no edge)

    while queue:
        current_loc, dist = queue.popleft()

        # Check neighbors in all directions
        # adj_list[current_loc] is a dict {direction: neighbor_loc}
        for neighbor_loc in adj_list.get(current_loc, {}).values():
            if neighbor_loc == end:
                return dist + 1
            if neighbor_loc not in visited and neighbor_loc not in obstacles:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))

    return float('inf') # Target unreachable

# Helper function for Manhattan distance
def manhattan_distance(loc1_name, loc2_name):
    """Calculates Manhattan distance between two locations based on parsed coordinates."""
    p1 = parse_loc(loc1_name)
    p2 = parse_loc(loc2_name)
    if p1 is None or p2 is None:
        # Cannot parse, maybe not a grid location?
        # This heuristic assumes a grid structure for this part.
        return float('inf')
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Helper function to get direction from loc1 to loc2
def get_direction(loc1_name, loc2_name, adj_list):
    """
    Finds the direction from loc1 to loc2 using the adjacency list.
    Returns the direction string (e.g., 'up', 'down', 'left', 'right')
    or None if loc2 is not directly adjacent to loc1.
    """
    # Check neighbors of loc1
    for direction, neighbor in adj_list.get(loc1_name, {}).items():
        if neighbor == loc2_name:
            return direction
    return None # Not adjacent

# Helper function to get opposite direction (not strictly needed with current logic, but good)
# opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal by summing costs for each box not yet
    at its goal location. The cost for a single box is the sum of:
    1. The estimated number of pushes required to move the box from its current
       location to its goal location (approximated by Manhattan distance on the grid).
    2. The estimated number of robot moves required to reach a location adjacent
       to the box from which it can push the box towards its goal. This is
       calculated using BFS on the grid graph, considering other boxes as obstacles.

    Assumptions:
    - The problem uses a grid-like structure where locations are named 'loc_R_C'.
    - Adjacency facts define a connected graph.
    - Goal facts specify a unique goal location for each box.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - Deadlocks (situations where a box cannot be moved) are handled by returning
      infinity for the robot's path to the required push location if that location
      is unreachable due to obstacles (other boxes). If no location allows pushing
      the box towards the goal (based on Manhattan distance reduction), the heuristic
      for that box is considered infinite.

    Heuristic Initialization:
    - Parses 'adjacent' facts from the static information to build an adjacency
      list representing the grid connectivity.
    - Parses goal facts to map each box to its target goal location.
    - Stores all known location names mentioned in adjacent facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify which boxes are not yet at their goal locations.
    3. If all boxes are at their goals, the heuristic is 0.
    4. Initialize the total heuristic value to 0.
    5. For each box that is not at its goal:
        a. Get the box's current location and its goal location.
        b. Calculate the estimated number of pushes needed for this box using
           Manhattan distance between its current location and goal location. Add this to the total heuristic.
           If locations cannot be parsed into grid coordinates, return infinity.
        c. Determine the set of locations adjacent to the box from which the robot
           could push the box to move it strictly closer to its goal (based on Manhattan distance).
           Iterate through all known locations `loc` in the grid. If `loc` is adjacent to the box's current location,
           find the direction `push_dir` from `loc` to the box. Calculate the box's next location
           `box_next_loc` if pushed in `push_dir`. If `box_next_loc` exists and is strictly closer
           to the goal (by Manhattan distance) than the current box location, add `loc` to the set
           of valid required robot locations.
        d. If no such valid required robot location exists, this box is considered
           stuck or requires complex setup; return infinity for the total heuristic.
        e. Calculate the shortest path distance for the robot from its current
           location to any of the valid required push locations identified in (c).
           This distance is calculated using BFS on the grid graph, treating
           locations occupied by *other* boxes (and the current box's location) as obstacles.
        f. If the robot cannot reach any of the required push locations, return
           infinity for the total heuristic.
        g. Add the minimum robot-to-push distance found in (e) to the total heuristic.
    6. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the Sokoban heuristic.

        Args:
            task: An instance of the Task class containing problem definition.
        """
        super().__init__(task) # Call parent constructor
        self.task = task
        self.adj_list = self._build_adj_list(task.static)
        self.box_goals = self._get_box_goals(task.goals)
        # Collect all unique locations mentioned in adjacent facts
        self.locations = set()
        for loc1, neighbors in self.adj_list.items():
            self.locations.add(loc1)
            for loc2 in neighbors.values():
                self.locations.add(loc2)


    def _build_adj_list(self, static_facts):
        """Builds an adjacency list from adjacent facts."""
        adj = collections.defaultdict(dict)
        for fact_str in static_facts:
            # Example: '(adjacent loc_4_2 loc_4_3 right)'
            if fact_str.startswith('(adjacent '):
                # Split by space, strip leading/trailing parens
                parts = fact_str.strip('()').split()
                if len(parts) == 4:
                    predicate, loc1, loc2, direction = parts
                    adj[loc1][direction] = loc2
        return dict(adj) # Convert defaultdict to dict for efficiency after building

    def _get_box_goals(self, goal_facts):
        """Maps boxes to their goal locations from goal facts."""
        goals = {}
        for fact_str in goal_facts:
            # Example: '(at box1 loc_2_4)'
            if fact_str.startswith('(at '):
                # Split by space, strip leading/trailing parens
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    predicate, box_name, loc_name = parts
                    goals[box_name] = loc_name
        return goals

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The current search node containing the state.

        Returns:
            The estimated cost (integer or float('inf')) to reach a goal state.
        """
        state = node.state

        # 1. Identify robot and box locations
        robot_loc = None
        box_locations = {} # {box_name: loc_name}
        # Convert frozenset to set for potentially faster lookups
        current_facts = set(state)

        for fact_str in current_facts:
            if fact_str.startswith('(at-robot '):
                # Example: '(at-robot loc_6_4)'
                parts = fact_str.strip('()').split()
                if len(parts) == 2:
                    robot_loc = parts[1]
            elif fact_str.startswith('(at '):
                # Example: '(at box1 loc_4_4)'
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    box_name = parts[1]
                    loc_name = parts[2]
                    box_locations[box_name] = loc_name

        # Basic check: Ensure robot and all goal boxes are present in the state
        # This might catch malformed states, though the planner should handle this.
        # Also check if any box is at a location not in our known grid.
        if robot_loc is None or len(box_locations) != len(self.box_goals) or \
           robot_loc not in self.locations or any(loc not in self.locations for loc in box_locations.values()):
             # State doesn't contain expected elements or contains unknown locations
             # Treat as unreachable
             return float('inf')

        # 2. Identify non-goal boxes
        non_goal_boxes = [
            box for box, loc in box_locations.items()
            if box in self.box_goals and loc != self.box_goals[box]
        ]

        # 3. If all boxes are at goals, heuristic is 0
        if not non_goal_boxes:
            return 0

        # 4. Initialize total heuristic
        total_h = 0
        all_box_locs = set(box_locations.values())

        # 5. Calculate cost for each non-goal box
        for box in non_goal_boxes:
            current_box_loc = box_locations[box]
            goal_loc = self.box_goals.get(box) # Use .get for safety

            # If somehow a non-goal box doesn't have a goal (shouldn't happen based on PDDL)
            if goal_loc is None:
                 return float('inf')

            # b. Box-to-goal distance (Manhattan distance approximation)
            # This estimates pushes needed assuming a clear path for the box
            box_to_goal_dist = manhattan_distance(current_box_loc, goal_loc)
            if box_to_goal_dist == float('inf'):
                 # Locations cannot be parsed into coordinates, or are somehow non-grid
                 # This heuristic relies on grid structure for Manhattan distance
                 return float('inf')
            # Add Manhattan distance for the box
            total_h += box_to_goal_dist

            # c. Determine valid required robot push locations
            # A location 'loc' is a valid required robot location if:
            # 1. 'loc' is adjacent to 'current_box_loc'.
            # 2. Pushing the box from 'loc' (in the direction from 'loc' to 'current_box_loc')
            #    moves the box to a 'box_next_loc' that is strictly closer to 'goal_loc'
            #    (by Manhattan distance) than 'current_box_loc'.
            valid_req_locs = []
            current_dist_to_goal = manhattan_distance(current_box_loc, goal_loc)

            # Iterate through all locations in the grid to find potential robot positions
            for loc in self.locations:
                 # Check if current_box_loc is adjacent to loc.
                 # If it is, find the direction from loc to current_box_loc.
                 push_dir = get_direction(loc, current_box_loc, self.adj_list)

                 if push_dir is not None:
                     # If robot is at 'loc', it pushes the box in 'push_dir'.
                     # Find where the box would move to.
                     box_next_loc = self.adj_list.get(current_box_loc, {}).get(push_dir)

                     if box_next_loc is not None:
                          # Check if this move reduces the Manhattan distance to the goal
                          next_dist_to_goal = manhattan_distance(box_next_loc, goal_loc)
                          if next_dist_to_goal < current_dist_to_goal:
                              valid_req_locs.append(loc)

            # d. If no location allows pushing towards the goal
            if not valid_req_locs:
                 # This box is likely in a position where a simple push doesn't
                 # reduce the straight-line distance to the goal (e.g., corner,
                 # or needs to be pushed away first). Treat as a potential deadlock.
                 return float('inf')

            # e. Calculate robot-to-push distance
            # Obstacles for the robot are all box locations, including the current one
            # (robot cannot occupy the same space as any box).
            obstacles_for_robot = all_box_locs

            min_robot_dist = float('inf')
            for req_loc in valid_req_locs:
                 # BFS finds the shortest path for the robot from its current location
                 # to the required push location, avoiding obstacles.
                 dist = bfs(robot_loc, req_loc, obstacles_for_robot, self.adj_list)
                 min_robot_dist = min(min_robot_dist, dist)

            # f. If robot cannot reach any required push location
            if min_robot_dist == float('inf'):
                 # Robot is blocked from getting into position to push this box.
                 # Treat as a potential deadlock.
                 return float('inf')

            # g. Add robot-to-push distance
            total_h += min_robot_dist

        # 6. Return total heuristic
        return total_h
