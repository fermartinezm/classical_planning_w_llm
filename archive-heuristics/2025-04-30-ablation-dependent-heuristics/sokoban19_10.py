# Imports needed
from collections import deque

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the shortest path distances for each box from its current location
        to its goal location, calculated on the empty grid (ignoring robot
        and other boxes as temporary obstacles). Additionally, it adds the
        shortest path distance from the robot's current location to the
        closest box that is not yet at its goal location.

    Assumptions:
        - The PDDL problem defines locations using the format 'loc_row_col'.
        - The goal state specifies a unique target location for each box
          using '(at box_name goal_location)' facts.
        - The 'adjacent' facts in the static information define a connected
          graph of locations relevant to the problem (initial, goal, and
          intermediate locations).
        - The heuristic assumes unit cost for actions.

    Heuristic Initialization:
        1. Stores the planning task object.
        2. Parses the goal facts from the task to map each box to its target location.
        3. Parses the static 'adjacent' facts from the task to build an adjacency list
           representation of the grid graph.
        4. Identifies all relevant locations mentioned in the initial state, goals,
           and adjacent facts.
        5. Computes all-pairs shortest paths (APSP) on this grid graph
           using Breadth-First Search (BFS) starting from each relevant location.
           These distances represent the minimum number of 'move' actions
           (or pushes along an empty path) between any two locations.

    Step-By-Step Thinking for Computing Heuristic:
        1. Given a state, first check if it is the goal state using the stored
           task object. If so, the heuristic value is 0.
        2. Extract the current location of the robot and each box from the state facts.
        3. Initialize the heuristic value `h` to 0.
        4. Identify boxes that are not yet at their goal locations.
        5. For each box not at its goal:
           a. Get its current location and its target goal location from the
              pre-parsed goal information.
           b. Look up the precomputed shortest path distance between the
              current box location and the goal location on the empty grid.
           c. Add this distance to `h`. If the goal is unreachable from the
              current box location based on precomputed distances, the state
              is likely unsolvable or a dead-end, and a very large heuristic
              value (infinity) is returned.
        6. If there are boxes not at their goal locations, find the minimum
           shortest path distance from the robot's current location to any
           of these boxes' current locations using the precomputed distances.
        7. Add this minimum robot-to-box distance to `h`. If the robot cannot
           reach any box that needs moving based on precomputed distances,
           return infinity.
        8. Return the final value of `h`.

    This heuristic is non-admissible because:
    - It sums distances for multiple boxes, ignoring potential conflicts or
      the fact that only one box can be pushed at a time.
    - The box-goal distance is calculated on the empty grid, ignoring other
      boxes and the robot as temporary obstacles.
    - The robot-to-box distance is added, which doesn't directly correspond
      to the number of pushes needed, but rather the cost for the robot
      to get near a box.
    However, it provides a reasonable estimate by considering both the box
    positions relative to goals and the robot's position relative to the boxes,
    aiming to guide the search effectively in practice.
    """

    def __init__(self, task):
        self.task = task # Store the task object

        # Parse goals: map box name to goal location
        self.box_goals = {}
        for goal_fact in self.task.goals:
            pred, args = self._parse_fact(goal_fact)
            # Goal facts are typically '(at box_name goal_location)'
            if pred == 'at' and args and len(args) == 2 and args[0].startswith('box'):
                box_name, goal_loc = args
                self.box_goals[box_name] = goal_loc

        # Build graph from adjacent facts
        self.adj = {}
        locations_from_adj = set()
        for static_fact in self.task.static:
            pred, args = self._parse_fact(static_fact)
            if pred == 'adjacent' and args and len(args) == 3:
                loc1, loc2, dir = args
                locations_from_adj.add(loc1)
                locations_from_adj.add(loc2)
                self.adj.setdefault(loc1, {})[loc2] = dir
                self.adj.setdefault(loc2, {})[loc1] = self._opposite_dir(dir)

        # Collect all relevant locations from initial state, goals, and adjacent facts
        all_relevant_locations = set(locations_from_adj)
        for goal_loc in self.box_goals.values():
             all_relevant_locations.add(goal_loc)
        # Need initial state locations too.
        for fact_string in self.task.initial_state:
             pred, args = self._parse_fact(fact_string)
             if pred in ('at-robot', 'at') and args and len(args) >= 1:
                  # Location is the last argument for 'at-robot' and 'at' facts
                  all_relevant_locations.add(args[-1])

        # Compute All-Pairs Shortest Paths (APSP) using BFS
        self.distances = {}
        for start_loc in all_relevant_locations:
             # BFS from start_loc to find distances to all reachable locations
             loc_distances = self._bfs(start_loc, self.adj)
             for end_loc, dist in loc_distances.items():
                 self.distances[(start_loc, end_loc)] = dist

        # Note: self.distances only contains entries for reachable pairs.
        # Looking up an unreachable pair (loc1, loc2) using self.distances.get((loc1, loc2), float('inf'))
        # will correctly return infinity.

    def __call__(self, state):
        # Check if goal is reached using the task object
        if self.task.goal_reached(state):
            return 0

        # Extract robot and box locations from the current state
        robot_loc = None
        box_locs = {} # box_name -> location
        # clear_locs = set() # Not used in this heuristic
        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'at-robot' and args and len(args) == 1:
                robot_loc = args[0]
            elif pred == 'at' and args and len(args) == 2 and args[0].startswith('box'):
                box_name, loc = args
                box_locs[box_name] = loc
            # elif pred == 'clear' and args and len(args) == 1:
            #      clear_locs.add(args[0])

        # If robot location is not found, the state is malformed or represents a failure.
        # Return infinity in this case.
        if robot_loc is None:
             return float('inf')

        h = 0
        boxes_not_at_goal = []

        # Calculate sum of box-to-goal distances
        for box_name, current_loc in box_locs.items():
            goal_loc = self.box_goals.get(box_name)
            # Only consider boxes that have a goal and are not there yet
            if goal_loc and current_loc != goal_loc:
                # Distance from current box location to its goal location
                # Use .get with default float('inf') to handle unreachable goals
                dist = self.distances.get((current_loc, goal_loc), float('inf'))
                if dist == float('inf'):
                     # This box cannot reach its goal from here on the empty grid
                     return float('inf') # Indicate unsolvable state / dead end
                h += dist
                boxes_not_at_goal.append(box_name)
            # If goal_loc is None, this box doesn't have a goal specified.
            # If current_loc == goal_loc, distance is 0, already handled.

        # Add minimum robot-to-box distance for boxes not at goal
        # Only add this component if there are boxes that still need moving
        if boxes_not_at_goal:
            min_robot_to_box_dist = float('inf')
            # robot_loc is guaranteed to be not None here due to check above
            for box_name in boxes_not_at_goal:
                current_box_loc = box_locs[box_name]
                # Distance from robot location to current box location
                # Use .get with default float('inf') to handle unreachable boxes
                dist = self.distances.get((robot_loc, current_box_loc), float('inf'))
                min_robot_to_box_dist = min(min_robot_to_box_dist, dist)

            if min_robot_to_box_dist == float('inf'):
                 # Robot cannot reach any box that needs moving
                 return float('inf') # Indicate unsolvable state / dead end

            h += min_robot_to_box_dist

        return h

    def _parse_fact(self, fact_string):
        # Removes leading/trailing parens and splits by space
        # Handle potential empty fact strings or malformed facts gracefully
        if not fact_string or not isinstance(fact_string, str):
             return None, []
        try:
            # Simple split might fail on complex PDDL, but works for this format
            parts = fact_string.strip("()").split()
            if not parts:
                 return None, []
            return parts[0], parts[1:] # predicate, arguments
        except Exception:
            # Catch any unexpected errors during parsing
            # print(f"Warning: Could not parse fact string: {fact_string}") # Optional: for debugging
            return None, []


    def _opposite_dir(self, direction):
        if direction == 'up': return 'down'
        if direction == 'down': return 'up'
        if direction == 'left': return 'right'
        if direction == 'right': return 'left'
        return None # Should not happen for valid directions

    def _bfs(self, start_loc, adj):
        """
        Performs Breadth-First Search from start_loc on the adjacency list adj.
        Returns a dictionary mapping reachable locations to their shortest distance from start_loc.
        """
        distances = {start_loc: 0}
        queue = deque([start_loc])
        visited = {start_loc}

        while queue:
            curr_loc = queue.popleft()
            dist = distances[curr_loc]

            # Check if curr_loc has neighbors in adj (handle isolated locations gracefully)
            if curr_loc in adj:
                for neighbor in adj[curr_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = dist + 1
                        queue.append(neighbor)
        return distances
