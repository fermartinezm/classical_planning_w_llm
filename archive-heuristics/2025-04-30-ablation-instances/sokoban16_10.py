# Assume Heuristic base class is available (e.g., from heuristics.heuristic_base import Heuristic)

class sokobanHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the minimum
    number of pushes required for each misplaced box and adding the minimum
    robot movement cost to get into a position to make the first push for
    any of those boxes.

    # Assumptions
    - Each box has a unique goal location.
    - The grid structure is defined by the 'adjacent' facts.
    - Robot movement is possible on any location not occupied by a box.
    - Box movement (push) is possible into any adjacent location that is clear.
    - The heuristic assumes solvable instances, returning infinity for states
      where a box cannot reach its goal or the robot cannot reach any push position.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds an adjacency list representation of the grid graph from 'adjacent'
      static facts.
    - Stores directional adjacency information to determine push positions.
    - Pre-calculates shortest path distances from all unique goal locations
      to all other locations on the grid graph.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Determine which boxes are misplaced (not at their goal location).
    3. If no boxes are misplaced, the state is a goal state, return 0.
    4. Initialize `total_heuristic` to 0 and `min_robot_approach_cost` to infinity.
    5. Identify locations currently occupied by boxes (these are forbidden for robot movement).
    6. Iterate through each misplaced box:
       a. Get the box's current location and its goal location.
       b. Look up the pre-calculated shortest path distance from the box's current
          location to its goal location on the grid (minimum pushes). Add this to
          `total_heuristic`. If the distance is infinity (box cannot reach its goal),
          return infinity immediately.
       c. If the box requires pushing (distance > 0):
          i. Find a location adjacent to the box's current location that lies
             on a shortest path towards its goal. This is the intended next
             location for the box.
          ii. Determine the required robot push position: the location adjacent
              to the box's current location, on the opposite side of the
              intended push direction.
          iii. If a valid push position is found:
              - Calculate the shortest path distance for the robot from its
                current location to this push position, avoiding locations
                occupied by boxes.
              - Update `min_robot_approach_cost` with the minimum distance
                found so far across all misplaced boxes' initial push positions.
    7. After iterating through all misplaced boxes, if `min_robot_approach_cost`
       is still infinity (meaning the robot cannot reach any initial push
       position for any misplaced box), and there are misplaced boxes, return
       infinity (as the state is likely unsolvable).
    8. Add `min_robot_approach_cost` to `total_heuristic`.
    9. Return the final `total_heuristic` value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.box_goals = {}
        for goal_fact in self.goals:
            parts = self._get_parts(goal_fact)
            # Assuming goal facts for boxes are like '(at box_name location_name)'
            if parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box_name = parts[1]
                goal_location = parts[2]
                self.box_goals[box_name] = goal_location

        # Build adjacency list and directional adjacency from static facts
        self.adjacency_list = {} # {loc: set(neighbors)}
        self.adjacent_dirs = {} # {(loc1, loc2): dir}
        self.opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'} # Assuming these directions

        for fact in static_facts:
            parts = self._get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                if loc1 not in self.adjacency_list:
                    self.adjacency_list[loc1] = set()
                if loc2 not in self.adjacency_list:
                    self.adjacency_list[loc2] = set()
                self.adjacency_list[loc1].add(loc2)
                self.adjacency_list[loc2].add(loc1) # Assuming symmetric adjacency
                self.adjacent_dirs[(loc1, loc2)] = direction
                # Add reverse direction mapping
                if direction in self.opposite_dir:
                     self.adjacent_dirs[(loc2, loc1)] = self.opposite_dir[direction]


        # Pre-calculate shortest path distances from all unique goal locations
        # to all other locations on the grid graph.
        self.dist_from_unique_goals = {} # {goal_loc: {loc: dist}}
        unique_goal_locations = set(self.box_goals.values())

        for goal_loc in unique_goal_locations:
            dist_from_goal = {}
            queue = [(goal_loc, 0)]
            visited = {goal_loc}
            while queue:
                current, d = queue.pop(0)
                dist_from_goal[current] = d
                if current in self.adjacency_list:
                    for neighbor in self.adjacency_list[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))
            self.dist_from_unique_goals[goal_loc] = dist_from_goal

    def _get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()

    def _shortest_path_distance(self, start_loc, end_loc, adjacency_list):
        """
        Calculates the shortest path distance between two locations on the grid graph.
        Assumes the graph is defined by adjacency_list.
        """
        if start_loc == end_loc:
            return 0
        queue = [(start_loc, 0)]
        visited = {start_loc}
        while queue:
            current_loc, dist = queue.pop(0)
            if current_loc == end_loc:
                return dist
            if current_loc in adjacency_list:
                for neighbor in adjacency_list[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return float('inf') # No path found

    def _shortest_path_distance_robot(self, start_loc, end_loc, adjacency_list, forbidden_locations):
        """
        Calculates the shortest path distance for the robot, avoiding forbidden locations.
        """
        if start_loc == end_loc:
            return 0
        queue = [(start_loc, 0)]
        visited = {start_loc} | forbidden_locations # Robot cannot start in a forbidden location
        if start_loc in forbidden_locations:
             return float('inf') # Should not happen in valid states

        while queue:
            current_loc, dist = queue.pop(0)
            if current_loc == end_loc:
                return dist
            if current_loc in adjacency_list:
                for neighbor in adjacency_list[current_loc]:
                    if neighbor not in forbidden_locations and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return float('inf') # No path found

    def _get_push_position(self, box_loc, next_loc, adjacency_list, adjacent_dirs, opposite_dir):
        """
        Finds the location the robot must be at to push the box from box_loc to next_loc.
        Assumes next_loc is adjacent to box_loc.
        Returns the push position location, or None if not found.
        """
        # Find the direction from box_loc to next_loc
        dir_box_to_next = adjacent_dirs.get((box_loc, next_loc))
        if not dir_box_to_next:
             return None

        # The robot must be adjacent to box_loc in the opposite direction
        required_robot_dir = opposite_dir[dir_box_to_next]

        # Find the location adjacent to box_loc in the required_robot_dir
        for neighbor in adjacency_list.get(box_loc, set()):
            if adjacent_dirs.get((box_loc, neighbor)) == required_robot_dir:
                return neighbor # Found the push position

        return None # No location found in the required direction (e.g., wall)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract robot location and box locations from the current state
        robot_loc = None
        box_locations = {} # {box_name: location}

        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box_name = parts[1]
                location = parts[2]
                box_locations[box_name] = location

        # Identify misplaced boxes
        misplaced_boxes = {b for b, loc in box_locations.items() if loc != self.box_goals.get(b)}

        # If no boxes are misplaced, it's a goal state
        if not misplaced_boxes:
            return 0

        total_heuristic = 0
        min_robot_approach_cost = float('inf')

        # Locations occupied by boxes are forbidden for robot movement
        robot_forbidden_locations = set(box_locations.values())

        for box_name in misplaced_boxes:
            box_loc = box_locations[box_name]
            goal_loc = self.box_goals.get(box_name) # Use .get for safety, though misplaced implies it exists

            # Get distances from goal for this box (using pre-calculated BFS)
            dist_from_this_goal = self.dist_from_unique_goals.get(goal_loc, {})
            box_path_dist = dist_from_this_goal.get(box_loc, float('inf'))

            # If a box cannot reach its goal, the state is likely unsolvable
            if box_path_dist == float('inf'):
                return float('inf')

            # Add minimum pushes required for this box
            total_heuristic += box_path_dist

            # If the box needs pushing (distance > 0)
            if box_path_dist > 0:
                # Find a neighbor of box_loc that is on a shortest path towards goal_loc
                # This is the intended next location for the box's first step
                first_step_loc = None
                if box_loc in self.adjacency_list:
                    for neighbor in self.adjacency_list[box_loc]:
                        if dist_from_this_goal.get(neighbor, float('inf')) == box_path_dist - 1:
                            first_step_loc = neighbor
                            break # Found a neighbor on a shortest path

                if first_step_loc:
                    # Find the required robot push position for box_loc -> first_step_loc
                    push_pos = self._get_push_position(box_loc, first_step_loc, self.adjacency_list, self.adjacent_dirs, self.opposite_dir)

                    if push_pos:
                        # Calculate robot distance to this push position, avoiding boxes
                        robot_dist = self._shortest_path_distance_robot(robot_loc, push_pos, self.adjacency_list, robot_forbidden_locations)
                        min_robot_approach_cost = min(min_robot_approach_cost, robot_dist)

        # Add the minimum initial robot approach cost (if any boxes need pushing)
        if min_robot_approach_cost == float('inf'):
             # If there are misplaced boxes but the robot cannot reach any initial push position,
             # the state is likely unsolvable.
             if misplaced_boxes:
                 return float('inf')
             else:
                 # This case should not be reached due to the initial check
                 return 0

        total_heuristic += min_robot_approach_cost

        return total_heuristic
