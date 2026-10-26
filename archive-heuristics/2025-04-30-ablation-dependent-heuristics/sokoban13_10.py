import re
import collections
import math # For float('inf')

# Define opposite directions
OPPOSITE_DIRECTION = {
    'up': 'down',
    'down': 'up',
    'left': 'right',
    'right': 'left',
}

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each box that is not yet at its goal location.
    The estimated cost for a single box is the sum of:
    1. The shortest path distance (in number of pushes) for the box from its
       current location to its goal location on the empty grid.
    2. The shortest path distance (in number of moves) for the robot from its
       current location to a position adjacent to the box, from which it
       can push the box towards its goal. This robot path considers other
       boxes as obstacles.

    Assumptions:
    - The problem instance uses 'loc_R_C' format for locations, where R and C
      are integers representing row and column. This allows parsing locations
      and potentially using grid-based reasoning, although the graph structure
      from 'adjacent' predicates is used for pathfinding. (Note: The current
      implementation primarily relies on the graph from 'adjacent' and doesn't
      strictly require the 'loc_R_C' format beyond parsing the names).
    - The goal state specifies a unique goal location for each box.
    - The grid is connected as defined by the 'adjacent' predicates.
    - The heuristic assumes that moving one box towards its goal is a step
      towards the overall goal, even if it might temporarily block other boxes.
      It is designed for greedy best-first search and does not need to be admissible.

    Heuristic Initialization:
    The constructor processes the static facts from the task.
    - It builds a graph representation (adjacency list) from the 'adjacent' predicates.
    - It identifies all unique locations in the grid.
    - It precomputes the shortest path distances between all pairs of locations
      on the empty grid graph (ignoring boxes and robot). This is used to
      determine the minimum number of pushes for a box and to find locations
      closer to the goal.
    - It stores the goal location for each box by parsing the task's goal state.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to find the robot's location and the current
        location of each box.
    2.  Identify which boxes are not currently at their assigned goal locations.
    3.  If all boxes are at their goals, the heuristic value is 0.
    4.  Initialize the total heuristic value `h` to 0.
    5.  For each box `B` that is not at its goal:
        a.  Get the box's current location `L_box` and its goal location `L_goal`.
        b.  Calculate the shortest path distance `d_box` for the box from `L_box`
            to `L_goal` using the precomputed distances on the empty grid. This
            represents the minimum number of pushes needed for this box. If
            `L_goal` is unreachable from `L_box` on the empty grid, the problem
            is likely unsolvable; return a large value (infinity).
        c.  Find all possible locations `L_next` adjacent to `L_box` such that
            moving the box to `L_next` would bring it strictly closer to `L_goal`
            (i.e., `precomputed_dist[L_next][L_goal] < d_box`).
        d.  For each such potential `L_next`, determine the required robot
            pushing position `L_push_pos`. This is the location adjacent to
            `L_box` in the direction opposite to the push direction (from `L_box`
            to `L_next`).
        e.  A push from `L_box` to `L_next` is only immediately possible if
            `L_next` is clear in the current state. Filter the potential `L_next`
            locations to only include those that are clear.
        f.  For each valid push direction (i.e., clear `L_next` that is strictly closer to goal):
            i.  Calculate the shortest path distance `d_robot` for the robot
                from its current location `L_robot` to the required pushing
                position `L_push_pos`. This pathfinding is done on the grid
                graph, treating locations occupied by *other* boxes (all boxes
                except the current one `B`) and walls (locations not in the graph)
                as obstacles.
            ii. If `L_push_pos` is unreachable by the robot (e.g., blocked by
                another box or wall), this specific push is not currently feasible
                via a direct path for the robot.
        g.  Find the minimum `d_robot` among all valid and reachable pushing
            positions for box `B`. If no valid pushing position is reachable
            (e.g., `L_box` is in a corner, or all paths for the robot are blocked
            by other boxes), the box might be stuck; return a large value (infinity).
        h.  Add `d_box + min_d_robot` to the total heuristic `h`.
    7.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.goal_locations = self._parse_goal(task.goals)
        self.graph, self.locations = self._build_graph(task.static)
        self.precomputed_dist = self._precompute_distances()

        # Optional: Check reachability of goals from initial box positions
        # This is more for debugging or pre-checking problem solvability
        # for box movement on empty grid. Robot reachability is state-dependent.
        # for box, goal_loc in self.goal_locations.items():
        #      initial_box_loc = None
        #      for fact in task.initial_state:
        #          if fact.startswith(f'(at {box} '):
        #              initial_box_loc = fact[1:-1].split()[2] # Extract location string
        #              break
        #      if initial_box_loc and self.precomputed_dist.get(initial_box_loc, {}).get(goal_loc, math.inf) == math.inf:
        #           print(f"Warning: Goal location {goal_loc} for box {box} is unreachable from initial location {initial_box_loc} on empty grid.")


    def _parse_goal(self, goals):
        """Parses the goal facts to find target locations for each box."""
        goal_locs = {}
        for goal in goals:
            # Goal facts are typically (at box_name loc_name)
            if goal.startswith('(at '):
                parts = goal[1:-1].split() # Remove brackets and split
                box_name = parts[1]
                loc_name = parts[2]
                goal_locs[box_name] = loc_name
        return goal_locs

    def _build_graph(self, static_facts):
        """Builds the adjacency graph from static 'adjacent' facts."""
        graph = collections.defaultdict(dict)
        locations = set()
        for fact in static_facts:
            if fact.startswith('(adjacent '):
                parts = fact[1:-1].split() # Remove brackets and split
                loc1 = parts[1]
                loc2 = parts[2]
                direction = parts[3]
                graph[loc1][direction] = loc2
                locations.add(loc1)
                locations.add(loc2)

        # Add reverse connections based on opposite directions
        # Iterate over a copy of the graph keys as we might modify it
        for loc1 in list(graph.keys()):
            for direction, loc2 in list(graph[loc1].items()):
                 if direction in OPPOSITE_DIRECTION:
                    opp_dir = OPPOSITE_DIRECTION[direction]
                    # Ensure the reverse connection exists
                    if opp_dir not in graph[loc2] or graph[loc2][opp_dir] != loc1:
                         graph[loc2][opp_dir] = loc1
                         locations.add(loc1) # Add again just in case
                         locations.add(loc2) # Add again just in case

        return dict(graph), list(locations) # Convert defaultdict back to dict

    def _precompute_distances(self):
        """
        Precomputes shortest path distances between all pairs of locations
        on the empty grid using BFS from each location.
        Returns a dict: distances[start_loc][end_loc] = distance.
        """
        distances = {}
        for start_loc in self.locations:
            distances[start_loc] = self._bfs_all_reachable(start_loc)
        return distances

    def _bfs_all_reachable(self, start_loc):
        """
        Performs BFS starting from start_loc to find distances to all reachable locations.
        Assumes no obstacles (empty grid).
        Returns a dict: distances[loc] = distance from start_loc.
        """
        distances = {loc: math.inf for loc in self.locations}
        queue = collections.deque()

        if start_loc not in self.locations:
             # Start location is not part of the known grid
             return distances # All distances remain inf

        distances[start_loc] = 0
        queue.append(start_loc)

        while queue:
            current_loc = queue.popleft()

            # Iterate through adjacent locations
            for direction, next_loc in self.graph.get(current_loc, {}).items():
                # No obstacles in this precomputation BFS
                if distances[next_loc] == math.inf:
                    distances[next_loc] = distances[current_loc] + 1
                    queue.append(next_loc)

        return distances

    def _bfs_single_target(self, start_loc, end_loc, obstacles):
        """
        Performs BFS starting from start_loc to find distance to end_loc.
        Considers obstacles (set of location strings).
        Returns distance or float('inf') if unreachable.
        """
        if start_loc in obstacles or start_loc not in self.locations:
             return math.inf # Cannot start from an obstacle or outside the graph
        if end_loc not in self.locations:
             return math.inf # Target is outside the graph
        if end_loc in obstacles:
             return math.inf # Target is an obstacle

        distances = {loc: math.inf for loc in self.locations}
        queue = collections.deque()

        distances[start_loc] = 0
        queue.append(start_loc)

        while queue:
            current_loc = queue.popleft()

            if current_loc == end_loc:
                return distances[current_loc]

            # Iterate through adjacent locations
            for direction, next_loc in self.graph.get(current_loc, {}).items():
                if next_loc not in obstacles and distances[next_loc] == math.inf:
                    distances[next_loc] = distances[current_loc] + 1
                    queue.append(next_loc)

        # end_loc not reached
        return math.inf


    def __call__(self, state, task):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @param task: The planning task object (can be ignored as info is precomputed).
        @return: The heuristic value (non-negative integer or float('inf')).
        """
        # Parse current state
        robot_loc = None
        box_locations = {} # {box_name: loc_name}
        # clear_locations = set() # Not needed directly, derived from occupied locations

        for fact in state:
            if fact.startswith('(at-robot '):
                robot_loc = fact[1:-1].split()[1]
            elif fact.startswith('(at '):
                parts = fact[1:-1].split()
                box_name = parts[1]
                loc_name = parts[2]
                box_locations[box_name] = loc_name
            # clear facts are not strictly needed if we derive clear locations
            # from all locations minus occupied locations.
            # elif fact.startswith('(clear '):
            #     loc_name = fact[1:-1].split()[1]
            #     clear_locations.add(loc_name)

        # Determine actual clear locations in the current state
        current_occupied = set(box_locations.values())
        if robot_loc:
             current_occupied.add(robot_loc)
        all_locations_in_graph = set(self.locations)
        actual_clear_locations = all_locations_in_graph - current_occupied


        total_heuristic = 0
        unsolvable = False

        # Identify boxes not at goal
        boxes_to_move = {
            box: loc for box, loc in box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        }

        if not boxes_to_move:
            return 0 # Goal reached

        # Obstacles for robot movement: locations occupied by other boxes
        # The robot cannot move through locations occupied by boxes other than the one it's trying to push.
        # It also cannot move through walls (locations not in self.locations, handled by BFS).
        # The robot's own location is the start of the BFS, not an obstacle for itself.
        robot_base_obstacles = set(box_locations.values()) # All box locations are potential obstacles for the robot

        for box, current_box_loc in boxes_to_move.items():
            goal_box_loc = self.goal_locations.get(box)

            # If a box doesn't have a goal location defined, it's likely an error
            # or the problem is unsolvable regarding this box. Treat as unsolvable.
            if goal_box_loc is None:
                 unsolvable = True
                 break

            # 1. Box distance (minimum pushes on empty grid)
            # Use precomputed distance. If start/end not in precomputed map, it's unreachable.
            d_box = self.precomputed_dist.get(current_box_loc, {}).get(goal_box_loc, math.inf)

            if d_box == math.inf:
                # Box goal is unreachable even on empty grid - indicates unsolvable problem
                unsolvable = True
                break # No need to check other boxes

            # 2. Robot distance to a valid pushing position
            min_d_robot = math.inf
            valid_push_possible = False # Can the box *potentially* be pushed towards goal?
            reachable_push_position_found = False # Can the robot *reach* a valid pushing spot?

            # Find locations adjacent to the box
            adjacent_to_box = self.graph.get(current_box_loc, {})

            # Consider pushing in each possible direction from the box's current location
            for push_dir, next_box_loc in adjacent_to_box.items():
                 # Check if pushing in this direction moves the box strictly closer to the goal
                 dist_after_push = self.precomputed_dist.get(next_box_loc, {}).get(goal_box_loc, math.inf)

                 # Check if next_box_loc is a valid location in the graph
                 if next_box_loc not in self.locations:
                      continue # Cannot push into a wall/non-existent location

                 # Condition for moving strictly closer:
                 # The distance from the next location to the goal must be less than the distance from the current location.
                 if dist_after_push >= d_box:
                      continue # This push direction doesn't move the box strictly closer

                 valid_push_possible = True # At least one direction exists to push the box towards goal

                 # Check if the target location for the box is clear in the current state
                 if next_box_loc not in actual_clear_locations:
                      continue # Cannot push box into an occupied spot

                 # Find the required robot position to push from current_box_loc to next_box_loc
                 # Robot must be adjacent to current_box_loc in the opposite direction of the push
                 required_robot_dir = OPPOSITE_DIRECTION.get(push_dir)
                 if required_robot_dir is None:
                      continue # Should not happen with standard directions

                 required_robot_loc = self.graph.get(current_box_loc, {}).get(required_robot_dir)

                 if required_robot_loc is None:
                      continue # No location exists to push from this side (e.g., against a wall)

                 # Calculate robot distance to the required pushing position
                 # Obstacles for the robot are all locations occupied by *other* boxes
                 # The current box (current_box_loc) is NOT an obstacle for the robot trying to reach required_robot_loc.
                 # The target location next_box_loc is clear (checked above), so it's not an obstacle.
                 # The required_robot_loc itself might be occupied by another box - this is handled by BFS obstacles.
                 other_boxes_locs = set(box_locations.values()) - {current_box_loc}
                 current_robot_obstacles = other_boxes_locs # Robot cannot move through other boxes

                 d_robot = self._bfs_single_target(robot_loc, required_robot_loc, current_robot_obstacles)

                 if d_robot != math.inf:
                      min_d_robot = min(min_d_robot, d_robot)
                      reachable_push_position_found = True # Found at least one reachable pushing position

            # If no valid pushing position was found or reachable for this box
            # A box is stuck if there's no direction towards the goal OR the robot cannot reach any such pushing spot.
            # Note: If valid_push_possible is False, it means the box is surrounded or in a corner such that
            # no push moves it strictly closer to the goal on the empty grid. This is a strong indicator of being stuck.
            # If valid_push_possible is True, but reachable_push_position_found is False (because min_d_robot is inf),
            # it means there are valid push directions, but the robot cannot currently reach any of the required spots.
            if not valid_push_possible or not reachable_push_position_found:
                 # This box is stuck or unreachable by the robot in a useful position
                 unsolvable = True
                 break # No need to check other boxes

            # Add cost for this box: box pushes + robot moves to get into position for the first push
            total_heuristic += d_box + min_d_robot

        if unsolvable:
             return math.inf # Problem is likely unsolvable from this state

        return total_heuristic
