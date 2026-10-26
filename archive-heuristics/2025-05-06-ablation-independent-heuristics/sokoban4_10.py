import collections
import re
import sys

# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Assuming Task class is available in the environment
# from task import Task


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum number of pushes
    required for each box to reach its goal location (ignoring dynamic obstacles
    for box movement) and adding the minimum number of robot moves required
    to reach a location adjacent to any box that needs to be moved (avoiding
    current box locations).

    Assumptions:
    - Each box mentioned in task.goals has a unique goal location specified as
      an '(at box_name location_name)' fact.
    - The grid structure is defined by 'adjacent' facts in task.static.
    - Location names are strings, typically in the format 'loc_R_C'.
    - Goal facts only specify box locations. The robot's final position is not
      part of the goal.

    Heuristic Initialization:
    - Parses 'adjacent' facts from task.static to build a graph representation
      of the grid (adjacency list). Maps location strings to integer IDs.
    - Pre-calculates all-pairs shortest path distances on this static grid graph.
    - Parses 'at' facts from task.goals to determine the target location for
      each box.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state using task.goal_reached. If yes, return 0.
    2. Parse the current state to find the robot's location and the location
       of each box.
    3. Identify which boxes are not yet at their goal locations. Only consider
       boxes that have a goal specified in task.goals.
    4. If no boxes need moving but the goal is not reached, this indicates a
       problematic state or goal definition (e.g., goal includes robot position
       or clear cells, violating assumptions). Return infinity.
    5. Calculate the sum of box-to-goal distances:
       - For each box not at its goal:
         - Find its current location and its goal location.
         - Compute the shortest path distance between these two locations
           using the pre-calculated distances on the static grid graph. Dynamic
           obstacles (other boxes, robot, clear status) are ignored for this
           box movement estimate.
         - If any box's goal is unreachable on the static grid, the state is
           likely a deadlock or unsolvable; return infinity.
         - Add this distance to the total box-to_goal sum.
    6. Calculate the minimum robot-to-box distance:
       - Find all locations that are adjacent to any box that needs to be moved.
       - Compute the shortest path distance from the robot's current location
         to each of these adjacent locations. The robot's path must avoid
         locations currently occupied by boxes. This is done using BFS on the
         static grid graph, treating current box locations as obstacles.
       - Find the minimum among these distances. If the robot cannot reach
         any location adjacent to a box that needs moving, return infinity.
    7. The heuristic value is the sum of the total box-to-goal distance and
       the minimum robot-to-box distance.
    """

    def __init__(self, task):
        super().__init__(task) # Pass task to base class if needed
        self.task = task # Store task reference
        self.box_goals = self._parse_box_goals(task.goals)
        self.location_to_id, self.id_to_location, self.adj_list = self._build_grid_graph(task.static)
        # Pre-calculate all-pairs shortest paths on the static grid for efficiency
        self._static_distances = self._calculate_all_pairs_distances()


    def _parse_box_goals(self, goals):
        """Parses goal facts to map boxes to their target locations."""
        box_goals = {}
        # Goal facts are like '(at box1 loc_R_C)'
        at_pattern = re.compile(r'\(at (\S+) (\S+)\)')
        for goal_fact in goals:
            match = at_pattern.match(goal_fact)
            if match:
                box_name = match.group(1)
                location_name = match.group(2)
                box_goals[box_name] = location_name
        return box_goals

    def _build_grid_graph(self, static_facts):
        """Builds the grid graph from adjacent facts."""
        location_to_id = {}
        id_to_location = []
        adj_list = collections.defaultdict(list)
        location_id_counter = 0

        # Parse adjacent facts like '(adjacent loc_R1_C1 loc_R2_C2 direction)'
        adjacent_pattern = re.compile(r'\(adjacent (\S+) (\S+) (\S+)\)')

        for fact in static_facts:
            match = adjacent_pattern.match(fact)
            if match:
                loc1_str = match.group(1)
                loc2_str = match.group(2)

                # Ensure both locations have IDs
                if loc1_str not in location_to_id:
                    location_to_id[loc1_str] = location_id_counter
                    id_to_location.append(loc1_str)
                    location_id_counter += 1
                if loc2_str not in location_to_id:
                    location_to_id[loc2_str] = location_id_counter
                    id_to_location.append(loc2_str)
                    location_id_counter += 1

                # Add adjacency (graph is undirected for distance calculation)
                id1 = location_to_id[loc1_str]
                id2 = location_to_id[loc2_str]
                adj_list[id1].append(id2)
                adj_list[id2].append(id1) # Assuming adjacency is symmetric

        # Remove duplicates from adjacency lists
        for loc_id in adj_list:
            adj_list[loc_id] = list(set(adj_list[loc_id]))

        return location_to_id, id_to_location, adj_list

    def _calculate_all_pairs_distances(self):
        """Calculates shortest distances between all pairs of locations on the static grid."""
        num_locations = len(self.location_to_id)
        distances = {} # Use dict for sparse graph or smaller number of reachable pairs

        for start_id in range(num_locations):
            start_loc_str = self.id_to_location[start_id]
            distances[start_loc_str] = {}
            queue = collections.deque([(start_id, 0)])
            visited = {start_id}

            while queue:
                current_id, dist = queue.popleft()
                current_loc_str = self.id_to_location[current_id]
                distances[start_loc_str][current_loc_str] = dist

                for neighbor_id in self.adj_list.get(current_id, []):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, dist + 1))
        return distances

    def _get_static_distance(self, loc1_str, loc2_str):
        """Retrieves pre-calculated static distance between two locations."""
        # Returns float('inf') if loc1 or loc2 are not in the graph or unreachable
        return self._static_distances.get(loc1_str, {}).get(loc2_str, float('inf'))


    def _bfs_with_obstacles(self, start_loc_str, target_loc_str, obstacles_loc_strs):
        """
        Performs BFS on the grid graph to find the shortest distance, avoiding obstacles.
        Obstacles are location strings to avoid.
        """
        if start_loc_str == target_loc_str:
            return 0

        start_id = self.location_to_id.get(start_loc_str)
        target_id = self.location_to_id.get(target_loc_str)

        if start_id is None or target_id is None:
            return float('inf') # Start or target not in graph

        obstacle_ids = {self.location_to_id[loc] for loc in obstacles_loc_strs if loc in self.location_to_id}

        # Check if start or target is an obstacle
        if start_id in obstacle_ids or target_id in obstacle_ids:
             return float('inf') # Cannot start or end on an obstacle

        queue = collections.deque([(start_id, 0)])
        visited = {start_id}

        while queue:
            current_id, dist = queue.popleft()

            if current_id == target_id:
                return dist

            for neighbor_id in self.adj_list.get(current_id, []):
                if neighbor_id not in visited and neighbor_id not in obstacle_ids:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, dist + 1))

        return float('inf') # Target not reachable

    def __call__(self, node):
        """
        Computes the sokoban heuristic for the given state.
        """
        state = node.state

        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # 2. Parse current state
        robot_l = None
        box_locations = {} # {box_name: location_string}
        at_pattern = re.compile(r'\(at (\S+) (\S+)\)')
        at_robot_pattern = re.compile(r'\(at-robot (\S+)\)')

        for fact in state:
            match_robot = at_robot_pattern.match(fact)
            if match_robot:
                robot_l = match_robot.group(1)
                continue

            match_box = at_pattern.match(fact)
            if match_box:
                box_name = match_box.group(1)
                location_name = match_box.group(2)
                box_locations[box_name] = location_name
                continue

        if robot_l is None:
             # Should not happen in a valid Sokoban state representation
             return float('inf') # Robot location unknown

        # 3. Identify boxes to move
        # Only consider boxes that have a goal specified and are not at that goal
        boxes_to_move = [b for b in self.box_goals if b in box_locations and box_locations[b] != self.box_goals[b]]

        # 4. Handle case where no relevant boxes need moving but goal isn't reached
        # This implies the goal includes conditions other than box positions,
        # or there's a state representation issue. Return infinity as heuristic
        # is based on box movement.
        if not boxes_to_move and not self.task.goal_reached(state):
             return float('inf')


        # 5. Calculate sum of box-to-goal distances (static grid distance)
        box_to_goal_sum = 0
        for box_name in boxes_to_move:
            box_l = box_locations[box_name]
            goal_l = self.box_goals[box_name]
            # Use pre-calculated static distance for box movement estimate
            dist = self._get_static_distance(box_l, goal_l)
            if dist == float('inf'):
                 # Box is in a location from which its goal is unreachable on the static grid
                 return float('inf') # Indicates a potential deadlock
            box_to_goal_sum += dist

        # 6. Calculate minimum robot-to-box distance (BFS with obstacles)
        robot_to_box_dist = float('inf')

        if boxes_to_move: # Only calculate if there are boxes to move
            potential_robot_pos = set() # Locations adjacent to boxes that need moving

            for box_name in boxes_to_move:
                box_l = box_locations[box_name]
                box_l_id = self.location_to_id.get(box_l)
                if box_l_id is not None:
                    # Find neighbors of the box location on the static grid
                    for neighbor_id in self.adj_list.get(box_l_id, []):
                        potential_robot_pos.add(self.id_to_location[neighbor_id])

            # Robot's path must avoid locations currently occupied by boxes
            # The robot itself is not an obstacle for its own path calculation
            robot_obstacles = set(box_locations.values())

            min_dist = float('inf')
            for target_pos in potential_robot_pos:
                 # BFS for robot avoids current box locations
                 dist = self._bfs_with_obstacles(robot_l, target_pos, obstacles_loc_strs=robot_obstacles)
                 min_dist = min(min_dist, dist)

            robot_to_box_dist = min_dist

            # If robot cannot reach any adjacent position to any box that needs moving
            if robot_to_box_dist == float('inf'):
                 return float('inf') # Robot is trapped or cannot reach any box

        # 7. Combine distances
        # If boxes_to_move was empty, box_to_goal_sum is 0, robot_to_box_dist is inf.
        # This case is handled at step 4, returning inf if goal is not reached.
        # If goal is reached, h=0 is returned at step 1.
        # If goal is not reached, boxes_to_move is non-empty.
        # If box_to_goal_sum is inf, h is inf.
        # If robot_to_box_dist is inf, h is inf.
        # If both are finite, h = box_to_goal_sum + robot_to_box_dist.
        # Since goal is not reached, box_to_goal_sum > 0 (unless a box is exactly 0 distance away, i.e., at goal, but was in boxes_to_move list - this shouldn't happen based on the list definition).
        # So box_to_goal_sum >= 0.
        # robot_to_box_dist >= 0.
        # If goal not reached, box_to_goal_sum > 0 (unless box is trapped, then inf).
        # If box_to_goal_sum is finite > 0, and robot_to_box_dist is finite >= 0, then h_value > 0.
        # The only potential issue is if a box is at its goal but needs to be moved temporarily. This heuristic doesn't penalize moving a box away from its goal. This is a common limitation of simple Sokoban heuristics. However, for greedy search, this might still work reasonably well by prioritizing states where boxes are closer to goals.

        return h_value
