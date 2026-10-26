from collections import deque
import re

# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def bfs(start_loc, target_locs, graph, obstacles):
    """
    Finds the shortest path distance from start_loc to any location in target_locs
    on the given graph, avoiding obstacles.
    Returns distance or float('inf') if unreachable.
    """
    if start_loc in obstacles: # Cannot start from an obstacle
        return float('inf')

    q = deque([(start_loc, 0)])
    visited = {start_loc}
    
    # Optimization: If start is a target and not an obstacle, distance is 0
    if start_loc in target_locs:
        return 0

    while q:
        current_loc, dist = q.popleft()

        # Check if current_loc is a valid node in the graph
        if current_loc not in graph:
             continue # Should not happen if start_loc is valid and graph is built correctly

        for neighbor_loc in graph[current_loc].values():
            if neighbor_loc not in visited and neighbor_loc not in obstacles:
                if neighbor_loc in target_locs:
                    return dist + 1 # Found a target
                visited.add(neighbor_loc)
                q.append((neighbor_loc, dist + 1))

    return float('inf') # Target(s) unreachable

def bfs_distances(start_loc, graph):
    """
    Finds shortest path distances from start_loc to all reachable locations
    on the given graph (ignoring dynamic obstacles like boxes).
    Returns a dictionary {location: distance}.
    """
    q = deque([(start_loc, 0)])
    distances = {start_loc: 0}
    
    while q:
        current_loc, dist = q.popleft()

        # Check if current_loc is a valid node in the graph
        if current_loc not in graph:
             continue # Should not happen if start_loc is valid and graph is built correctly

        for neighbor_loc in graph[current_loc].values():
            if neighbor_loc not in distances: # Not visited
                distances[neighbor_loc] = dist + 1
                q.append((neighbor_loc, dist + 1))

    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box
    not at its goal, the minimum number of pushes required for the box to reach
    its goal plus the minimum number of robot moves required to get into a
    position to make the first necessary push towards that goal.

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - Locations are named `loc_row_col`.
    - The goal is defined by `(at box goal_location)` facts for one or more boxes.
    - A box can only be moved by the robot pushing it.
    - The robot can only move to `clear` locations.
    - A box can only be pushed into a `clear` location.
    - Other boxes are obstacles for both robot movement and box movement (pushes).

    # Heuristic Initialization
    - Build the grid graph from `adjacent` facts, including directional adjacency.
    - Build a reverse graph mapping `(adjacent_loc, direction)` back to `start_loc`
      (i.e., `reverse_graph[l2][d] = l1` if `adjacent(l1, l2, d)`). This helps find
      the robot's required push position.
    - Store the goal location for each box from the task's goal conditions.
    - Pre-calculate shortest path distances from each goal location to all other
      locations on the full grid (ignoring dynamic obstacles).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Identify the robot's current location.
    3. Identify the current location of each box relevant to the goals.
    4. Create a set of locations occupied by these boxes.
    5. Initialize total heuristic cost to 0.
    6. For each box that is not yet at its goal location:
        a. Calculate the shortest path distance for the box from its current location
           to its goal location on the grid graph, considering other boxes as obstacles.
           This estimates the minimum number of pushes required for this box. If
           the goal is unreachable for the box (e.g., trapped by other boxes or walls),
           the state is likely unsolvable; return infinity.
        b. Determine the set of potential "required robot push positions". These are
           locations adjacent to the box's current location from which a push would
           move the box onto a shortest path towards its goal.
           - Use the pre-calculated distances from the box's goal location to find
             neighbors of the box's current location that are strictly closer to the goal.
           - For each such neighbor, determine the push direction required to move
             the box there.
           - Use the reverse graph to find the location the robot must occupy to
             perform that push. Add this location to the set of required robot positions.
        c. If no such required robot push positions exist (e.g., box is in a corner
           and cannot move towards goal), return infinity.
        d. Calculate the shortest path distance for the robot from its current location
           to any of the required robot push positions, considering all boxes as obstacles
           for the robot's movement. If none are reachable, return infinity.
        e. Add the box's push distance (from 6a) and the robot's distance to the
           closest required push position (from 6d) to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task) # Store task for goal_reached check
        self.goals = task.goals
        static_facts = task.static

        self.graph = {} # {loc: {direction: adjacent_loc}}
        self.reverse_graph = {} # {adjacent_loc: {direction: loc}}

        # Build graph and reverse graph from adjacent facts
        # The direction in adjacent(l1, l2, d) means moving from l1 to l2 is in direction d.
        # If robot is at l1 and box at l2, pushing in direction d moves box.
        # So, if box is at l2 and we want to push in direction d, robot needs to be at l1.
        # The reverse_graph[l2][d] = l1 maps the box location and push direction to the required robot location.
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                
                # Build graph: l1 -> l2 in direction 'direction'
                if l1 not in self.graph:
                    self.graph[l1] = {}
                self.graph[l1][direction] = l2

                # Build reverse graph: l2 -> l1 in direction 'direction'
                # This means to push a box *from* l2 *in direction* 'direction',
                # the robot must be at l1.
                if l2 not in self.reverse_graph:
                    self.reverse_graph[l2] = {}
                self.reverse_graph[l2][direction] = l1

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Pre-calculate distances from each goal location to all other locations on the full grid
        self.goal_distances_map = {} # {goal_loc: {loc: dist}}
        for goal_loc in set(self.goal_locations.values()):
             self.goal_distances_map[goal_loc] = bfs_distances(goal_loc, self.graph)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if the state is a goal state.
        if self.task.goal_reached(state):
            return 0

        # 2. Identify the robot's current location.
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
                break # Found robot location

        # 3. Identify the current location of each box relevant to the goals.
        box_locations = {} # {box_name: loc_name}
        locations_with_boxes = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.goal_locations: # Only track boxes relevant to goals
                 box_name, loc_name = parts[1], parts[2]
                 box_locations[box_name] = loc_name
                 locations_with_boxes.add(loc_name)

        # 5. Initialize total heuristic cost to 0.
        total_cost = 0

        # 6. For each box that is not yet at its goal location:
        for box_name, goal_loc in self.goal_locations.items():
            box_loc = box_locations.get(box_name) # Get current box location

            # If box is not in the state (shouldn't happen in valid states) or already at goal, skip
            if box_loc is None or box_loc == goal_loc:
                continue

            # 6a. Calculate box push distance to goal (considering other boxes as obstacles)
            # Obstacles for the box are locations occupied by *other* boxes.
            box_obstacles = locations_with_boxes - {box_loc}
            box_to_goal_dist = bfs(box_loc, {goal_loc}, self.graph, obstacles=box_obstacles)

            if box_to_goal_dist == float('inf'):
                # Box cannot reach its goal (e.g., trapped by other boxes or walls)
                return float('inf')

            # 6b. Determine required robot push positions
            required_robot_pos = set()
            
            # Get distances from the goal for this box (pre-calculated on full grid)
            dist_to_goal = self.goal_distances_map.get(goal_loc, {})
            
            # If the box_loc is not reachable from the goal on the full grid (shouldn't happen on connected grid)
            if box_loc not in dist_to_goal:
                 # This implies the goal location is unreachable from this box location even on the empty grid
                 return float('inf') 

            box_dist_from_goal = dist_to_goal[box_loc]

            # Find neighbors of box_loc that are on a shortest path towards goal_loc
            # Iterate through directions from box_loc
            for push_direction, next_box_loc in self.graph.get(box_loc, {}).items():
                 # Check if this neighbor is closer to the goal
                 if dist_to_goal.get(next_box_loc, float('inf')) == box_dist_from_goal - 1:
                      # next_box_loc is on a shortest path from box_loc to goal_loc
                      
                      # The robot needs to be at the location adjacent to box_loc in the *same* direction
                      # as the push direction (from robot_pos -> box_loc -> next_box_loc)
                      # We need the location L such that adjacent(L, box_loc, push_direction)
                      # This is the reverse mapping: reverse_graph[box_loc][push_direction]
                      
                      if box_loc in self.reverse_graph and push_direction in self.reverse_graph[box_loc]:
                          robot_push_loc = self.reverse_graph[box_loc][push_direction]
                          required_robot_pos.add(robot_push_loc)
                      # else: This push direction is impossible (e.g., box against wall) - handled by only considering valid next_box_locs

            # If after checking all potential next steps, no valid robot push position was found
            # (e.g., box is in a corner and cannot move towards goal in any valid direction)
            if not required_robot_pos:
                 # This implies the box is stuck even on the full grid without dynamic obstacles
                 return float('inf') # Box is stuck

            # 6d. Calculate robot distance to any required push position (considering all boxes as obstacles)
            robot_obstacles = locations_with_boxes # Robot cannot move onto a square with any box
            robot_to_push_pos_dist = bfs(robot_loc, required_robot_pos, self.graph, obstacles=robot_obstacles)

            if robot_to_push_pos_dist == float('inf'):
                # Robot cannot reach any required push position
                return float('inf')

            # 6e. Add costs for this box
            # Cost = box_pushes + robot_moves_to_enable_first_push
            total_cost += box_to_goal_dist + robot_to_push_pos_dist

        # 7. Return the total heuristic cost.
        return total_cost
