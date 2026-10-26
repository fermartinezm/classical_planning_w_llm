# Import necessary modules
from collections import deque
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Define the heuristic class inheriting from Heuristic (uncomment the inheritance in the target environment)
# class sokobanHeuristic(Heuristic):
class sokobanHeuristic:

    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to solve the puzzle by summing, for each box not yet at its goal,
    the minimum number of pushes required to move the box to its goal plus the minimum number of robot
    moves required to get the robot into a position to perform the *first* necessary push for that box.

    # Assumptions
    - The cost of moving a box one step (a 'push' action) is 1.
    - The cost of moving the robot one step (a 'move' action) is 1.
    - The robot movement cost between pushes of the same box or between pushing different boxes is approximated by the initial approach cost for each box.
    - The minimum number of pushes required for a box is the shortest path distance from its current location to its goal location on the graph of all locations, ignoring other boxes as obstacles for the box itself.
    - The robot's movement is restricted by walls and locations occupied by boxes.

    # Heuristic Initialization
    - Builds a graph of locations based on `adjacent` facts.
    - Builds a reverse graph for efficient distance calculation from goals.
    - Stores the mapping between directions and their opposites.
    - Extracts the goal location for each box from the goal conditions.
    - Precomputes shortest path distances from each goal location to all other locations on the full graph (used for box movement cost).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box.
    2. For each box that is not yet at its goal location:
        a. Retrieve the precomputed shortest path distance from the box's current location to its goal location (using the map computed from the goal on the reverse graph). This is the minimum number of pushes required for this box. If the goal is unreachable, the state is likely unsolvable, return infinity.
        b. Determine a required robot location for the *first* push of this box towards its goal. This location is adjacent to the box's current location, on the side opposite the direction of the first step of a shortest path from the box to its goal. This is found by checking neighbors of the box's current location and using the precomputed distances to identify a neighbor that is one step closer to the goal.
        c. Calculate the shortest path distance for the robot from its current location to this required push location. The robot's movement is restricted by locations occupied by boxes. If the required push location is unreachable by the robot, return infinity.
        d. Add the number of pushes required (step 2a) and the robot approach cost (step 2c) to the total heuristic value.
    3. The total heuristic value is the sum accumulated in step 2.
    4. If all boxes are at their goals, the heuristic is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        # The task object is assumed to have 'goals' and 'static' attributes
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph and reverse graph from adjacent facts
        self.graph = {}
        self.reverse_graph = {}
        self.opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                if loc1 not in self.reverse_graph:
                    self.reverse_graph[loc1] = []
                if loc2 not in self.reverse_graph:
                    self.reverse_graph[loc2] = []

                self.graph[loc1].append((loc2, direction))
                # In the reverse graph, an edge from loc1 to loc2 with direction 'dir'
                # corresponds to an edge from loc2 to loc1 with direction 'dir'
                self.reverse_graph[loc2].append((loc1, direction))

                # Add the symmetric edge in the graph if the opposite direction is known
                opposite_dir = self.opposite_direction.get(direction)
                if opposite_dir:
                    # Check if the opposite edge is already listed to avoid duplicates
                    found_opposite = False
                    for neighbor, d in self.graph.get(loc2, []):
                        if neighbor == loc1 and d == opposite_dir:
                            found_opposite = True
                            break
                    if not found_opposite:
                         self.graph[loc2].append((loc1, opposite_dir))
                         # Add the corresponding edge in the reverse graph
                         self.reverse_graph[loc1].append((loc2, opposite_dir))


        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming objects starting with 'box' are the boxes we care about for goals
            if parts[0] == 'at' and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # Precompute shortest path distances from each goal location on the reverse graph
        # This gives distances *to* the goal on the original graph (box movement cost)
        self._dist_from_goal = {}
        # Use a set of goal locations to avoid recomputing BFS for the same goal location
        for goal_loc in set(self.box_goals.values()):
             # BFS on the reverse graph starting from the goal location
             self._dist_from_goal[goal_loc] = self._bfs(goal_loc, self.reverse_graph)


    def _bfs(self, start, graph_to_use, occupied_locations=None):
        """
        Performs BFS to find shortest path distances from start to all reachable locations
        on the specified graph, avoiding occupied_locations.
        Args:
            start (str): The starting location.
            graph_to_use (dict): The adjacency graph {location: [(neighbor, dir), ...]}.
            occupied_locations (set, optional): Locations that cannot be entered. Defaults to None (no obstacles).
        Returns:
            dict: A dictionary mapping reachable locations to their shortest distance from start.
                  Returns {start: 0} if start is not in graph_to_use or has no outgoing edges.
        """
        # Initialize distances with infinity for all known locations in the graph, except start
        # A location is "known" if it's a key in either graph or reverse_graph
        all_known_locations = set(self.graph.keys()).union(self.reverse_graph.keys())
        distances = {loc: float('inf') for loc in all_known_locations}

        if start in distances:
             distances[start] = 0
        else:
             # If start location is not part of the defined graph structure at all
             # Distance to itself is 0, others remain inf.
             distances[start] = 0 # Add the start node itself


        queue = deque([start])
        visited = {start}

        while queue:
            current_loc = queue.popleft()

            # Get neighbors from the specified graph
            # Use .get() with default [] in case current_loc has no outgoing edges in this graph
            edges = graph_to_use.get(current_loc, [])

            for neighbor_loc, _ in edges:
                if occupied_locations is not None and neighbor_loc in occupied_locations:
                    continue # Cannot move into occupied location

                # Check if neighbor_loc is a known location in the graph structure
                # This prevents BFS from exploring nodes not defined in the adjacent facts
                if neighbor_loc not in all_known_locations:
                     continue # Skip locations not part of the main graph structure

                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    distances[neighbor_loc] = distances[current_loc] + 1
                    queue.append(neighbor_loc)

        # Filter out unreachable locations (distance remains inf) if needed, or keep them.
        # Keeping them is fine, bfs_distance will return inf.
        return distances

    def bfs_distance(self, start, end, graph_to_use, occupied_locations=None):
         """Helper to get distance between two points from BFS result."""
         # Check if start or end are not in the graph at all
         # This check is important because _bfs only explores from 'start'.
         # If 'end' is not in the graph, _bfs won't find it.
         # We can check if they are keys or values in the graph/reverse_graph.
         all_known_locations = set(self.graph.keys()).union(self.reverse_graph.keys())
         is_start_in_graph = start in all_known_locations
         is_end_in_graph = end in all_known_locations

         if not is_start_in_graph or not is_end_in_graph:
              # If either location is not part of the defined graph structure,
              # they are likely unreachable from each other unless they are the same.
              return 0 if start == end else float('inf')


         distances = self._bfs(start, graph_to_use, occupied_locations=occupied_locations)
         return distances.get(end, float('inf'))


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Parse current state
        robot_loc = None
        box_locations = {} # {box_name: location}
        # occupied_locations = set() # Locations occupied by robot or boxes - not needed directly for robot BFS obstacles

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
                # robot_loc is the start of robot BFS, not an obstacle for itself
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                box_locations[box] = location
                # Box locations are obstacles for robot BFS

        total_heuristic = 0

        # Locations occupied by boxes (for robot movement obstacles)
        box_locations_set = set(box_locations.values())

        for box, current_box_loc in box_locations.items():
            goal_box_loc = self.box_goals.get(box)

            if goal_box_loc is None:
                 # This box doesn't have a goal specified, ignore it or handle error
                 continue # Or return float('inf') if any box without a goal makes the state invalid

            if current_box_loc == goal_box_loc:
                # Box is already at its goal
                continue

            # Get the precomputed distance map from the goal for this box
            dist_from_goal_map = self._dist_from_goal.get(goal_box_loc)
            if dist_from_goal_map is None:
                 # Goal location not found in precomputed map - implies goal_loc wasn't in graph
                 return float('inf') # Should not happen in valid problems

            # 1. Calculate minimum pushes needed for this box
            # Distance from current_box_loc to goal_box_loc using the map from goal (on reverse graph)
            pushes_needed = dist_from_goal_map.get(current_box_loc, float('inf'))

            if pushes_needed == float('inf'):
                # Box cannot reach its goal location
                return float('inf')

            # 2. Find a required robot location for the first push
            # Find a neighbor 'next_loc' of current_box_loc that is one step closer to the goal
            required_robot_push_loc = None
            first_step_dir = None

            # Iterate through neighbors of the box's current location in the standard graph
            # We need to find a neighbor that is reachable and is one step closer to the goal
            # The box itself is not an obstacle for finding the path for the box (conceptually)
            # But the neighbor must be a valid location in the graph.
            for neighbor_loc, dir in self.graph.get(current_box_loc, []):
                 # Check if this neighbor is one step closer to the goal using the precomputed map
                 # Ensure neighbor_loc is also a location that can reach the goal
                 if dist_from_goal_map.get(neighbor_loc, float('inf')) == pushes_needed - 1:
                      # Found a valid first step: current_box_loc -> neighbor_loc (in direction dir)
                      first_step_dir = dir
                      break # Found a valid first step direction

            if first_step_dir is None:
                 # This implies pushes_needed > 0 but no neighbor is closer to the goal.
                 # This shouldn't happen in a valid graph if pushes_needed is finite and > 0.
                 # It might indicate a dead-end state for the box where it's stuck.
                 return float('inf') # Box is stuck

            # Find the location adjacent to current_box_loc in the opposite direction of the first step
            opposite_dir = self.opposite_direction.get(first_step_dir)
            if opposite_dir:
                 # Search for the location 'behind' the box in the standard graph
                 for potential_push_loc, push_dir in self.graph.get(current_box_loc, []):
                      if push_dir == opposite_dir:
                           required_robot_push_loc = potential_push_loc
                           break # Found the required push location

            if required_robot_push_loc is None:
                 # Robot cannot get into a position to push the box in the required direction.
                 # This might happen if the location "behind" the box doesn't exist in the graph.
                 return float('inf') # Robot cannot get into position


            # 3. Calculate robot approach cost
            # Robot cannot move into locations occupied by boxes
            # The robot's current location is not an obstacle for the BFS starting from there.
            # Obstacles are the locations currently occupied by boxes.
            robot_approach_cost = self.bfs_distance(robot_loc, required_robot_push_loc, self.graph, box_locations_set)

            if robot_approach_cost == float('inf'):
                 # Robot cannot reach the required push location
                 return float('inf')

            # Add costs for this box
            total_heuristic += pushes_needed + robot_approach_cost

        return total_heuristic
