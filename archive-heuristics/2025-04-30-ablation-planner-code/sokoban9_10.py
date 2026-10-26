from collections import deque
from heuristics.heuristic_base import Heuristic
import math # Use math.inf for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at ball1 rooma)" -> ["at", "ball1", "rooma"]
    # Remove surrounding parentheses and split by space.
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distances for each box to its goal location and adding the
    shortest path distance from the robot to the closest box that is not yet
    at its goal. The shortest path distances are computed on the graph defined
    by the 'adjacent' predicates.

    # Assumptions
    - The grid structure and connectivity are fully defined by the 'adjacent'
      predicates in the static facts.
    - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A).
    - All locations mentioned in initial state, goal state, or adjacent facts
      are part of the traversable graph.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - The heuristic assumes that moving a box one step towards its goal costs
      at least 1 push action, and the robot needs to reach the box to push it.

    # Heuristic Initialization
    - Parses the goal conditions to identify all boxes and their target locations.
    - Parses the static facts to build an undirected graph representing the
      connectivity between locations based on 'adjacent' predicates.
    - Computes all-pairs shortest paths on this graph using Breadth-First Search (BFS)
      and stores these distances for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box that needs to reach a goal.
    3. Initialize the total heuristic value to 0.
    4. Initialize a variable `min_robot_box_dist` to infinity, to track the
       minimum distance from the robot to any box that is not yet at its goal.
    5. Iterate through each box identified in the goal conditions:
       a. Get the box's current location from the state and its goal location
          (stored during initialization).
       b. If the box is already at its goal location, continue to the next box.
       c. If the box is not at its goal:
          i. Calculate the shortest path distance from the box's current location
             to its goal location using the pre-computed distances. This distance
             represents the minimum number of 'push' actions required for this box
             if it could move freely. Add this distance to the total heuristic value.
          ii. Calculate the shortest path distance from the robot's current location
              to the box's current location using the pre-computed distances. This
              distance estimates the cost for the robot to reach this box to start
              pushing it. Update `min_robot_box_dist` with the minimum distance found
              so far across all boxes not at their goals.
          iii. If any required distance lookup fails (locations are disconnected),
               return infinity, indicating a potentially unsolvable state or deadlock.
    6. If all boxes were found to be at their goal locations, the state is a goal state,
       and the heuristic value is 0.
    7. Otherwise (at least one box is not at its goal), add `min_robot_box_dist`
       to the total heuristic value. This accounts for the robot needing to reach
       at least one box to start making progress.
    8. Return the final total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        and analyzing the location graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        # Keep track of all box names.
        self.boxes = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at box_name goal_location)
            if parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                goal_loc = parts[2]
                self.goal_locations[box_name] = goal_loc
                self.boxes.add(box_name)

        # Build the graph of locations based on 'adjacent' facts.
        self.graph = {}
        all_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.graph.setdefault(l1, set()).add(l2)
                # Assuming adjacency is symmetric
                self.graph.setdefault(l2, set()).add(l1)
                all_locations.add(l1)
                all_locations.add(l2)

        # Add locations from goals and initial state that might not be in adjacent facts
        # (e.g., isolated goal locations or initial positions).
        # We need to parse the initial state to get all locations mentioned there.
        # However, the heuristic __init__ only receives the task, not the initial state.
        # Let's assume all relevant locations are in adjacent facts or goals for graph building.
        # If a location from the state/goal is not in the graph, get_distance will return inf.

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_node in self.graph:
            self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all reachable nodes.
        Stores results in self.distances.
        """
        q = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0 # Distance to self is 0

        while q:
            curr, d = q.popleft()

            # Store distance from start_node to curr
            # Note: Already stored start_node to start_node = 0

            if curr not in self.graph:
                 # Should not happen if graph was built correctly from adjacent facts
                 continue

            for neighbor in self.graph[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.distances[(start_node, neighbor)] = d + 1
                    q.append((neighbor, d + 1))

    def get_distance(self, loc1_str, loc2_str):
        """
        Looks up the pre-computed shortest path distance between two locations.
        Returns float('inf') if no path exists or locations are not in the graph.
        """
        # Check if locations are in the graph nodes before lookup
        if loc1_str not in self.graph or loc2_str not in self.graph:
             # This can happen if the state contains a location not linked by adjacent facts
             # or not mentioned in goals/initial state used for graph building.
             # Treat as unreachable.
             return math.inf

        return self.distances.get((loc1_str, loc2_str), math.inf)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of strings).

        # Extract robot and box locations from the current state.
        robot_loc_str = None
        current_box_locations = {}
        # We don't need clear locations for this heuristic calculation.

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: # Skip empty facts if any
                continue

            predicate = parts[0]

            if predicate == 'at-robot' and len(parts) == 2:
                robot_loc_str = parts[1]
            elif predicate == 'at' and len(parts) == 3:
                obj_name = parts[1]
                loc_str = parts[2]
                if obj_name in self.boxes:
                    current_box_locations[obj_name] = loc_str

        # If robot location is unknown, state is invalid/unreachable.
        if robot_loc_str is None:
             return math.inf

        total_box_goal_dist = 0
        min_robot_box_dist = math.inf
        all_boxes_at_goal = True

        # Calculate heuristic components for each box not at its goal.
        for box_name in self.boxes:
            current_loc_str = current_box_locations.get(box_name)
            goal_loc_str = self.goal_locations.get(box_name)

            # If a box is not found in the current state, it's an invalid state.
            if current_loc_str is None or goal_loc_str is None:
                 return math.inf # Should not happen in valid planning states

            if current_loc_str != goal_loc_str:
                all_boxes_at_goal = False

                # Distance for the box to reach its goal (minimum pushes).
                box_goal_dist = self.get_distance(current_loc_str, goal_loc_str)

                # Distance for the robot to reach the box.
                robot_box_dist = self.get_distance(robot_loc_str, current_loc_str)

                # If any required path is blocked/non-existent, return infinity.
                if box_goal_dist == math.inf or robot_box_dist == math.inf:
                    return math.inf

                total_box_goal_dist += box_goal_dist
                min_robot_box_dist = min(min_robot_box_dist, robot_box_dist)

        # If all boxes are at their goals, the heuristic is 0.
        if all_boxes_at_goal:
            return 0
        else:
            # Heuristic is the sum of minimum pushes for all boxes
            # plus the cost for the robot to reach the closest box needing movement.
            return total_box_goal_dist + min_robot_box_dist

