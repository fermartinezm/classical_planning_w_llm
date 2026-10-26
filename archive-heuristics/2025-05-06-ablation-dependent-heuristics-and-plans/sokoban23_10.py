# Import necessary modules
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This case should ideally not happen with standard planner state representations
        # print(f"Warning: Unexpected fact format in get_parts: {fact}")
        return []
    return fact[1:-1].split()

# Helper function to build the location graph from adjacent facts
def build_graph(static_facts):
    """Build adjacency list graph from adjacent facts."""
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        # Check if the fact is an 'adjacent' predicate with the correct number of arguments
        if parts and parts[0] == 'adjacent' and len(parts) == 4:
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            # Ensure locations are added to the graph even if they only appear once
            if loc1 not in graph:
                graph[loc1] = []
            if loc2 not in graph:
                graph[loc2] = []
            # Add bidirectional edges (assuming adjacency is symmetric in Sokoban)
            graph[loc1].append(loc2)
            graph[loc2].append(loc1)
        # else:
            # print(f"Warning: Skipping non-'adjacent' or malformed static fact in build_graph: {fact}")
    return graph

# Helper function to compute shortest path distance using BFS
def bfs_distance(graph, start, end):
    """Compute shortest path distance using BFS."""
    if start == end:
        return 0
    # Check if start or end locations exist as nodes in the graph
    if start not in graph or end not in graph:
        # This indicates that one of the locations is not part of the connected graph
        # defined by 'adjacent' facts. It's likely unreachable.
        # print(f"Warning: Location {start} or {end} not found in graph nodes.")
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Check if current_loc is a key in the graph before accessing neighbors
        # This check is technically redundant if all nodes added to queue are from graph keys,
        # but adds robustness.
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    # If BFS completes without finding the end, it's unreachable
    # print(f"Warning: Path not found from {start} to {end}.")
    return float('inf')

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It is calculated as the sum of the shortest path
    distances for each box to its goal, plus the shortest path distance from
    the robot to the nearest box that is not yet at its goal location.
    Distances are computed on the static graph of locations defined by the
    'adjacent' predicates, ignoring dynamic obstacles (like other boxes or
    the robot occupying a location).

    # Assumptions
    - The locations form a graph defined by the 'adjacent' predicates.
    - The cost of moving the robot between adjacent locations is 1.
    - The cost of a push action is 1.
    - The heuristic does not explicitly check for or penalize deadlock states.
    - The heuristic assumes adjacency is symmetric (if A is adjacent to B, B is adjacent to A).
    - The goal is defined solely by the target locations of specific boxes using the '(at ?box ?location)' predicate.
    - Box names start with "box".

    # Heuristic Initialization
    - Build the location graph from the 'adjacent' static facts.
    - Extract the goal locations for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state facts.
    2. Identify the current location of all boxes from the state facts. We only care about boxes that are listed in the task goals.
    3. Initialize the total heuristic cost `total_heuristic` to 0.
    4. Initialize a variable `min_robot_to_box_dist` to infinity, to track the minimum shortest path distance from the robot to any box that needs moving.
    5. Initialize a counter `needs_moving_boxes_count` to 0.
    6. Iterate through each box and its goal location stored during initialization:
       a. Get the box's name (`box_name`) and its goal location (`goal_location`).
       b. Find the box's current location (`current_location`) in the state using the `box_locations` dictionary. If the box is not found in the state, this indicates a potentially invalid state (e.g., a goal box is missing); return infinity.
       c. If the box's `current_location` is the same as its `goal_location`, the cost for this box is 0; continue to the next box.
       d. If the box is not at its goal:
          i. Increment `needs_moving_boxes_count`.
          ii. Calculate the shortest path distance between the box's `current_location` and its `goal_location` using the pre-computed `location_graph` and BFS. This distance represents the minimum number of pushes required for this box if there were no obstacles. Add this distance to the `total_heuristic`.
          iii. If the box-to-goal distance is infinity, the box cannot reach its goal; return infinity immediately as the state is unsolvable.
          iv. Calculate the shortest path distance between the robot's `robot_location` and the box's `current_location` using BFS.
          v. If the robot-to-box distance is infinity, the robot cannot reach this box; return infinity immediately as the state is unsolvable.
          vi. Update `min_robot_to_box_dist` if the current robot-to_box distance is smaller.
    7. After iterating through all goal boxes:
       a. If `needs_moving_boxes_count` is greater than 0 (meaning there is at least one box not at its goal) and `min_robot_to_box_dist` is not infinity, add `min_robot_to_box_dist` to the `total_heuristic`. This accounts for the robot's initial effort to reach a box it needs to push.
    8. Return the final `total_heuristic` value. If any step resulted in infinity, infinity is returned. The heuristic is 0 if and only if all goal boxes are at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and extracting
        goal locations for boxes.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        # Assuming task object has 'goals' (frozenset of goal facts) and 'static' (frozenset of static facts)
        self.goals = task.goals
        static_facts = task.static

        # Build the graph of locations based on adjacent facts
        # This graph represents the traversable grid/map for distance calculations.
        self.location_graph = build_graph(static_facts)

        # Store goal locations for each box that is part of the goal
        self.goal_locations = {} # {box_name: goal_location_string}
        for goal_fact in self.goals:
            # Goal facts are typically '(at box_name loc_name)'
            parts = get_parts(goal_fact)
            # Check if the fact is an 'at' predicate with two arguments (object and location)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                # Assume any object mentioned in an 'at' goal fact is a box
                # based on the domain definition and problem examples.
                self.goal_locations[obj_name] = loc_name
            # Note: We ignore other types of goal facts if they existed,
            # as the heuristic focuses on box placement.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer or float('inf') representing the estimated cost to the goal.
        """
        state = node.state  # frozenset of facts in the current state

        # Find robot location and box locations in the current state
        robot_location = None
        box_locations = {} # {box_name: current_location_string}

        for fact in state:
            parts = get_parts(fact)
            # Identify robot location
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            # Identify box locations (only for boxes we care about based on goals)
            elif parts and parts[0] == "at" and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 # Only track locations for objects that are goal boxes
                 if obj_name in self.goal_locations:
                     box_locations[obj_name] = loc_name

        # If robot location is not found in the state, it's an invalid state
        if robot_location is None:
             # print("Error: Robot location not found in state.")
             return float('inf') # Cannot solve from a state without a robot

        total_heuristic = 0
        min_robot_to_box_dist = float('inf')
        needs_moving_boxes_count = 0

        # Calculate cost for each box that is listed in the goals
        for box_name, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box_name)

            # If a box required by the goal is not present in the state, it's unsolvable
            if current_location is None:
                 # print(f"Error: Box {box_name} required by goal not found in state.")
                 return float('inf')

            # If box is already at goal, cost is 0 for this box
            if current_location == goal_location:
                continue

            # This box needs moving
            needs_moving_boxes_count += 1

            # Cost 1: Distance from box to its goal (minimum pushes)
            # Compute shortest path distance on the static location graph
            box_to_goal_dist = bfs_distance(self.location_graph, current_location, goal_location)
            if box_to_goal_dist == float('inf'):
                 # Box cannot reach goal from its current location on the static graph.
                 # This indicates an unsolvable state (e.g., box trapped).
                 # print(f"Error: Box {box_name} at {current_location} cannot reach goal {goal_location}.")
                 return float('inf') # Unsolvable state

            total_heuristic += box_to_goal_dist

            # Cost 2: Robot distance to this box
            # Compute shortest path distance on the static location graph
            robot_to_box_dist = bfs_distance(self.location_graph, robot_location, current_location)
            if robot_to_box_dist == float('inf'):
                 # Robot cannot reach the box on the static graph.
                 # This indicates an unsolvable state.
                 # print(f"Error: Robot at {robot_location} cannot reach box {box_name} at {current_location}.")
                 return float('inf') # Unsolvable state

            # Update minimum robot distance to any box that needs moving
            min_robot_to_box_dist = min(min_robot_to_box_dist, robot_to_box_dist)

        # Add the minimum robot distance to a box that needs moving, if any box needs moving.
        # This accounts for the robot's initial effort to reach a box it needs to push.
        # If no boxes need moving, needs_moving_boxes_count is 0, and this term is not added.
        if needs_moving_boxes_count > 0 and min_robot_to_box_dist != float('inf'):
             total_heuristic += min_robot_to_box_dist

        # The heuristic value must be non-negative. BFS distances are non-negative,
        # so the sum is non-negative unless it's infinity.
        # Return infinity if any part of the calculation resulted in infinity, otherwise return the sum.
        return total_heuristic if total_heuristic != float('inf') else float('inf')
