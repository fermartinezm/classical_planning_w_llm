from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input format defensively, though valid PDDL facts should match
        return []
    return fact[1:-1].split()

# Helper function to match a PDDL fact string against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to compute shortest paths in an unweighted graph
def bfs(start_node, graph):
    """
    Performs Breadth-First Search from a start node to find shortest distances
    to all reachable nodes in an unweighted graph.

    Args:
        start_node: The node to start the search from.
        graph: A dictionary representing the graph, where keys are nodes
               and values are lists of adjacent nodes.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from the start_node.
        Returns {start_node: 0} if start_node is not in the graph or has no neighbors.
    """
    # If start_node is not in the graph, it's isolated or invalid.
    # BFS from here can only reach itself with distance 0.
    if start_node not in graph:
         return {start_node: 0}

    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is still a valid key in the graph (should be if added initially)
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each misplaced box to its goal location and adding the
    minimum shortest path distance from the robot to any of the misplaced boxes.
    It uses Breadth-First Search (BFS) on the location graph defined by
    `adjacent` facts to compute distances.

    # Assumptions:
    - The goal specifies a unique target location for each box using `(at box_name location_name)` predicates.
    - The cost of moving the robot one step is 1.
    - The cost of pushing a box one step is 1 (this implicitly includes the robot's
      movement to the pushing position, which is a simplification).
    - The heuristic ignores complex Sokoban constraints like dead ends, requiring
      multiple boxes to cooperate, or obstacles (other boxes, walls) when calculating
      shortest paths for boxes and the robot. Distances are calculated on the empty grid graph.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
    - Builds an undirected graph representation of the locations based on the
      `adjacent` static facts. Includes all locations mentioned in `adjacent`
      facts and goal conditions as nodes.
    - Computes all-pairs shortest paths between all identified locations using
      BFS and stores these distances in a dictionary-of-dictionaries structure.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state facts.
    2. Identify the current location of each box from the state facts.
    3. Determine which boxes are not currently at their assigned goal locations
       by comparing current box locations with the precomputed goal locations.
    4. If there are no misplaced boxes, the current state is a goal state, and
       the heuristic value is 0.
    5. If there are misplaced boxes:
       a. Calculate the sum of the shortest path distances for each misplaced box
          from its current location to its assigned goal location. This estimates
          the minimum number of pushes required in total for all boxes, assuming
          paths are clear. Use the precomputed distances. If any box's goal is
          unreachable from its current location, the state is likely unsolvable,
          and the heuristic returns infinity.
       b. Calculate the minimum shortest path distance from the robot's current
          location to the current location of any of the misplaced boxes. This
          estimates the minimum effort for the robot to reach a box it needs to
          start working on. Use the precomputed distances. If the robot cannot
          reach any misplaced box, the state is likely unsolvable, and the
          heuristic returns infinity.
       c. The total heuristic value is the sum of the value from step 5a (total
          box movement effort) and step 5b (minimum robot approach effort).
          This sum provides a combined, non-admissible estimate of the remaining
          work.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        location graph, and computing shortest paths.
        """
        # Assuming task object has 'goals' and 'static' attributes as per example
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        # Collect all potential locations mentioned in goals and static facts
        all_potential_locations = set()

        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Goal facts are typically (at box_name location_name)
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.goal_locations[box] = location
                all_potential_locations.add(location)

        # Build the location graph from adjacent facts.
        # The graph is undirected for distance calculation purposes.
        # Initialize graph with all potential locations as nodes.
        self.location_graph = {loc: [] for loc in all_potential_locations}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Add locations to the set if not already present
                all_potential_locations.add(loc1)
                all_potential_locations.add(loc2)

                # Ensure locations are keys in the graph dictionary (should be already if added to set first)
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []

                # Add edges in both directions for distance calculation
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1)

        # Ensure all collected locations are keys in the graph dictionary,
        # even if they have no adjacent facts (isolated nodes). This handles
        # locations mentioned only in goals or initial state (if initial state
        # locations were also collected here, which they aren't currently,
        # but goal locations are).
        for loc in all_potential_locations:
             if loc not in self.location_graph:
                 self.location_graph[loc] = []


        # Compute all-pairs shortest paths using BFS from each location node in the graph.
        self.shortest_paths = {}
        for start_loc in self.location_graph.keys():
             self.shortest_paths[start_loc] = bfs(start_loc, self.location_graph)


    def get_distance(self, loc1, loc2):
        """
        Gets the shortest path distance between two locations using precomputed paths.
        Returns float('inf') if locations are not connected or if either location
        was not part of the graph built during initialization.
        """
        # Check if the starting location for the path lookup exists in our precomputed paths
        if loc1 not in self.shortest_paths:
             # This location wasn't in the graph built from static facts and goals.
             # It might be an initial robot/box location not connected to anything.
             # Distance to itself is 0, otherwise unreachable.
             return 0 if loc1 == loc2 else float('inf')

        # Check if the target location was reached from the starting location during BFS
        if loc2 not in self.shortest_paths[loc1]:
             # loc2 is not reachable from loc1 in the graph.
             return float('inf')

        return self.shortest_paths[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Find robot's current location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        # If robot location is not found, the state is invalid or unsolvable from here
        if robot_location is None:
             return float('inf')

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            # Match facts like '(at box1 loc_3_5)'
            if match(fact, "at", "box*", "*"):
                parts = get_parts(fact)
                # Ensure fact has enough parts before accessing indices
                if len(parts) >= 3:
                    box, location = parts[1], parts[2]
                    current_box_locations[box] = location
                # else: ignore malformed 'at' fact

        # Identify misplaced boxes (boxes not at their goal location)
        misplaced_boxes = []
        for box, goal_loc in self.goal_locations.items():
            # Check if the box exists in the current state and is not at its goal
            if box in current_box_locations and current_box_locations[box] != goal_loc:
                misplaced_boxes.append(box)
            # Note: If a box is in goal_locations but not in current_box_locations,
            # it implies the box is not in the state, which is likely an invalid state
            # or problem definition. We assume all goal boxes are present in the initial
            # state and thus in current_box_locations in any reachable state.

        # If all boxes are in their goal locations, the heuristic is 0
        if not misplaced_boxes:
            return 0

        # Calculate sum of box-to-goal distances for misplaced boxes
        box_to_goal_distance_sum = 0
        for box in misplaced_boxes:
            current_loc = current_box_locations.get(box) # Use .get for safety
            goal_loc = self.goal_locations[box]

            # If a box in the goal list is somehow not in the current state, treat as unsolvable
            if current_loc is None:
                 return float('inf')

            dist = self.get_distance(current_loc, goal_loc)
            # If any box cannot reach its goal, the state is likely unsolvable
            if dist == float('inf'):
                 return float('inf')
            box_to_goal_distance_sum += dist

        # Calculate minimum robot-to-misplaced-box distance
        min_robot_to_box_distance = float('inf')
        robot_can_reach_any_box = False
        for box in misplaced_boxes:
            box_loc = current_box_locations.get(box) # Use .get for safety
            if box_loc is None: continue # Should not happen based on misplaced_boxes logic

            dist = self.get_distance(robot_location, box_loc)
            # Only consider reachable boxes for the minimum distance
            if dist != float('inf'):
                 min_robot_to_box_distance = min(min_robot_to_box_distance, dist)
                 robot_can_reach_any_box = True

        # If the robot cannot reach any misplaced box, the state is likely unsolvable
        if not robot_can_reach_any_box:
             return float('inf')

        # The heuristic is the sum of the total box movement distance and the
        # minimum robot approach distance.
        return box_to_goal_distance_sum + min_robot_to_box_distance
