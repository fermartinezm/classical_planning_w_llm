from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to extract the components of a PDDL fact string.
def get_parts(fact):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    e.g., "(at box1 loc_4_4)" -> ["at", "box1", "loc_4_4"]
    """
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the minimum
    number of pushes required for each box to reach its goal location, plus
    the minimum number of robot moves required to reach a position where it
    can push one of the boxes that needs to be moved.

    # Assumptions
    - The goal specifies the target location for each box using the `(at ?box ?location)` predicate.
    - The grid structure and possible movements are defined by the static `adjacent` predicates.
    - The heuristic does not explicitly check for or penalize deadlocks (e.g., a box pushed into a corner).
    - The cost of a 'move' action (robot only) and a 'push' action (robot moves and pushes box) are both considered as 1 unit of cost by the search algorithm. The heuristic estimates the number of such actions.

    # Heuristic Initialization
    - Builds a graph representing the locations and their adjacencies based on
      the static 'adjacent' facts. This graph is used to calculate shortest path
      distances between locations.
    - Stores the goal location for each box by parsing the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current State:** Determine the current location of the robot and all boxes by parsing the facts in the current state.
    2.  **Check for Goal State:** If all boxes are already at their designated goal locations, the heuristic value is 0.
    3.  **Calculate Box Distances:** For each box that is *not* currently at its goal location:
        a.  Find the shortest path distance on the location graph from the box's current location to its goal location using Breadth-First Search (BFS). This distance represents the minimum number of 'push' actions required for this box to reach its goal, assuming the path is clear and the robot can always get into position.
        b.  Sum these shortest path distances for all boxes that need to be moved. This sum is the estimated total number of box pushes required.
        c.  If any box's goal location is unreachable from its current location, the state is likely unsolvable or in a deadlock, and the heuristic returns a large value (`self.UNREACHABLE_COST`).
    4.  **Calculate Robot Positioning Cost:** Estimate the cost for the robot to get into a position where it can start pushing a box that needs to be moved.
        a.  Identify all boxes that are not yet at their goal locations.
        b.  For each such box, find all locations that are directly adjacent to it according to the location graph. These are potential locations where the robot could stand to push the box.
        c.  Calculate the shortest path distance on the location graph from the robot's current location to *each* of these potential robot locations (adjacent to any box needing movement) using BFS.
        d.  The minimum of all these distances (over all boxes needing movement and all adjacent locations) is the estimated minimum number of robot 'move' actions required to get into a pushing position.
        e.  If the robot cannot reach any location adjacent to any box that needs moving, the state is likely unsolvable, and the heuristic returns a large value (`self.UNREACHABLE_COST`).
    5.  **Combine Costs:** The total heuristic value is the sum of the total estimated box pushes (from step 3) and the estimated robot positioning cost (from step 4). This sum provides a non-admissible estimate of the remaining effort.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and storing
        goal locations.
        """
        self.goals = task.goals
        self.static = task.static

        # Build the location graph from adjacent facts
        self.graph = {}
        # We don't strictly need opposite directions for BFS distance,
        # but building a symmetric graph is standard if adjacency is symmetric.
        # Sokoban adjacent facts are usually provided in both directions.
        # Let's build the graph as provided, assuming it's sufficient for reachability.

        # Collect all locations mentioned in static facts (adjacent)
        locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, d = parts[1:]
                locations.add(l1)
                locations.add(l2)

        # Initialize graph with all locations found
        for loc in locations:
            self.graph[loc] = {}

        # Add edges based on adjacent facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, d = parts[1:]
                # Store neighbor and direction (direction not strictly needed for BFS distance)
                self.graph[l1][l2] = d


        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goal facts are always (at box loc)
            if parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1:]
                self.goal_locations[box] = location

        # Define a large value for unreachable states
        self.UNREACHABLE_COST = 1000000

    def bfs(self, start_location, target_locations):
        """
        Performs BFS to find the shortest distance from start_location to any
        location in target_locations.

        @param start_location: The starting location.
        @param target_locations: A set of target locations.
        @return: The shortest distance, or self.UNREACHABLE_COST if no target is reachable.
        """
        if not target_locations:
             return 0 # Should not happen if called correctly with non-empty targets

        # Check if start location is a valid node in our graph
        if start_location not in self.graph:
             return self.UNREACHABLE_COST

        queue = deque([(start_location, 0)])
        visited = {start_location}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc in target_locations:
                return dist

            # Iterate over neighbors from the graph
            # We already checked if current_loc is in self.graph before starting BFS
            for neighbor in self.graph.get(current_loc, {}): # Use .get for safety, though should be in graph
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return self.UNREACHABLE_COST # No target location was reached

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify current locations
        robot_loc = None
        box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1:]
                box_locations[obj] = loc

        # Defensive check: robot must exist and be at a known location
        if robot_loc is None or robot_loc not in self.graph:
             return self.UNREACHABLE_COST

        # 2. Calculate sum of box distances and identify boxes to move
        box_dist_sum = 0
        boxes_to_move = []

        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            # Defensive check: box must exist and be at a known location
            if current_loc is None or current_loc not in self.graph:
                 return self.UNREACHABLE_COST

            if current_loc != goal_loc:
                boxes_to_move.append(box)
                dist = self.bfs(current_loc, {goal_loc})
                if dist == self.UNREACHABLE_COST:
                    # A box cannot reach its goal location
                    return self.UNREACHABLE_COST
                box_dist_sum += dist

        # If all boxes are at goal, heuristic is 0
        if not boxes_to_move:
            return 0

        # 3. Calculate minimum robot distance to a push location
        # A push location for a box at box_loc is any location rloc such that
        # rloc is adjacent to box_loc.
        potential_robot_locations = set()
        for box in boxes_to_move:
            box_loc = box_locations[box]
            # Add all neighbors of the box's location as potential robot locations
            if box_loc in self.graph:
                for rloc in self.graph[box_loc]:
                    # Ensure the potential robot location is also a valid graph node
                    if rloc in self.graph:
                         potential_robot_locations.add(rloc)

        # If there are no valid locations adjacent to any box that needs moving
        if not potential_robot_locations:
             # This might happen in strange grid layouts or deadlocks
             return self.UNREACHABLE_COST

        # Find the minimum distance from the robot to any of these potential locations
        min_robot_dist = self.bfs(robot_loc, potential_robot_locations)

        # If robot cannot reach any push location for any box
        if min_robot_dist == self.UNREACHABLE_COST:
             return self.UNREACHABLE_COST

        # 4. Total heuristic
        # Sum of minimum pushes for all boxes + minimum moves for robot to get into position
        return box_dist_sum + min_robot_dist
