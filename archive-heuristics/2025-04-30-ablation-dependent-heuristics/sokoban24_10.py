import math
import collections

def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Remove parentheses and split by space
    parts = fact_string.strip("()").split()
    return tuple(parts)

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each box to reach its goal location.
    For each box not at its goal, the estimated cost is the sum of:
    1. The shortest path distance (in pushes) for the box from its current
       location to its goal location.
    2. The minimum shortest path distance (in robot moves) for the robot
       from its current location to a position adjacent to the box, from
       which the box can be pushed along a shortest path towards its goal.
       This minimum is taken over all possible first steps on shortest paths
       for the box.

    Assumptions:
    - The PDDL instance uses 'loc_R_C' format for locations, although the
      heuristic relies only on the 'adjacent' facts for graph structure.
    - The goal state specifies the target location for each box using
      '(at box_name goal_location)' facts.
    - The 'adjacent' facts define a graph for reachable locations.
    - Unreachable goals or required robot positions are handled by returning
      infinity, indicating an likely unsolvable state from that point.

    Heuristic Initialization:
    The heuristic is initialized with the planning task. During initialization:
    1. The adjacency graph of locations is built from the static 'adjacent' facts.
       This graph maps a location to a dictionary of adjacent locations by direction.
       A reverse graph is also built: reverse_graph[loc2][dir] = loc1 if
       adjacent(loc1, loc2, dir) is true. This helps find the location
       'behind' a box relative to a push direction.
    2. All-pairs shortest path distances between locations are precomputed using BFS
       starting from each location. This allows quick lookup of distances
       d(loc1, loc2).
    3. The goal locations for each box are extracted from the task's goal state.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the current location of the robot and each box by parsing the state facts.
    2. Initialize the total heuristic value h = 0.
    3. For each box specified in the goal state:
        a. Get the box's current location (box_loc) and its goal location (goal_loc).
        b. If box_loc is already goal_loc, the box is at its goal; continue to the next box.
        c. Calculate the shortest path distance for the box: box_dist = distances[box_loc][goal_loc].
           If box_dist is infinity, the box cannot reach its goal; return infinity for the total heuristic.
        d. Initialize minimum robot cost for pushing this box: min_robot_cost_for_push = infinity.
        e. Find candidate first steps for the box towards the goal: Iterate through neighbors (next_loc) of box_loc. A neighbor next_loc is a candidate if distances[next_loc][goal_loc] == box_dist - 1.
        f. For each candidate next_loc:
            i. Determine the direction (push_dir) from box_loc to next_loc. This is the direction such that graph[box_loc][push_dir] == next_loc.
            ii. Determine the required robot position (r_pos) adjacent to box_loc in the *same* direction (push_dir). This is the location such that adjacent(r_pos, box_loc, push_dir) is true. Look this up in the reverse graph: r_pos = reverse_graph[box_loc].get(push_dir).
            iii. If r_pos exists and is a known location:
                - Calculate the shortest path distance for the robot: robot_dist = distances[robot_loc][r_pos].
                - Update min_robot_cost_for_push = min(min_robot_cost_for_push, robot_dist) if robot_dist is not infinity.
        g. If min_robot_cost_for_push is still infinity (no valid pushing position found for any shortest path step, or robot cannot reach any required push position), the box might be unpushable towards the goal from its current location along a shortest path; return infinity for the total heuristic.
        h. Add the cost for this box (box_dist + min_robot_cost_for_push) to the total heuristic h.
    4. Return the total heuristic value h.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing graph information and box goals.

        @param task: The planning task object.
        """
        self.task = task
        self.locations = set()
        self.graph = collections.defaultdict(dict) # loc -> {dir -> loc}
        # reverse_graph[loc2][dir] = loc1 if adjacent(loc1, loc2, dir)
        # This means loc1 is the location FROM which you move in 'dir' to reach loc2.
        # This is the required robot position to push from loc1 to loc2 in dir.
        self.reverse_graph = collections.defaultdict(dict)

        # 1. Build graph and reverse graph from adjacent facts
        for fact_string in task.static:
            fact = parse_fact(fact_string)
            if fact[0] == 'adjacent':
                loc1, loc2, direction = fact[1], fact[2], fact[3]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.graph[loc1][direction] = loc2
                self.reverse_graph[loc2][direction] = loc1 # loc1 is FROM which you move in 'direction' to get to loc2


        # 2. Precompute all-pairs shortest paths using BFS
        self.distances = {}
        # Ensure all locations mentioned in static facts are considered for BFS starts
        all_locations_in_graph = list(self.locations)
        for start_loc in all_locations_in_graph:
            self.distances[start_loc] = self._bfs(start_loc)

        # 3. Extract box goals
        self.box_goals = {} # box -> goal_loc
        for goal_fact_string in task.goals:
            goal_fact = parse_fact(goal_fact_string)
            if goal_fact[0] == 'at' and len(goal_fact) == 3: # (at box goal_loc)
                box, goal_loc = goal_fact[1], goal_fact[2]
                self.box_goals[box] = goal_loc
                # Add goal_loc to locations if not already present in graph (e.g., isolated goal)
                if goal_loc not in self.locations:
                     self.locations.add(goal_loc)
                     # Note: Distances to/from this isolated goal will be infinity unless
                     # it gets connected via state changes (not possible in Sokoban PDDL).
                     # BFS from existing nodes won't reach it. If needed, a BFS from goal_loc
                     # could be added here, but it would only find distances to nodes
                     # reachable *from* the goal (which is usually not relevant).
                     # The current BFS approach correctly results in inf distances if goal is isolated.


    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find distances to all reachable nodes.

        @param start_node: The starting location for BFS.
        @return: A dictionary mapping reachable locations to their distance from start_node.
        """
        # Initialize distances for all known locations
        distances = {node: float('inf') for node in self.locations}

        # If the start node isn't even in the set of locations derived from adjacent facts,
        # it's isolated from the main graph. No nodes are reachable.
        if start_node not in self.locations:
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Neighbors are locations reachable by any direction from current_node
            # Iterate through directions from current_node using the graph
            for direction, neighbor in self.graph[current_node].items():
                 # Ensure the neighbor is a known location before checking/updating distance
                 if neighbor in self.locations and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost (integer or infinity).
        """
        robot_loc = None
        box_locations = {} # box -> loc

        # Find robot and box locations in the current state
        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at-robot' and len(fact) == 2:
                robot_loc = fact[1]
            elif fact[0] == 'at' and len(fact) == 3:
                box, loc = fact[1], fact[2]
                box_locations[box] = loc

        # Robot must be present and in a known location
        if robot_loc is None or robot_loc not in self.locations:
             return float('inf')

        total_heuristic = 0

        # Calculate heuristic contribution for each box
        for box, goal_loc in self.box_goals.items():
            box_loc = box_locations.get(box)

            # Box must be present and in a known location
            if box_loc is None or box_loc not in self.locations:
                 return float('inf')

            # If box is already at goal, cost is 0 for this box
            if box_loc == goal_loc:
                continue

            # 1. Box distance to goal
            # Ensure goal_loc is in the distances map for box_loc (i.e., reachable)
            if goal_loc not in self.distances[box_loc]:
                 # Goal is unreachable from box location
                 return float('inf')

            box_dist = self.distances[box_loc][goal_loc]

            if box_dist == float('inf'):
                # Box cannot reach its goal (redundant check, but safe)
                return float('inf')

            # 2. Robot distance to a pushing position for the first step
            min_robot_cost_for_push = float('inf')
            found_valid_path_step = False # Flag to check if any shortest path step is possible

            # Iterate through all directions from the box's current location
            # Check neighbors reachable FROM box_loc
            for push_dir, next_loc in self.graph[box_loc].items():
                 # Check if moving box to next_loc is a step on a shortest path to goal
                 # Need to ensure next_loc is a valid location in the distance map from box_loc
                 if next_loc in self.distances[box_loc] and self.distances[next_loc].get(goal_loc, float('inf')) == box_dist - 1:
                    # next_loc is on a shortest path from box_loc to goal_loc
                    found_valid_path_step = True

                    # Find the required robot pushing position (r_pos)
                    # Robot must be at r_pos such that adjacent(r_pos, box_loc, push_dir)
                    # This means r_pos is reverse_graph[box_loc][push_dir]
                    r_pos = self.reverse_graph[box_loc].get(push_dir)

                    # Ensure r_pos exists and is a known location, and is reachable from robot_loc
                    if r_pos and r_pos in self.locations and r_pos in self.distances[robot_loc]:
                        # Calculate robot distance to this pushing position
                        robot_dist = self.distances[robot_loc][r_pos]
                        # Only consider finite robot distances
                        if robot_dist != float('inf'):
                            min_robot_cost_for_push = min(min_robot_cost_for_push, robot_dist)


            # If no shortest path step was possible from box_loc, or robot cannot reach any required push position
            # Note: min_robot_cost_for_push could still be infinity if found_valid_path_step is True
            # but all required r_pos are unreachable from robot_loc.
            if not found_valid_path_step or min_robot_cost_for_push == float('inf'):
                 # This state might be a trap or require complex maneuvers.
                 # Return infinity.
                 return float('inf')

            # Add cost for this box: box pushes + robot moves to get into position for the first push
            total_heuristic += box_dist + min_robot_cost_for_push

        return total_heuristic
