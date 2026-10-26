import collections
import math

def parse_fact_string(fact_string):
    """
    Parses a PDDL fact string like '(predicate arg1 arg2)' into a tuple
    (predicate, [arg1, arg2]).
    """
    # Remove surrounding brackets and split by spaces
    parts = fact_string.strip('()').split()
    predicate = parts[0]
    args = parts[1:]
    return (predicate, args)

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing two components:
    1. The sum of shortest path distances for each box from its current location
       to its designated goal location. This estimates the minimum number of pushes
       required for all boxes, ignoring obstacles and robot position.
    2. The minimum shortest path distance for the robot from its current location
       to any location adjacent to any box that is not yet at its goal. This
       estimates the robot's effort to get into a position to start pushing
       a box towards its goal.

    The shortest path distances are computed on the graph of locations defined
    by the 'adjacent' predicates, ignoring dynamic obstacles (boxes, robot)
    for efficiency.

    Assumptions:
    - The input state and task objects conform to the structure described
      in the problem description (frozenset of fact strings).
    - Location names are unique strings.
    - Box names are unique strings.
    - The 'adjacent' predicates define a connected graph of locations relevant
      to the problem instance.
    - The goal state specifies a unique target location for each box using '(at box loc)' facts.
    - The heuristic is designed for greedy best-first search and is not admissible.

    Heuristic Initialization:
    The heuristic constructor precomputes static information from the task:
    - Builds a graph of locations based on 'adjacent' predicates.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Extracts the goal location for each box from the task's goal state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Given a state (frozenset of fact strings):
    2. Extract the robot's current location by finding the fact '(at-robot ?l)'.
    3. Extract the current location of each box by finding facts '(at ?b ?l)'.
    4. Initialize total_box_distance = 0.
    5. Initialize a list of boxes_not_at_goal.
    6. For each box 'b' that has a goal location specified in self.goal_locations:
        a. Get the box's current location (current_loc) from the state.
        b. Get the box's goal location (goal_loc) from self.goal_locations.
        c. If current_loc is not equal to goal_loc:
            i. Add 'b' to boxes_not_at_goal.
            ii. Look up the precomputed shortest path distance between current_loc
                and goal_loc from self.distances. Add this distance to total_box_distance.
    7. If total_box_distance is 0, it means all boxes are at their goal locations.
       Since the goal only specifies box locations, the goal is reached. Return 0.
    8. If total_box_distance > 0, calculate the robot's contribution.
    9. Initialize min_robot_reach_box_adj_distance = infinity.
    10. For each box 'b' in boxes_not_at_goal:
        a. Get the box's current location (box_loc) from the state.
        b. Find all locations adjacent to box_loc using the precomputed self.graph.
        c. For each adjacent location (adj_loc):
            i. Look up the precomputed shortest path distance between the robot's
               current location and adj_loc from self.distances.
            ii. Update min_robot_reach_box_adj_distance with the minimum distance found so far.
    11. The heuristic value is total_box_distance + min_robot_reach_box_adj_distance.
    """

    def __init__(self, task):
        self.task = task
        self.graph = collections.defaultdict(list)
        self.locations = set()

        # 1. Build the location graph from static adjacent facts
        for fact_string in task.static:
            pred, args = parse_fact_string(fact_string)
            if pred == 'adjacent' and len(args) == 3:
                l1, l2, direction = args
                self.graph[l1].append(l2)
                # Assuming adjacency is symmetric, add the reverse edge
                self.graph[l2].append(l1)
                self.locations.add(l1)
                self.locations.add(l2)

        # Ensure all locations mentioned in goals are in the graph nodes
        # This handles cases where a goal location might not be adjacent to anything
        # in the initial static facts, but is still a valid location node.
        for goal_fact_string in task.goals:
             pred, args = parse_fact_string(goal_fact_string)
             if pred == 'at' and len(args) == 2:
                 box, loc = args
                 self.locations.add(loc)
                 # Ensure the location exists in the graph dictionary even if it has no adjacencies listed yet
                 if loc not in self.graph:
                     self.graph[loc] = []


        # 2. Compute all-pairs shortest paths on the location graph
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 3. Extract goal locations for boxes
        self.goal_locations = {}
        for goal_fact_string in task.goals:
            pred, args = parse_fact_string(goal_fact_string)
            if pred == 'at' and len(args) == 2:
                box, loc = args
                self.goal_locations[box] = loc

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all reachable nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {node: math.inf for node in self.locations}
        if start_node not in self.locations:
             # Start node is not in the known locations (e.g., robot starts outside the graph?)
             # This shouldn't happen in valid PDDL, but handle defensively.
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = collections.deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Use .get() with default empty list for robustness if a node somehow
            # got into self.locations but not as a key in self.graph (shouldn't happen
            # with current graph building, but safe).
            for neighbor in self.graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        robot_location = None
        box_locations = {}
        # clear_locations = set() # Not used in this heuristic calculation

        # Parse current state
        for fact_string in state:
            pred, args = parse_fact_string(fact_string)
            if pred == 'at-robot' and len(args) == 1:
                robot_location = args[0]
            elif pred == 'at' and len(args) == 2:
                box, loc = args
                box_locations[box] = loc
            # elif pred == 'clear' and len(args) == 1:
            #     clear_locations.add(args[0]) # Not used

        # Check if robot location is known (should be)
        if robot_location is None:
             # This state is malformed or unreachable in a standard Sokoban problem
             return math.inf

        # Calculate total box distance to goals
        total_box_distance = 0
        boxes_not_at_goal = []

        # Iterate through the boxes that *have* a goal specified
        for box, goal_loc in self.goal_locations.items():
            # Ensure the box exists in the current state (should be true)
            if box not in box_locations:
                 # This indicates a problem state (box disappeared?)
                 return math.inf # Cannot reach goal if a required box is missing

            current_loc = box_locations[box]

            if current_loc != goal_loc:
                boxes_not_at_goal.append(box)
                # Distance from current box location to its goal location
                # Check if current_loc is a valid node and goal_loc is reachable
                if current_loc not in self.distances or goal_loc not in self.distances[current_loc]:
                     # This means the box is in a location not in the graph, or its goal is unreachable
                     return math.inf # Box cannot reach goal

                total_box_distance += self.distances[current_loc][goal_loc]

        # If all boxes are at their goals, heuristic is 0
        if total_box_distance == 0:
            return 0

        # Calculate minimum robot distance to a location adjacent to a box needing a push
        min_robot_reach_box_adj_distance = math.inf

        # Check if robot_location is a valid node in the graph
        if robot_location not in self.distances:
             # Robot is in a location not in the graph
             return math.inf # Robot cannot move

        for box in boxes_not_at_goal:
            box_loc = box_locations[box]
            # Ensure box_loc is a valid node in the graph
            if box_loc not in self.graph:
                 # Box is in a location not in the graph (shouldn't happen if locations are consistent)
                 continue # Skip this box, or return inf? Let's skip for now, assuming graph covers relevant areas.

            # Find locations adjacent to the box
            for adj_loc in self.graph.get(box_loc, []):
                 # Distance from robot's current location to the adjacent location
                 # Check if adj_loc is reachable from robot_location
                 dist = self.distances[robot_location].get(adj_loc, math.inf)
                 min_robot_reach_box_adj_distance = min(min_robot_reach_box_adj_distance, dist)

        # If robot cannot reach any adjacent location of any box needing a push,
        # the state might be unsolvable or require complex maneuvers not captured
        # by this simple distance. Return infinity.
        if min_robot_reach_box_adj_distance == math.inf:
             # This happens if boxes_not_at_goal is empty (already handled),
             # or if the robot's location component is disconnected from all
             # locations adjacent to boxes needing pushes.
             return math.inf

        # The heuristic is the sum of box movement cost and robot approach cost
        return total_box_distance + min_robot_reach_box_adj_distance
