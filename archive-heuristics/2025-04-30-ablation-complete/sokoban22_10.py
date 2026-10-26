from heuristics.heuristic_base import Heuristic
from collections import deque
# No need for fnmatch or math for this implementation

# Helper function to extract components from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a given pattern
def match(fact, *pattern_args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `pattern_args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    fact_parts = get_parts(fact)
    if len(fact_parts) != len(pattern_args):
        return False
    return all(fact_part == pattern_arg or pattern_arg == '*' for fact_part, pattern_arg in zip(fact_parts, pattern_args))


# BFS function to compute shortest paths
def bfs(start_node, graph):
    """
    Performs BFS from start_node on the given graph to find distances to all reachable nodes.
    Graph is represented as a dictionary: node -> list of neighbors.
    Returns a dictionary: node -> distance from start_node.
    """
    distances = {}
    queue = deque([(start_node, 0)])
    visited = {start_node}
    distances[start_node] = 0

    while queue:
        current_node, dist = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box not at its goal,
    the shortest path distance the box needs to be pushed to reach its goal plus the shortest
    path distance the robot needs to move to reach the box's current location.

    # Assumptions
    - The grid structure and connectivity are defined by the 'adjacent' predicates.
    - Each box has a unique goal location specified in the task goals.
    - The cost of moving the robot is 1 per step (adjacent location).
    - The cost of pushing a box one step is 1 action.
    - The heuristic sums costs for each box independently, ignoring interactions between boxes
      or the specific robot positioning required for a push (it only considers robot distance to the box).
    - Unreachable goals for boxes or boxes unreachable by the robot indicate a potentially unsolvable state,
      assigned a large heuristic value.

    # Heuristic Initialization
    - Build both a directed graph (for box movement) and an undirected graph (for robot movement)
      from the 'adjacent' static facts.
    - Compute all-pairs shortest paths on both graphs using BFS.
    - Extract the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and each box from the state.
    2. Check if all boxes are already at their goal locations. If yes, the heuristic is 0.
    3. Initialize the total heuristic cost to 0.
    4. For each box that is not at its goal:
       a. Get its current location and its goal location.
       b. Calculate the shortest path distance for the box from its current location to its goal location
          on the directed adjacency graph (representing minimum pushes). If the goal is unreachable for the box,
          the state is likely unsolvable, return a large heuristic value.
       c. Calculate the shortest path distance for the robot from its current location to the box's
          current location on the undirected adjacency graph (representing robot approach moves). If the box
          location is unreachable for the robot, the state is likely unsolvable, return a large heuristic value.
       d. Add the box distance and the robot distance to the total heuristic cost for this box.
    5. Return the total heuristic cost summed over all boxes not at their goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building graphs and computing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        self.adj_undirected = {}
        self.adj_directed = {}
        all_locations = set()

        # Build adjacency graphs from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                l1, l2, direction = parts[1], parts[2], parts[3]
                all_locations.add(l1)
                all_locations.add(l2)

                # Undirected graph for robot movement
                if l1 not in self.adj_undirected:
                    self.adj_undirected[l1] = []
                if l2 not in self.adj_undirected:
                    self.adj_undirected[l2] = []
                # Add edge only if it doesn't exist to avoid duplicates
                if l2 not in self.adj_undirected[l1]:
                    self.adj_undirected[l1].append(l2)
                if l1 not in self.adj_undirected[l2]:
                    self.adj_undirected[l2].append(l1) # Assuming bidirectional movement for robot

                # Directed graph for box movement (push)
                if l1 not in self.adj_directed:
                    self.adj_directed[l1] = []
                self.adj_directed[l1].append(l2)

        self.all_locations = list(all_locations) # Store as list for consistent iteration

        # Compute all-pairs shortest paths on both graphs
        self.dist_undirected = {}
        for start_loc in self.all_locations:
            self.dist_undirected[start_loc] = bfs(start_loc, self.adj_undirected)

        self.dist_directed = {}
        for start_loc in self.all_locations:
             self.dist_directed[start_loc] = bfs(start_loc, self.adj_directed)

        # Store goal locations for each box
        self.goal_locations = {}
        # The goals are represented as a frozenset of facts, e.g., {(at box1 loc_2_4)}
        # We assume goals are always (at ?box ?location) for boxes
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Define a large value for unreachable states
        self.UNREACHABLE_COST = 1000000 # Use a large integer

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # We only care about the boxes defined in the goals
                if obj in self.goal_locations:
                     box_locations[obj] = loc

        # Check if all goals are already met
        all_goals_met = True
        for box, goal_loc in self.goal_locations.items():
             # Check if the box exists in the current state and is at the goal location
             # A box might not be in box_locations if it's not mentioned in 'at' facts,
             # which shouldn't happen in valid Sokoban states, but check defensively.
             if box not in box_locations or box_locations[box] != goal_loc:
                  all_goals_met = False
                  break

        if all_goals_met:
             return 0 # Goal state reached

        total_cost = 0

        # Calculate cost for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            box_loc = box_locations.get(box) # Get current location of the box

            # If the box is not in the state (shouldn't happen) or not at its goal location
            if box_loc is None or box_loc != goal_loc:
                # Get box-to-goal distance (pushes needed)
                # Use .get() with default empty dict to handle cases where start_loc might not be in precomputed distances
                # If box_loc is None, dist_directed.get(None, {}) will return {}, then .get(goal_loc, ...) returns UNREACHABLE_COST
                box_dist = self.dist_directed.get(box_loc, {}).get(goal_loc, self.UNREACHABLE_COST)

                # Get robot-to-box distance (robot moves needed to approach box)
                # If robot_loc is None (shouldn't happen in valid states), dist_undirected.get(None, {}) will return {}, then .get(box_loc, ...) returns UNREACHABLE_COST
                robot_dist = self.dist_undirected.get(robot_loc, {}).get(box_loc, self.UNREACHABLE_COST)

                # If either is unreachable, the state is likely unsolvable for this box
                if box_dist >= self.UNREACHABLE_COST or robot_dist >= self.UNREACHABLE_COST:
                    # Return a large value immediately if any box is in an unreachable state
                    return self.UNREACHABLE_COST

                # Add costs for this box
                total_cost += box_dist + robot_dist

        # If we reached here, all boxes are either at the goal or reachable.
        # If total_cost is 0, it means all boxes were at their goals (handled above).
        # If total_cost > 0, it's the sum of distances.
        return total_cost
