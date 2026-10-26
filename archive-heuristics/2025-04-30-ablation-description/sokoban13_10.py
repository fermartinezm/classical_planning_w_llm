from fnmatch import fnmatch
from collections import deque
import math

# Assuming Heuristic base class is available from heuristics.heuristic_base
# If running standalone for testing, you might need a dummy class like:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for arity match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distances for each box to its goal location and adding the
    shortest path distance from the robot to the nearest box that is not yet
    at its goal location. The distances are calculated based on the adjacency
    graph defined by the 'adjacent' predicates.

    # Assumptions
    - The grid structure and adjacency are defined by the 'adjacent' predicates.
    - All locations relevant to the problem (initial robot/box locations, goal locations,
      and locations connected by 'adjacent') are part of a connected graph, or at least
      reachable from each other as required by the problem.
    - The cost of moving the robot one step is 1.
    - The cost of pushing a box one step is 1 (implicitly includes robot movement).
    - The heuristic ignores dynamic obstacles (other boxes or the robot blocking paths).
    - The heuristic assumes that each box has a unique goal location specified in the task goals.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task goals.
    - Build an undirected adjacency graph of locations based on the 'adjacent' static facts.
    - Precompute the shortest path distances between all pairs of locations using BFS on this graph.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state.
    2. Identify the current location of each box that has a specified goal location.
    3. Initialize the total heuristic value to 0.
    4. Identify the set of boxes that are currently not at their goal locations.
    5. For each box in the set of ungoaled boxes:
       a. Get the box's current location and its goal location.
       b. Look up the precomputed shortest path distance between the box's current
          location and its goal location.
       c. Add this distance to the total heuristic value. If the goal is unreachable
          from the box's current location, the state is likely unsolvable or requires
          moving other objects first; return infinity.
    6. If there are any ungoaled boxes:
       a. Initialize a variable `min_robot_dist` to infinity.
       b. For each box in the set of ungoaled boxes:
          i. Get the box's current location.
          ii. Look up the precomputed shortest path distance between the robot's
              current location and the box's current location.
          iii. Update `min_robot_dist` with the minimum distance found so far.
          iv. If any ungoaled box is unreachable from the robot's current location,
              return infinity.
       c. Add `min_robot_dist` to the total heuristic value.
    7. Return the calculated total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for distance calculations.
        """
        # Assuming task object has attributes goals and static
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the adjacency graph from static facts.
        self.adj_graph = {}
        self.locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.adj_graph:
                    self.adj_graph[loc1] = set()
                if loc2 not in self.adj_graph:
                    self.adj_graph[loc2] = set()
                # Add both directions as adjacency is symmetric for movement
                self.adj_graph[loc1].add(loc2)
                self.adj_graph[loc2].add(loc1)

        # Precompute shortest path distances between all pairs of locations using BFS.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search from a start node to find distances
        to all reachable nodes in the adjacency graph.
        """
        distances = {node: math.inf for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in self.adj_graph:
                for neighbor in self.adj_graph[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        Heuristic = Sum of (box_distance_to_goal) + (robot_distance_to_nearest_ungooled_box).
        """
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_loc = get_parts(fact)
                break

        # If robot location isn't found (shouldn't happen in valid state), return inf
        if robot_loc is None:
             # print("Warning: Robot location not found in state.")
             return math.inf

        # Find current box locations for boxes with goals
        current_box_locations = {}
        boxes_not_at_goal = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                if box in self.goal_locations:
                    current_box_locations[box] = loc
                    if loc != self.goal_locations[box]:
                        boxes_not_at_goal.add(box)

        total_heuristic = 0

        # Add box distances to goal
        for box in boxes_not_at_goal:
            current_loc = current_box_locations[box]
            goal_loc = self.goal_locations[box]

            # Ensure locations are in the precomputed distances and reachable
            if current_loc not in self.distances or goal_loc not in self.distances[current_loc] or self.distances[current_loc][goal_loc] == math.inf:
                 # This indicates the goal is unreachable from the current box location
                 # print(f"Warning: Goal location {goal_loc} unreachable from box at {current_loc}")
                 return math.inf

            total_heuristic += self.distances[current_loc][goal_loc]

        # Add robot distance to the nearest ungoaled box
        if boxes_not_at_goal:
            min_robot_dist = math.inf
            for box in boxes_not_at_goal:
                box_loc = current_box_locations[box]

                # Ensure robot location and box location are in precomputed distances and reachable
                if robot_loc not in self.distances or box_loc not in self.distances[robot_loc] or self.distances[robot_loc][box_loc] == math.inf:
                    # This indicates a box is unreachable from the robot
                    # print(f"Warning: Box location {box_loc} unreachable from robot at {robot_loc}")
                    return math.inf

                min_robot_dist = min(min_robot_dist, self.distances[robot_loc][box_loc])

            # min_robot_dist will be finite if boxes_not_at_goal is not empty and all boxes are reachable from robot
            total_heuristic += min_robot_dist

        return total_heuristic
