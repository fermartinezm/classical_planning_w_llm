from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like '(at-robot loc_1_1)' or '(at box1 loc_2_4)'
    # Ensure we don't split on spaces within parameters if they existed (not the case in this domain)
    return fact[1:-1].split()

# Helper function to build the adjacency graph from 'adjacent' facts
def build_graph(static_facts):
    """Builds an adjacency list graph from adjacent facts."""
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        # Check if the fact is an 'adjacent' predicate with 3 arguments after the predicate name
        if len(parts) == 4 and parts[0] == 'adjacent':
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            if loc1 not in graph:
                graph[loc1] = []
            if loc2 not in graph:
                graph[loc2] = []
            # Add edges in both directions as movement is bidirectional in Sokoban grid
            graph[loc1].append(loc2)
            graph[loc2].append(loc1) # Assuming adjacency is symmetric

    # Remove duplicates from adjacency lists
    for loc in graph:
        graph[loc] = list(set(graph[loc]))

    return graph

# Helper function for Breadth-First Search
def bfs(graph, start_node):
    """Computes shortest path distances from a start_node to all reachable nodes."""
    # Initialize distances to infinity for all nodes in the graph
    distances = {node: math.inf for node in graph}
    # If start_node is not in the graph (e.g., isolated location), return empty distances
    if start_node not in graph:
        return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node exists in graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor exists in graph keys (and thus in distances dict)
                if neighbor in graph and distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

# Helper function to compute all-pairs shortest paths
def compute_all_pairs_distances(graph):
    """Computes shortest path distances between all pairs of nodes in the graph."""
    all_distances = {}
    # Iterate over all nodes that are keys in the graph (i.e., have at least one adjacency defined)
    for start_node in graph:
        all_distances[start_node] = bfs(graph, start_node)
    return all_distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the required number of actions (moves and pushes)
    to reach a goal state. It sums the estimated cost for each box that is not
    at its goal location. The estimated cost for a single box includes the
    minimum number of pushes required to move the box to its goal, plus the
    minimum number of robot moves required to get into a position to make the
    first "useful" push towards that goal. A "useful" push is one that moves
    the box along a shortest path towards its goal in the grid graph.

    # Assumptions
    - The world is a grid-like structure defined by 'adjacent' facts.
    - The robot can only push one box at a time.
    - The robot must be in a location adjacent to the box, on the side opposite
      the desired push direction, to push the box.
    - The heuristic uses shortest path distances on the location graph, ignoring
      obstacles (other boxes or walls not explicitly in the graph) for box movement.
      This is a relaxation.
    - The heuristic assumes that for any box not at its goal, if the state is
      solvable, there is at least one adjacent location from which a push would
      move the box towards the goal along a shortest path in the grid graph.
      If no such location exists for a box that needs moving, the state is
      considered a potential deadlock or unreachable state, and a large heuristic
      value (infinity) is returned.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - Builds an adjacency graph of locations based on 'adjacent' static facts.
    - Precomputes all-pairs shortest path distances between locations using BFS
      on the built graph. This is used to estimate robot movement cost and
      minimum box push distance.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Initialize the total heuristic cost to 0.
    3. For each box specified in the goal conditions:
        a. Get the box's current location and its goal location.
        b. If the box is already at its goal location, add 0 cost for this box and continue to the next box.
        c. If the box is not at its goal location:
            i. Calculate the minimum number of pushes required for the box to reach its goal. This is the shortest path distance between the box's current location and its goal location in the precomputed location graph (`box_dist`).
            ii. If `box_dist` is infinity, the goal is unreachable for this box in the grid, return infinity.
            iii. Find the set of locations adjacent to the box's current location using the graph.
            iv. Among these adjacent locations, identify those from which pushing the box would move it "towards" its goal. A location `adj_loc` is considered a valid push location if the shortest path distance from `adj_loc` to the goal is exactly one more than `box_dist`. This implies `adj_loc` is "behind" the box relative to the goal along a shortest path in the grid.
            v. If there are valid push locations, find the one that is closest to the robot's current location (using precomputed robot distances). This is the `best_push_loc`.
            vi. Calculate the minimum robot movement cost to reach `best_push_loc` from the robot's current location (`robot_approach_cost`).
            vii. Add `box_dist + robot_approach_cost` to the total heuristic cost.
            viii. If no valid push location is found (i.e., no adjacent location satisfies the distance condition), and the box needs moving (`box_dist > 0`), this state is likely a deadlock or unreachable by simple pushes. Return infinity.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        building the location graph, and computing all-pairs distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the adjacency graph from 'adjacent' facts
        self.graph = build_graph(static_facts)

        # Compute all-pairs shortest path distances on the graph
        self.distances = compute_all_pairs_distances(self.graph)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if the goal is an 'at' predicate with a box and a location
            if len(parts) == 3 and parts[0] == "at" and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot and box locations in the current state
        robot_location = None
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                box_locations[box] = location

        # If robot location is not found, the state is invalid. Return infinity.
        if robot_location is None:
             return math.inf

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each box that is not at its goal
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box) # Get box location, None if box is missing (invalid state)

            # If box is missing or already at goal, no cost for this box
            if current_location is None or current_location == goal_location:
                continue

            # Ensure current_location and goal_location exist in the graph distances
            # (Should be true for valid problems, but check defensively)
            if current_location not in self.distances or goal_location not in self.distances.get(current_location, {}):
                 # This indicates a problem with the graph or locations.
                 # The box or goal location is isolated or not in the precomputed distances.
                 # This state is likely unreachable or invalid.
                 return math.inf

            # Minimum pushes required for the box itself (ignoring obstacles)
            box_dist = self.distances[current_location][goal_location]

            # If box_dist is infinity, the goal is unreachable for this box in the grid
            if box_dist == math.inf:
                 # This box cannot reach its goal in the grid graph. Likely a deadlock.
                 return math.inf

            # Find the best location for the robot to push from
            best_robot_approach_cost = math.inf
            found_valid_push_loc = False

            # Iterate through locations adjacent to the box's current location
            # Ensure current_location has neighbors in the graph
            if current_location in self.graph:
                for adj_loc in self.graph[current_location]:
                    # Check if adj_loc is in distances (should be if graph is built correctly)
                    # and if goal_location is reachable from adj_loc
                    if adj_loc in self.distances and goal_location in self.distances.get(adj_loc, {}):
                        # Check if pushing from adj_loc through current_location moves towards goal_location
                        # This is true if dist(adj_loc, goal_location) == dist(current_location, goal_location) + 1
                        # This identifies the location "behind" the box relative to the goal along a shortest path.
                        if self.distances[adj_loc][goal_location] == box_dist + 1:
                            # This is a valid push location to move the box towards the goal
                            found_valid_push_loc = True
                            # Calculate robot cost to reach this valid push location
                            # Ensure robot_location is in distances and adj_loc is reachable from robot_location
                            if robot_location in self.distances and adj_loc in self.distances.get(robot_location, {}):
                                robot_approach_cost = self.distances[robot_location][adj_loc]
                                # Find the minimum robot cost among all valid push locations
                                best_robot_approach_cost = min(best_robot_approach_cost, robot_approach_cost)
                            else:
                                # Should not happen in valid states, but handle defensively
                                # Robot location or adjacent location not in distances graph
                                # This path is likely invalid or leads to an unreachable state
                                return math.inf

            # Add cost for this box: box movement + robot approach
            if found_valid_push_loc:
                 # Add box distance (pushes) + minimum robot moves to get into position for the first push
                 # Ensure best_robot_approach_cost is not infinity (should be true if found_valid_push_loc is True
                 # and robot_location is valid)
                 if best_robot_approach_cost == math.inf:
                      return math.inf # Should not happen if logic is correct and state is valid
                 total_cost += box_dist + best_robot_approach_cost
            else:
                 # If no valid push location found (e.g., box is in a corner relative to goal path)
                 # and the box needs moving (box_dist > 0), this state is likely a deadlock
                 # or requires complex maneuvers not captured by this heuristic.
                 # Return infinity to prune this path.
                 if box_dist > 0:
                     return math.inf # Box needs moving but cannot be pushed towards goal along shortest path

        # The heuristic is 0 only if all boxes are at their goal locations.
        # If total_cost is 0 here, it means all boxes were skipped because
        # current_location == goal_location for all of them.
        # This is the goal state.
        return total_cost
