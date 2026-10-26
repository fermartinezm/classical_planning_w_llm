from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class is available

# Utility functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def build_location_graph(static_facts):
    """Builds an adjacency list graph from adjacent facts."""
    graph = {}
    reverse_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            if loc1 not in graph:
                graph[loc1] = []
            if loc2 not in graph:
                graph[loc2] = []
            # Store neighbor and direction from loc1 to loc2
            graph[loc1].append((loc2, direction))
            # Store reverse edge (loc2 to loc1) with opposite direction
            graph[loc2].append((loc1, reverse_dir[direction]))
    return graph

def compute_all_pairs_shortest_paths(graph):
    """Computes all-pairs shortest paths using BFS."""
    distances = {}
    locations = list(graph.keys())
    for start_loc in locations:
        distances[(start_loc, start_loc)] = 0
        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft() # BFS queue

            if current_loc not in graph:
                 # This start_loc might be in the problem definition but isolated
                 # from all other locations via 'adjacent' facts. Distances from/to it
                 # to other nodes will remain infinite, which is correct.
                 continue

            for neighbor_loc, _ in graph[current_loc]:
                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    distances[(start_loc, neighbor_loc)] = dist + 1
                    queue.append((neighbor_loc, dist + 1))
    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It considers the minimum number of pushes needed
    for each box (approximated by shortest path distance) and the cost for
    the robot to reach a position from which it can make the first push
    towards a goal for any box.

    # Assumptions
    - The grid structure and connectivity are defined by the 'adjacent' predicates.
    - Shortest path distance on the location graph is a reasonable estimate for
      the number of moves/pushes required.
    - Moving a box one step requires one 'push' action.
    - The robot needs to reach a specific location adjacent to the box to push it.
    - The cost of moving the robot between pushes is simplified or ignored,
      except for the initial approach to the first box.
    - Deadlocks (situations where a box cannot be moved to its goal) are
      partially handled by returning infinity if no valid push position is reachable.
      The heuristic does *not* check if the target location of a push is clear,
      nor does it explicitly handle complex box-on-box or box-on-wall deadlocks.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task goals.
    - Builds a graph of locations based on 'adjacent' static facts.
    - Computes all-pairs shortest paths between locations on this graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and each box.
    2. Initialize total heuristic cost `h = 0`.
    3. Initialize minimum robot approach cost `min_robot_approach_cost = infinity`.
    4. Create a set `all_valid_robot_start_locs` to store potential robot positions
       for the first push of any box towards its goal.
    5. Iterate through each box and its goal location:
       a. Get the box's current location.
       b. If the box is not at its goal:
          i. Add the shortest path distance from the box's current location to its
             goal location to `h`. This estimates the number of pushes for this box.
             If the goal is unreachable from the box, return infinity.
          ii. Find all locations `r_loc` adjacent to the box's current location
              from which a push would move the box strictly closer to its goal.
              A location `r_loc` is valid if it's adjacent to the box's location
              `b_loc`, and pushing from `r_loc` through `b_loc` to `next_loc`
              results in `next_loc` being strictly closer to the box's goal `g_loc`
              than `b_loc` is. Add these `r_loc` locations to `all_valid_robot_start_locs`.
    6. If there are no boxes needing to be moved (all are at goals), return `h` (which is 0).
    7. If there are boxes needing to be moved, iterate through all locations in
       `all_valid_robot_start_locs`. For each `r_loc`, calculate the shortest
       path distance from the robot's current location to `r_loc`. Update
       `min_robot_approach_cost` with the minimum distance found.
    8. If `min_robot_approach_cost` is still infinity (meaning the robot cannot
       reach any valid push position for any box), return infinity.
    9. Otherwise, the total heuristic value is `h + min_robot_approach_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        building the location graph, and computing shortest paths.
        """
        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the location graph from adjacent facts.
        self.location_graph = build_location_graph(task.static)

        # Compute all-pairs shortest paths on the location graph.
        self.distances = compute_all_pairs_shortest_paths(self.location_graph)

        # Map direction string to its reverse.
        self.reverse_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot's current location
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_location = parts[1]
                break

        # This should not happen in a valid Sokoban state, but handle defensively
        if robot_location is None:
             return float('inf') # Robot location unknown or state is invalid

        # Find current location for each box
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is an 'at' predicate for a box that is in our goals
            if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.goal_locations:
                 box_locations[parts[1]] = parts[2]

        total_pushes_cost = 0
        min_robot_approach_cost = float('inf')
        found_box_to_move = False

        # Collect all potential robot start locations for pushing any box towards its goal
        all_valid_robot_start_locs = set()

        for box, goal_location in self.goal_locations.items():
            current_box_location = box_locations.get(box)

            # If box is not in state or already at goal, skip
            if current_box_location is None or current_box_location == goal_location:
                continue

            found_box_to_move = True

            # Add box-goal distance (minimum pushes)
            box_dist_to_goal = self.distances.get((current_box_location, goal_location))
            if box_dist_to_goal is None:
                # Box goal is unreachable from its current location on the graph
                return float('inf')
            total_pushes_cost += box_dist_to_goal

            # Find valid robot start locations for this box
            if current_box_location in self.location_graph:
                # Iterate through neighbors of the box's current location.
                # These neighbors are potential 'next_loc' after a push.
                for next_loc_candidate, dir_from_box in self.location_graph[current_box_location]:
                    # Check if pushing to next_loc_candidate moves the box strictly closer to the goal
                    dist_next_to_goal = self.distances.get((next_loc_candidate, goal_location))
                    if dist_next_to_goal is not None and dist_next_to_goal < box_dist_to_goal:

                        # This push direction (dir_from_box) is towards the goal.
                        # The robot needs to be at the location adjacent to current_box_location
                        # in the opposite direction.
                        required_robot_dir = self.reverse_dir.get(dir_from_box)
                        if required_robot_dir is None:
                            # Should not happen if reverse_dir is complete
                            continue

                        # Find the location adjacent to current_box_location in the required_robot_dir
                        if current_box_location in self.location_graph:
                             for neighbor_of_box, dir_from_neighbor in self.location_graph[current_box_location]:
                                 if dir_from_neighbor == required_robot_dir:
                                      # This neighbor_of_box is a valid position for the robot
                                      # to push the box towards the goal.
                                      all_valid_robot_start_locs.add(neighbor_of_box)

        # If no boxes need moving, heuristic is 0
        if not found_box_to_move:
            return 0

        # Calculate minimum robot approach cost to any valid push position
        # Only consider robot locations that are part of the graph
        if robot_location in self.location_graph:
            for r_loc in all_valid_robot_start_locs:
                 robot_dist_to_r_loc = self.distances.get((robot_location, r_loc))
                 if robot_dist_to_r_loc is not None:
                     min_robot_approach_cost = min(min_robot_approach_cost, robot_dist_to_r_loc)
        # else: robot_location is not in graph, likely invalid state, min_robot_approach_cost remains inf

        # If robot cannot reach any valid push position, return infinity
        if min_robot_approach_cost == float('inf'):
             return float('inf')

        # Total heuristic is sum of pushes + robot approach cost
        return total_pushes_cost + min_robot_approach_cost
