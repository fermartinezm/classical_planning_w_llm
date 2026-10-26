from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing two components for each box not yet at its goal:
    1. The shortest path distance for the box from its current location to its goal location on the full grid graph (ignoring obstacles).
    2. The shortest path distance for the robot from its current location to a valid pushing position adjacent to the box, considering current obstacles (other boxes).

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - Locations are named `loc_R_C`.
    - The cost of a move action is 1.
    - The cost of a push action is 1.
    - The heuristic assumes that moving a box one step towards its goal costs 1 (the push) plus the cost for the robot to get into the necessary pushing position for that step. It simplifies this by only calculating the robot cost for the *first* step towards the goal.

    # Heuristic Initialization
    - Parses `adjacent` facts to build the full undirected graph of locations.
    - Computes all-pairs shortest path distances on this full grid graph.
    - Stores box goal locations from the goal state.
    - Stores adjacency information including directions for finding pushing positions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. Identify all locations currently occupied by boxes (these are obstacles for robot movement).
    3. Build a robot movement graph where edges exist between adjacent locations if the target location is not occupied by a box.
    4. Run a Breadth-First Search (BFS) from the robot's current location on the robot movement graph to find the shortest distance from the robot to every reachable location.
    5. Initialize the total heuristic value `h` to 0.
    6. For each box that has a goal location:
        a. If the box is already at its goal location, its contribution is 0.
        b. If the box is not at its goal:
            i. Get the box's current location `L_box` and its goal location `L_goal`.
            ii. Calculate the shortest path distance `box_dist` from `L_box` to `L_goal` on the full grid graph (precomputed in `__init__`). This is the minimum number of pushes required for this box, ignoring dynamic obstacles.
            iii. Find all locations `L_push` adjacent to `L_box` that would allow pushing the box one step towards `L_goal` along a shortest path on the full grid. A location `L_neighbor` is a valid next step if its distance to `L_goal` is `box_dist - 1`. The corresponding pushing position `L_push` is adjacent to `L_box` in the opposite direction of the push (from `L_box` to `L_neighbor`).
            iv. Find the minimum robot distance `min_robot_dist` from the robot's current location to any of the valid pushing positions `L_push` found in the previous step, using the distances computed by the BFS on the robot movement graph.
            v. If no valid pushing position is reachable by the robot, this box might be in a difficult situation; add a large penalty (infinity).
            vi. The contribution for this box is `box_dist + min_robot_dist`.
            vii. Add this contribution to the total heuristic value `h`.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build the full undirected graph from adjacent facts
        self.graph = {}
        self.adj_info = {} # Store direction info: loc -> [(neighbor, dir), ...]
        self.opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                    self.adj_info[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                    # self.adj_info[loc2] = [] # We only need adj_info for the source location

                # Add undirected edge
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

                # Store directed adjacency info (from loc1 to loc2 in direction)
                self.adj_info[loc1].append((loc2, direction))

        self.locations = list(locations) # Store list of all locations

        # 2. Compute all-pairs shortest path distances on the full grid graph
        self.location_distances = {}
        for start_node in self.locations:
            self.location_distances[start_node] = self._bfs(start_node, self.graph)

        # 3. Store box goal locations
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def _bfs(self, start_node, graph, obstacles=None):
        """
        Performs BFS from a start_node on the given graph to find distances to all other nodes.
        Optionally takes a set of obstacle nodes that cannot be entered.
        Returns a dictionary mapping node to distance.
        """
        if obstacles is None:
            obstacles = set()

        distances = {node: float('inf') for node in graph}
        
        # Start node is reachable unless it's an obstacle itself (shouldn't happen for robot start)
        if start_node not in obstacles:
             distances[start_node] = 0
             queue = deque([start_node])
        else:
             # Robot starts in an obstacle? Should not happen in valid problems.
             # Or maybe the start_node is a pushing position that is currently occupied.
             # If the start_node is an obstacle, it's unreachable from itself for BFS purposes.
             queue = deque([])


        while queue:
            current_node = queue.popleft()

            if current_node in graph:
                 for neighbor in graph[current_node]:
                    if neighbor not in obstacles and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current robot and box locations
        robot_location = None
        box_locations = {} # box_name -> location
        
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        # 2. Identify locations occupied by boxes (obstacles for robot movement)
        # The robot's own location is not an obstacle for itself to move from.
        box_obstacle_locations = set(box_locations.values())

        # 3. Build robot movement graph for the current state and run BFS
        # Robot can move to any adjacent location that is NOT occupied by a box.
        # The graph nodes are all possible locations. Edges are restricted by obstacles.
        robot_distances = self._bfs(robot_location, self.graph, obstacles=box_obstacle_locations)

        # 4. Calculate heuristic sum for each box
        total_heuristic = 0

        for box, goal_location in self.goal_locations.items():
            current_box_location = box_locations.get(box)

            # If box is not in state or already at goal, its contribution is 0
            if current_box_location is None or current_box_location == goal_location:
                continue

            # i. Box distance to goal on full grid
            # Handle case where box_location or goal_location might not be in the precomputed distances
            # (e.g., if the location set was incomplete, or problem is malformed)
            if current_box_location not in self.location_distances or goal_location not in self.location_distances[current_box_location]:
                 # Cannot calculate box distance, treat as unsolvable or very high cost
                 return float('inf')

            box_dist = self.location_distances[current_box_location][goal_location]

            # If box_dist is inf, it's unreachable on the grid graph - unsolvable
            if box_dist == float('inf'):
                 return float('inf')

            # ii. Find valid pushing positions for the first step towards goal
            valid_push_positions = set()
            
            # Find neighbors of the box's current location on the full grid
            if current_box_location in self.adj_info:
                for neighbor_loc, direction in self.adj_info[current_box_location]:
                    # Check if this neighbor is on a shortest path to the goal on the full grid
                    # Ensure neighbor_loc is in precomputed distances
                    if neighbor_loc in self.location_distances[current_box_location] and \
                       self.location_distances[current_box_location][neighbor_loc] == 1 and \
                       self.location_distances[neighbor_loc][goal_location] == box_dist - 1:

                        # This neighbor_loc is a valid next step for the box.
                        # The robot needs to be adjacent to current_box_location
                        # in the opposite direction of the push (current_box_location -> neighbor_loc).
                        opposite_dir = self.opposite_direction.get(direction)
                        if opposite_dir:
                            # Find the location adjacent to current_box_location in the opposite direction
                            # Look through adj_info from current_box_location
                            if current_box_location in self.adj_info:
                                for potential_push_loc, push_dir in self.adj_info[current_box_location]:
                                    if push_dir == opposite_dir:
                                        valid_push_positions.add(potential_push_loc)
                                        break # Found the location in the opposite direction

            # iii. Find minimum robot distance to a valid pushing position
            min_robot_dist = float('inf')
            if valid_push_positions:
                for push_pos in valid_push_positions:
                    # Check distance from the robot's BFS result
                    if push_pos in robot_distances:
                         min_robot_dist = min(min_robot_dist, robot_distances[push_pos])
                    # else: push_pos is unreachable by robot in the current state, distance remains inf

            # iv. Calculate contribution for this box
            if min_robot_dist == float('inf'):
                 # Robot cannot reach any position to push the box towards the goal
                 # along a shortest path on the grid.
                 # This might indicate a deadlock or require moving other boxes first.
                 # Return infinity to prune this state in greedy search.
                 return float('inf')

            # Heuristic contribution: box distance + robot distance to get into position for the first push
            total_heuristic += box_dist + min_robot_dist

        return total_heuristic
