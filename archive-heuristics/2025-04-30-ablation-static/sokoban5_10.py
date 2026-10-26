from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base is in a 'heuristics' directory relative to where this code runs
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check to prevent index errors if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function for shortest path on the location graph
def bfs(start_loc, graph):
    """Computes shortest path distances from start_loc to all other locations."""
    distances = {loc: float('inf') for loc in graph}
    if start_loc not in distances:
         # Start location is not in the graph of locations derived from adjacent facts.
         # This might happen if a goal/initial location is isolated.
         # We can't compute distances from here.
         return {} # Return empty dict or handle as unreachable

    distances[start_loc] = 0
    queue = deque([start_loc])
    while queue:
        curr_loc = queue.popleft()
        # graph stores neighbors as a list of locations
        for neighbor in graph.get(curr_loc, []):
            if distances.get(neighbor, float('inf')) == float('inf'):
                distances[neighbor] = distances[curr_loc] + 1
                queue.append(neighbor)
    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components for each box not at its goal:
    1. The shortest path distance (number of pushes) for the box from its current location to its goal location.
    2. The shortest path distance for the robot from its current location to the required position to perform the *first* push of the box along a shortest path towards its goal.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - A box can only be pushed if the location behind it is clear and the robot is in the correct adjacent position.
    - The heuristic assumes the box can be pushed along a shortest path, ignoring potential blockages by other boxes or the robot itself beyond the first step's required robot position.
    - The heuristic returns infinity if a box goal is unreachable or if the required robot position for the first push is unreachable or non-existent (e.g., box against a wall).

    # Heuristic Initialization
    - Extracts goal locations for each box that appears in the goal state.
    - Builds the location graph based on `adjacent` facts, treating edges as undirected for distance calculation.
    - Stores `adjacent` relations including directions for finding pushing positions.
    - Precomputes all-pairs shortest path distances between locations using BFS on the undirected graph.
    - Defines mapping for reverse directions.
    - Collects all unique locations mentioned in static facts, initial state, and goals to ensure the distance map covers all relevant locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state by verifying if all goal facts are present in the state. If yes, return 0.
    2. Identify the current location of the robot from the state facts. If the robot's location is not found, return infinity (invalid state).
    3. Identify the current location of each box that has a goal location specified in the task's goals. If any such box is not found in the state facts, return infinity (invalid state).
    4. Initialize the total heuristic cost `h` to 0.
    5. Iterate through each box that has a goal location:
        a. Get the box's current location `loc_b_curr` and its goal location `loc_b_goal`.
        b. Calculate the shortest path distance `box_dist` between `loc_b_curr` and `loc_b_goal` using the precomputed distances. If either location is not in the distance map (e.g., an isolated location not connected to the main graph), the distance is considered infinity.
        c. If `box_dist` is infinity, the goal is unreachable for this box; return infinity for the heuristic, as the problem is likely unsolvable from this state.
        d. Add `box_dist` to `h`. This component estimates the minimum number of push actions required for this box.
        e. If `box_dist > 0` (meaning the box is not yet at its goal and needs to be moved):
            i. Find a location `p_1` that is adjacent to `loc_b_curr` and lies on a shortest path from `loc_b_curr` to `loc_b_goal`. This is a neighbor `p_1` of `loc_b_curr` in the undirected graph such that the precomputed distance from `p_1` to `loc_b_goal` is exactly one less than `box_dist`.
            ii. If no such `p_1` is found, it implies the box cannot move closer to the goal along a shortest path from its current position (potentially a dead end or calculation issue); return infinity.
            iii. Determine the direction `dir` required to move from `loc_b_curr` to `p_1` by looking up the directed relation in `self.adj_relations`.
            iv. Determine the required robot location `loc_r` to push the box from `loc_b_curr` to `p_1`. This `loc_r` must be adjacent to `loc_b_curr` in the direction opposite to `dir`.
            v. If no such `loc_r` exists (e.g., `loc_b_curr` is against a wall on the side the robot needs to be), return infinity, as the required push cannot be initiated from this position.
            vi. Calculate the shortest path distance `robot_dist` from the robot's current location (`robot_loc`) to the required robot position (`loc_r`) using the precomputed distances. If either location is not in the distance map, distance is infinity.
            vii. If `robot_dist` is infinity, the robot cannot reach the required pushing position; return infinity.
            viii. Add `robot_dist` to `h`. This component estimates the robot movement cost to enable the first push for this box.
    6. After processing all boxes, return the total accumulated heuristic cost `h`.

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Extract goal locations for each box that appears in the goal state.
        self.box_goals = {}
        self.goal_boxes = set() # Only consider boxes that have a goal location
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming facts like (at box1 loc_...) in goals refer to boxes
            if predicate == "at" and len(args) == 2 and args[0].startswith('box'):
                box, location = args
                self.box_goals[box] = location
                self.goal_boxes.add(box)

        # 2. Build the location graph (undirected for distance) and store adjacent relations (directed).
        self.graph = {}
        self.adj_relations = {} # Stores (l1, l2) -> direction
        locations = set() # Collect all unique locations

        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent" and len(args) == 3:
                l1, l2, direction = args
                locations.add(l1)
                locations.add(l2)
                if l1 not in self.graph:
                    self.graph[l1] = []
                if l2 not in self.graph:
                     self.graph[l2] = []
                # Add edge for distance calculation (undirected)
                # Avoid adding duplicate edges if PDDL lists both directions explicitly
                if l2 not in self.graph[l1]:
                    self.graph[l1].append(l2)
                if l1 not in self.graph[l2]:
                    self.graph[l2].append(l1)

                # Store directed relation
                self.adj_relations[(l1, l2)] = direction

        # Ensure all locations mentioned in goals/initial state are included, even if isolated
        # This is important so BFS can compute distances to/from these locations.
        # Add locations from goals
        for loc in self.box_goals.values():
             locations.add(loc)
             if loc not in self.graph: self.graph[loc] = []
        # Add locations from initial state (robot and boxes)
        for fact in initial_state:
             predicate, *args = get_parts(fact)
             # Assuming location is the last argument for 'at-robot' and 'at'
             if predicate in ["at-robot", "at"] and len(args) > 0:
                 loc = args[-1]
                 locations.add(loc)
                 if loc not in self.graph: self.graph[loc] = []


        # 3. Precompute all-pairs shortest path distances using BFS.
        self.dist = {}
        # Use the collected set of all relevant locations
        all_locations_list = list(locations)
        for start_loc in all_locations_list:
            # Ensure start_loc is in graph keys, even if it has no neighbors
            if start_loc not in self.graph:
                 self.graph[start_loc] = []
            self.dist[start_loc] = bfs(start_loc, self.graph)

        # 4. Define mapping for reverse directions.
        self.reverse_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Identify robot location.
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
             # Should not happen in valid Sokoban states
             return float('inf') # Robot location unknown

        # 3. Identify current box locations for boxes that have goals.
        current_box_locations = {}
        for box in self.goal_boxes:
             found = False
             for fact in state:
                 if match(fact, "at", box, "*"):
                     current_box_locations[box] = get_parts(fact)[2]
                     found = True
                     break
             # If a box with a goal is not 'at' a location in the current state,
             # the state is likely invalid or unsolvable.
             if not found:
                 return float('inf')


        # 4. Initialize heuristic cost.
        total_cost = 0

        # 5. For each box not at its goal location:
        for box in self.goal_boxes:
            current_location = current_box_locations.get(box)
            goal_location = self.box_goals.get(box)

            # These checks should pass based on the loop over self.goal_boxes and check above,
            # but kept for clarity/safety.
            if current_location is None or goal_location is None:
                 return float('inf')

            if current_location != goal_location:
                # a. Calculate box distance
                # Use .get with inf default in case current_location or goal_location
                # wasn't in the locations collected during init (e.g., isolated)
                box_dist = self.dist.get(current_location, {}).get(goal_location, float('inf'))

                # c. If box_dist is infinity, goal is unreachable for this box
                if box_dist == float('inf'):
                    return float('inf')

                # d. Add box_dist to total cost (estimated pushes)
                total_cost += box_dist

                # e. If box needs to move (box_dist > 0)
                if box_dist > 0:
                    # i. Find the next location p_1 on a shortest path
                    p_1 = None
                    # Neighbors in the graph are already the locations adjacent to current_location
                    # Iterate through neighbors in the undirected graph
                    for neighbor in self.graph.get(current_location, []):
                        # Check if this neighbor is one step closer to the goal in the precomputed distances
                        # Use .get with inf default for safety
                        if self.dist.get(neighbor, {}).get(goal_location, float('inf')) == box_dist - 1:
                            p_1 = neighbor
                            break # Found one such neighbor, pick it

                    # If no such neighbor exists, something is wrong with the distance calculation
                    # or the state is a dead end where the box cannot move closer along a shortest path.
                    if p_1 is None:
                         return float('inf') # Cannot make progress towards goal

                    # ii. Determine the direction from current_location to p_1
                    # Look up the direction in adj_relations (directed graph)
                    direction = self.adj_relations.get((current_location, p_1))
                    if direction is None:
                         # This implies p_1 is adjacent to current_location in the undirected graph,
                         # but there's no explicit directed adjacent fact (current_location, p_1, dir).
                         # This shouldn't happen if the graph and adj_relations are built correctly
                         # from the same adjacent facts which should be bidirectional.
                         return float('inf') # Consistency check failed

                    # iii. Determine the required robot location loc_r
                    reverse_dir = self.reverse_direction.get(direction)
                    if reverse_dir is None:
                         # Should not happen if reverse_direction is complete
                         return float('inf') # Consistency check failed

                    loc_r = None
                    # Find the location adjacent to current_location in the reverse direction
                    # Iterate through neighbors in the undirected graph
                    for neighbor_r in self.graph.get(current_location, []):
                         # Check if the directed relation from current_location to neighbor_r
                         # matches the required reverse direction.
                         if self.adj_relations.get((current_location, neighbor_r)) == reverse_dir:
                              loc_r = neighbor_r
                              break # Found the required robot position

                    # iv. If no such loc_r exists
                    if loc_r is None:
                         # The box is against a wall or obstacle in the direction the robot needs to be.
                         # This state might be a dead end for this box.
                         return float('inf') # Unsolvable state for this box


                    # v. Calculate robot distance to loc_r
                    # Use .get with inf default in case robot_loc or loc_r
                    # wasn't in the locations collected during init (e.g., isolated)
                    robot_dist = self.dist.get(robot_loc, {}).get(loc_r, float('inf'))

                    # vi. If robot_dist is infinity, robot cannot reach pushing position
                    if robot_dist == float('inf'):
                        return float('inf')

                    # vii. Add robot_dist to total cost
                    total_cost += robot_dist

        # 6. Return total heuristic cost
        return total_cost
