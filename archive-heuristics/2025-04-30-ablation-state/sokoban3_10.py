from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def opposite_dir(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen in this domain

def bfs_distance(start, end, graph, obstacles=None):
    """
    Calculates the shortest path distance between start and end in the graph,
    avoiding obstacles.

    Args:
        start (str): The starting location.
        end (str): The target location.
        graph (dict): The adjacency list representation of the grid graph.
        obstacles (set, optional): A set of locations that cannot be visited. Defaults to None.

    Returns:
        int or float('inf'): The shortest distance, or infinity if the end is unreachable.
    """
    if obstacles is None:
        obstacles = set()

    # Cannot start or end in an obstacle (unless start is the obstacle itself, which is handled by BFS)
    if end in obstacles:
        return math.inf

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Check neighbors from the graph
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                # Cannot move to an obstacle location
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return math.inf # End not reachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It is calculated as the sum, over all misplaced boxes, of:
    1. The minimum number of push actions required to move the box from its current
       location to its goal location on the empty grid (box distance).
    2. The minimum number of move actions required for the robot to reach the
       correct position to push the box one step towards its goal, considering
       other boxes as obstacles (robot approach distance).

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - The goal is to move specific boxes to specific locations.
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Parses `adjacent` facts from static information to build the grid graph (adjacency list).
    - Computes all-pairs shortest path distances on the empty grid graph using BFS.
    - Parses goal conditions to map each box to its goal location.
    - Parses `adjacent` facts to map adjacent location pairs to directions, used for finding robot push positions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. For each box that is not at its goal location:
       a. Calculate the minimum number of push actions (`box_distance`) needed for this box
          to reach its goal on the empty grid (using pre-computed distances). This is the
          shortest path distance for the box on the grid graph.
       b. If `box_distance` is 0 (box is at goal) or infinity (goal unreachable), its contribution is just `box_distance`.
       c. If `box_distance` > 0:
          i. Find a location `next_loc_b` adjacent to the box's current location that is
             one step closer to the goal on the empty grid. There might be multiple such locations;
             we pick one arbitrarily (the first one found).
          ii. Determine the direction (`push_dir`) from the box's current location (`box_loc`) to `next_loc_b`.
          iii. Determine the required robot position (`robot_push_pos`). This is the location
              adjacent to `box_loc` such that moving from `robot_push_pos` to `box_loc` is
              in the same direction as `push_dir`. This is the location "behind" the box.
          iv. Calculate the minimum number of move actions (`robot_approach_distance`) for the
              robot to get from its current location (`robot_loc`) to `robot_push_pos`. This is done
              using BFS on the grid graph, treating locations occupied by *other* boxes
              (excluding the box being considered) as obstacles. If `robot_push_pos` is
              occupied by another box, this distance is considered infinite.
       d. Add `box_distance + robot_approach_distance` to the total heuristic value.
    3. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build grid graph and direction mapping from adjacent facts
        self.grid_graph = {}
        self.directions = {} # Maps (loc1, loc2) -> direction
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                if l1 not in self.grid_graph:
                    self.grid_graph[l1] = []
                if l2 not in self.grid_graph:
                    self.grid_graph[l2] = []
                # Add bidirectional edges for robot movement
                self.grid_graph[l1].append(l2)
                self.grid_graph[l2].append(l1)
                # Store directed adjacency for push logic
                self.directions[(l1, l2)] = direction
                self.directions[(l2, l1)] = opposite_dir(direction)

        # Compute all-pairs shortest paths on the empty grid
        self.distances = {}
        all_locations = list(self.grid_graph.keys())
        for start_loc in all_locations:
            self.distances[start_loc] = {}
            # Use BFS to find distances from start_loc to all other locations
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while queue:
                current_loc, dist = queue.popleft()

                if current_loc in self.grid_graph:
                    for neighbor in self.grid_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

        # Store goal locations for each box and identify all box names
        self.goal_locations = {}
        self.box_names = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location
                self.box_names.add(box)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify robot and box locations
        robot_loc = None
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1] in self.box_names:
                box, loc = parts[1], parts[2]
                current_box_locations[box] = loc

        total_heuristic = 0

        # Calculate heuristic for each misplaced box
        for box, goal_loc in self.goal_locations.items():
            box_loc = current_box_locations.get(box)

            # If box location is unknown or it's already at the goal, skip
            if box_loc is None or box_loc == goal_loc:
                continue

            # 1. Box distance (minimum pushes on empty grid)
            # Use pre-computed distances. Default to infinity if locations are not in graph or unreachable.
            box_dist = self.distances.get(box_loc, {}).get(goal_loc, math.inf)

            if box_dist == math.inf:
                # Box cannot reach goal even on empty grid (e.g., trapped)
                total_heuristic += math.inf
                continue # Cannot calculate robot distance meaningfully if box is stuck

            total_heuristic += box_dist

            # 2. Robot approach distance (to push the box one step towards goal)
            if box_dist > 0:
                # Find a neighbor that is one step closer to the goal on the empty grid
                next_loc_b = None
                # Iterate through neighbors of the box's current location
                for neighbor in self.grid_graph.get(box_loc, []):
                    # Check if this neighbor is one step closer to the goal
                    if self.distances.get(neighbor, {}).get(goal_loc, math.inf) == box_dist - 1:
                        next_loc_b = neighbor
                        break # Found a suitable next step

                if next_loc_b is None:
                    # This case implies the box is not on a path towards the goal
                    # or the pre-computed distances are inconsistent.
                    # For a non-admissible heuristic, we can add a penalty or infinity.
                    # Adding infinity signals a potential dead-end or complex situation.
                    total_heuristic += math.inf
                    continue

                # Determine the required push direction from box_loc to next_loc_b
                push_dir = self.directions.get((box_loc, next_loc_b))
                if push_dir is None:
                     # Should not happen if grid_graph and directions are built correctly
                     total_heuristic += math.inf
                     continue

                # Determine the required robot push position
                # This is the location adjacent to box_loc such that moving from
                # robot_push_pos to box_loc is in the same direction as push_dir.
                robot_push_pos = None
                # Iterate through neighbors of the box's current location
                for neighbor_of_box in self.grid_graph.get(box_loc, []):
                    # Check if the direction from this neighbor to box_loc is the push_dir
                    if self.directions.get((neighbor_of_box, box_loc)) == push_dir:
                        robot_push_pos = neighbor_of_box
                        break # Found the push position

                if robot_push_pos is None:
                    # Should not happen if grid_graph and directions are built correctly
                    total_heuristic += math.inf
                    continue

                # Calculate robot distance considering obstacles (other boxes)
                # Obstacles are locations occupied by any box *other than* the one we are considering pushing.
                obstacles = {loc for b, loc in current_box_locations.items() if b != box}

                # Calculate the shortest path distance for the robot from its current location
                # to the required push position, avoiding obstacle locations.
                robot_dist = bfs_distance(robot_loc, robot_push_pos, self.grid_graph, obstacles)

                total_heuristic += robot_dist

        # The heuristic is 0 if and only if all boxes are at their goal locations,
        # as the loop calculates 0 in that case. If any box is misplaced and solvable,
        # box_dist will be > 0, making the heuristic > 0. If unsolvable, it will be inf.

        return total_heuristic
