from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Helper function to split a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def opposite_of(dir):
    """Helper function to get the opposite direction."""
    if dir == 'up': return 'down'
    if dir == 'down': return 'up'
    if dir == 'left': return 'right'
    if dir == 'right': return 'left'
    return None # Should not happen in this domain

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    contributions for each box that is not yet at its goal location.
    For each such box, the contribution is the sum of:
    1. The shortest path distance for the robot to reach the location
       immediately behind the box, allowing it to push the box towards
       its goal.
    2. The shortest path distance for the box to reach its goal location.
    Shortest path distances are computed using Breadth-First Search (BFS)
    on the grid graph defined by the 'adjacent' predicates. The robot's
    path calculation considers other boxes as obstacles.

    Assumptions:
    - The locations form a grid-like structure where 'adjacent' predicates
      define the traversable graph.
    - 'adjacent' predicates are mostly symmetric (if A is adjacent to B,
      B is adjacent to A), allowing BFS from goals.
    - The goal specifies a unique target location for each box (or multiple
      boxes can share a goal location).
    - The robot cannot move onto a location occupied by a box.
    - The heuristic assumes a box can be pushed towards its goal if the
      location behind it exists and the path for the box is clear (which
      is implicitly handled by the box distance calculation).

    Heuristic Initialization:
    The constructor performs the following steps once:
    1. Parses the goal facts to create a mapping from each box object to its
       target location (`self.goal_locations`).
    2. Builds an adjacency list representation of the grid graph from the
       'adjacent' static facts (`self.adj`).
    3. Builds a map from a location and a direction to the resulting neighbor
       location (`self.location_neighbor_in_dir`).
    4. Builds a map from a pair of adjacent locations to the direction required
       to move between them (`self.dir_map`).
    5. Pre-calculates the shortest path distances from all *unique goal locations*
       to all reachable locations using BFS on the graph. This allows efficient
       lookup of the box-to-goal distance and finding the next step on a
       shortest path during heuristic computation (`self.dist_to_goal`).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the current location of the robot and each box by parsing
       the state facts.
    2. Initialize the total heuristic value `total_h` to 0.
    3. For each box and its assigned goal location from `self.goal_locations`:
        a. Get the box's current location.
        b. If the box is already at its goal location, continue to the next box.
        c. Look up the pre-calculated shortest path distance (`box_dist`) from the
           box's current location to its goal location using `self.dist_to_goal`.
           If the goal is unreachable for the box, the state is likely unsolvable,
           and the heuristic returns infinity.
        d. Find a location adjacent to the box's current location that is one step
           closer to the goal according to the pre-calculated distances. This
           determines the required direction (`push_dir`) for the first push.
           If no such location exists (should only happen if box_dist is 0,
           which is handled), return infinity.
        e. Determine the `required_robot_loc`, which is the location adjacent
           to the box's current location in the direction opposite to `push_dir`.
           If no such location exists (e.g., wall behind the box), return infinity.
        f. Calculate the shortest path distance (`robot_dist`) for the robot
           to move from its current location to `required_robot_loc`. This BFS
           considers all *other* box locations as obstacles. If the robot cannot
           reach the required position, return infinity.
        g. Add `robot_dist + box_dist` to `total_h`.
    4. Return the final `total_h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goals: map box to goal location
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Data structures for graph and location relationships
        self.adj = {} # Adjacency list: {loc: [neighbor1, neighbor2, ...]}
        self.location_neighbor_in_dir = {} # Map: {(loc, dir): neighbor_loc}
        self.dir_map = {} # Map: {(loc1, loc2): dir}
        self.opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # 2, 3, 4. Build graph structures from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                _, loc1, loc2, dir = parts
                self.adj.setdefault(loc1, []).append(loc2)
                self.location_neighbor_in_dir[(loc1, dir)] = loc2
                self.dir_map[(loc1, loc2)] = dir

        # 5. Pre-calculate distances from goals using BFS
        self.dist_to_goal = {}
        all_goal_locations = set(self.goal_locations.values())
        for goal_loc in all_goal_locations:
            self.dist_to_goal[goal_loc] = self._bfs_from_goal(goal_loc)

    def _bfs_distance(self, start, end, adj, obstacles=None):
        """
        Performs BFS to find the shortest path distance between start and end.
        Optionally avoids locations in the obstacles set.
        """
        if start == end:
            return 0
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current_loc, dist = queue.popleft()
            if current_loc == end:
                return dist
            # Check neighbors from the adjacency list
            for neighbor in adj.get(current_loc, []):
                if obstacles and neighbor in obstacles:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf') # Unreachable

    def _bfs_from_goal(self, goal_loc):
        """
        Performs BFS starting from the goal location to find distances
        of all reachable locations *to* the goal. Uses the reverse graph.
        """
        distances = {goal_loc: 0}
        queue = deque([goal_loc])

        # Build reverse adjacency list for BFS from goal
        reverse_adj = {}
        for loc, neighbors in self.adj.items():
            for neighbor in neighbors:
                reverse_adj.setdefault(neighbor, []).append(loc)

        while queue:
            current_loc = queue.popleft()
            dist = distances[current_loc]
            # Iterate through locations that can reach current_loc
            for neighbor in reverse_adj.get(current_loc, []):
                 if neighbor not in distances:
                     distances[neighbor] = dist + 1
                     queue.append(neighbor)
        return distances


    def __call__(self, node):
        state = node.state

        # 1. Identify robot and box locations
        robot_loc = None
        current_box_locations = {}
        # clear_locations = set() # Not needed for this heuristic
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box_name = parts[1]
                location = parts[2]
                current_box_locations[box_name] = location
            # elif parts[0] == 'clear':
            #      clear_locations.add(parts[1])

        if robot_loc is None:
             # Should not happen in a valid state
             return float('inf')

        # 2. Initialize total heuristic
        total_h = 0

        # 3. Calculate contribution for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box) # Use .get for safety

            if current_loc is None:
                 # Box not found in state facts - should not happen in valid states
                 return float('inf')

            # a. Check if box is at goal
            if current_loc == goal_loc:
                continue # Box is home, no cost

            # c. Get box distance to goal
            # Use pre-calculated distances from goal
            box_dist = self.dist_to_goal.get(goal_loc, {}).get(current_loc, float('inf'))

            if box_dist == float('inf'):
                # Box cannot reach its goal
                return float('inf')

            # d. Find next_loc on a shortest path and push_dir
            next_loc = None
            push_dir = None
            # Iterate through neighbors of current_loc
            for neighbor in self.adj.get(current_loc, []):
                # Check if neighbor is one step closer to the goal
                if neighbor in self.dist_to_goal.get(goal_loc, {}) and self.dist_to_goal[goal_loc][neighbor] == box_dist - 1:
                    next_loc = neighbor
                    # Find the direction from current_loc to next_loc
                    push_dir = self.dir_map.get((current_loc, next_loc))
                    if push_dir: # Found a valid next step and direction
                        break

            if next_loc is None:
                 # This should ideally not happen if box_dist > 0 and finite,
                 # assuming a well-formed graph where paths exist.
                 # Return inf as a safeguard.
                 if box_dist > 0:
                     return float('inf')
                 else: # box_dist is 0, already at goal, handled by continue
                     continue # Should not reach here

            # e. Determine required_robot_loc
            # Robot must be at the location behind current_loc relative to next_loc
            required_robot_loc = self.location_neighbor_in_dir.get((current_loc, opposite_of(push_dir)))

            if required_robot_loc is None:
                # No location exists behind the box in the required push direction (e.g., wall)
                # This state might be unsolvable via this path
                return float('inf')

            # f. Calculate robot distance to required_robot_loc, avoiding other boxes
            # Obstacles for the robot are all locations occupied by *any* box
            robot_obstacles = set(current_box_locations.values())

            robot_dist = self._bfs_distance(robot_loc, required_robot_loc, self.adj, obstacles=robot_obstacles)

            if robot_dist == float('inf'):
                # Robot cannot reach the position to push the box
                return float('inf')

            # g. Add contribution
            total_h += robot_dist + box_dist

        # 4. Return total heuristic value
        return total_h

