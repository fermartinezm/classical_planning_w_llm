# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the environment
from fnmatch import fnmatch
import collections # For BFS queue

# Dummy Heuristic base class for standalone testing
# In the actual environment, this would be provided.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum of shortest path distances for each box
    to its goal, plus the minimum shortest path distance for the robot
    to reach a position from which it can push any box towards its goal.

    This heuristic is not admissible but aims to guide a greedy best-first
    search efficiently.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and
        precomputing all-pairs shortest paths.
        """
        super().__init__(task)

        # Build grid graph from adjacent facts
        self.locations = set()
        # adj_graph[loc][dir] = neighbor_loc (directed)
        self.adj_graph = {}
        # adj_list[loc] = [neighbor_loc1, neighbor_loc2, ...] (undirected for BFS)
        self.adj_list = {}

        # Map direction string to its opposite
        self.opposite_dir = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }

        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)

                if loc1 not in self.adj_graph:
                    self.adj_graph[loc1] = {}
                self.adj_graph[loc1][direction] = loc2

                if loc1 not in self.adj_list:
                    self.adj_list[loc1] = []
                if loc2 not in self.adj_list:
                    self.adj_list[loc2] = []

                # Add undirected edges for BFS distance calculation
                if loc2 not in self.adj_list[loc1]:
                     self.adj_list[loc1].append(loc2)
                if loc1 not in self.adj_list[loc2]:
                     self.adj_list[loc2].append(loc1)


        # Compute all-pairs shortest paths using BFS from each location
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

        # Store goal locations for boxes
        self.goal_locations = {} # box -> location
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, location = get_parts(goal)
                self.goal_locations[box] = location

    def _bfs(self, start_loc):
        """Compute shortest path distances from start_loc to all other locations using BFS."""
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_loc] = 0
        queue = collections.deque([start_loc])
        visited = {start_loc}

        while queue:
            current_loc = queue.popleft()

            # Use the undirected adjacency list for distance calculation
            neighbors = self.adj_list.get(current_loc, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        The heuristic is the sum of:
        1. Sum of shortest path distances for each box to its goal location.
        2. Minimum shortest path distance for the robot to reach a location
           from which it can push any box towards its goal.
        """
        state = node.state

        # Find current robot and box locations
        robot_loc = None
        box_locations = {} # box -> location

        for fact in state:
            if match(fact, "at-robot", "*"):
                _, loc = get_parts(fact)
                robot_loc = loc
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        # Identify boxes that are not at their goal
        boxes_to_move = {b for b, loc in box_locations.items() if self.goal_locations.get(b) and loc != self.goal_locations[b]}

        if not boxes_to_move:
            return 0 # Goal reached

        # Component 1: Sum of box-to-goal distances
        h_boxes = 0
        unreachable_goal = False
        for box in boxes_to_move:
            loc_b = box_locations[box]
            loc_g = self.goal_locations[box]
            # Use precomputed shortest path distance on the grid graph
            box_dist = self.dist.get(loc_b, {}).get(loc_g, float('inf'))
            if box_dist == float('inf'):
                 # Box goal is unreachable on the grid (implies deadlock or unsolvable)
                 unreachable_goal = True
                 break
            h_boxes += box_dist

        if unreachable_goal:
             return float('inf')

        # Component 2: Minimum robot distance to a required push position
        # A required push position for box b at loc_b towards goal loc_g
        # is a location loc_r such that adjacent(loc_r, loc_b, push_dir)
        # where push_dir is the direction from loc_r to loc_b, AND
        # adjacent(loc_b, loc_next, push_dir) where loc_next is the location
        # the box moves to, AND dist(loc_next, loc_g) < dist(loc_b, loc_g).
        # This means loc_r is adjacent to loc_b in the direction *opposite* of the push direction.
        # If box moves from loc_b to loc_next in direction d, robot must be at loc_r
        # such that adjacent(loc_r, loc_b, opposite(d)).

        min_robot_dist_to_push_pos = float('inf')
        robot_can_reach_any_push_pos = False

        for box in boxes_to_move:
            loc_b = box_locations[box]
            loc_g = self.goal_locations[box]

            current_box_dist = self.dist.get(loc_b, {}).get(loc_g, float('inf'))
            if current_box_dist == float('inf'):
                 # Should be caught by unreachable_goal check, but defensive
                 continue

            # Find valid push directions from loc_b that reduce distance to loc_g
            # Iterate through neighbors of loc_b using the directed graph to get directions
            for push_dir, loc_next in self.adj_graph.get(loc_b, {}).items():
                 # Check if moving box to loc_next reduces distance to goal
                 dist_next_to_goal = self.dist.get(loc_next, {}).get(loc_g, float('inf'))

                 if dist_next_to_goal < current_box_dist:
                      # loc_next is a step towards the goal by pushing in push_dir.
                      # Find the required robot location behind loc_b for this push direction.
                      # The robot must be at loc_r such that adjacent(loc_r, loc_b, push_dir).
                      # This means loc_r is adjacent to loc_b in the direction opposite to push_dir.
                      required_robot_pos = self.adj_graph.get(loc_b, {}).get(self.opposite_dir.get(push_dir))

                      if required_robot_pos:
                           # Calculate distance from robot's current location to this required position
                           robot_dist_to_this_pos = self.dist.get(robot_loc, {}).get(required_robot_pos, float('inf'))

                           if robot_dist_to_this_pos != float('inf'):
                                min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_dist_to_this_pos)
                                robot_can_reach_any_push_pos = True

        # If no push position was found for any box that reduces distance to goal,
        # or if the robot cannot reach any such position.
        # This implies the state is a local minimum or requires non-greedy moves.
        # Return infinity if robot cannot reach *any* push position that leads to progress.
        if not robot_can_reach_any_push_pos and boxes_to_move:
             # This covers cases where boxes are in local minima (no neighbor is closer to goal)
             # or robot is partitioned away from all required push positions.
             return float('inf')

        # The heuristic is the sum of box distances plus the minimum robot distance
        # to a position that enables a goal-reducing push for any box.
        total_h = h_boxes + min_robot_dist_to_push_pos

        return total_h
