from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math
from collections import deque

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the minimum
    number of pushes required for each box to reach its goal location (shortest
    path distance on the static graph) and the minimum number of robot moves
    required to reach a position from which it can push any box towards its goal.

    # Assumptions
    - The grid structure and adjacency are defined by the `adjacent` facts.
    - Distances are computed on the static graph, ignoring dynamic obstacles
      (other boxes, robot, non-clear locations).
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Builds a graph of locations based on `adjacent` facts.
    - Precomputes all-pairs shortest path distances on this static graph using BFS.
    - Extracts goal locations for each box.
    - Maps directions to their opposites.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current location of the robot and each box.
    2. For each box that is not yet at its goal location:
       a. Add the shortest path distance from the box's current location to its
          goal location (on the static graph) to a running total. This estimates
          the minimum number of pushes required for this box.
       b. Identify potential "next step" locations for the box on a shortest path
          towards its goal.
       c. For each such potential next step, determine the required robot position
          from which it can push the box to that next location. This position is
          adjacent to the box's current location in the direction *opposite* to
          the push direction.
       d. Calculate the shortest path distance from the robot's current location
          to this required pushing position.
       e. Keep track of the minimum such robot distance found across all boxes
          that need moving and all their valid next steps.
    3. The total heuristic value is the sum of the box-to-goal distances (from step 2a)
       plus the minimum robot-to-pushing-position distance (from step 2e).
    4. If all boxes are at their goal locations, the heuristic is 0.
    5. If any required distance calculation results in infinity (meaning locations
       are disconnected in the static graph), the state is likely unsolvable,
       and the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        self.locations = set()
        self.adj_graph = {}  # loc -> list of (neighbor, direction)
        self.rev_adj_graph = {} # loc -> list of (neighbor, direction)

        # Build graph from adjacent facts
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, l1, l2, direction = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                self.adj_graph.setdefault(l1, []).append((l2, direction))
                # The reverse graph edge (l2, l1) has the same direction name
                # as the forward edge (l1, l2) in the PDDL adjacent predicate.
                # adjacent(l1, l2, dir) means moving l1 -> l2 is dir.
                # adjacent(l2, l1, opp_dir) means moving l2 -> l1 is opp_dir.
                # The PDDL lists both directions explicitly.
                # We need the reverse graph to find l_push:
                # If adjacent(l_push, l_box, dir) is true, then l_box is adjacent to l_push in opp(dir).
                # So, we look for l_push such that (l_box, opp(dir)) is in adj_graph[l_push].
                # Or, equivalently, (l_push, dir) is in rev_adj_graph[l_box].
                self.rev_adj_graph.setdefault(l2, []).append((l1, direction))


        # Define opposite directions
        self.opposite_direction = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }

        # Precompute all-pairs shortest paths on the static graph
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def _bfs(self, start_node):
        """Perform BFS to find distances from start_node to all reachable nodes."""
        distances = {node: math.inf for node in self.locations}
        if start_node not in self.locations:
             # Start node might not be in locations if it only appears in init/goal
             # but not in any adjacent facts (e.g., a goal in an isolated area).
             # In a connected grid, all locations should appear in adjacent facts.
             # Handle defensively, though this might indicate an issue with the problem definition.
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            curr_node = queue.popleft()
            curr_dist = distances[curr_node]

            # Iterate through neighbors in the forward graph
            for neighbor, _ in self.adj_graph.get(curr_node, []):
                if distances[neighbor] == math.inf:
                    distances[neighbor] = curr_dist + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current robot and box locations
        robot_location = None
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_location = parts[1]
            elif parts[0] == "at" and parts[1] in self.goal_locations:
                 box, location = parts[1], parts[2]
                 current_box_locations[box] = location

        # If robot location is unknown or not in our graph, state is likely invalid/unreachable
        if robot_location is None or robot_location not in self.distances:
             return math.inf

        total_box_dist = 0
        min_robot_dist = math.inf # Minimum distance for the robot to reach *any* required push position
        boxes_to_move = []

        # Calculate sum of box-to-goal distances and identify boxes needing movement
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)

            # If a box with a goal isn't found in the state, something is wrong.
            # Or maybe it's a multi-agent scenario where another agent moved it?
            # Assuming single agent and all goal boxes are always present.
            if current_loc is None:
                 return math.inf # Should not happen in standard Sokoban

            if current_loc != goal_loc:
                boxes_to_move.append(box)
                # Add box-to-goal distance using precomputed distances
                if current_loc in self.distances and goal_loc in self.distances[current_loc]:
                    total_box_dist += self.distances[current_loc][goal_loc]
                else:
                    # Box is in a location unreachable from its goal (or vice versa) in the static graph
                    return math.inf # Unsolvable state based on static graph

        # If no boxes need moving, the goal is reached
        if not boxes_to_move:
            return 0

        # Calculate minimum robot distance to a pushing position for any box needing a push
        for box in boxes_to_move:
            l_box = current_box_locations[box]
            g_b = self.goal_locations[box]

            # Find potential next steps for the box towards the goal
            # Iterate through neighbors of l_box in the forward graph
            for l_next, dir in self.adj_graph.get(l_box, []):
                 # Check if moving to l_next is a step towards the goal using precomputed distances
                 # Ensure both current_loc and next_loc are in the distance map for the goal
                 if l_box in self.distances and g_b in self.distances[l_box] and \
                    l_next in self.distances and g_b in self.distances[l_next] and \
                    self.distances[l_box][g_b] > self.distances[l_next][g_b]:

                    # Find the required robot pushing position l_push
                    # To push box from l_box to l_next in direction `dir`,
                    # the robot must be at l_push such that adjacent(l_push, l_box, dir) is true.
                    # This means l_box is adjacent to l_push in the opposite direction `opp_dir`.
                    # So, l_push is the neighbor of l_box reachable by moving in `opp_dir`.
                    opp_dir = self.opposite_direction.get(dir)
                    if opp_dir:
                        l_push = None
                        # Find the neighbor of l_box in the opposite direction
                        for neighbor, ndir in self.adj_graph.get(l_box, []):
                             if ndir == opp_dir:
                                 l_push = neighbor
                                 break

                        if l_push is not None:
                            # Calculate distance from current robot location to l_push
                            if robot_location in self.distances and l_push in self.distances[robot_location]:
                                min_robot_dist = min(min_robot_dist, self.distances[robot_location][l_push])
                            # else: Robot cannot reach this specific pushing position in the static graph.
                            # This specific path for the box might be blocked for the robot.
                            # We continue searching for other possible push positions for this or other boxes.

        # If min_robot_dist is still infinity, it means the robot cannot reach any
        # valid pushing position for any box needing a push, based on the static graph.
        # This state might be a deadlock or require complex moves not captured by the heuristic.
        # If there were boxes to move but no reachable push position, return infinity.
        if boxes_to_move and min_robot_dist == math.inf:
             return math.inf # Unsolvable or effectively blocked

        # If boxes_to_move was empty, we returned 0 earlier.
        # If boxes_to_move is not empty and min_robot_dist is finite, return the sum.
        return total_box_dist + min_robot_dist

