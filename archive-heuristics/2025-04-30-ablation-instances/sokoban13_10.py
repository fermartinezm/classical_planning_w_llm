from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available from the planning framework
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, uncomment this dummy base class:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#
#     def __call__(self, node):
#         raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_3_5)".
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

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box to its goal location and adding the shortest
    path distance for the robot to reach a valid pushing position for the
    nearest box that needs to be moved. It ignores the 'clear' predicate
    and potential deadlocks for computational efficiency.

    # Assumptions
    - The goal state specifies a unique target location for each box.
    - The locations form a graph defined by 'adjacent' predicates.
    - Shortest paths are computed on this location graph.
    - The heuristic does not explicitly check for or penalize deadlocks.
    - The heuristic does not explicitly account for clearing paths or goal locations.
    - Any object with an 'at' predicate in the goal is considered a box that needs moving.

    # Heuristic Initialization
    - Parses 'adjacent' facts from static information to build:
        - A set of all location names.
        - An adjacency map (location -> {direction: adjacent_loc}) representing the grid/graph.
        - A pushing position map ((box_loc, push_direction) -> robot_loc) indicating where the robot must be to push a box at box_loc in push_direction.
    - Precomputes all-pairs shortest path distances and predecessors between locations using BFS on the adjacency graph.
    - Extracts goal locations for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state.
    2. Identify the current location of each box that has a goal location defined.
    3. Initialize `total_box_distance = 0` and `min_robot_distance_to_push_pos = infinity`.
    4. Create a list of boxes that are not currently at their goal locations.
    5. If the list of boxes not at goal is empty, the state is a goal state, return 0.
    6. For each box in the list of boxes not at goal:
       - Get the box's current location and its goal location.
       - Calculate the shortest path distance from the box's current location to its goal location using the precomputed distances. If no path exists, the state is likely unsolvable, return infinity. Add this distance to `total_box_distance`.
       - Find a shortest path from the box's current location to its goal location using the precomputed predecessors.
       - If the path has fewer than 2 locations (meaning the box is already at the goal or there's an issue), skip the robot distance calculation for this box (though this case should be handled by the 'boxes not at goal' list).
       - Get the second location on the shortest path; this is the next location the box needs to move to.
       - Determine the required robot pushing position for moving the box from its current location to this next location. This is found using the precomputed pushing position map: `push_pos = push_pos_map[(current_box_loc, direction_of_push)]`. The direction of push is the direction from `current_box_loc` to `next_box_loc`. If no such pushing position exists in the map, return infinity as the state might be a deadlock or impossible to move from.
       - Calculate the shortest path distance from the robot's current location to this required pushing position. If no path exists, return infinity.
       - Update `min_robot_distance_to_push_pos` with the minimum distance found so far across all boxes not at goal.
    7. The total heuristic value is `total_box_distance + min_robot_distance_to_push_pos`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.locations = set()
        # self.adj: location -> {direction: adjacent_location}
        self.adj = {}
        # self.push_pos_map: (box_loc, push_direction) -> robot_loc
        # This maps the location of the box and the direction it needs to be pushed
        # to the location where the robot must be to perform that push.
        # Derived from (adjacent l1 l2 dir) facts: l1 is robot_loc, l2 is box_loc, dir is push_direction
        self.push_pos_map = {}

        # Build graph and push_pos_map from adjacent facts
        # (adjacent l1 l2 dir) means l1 is adjacent to l2 in direction dir
        # This implies l1 is the robot position to push a box at l2 in direction dir
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.locations.add(l1)
                self.locations.add(l2)

                if l1 not in self.adj:
                    self.adj[l1] = {}
                self.adj[l1][direction] = l2

                # l1 is the robot location to push a box at l2 in direction 'direction'
                self.push_pos_map[(l2, direction)] = l1

        # Precompute all-pairs shortest paths and predecessors
        self.distances, self.predecessors = self._compute_all_pairs_bfs()

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming objects with 'at' predicate in goal are boxes
            if predicate == "at" and len(args) == 2:
                 obj, location = args
                 # Further check if it's likely a box based on naming convention
                 # A more robust solution would parse object types from the problem file
                 if obj.startswith('box'):
                     self.goal_locations[obj] = location
                 # Ignore (at-robot loc) goals if they exist

    def _compute_all_pairs_bfs(self):
        """Computes shortest path distances and predecessors between all pairs of locations using BFS."""
        distances = {loc: {l: math.inf for l in self.locations} for loc in self.locations}
        predecessors = {loc: {l: None for l in self.locations} for loc in self.locations}

        for start_node in self.locations:
            distances[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node}

            while queue:
                u = queue.popleft()

                # Iterate through adjacent locations
                if u in self.adj:
                    for direction, v in self.adj[u].items():
                        if v not in visited:
                            visited.add(v)
                            distances[start_node][v] = distances[start_node][u] + 1
                            predecessors[start_node][v] = u
                            queue.append(v)
        return distances, predecessors

    def _get_shortest_path(self, start_loc, end_loc):
        """Reconstructs a shortest path from start_loc to end_loc using predecessors."""
        if self.distances.get(start_loc, {}).get(end_loc, math.inf) == math.inf:
            return None # No path exists

        path = []
        current = end_loc
        while current is not None:
            path.append(current)
            if current == start_loc:
                break
            # Use .get() with None default for safety, though precomputed predecessors should cover all reachable pairs
            current = self.predecessors.get(start_loc, {}).get(current)
            if current is None and path[-1] != start_loc:
                 # Path reconstruction failed, should not happen if BFS was correct
                 return None


        path.reverse()
        return path

    def _get_pushing_position(self, box_loc, next_box_loc):
        """
        Given a box location (bloc) and the next location (floc) on its path,
        returns the required robot pushing position (rloc).
        Assumes next_box_loc is adjacent to box_loc.
        Finds dir such that (adjacent box_loc next_box_loc dir).
        Then finds rloc such that (adjacent rloc box_loc dir).
        """
        # Find the direction from box_loc to next_box_loc (this is the push direction)
        push_direction = None
        if box_loc in self.adj:
            for d, loc in self.adj[box_loc].items():
                if loc == next_box_loc:
                    push_direction = d
                    break

        if push_direction is None:
            # Should not happen if next_box_loc is truly adjacent and part of the graph
            return None # Indicates an issue or invalid move

        # The pushing position is rloc such that (adjacent rloc box_loc push_direction)
        # We built push_pos_map as (box_loc, push_direction) -> rloc
        return self.push_pos_map.get((box_loc, push_direction))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
                break

        if robot_loc is None:
             # Robot location not found, state is likely invalid
             return math.inf

        # Find box locations for boxes that have a goal
        box_locations = {} # box_name -> location
        for fact in state:
            parts = get_parts(fact)
            # Check if it's an 'at' predicate for an object that is a goal box
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] in self.goal_locations:
                 box_locations[parts[1]] = parts[2]

        total_box_distance = 0
        min_robot_distance_to_push_pos = math.inf
        boxes_not_at_goal = []

        # Identify boxes not at goal and calculate their distances
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            if current_loc is None:
                 # Box with a goal is not found in the state - likely invalid
                 return math.inf

            if current_loc != goal_loc:
                boxes_not_at_goal.append(box)

                # Add box distance to total
                dist_box_to_goal = self.distances.get(current_loc, {}).get(goal_loc, math.inf)
                if dist_box_to_goal == math.inf:
                    # Box cannot reach goal - likely unsolvable
                    return math.inf

                total_box_distance += dist_box_to_goal

        # If all boxes are at the goal, the heuristic is 0
        if not boxes_not_at_goal:
            return 0

        # Find the minimum robot distance to a pushing position for any box not at goal
        for box in boxes_not_at_goal:
            current_loc = box_locations[box]
            goal_loc = self.goal_locations[box]

            # Find pushing position for the first step towards the goal
            shortest_path = self._get_shortest_path(current_loc, goal_loc)

            # Path must exist and have at least 2 nodes if box is not at goal
            if shortest_path is None or len(shortest_path) < 2:
                 # This should not happen if dist_box_to_goal was finite and > 0
                 # Indicates an issue with pathfinding or state
                 return math.inf

            next_loc = shortest_path[1] # The location after current_loc on the shortest path
            push_pos = self._get_pushing_position(current_loc, next_loc)

            if push_pos is None:
                # Cannot find a valid pushing position for the first step.
                # This might indicate a deadlock or an impossible move sequence.
                # Return inf as it suggests an issue with the path or state.
                 return math.inf

            dist_robot_to_push_pos = self.distances.get(robot_loc, {}).get(push_pos, math.inf)

            if dist_robot_to_push_pos == math.inf:
                 # Robot cannot reach the required pushing position
                 return math.inf # Unsolvable from here

            min_robot_distance_to_push_pos = min(min_robot_distance_to_push_pos, dist_robot_to_push_pos)

        # The heuristic is the sum of box distances plus the robot's distance
        # to the nearest pushing position needed.
        # min_robot_distance_to_push_pos should be finite here if we haven't returned inf.
        if min_robot_distance_to_push_pos == math.inf:
             # Safety check, should have been caught earlier
             return math.inf

        return total_box_distance + min_robot_distance_to_push_pos

