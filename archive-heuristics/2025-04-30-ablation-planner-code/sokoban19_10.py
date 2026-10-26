from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available as specified in the problem description context
# from heuristics.heuristic_base import Heuristic

# If running this code standalone without the planning framework,
# you would need a dummy Heuristic class definition here.
# For the final output, we assume the import works.
# Example dummy class (remove if using the framework):
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        pass # To be implemented by subclasses


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing, for each misplaced box,
    the shortest path distance for the box to its goal location plus the shortest path distance
    for the robot to reach the required position to initiate the first push towards the goal.

    # Assumptions
    - The grid is defined by the 'adjacent' predicates.
    - Shortest paths for boxes and the robot are calculated on the full grid graph, ignoring
      dynamic obstacles (other boxes, robot) and the 'clear' predicate. This is a relaxation.
    - The cost of moving a box is the number of pushes (box path length).
    - The robot's cost is primarily getting into position for the *first* push for each box.
      Subsequent robot movements needed to follow the box are implicitly covered by the push action
      moving the robot to the box's previous location.
    - The heuristic sums costs for each box independently, ignoring potential interactions
      or optimal box movement order.

    # Heuristic Initialization
    - Parses goal conditions to store the target location for each box.
    - Parses static 'adjacent' facts to build a graph representation of the locations.
    - Pre-calculates all-pairs shortest path distances between locations on the grid graph using BFS.
    - Stores a mapping of directions to their opposites.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box.
    2. Initialize the total heuristic value to 0.
    3. For each box:
        a. If the box is already at its goal location, add 0 to the total.
        b. If the box is not at its goal:
            i. Calculate the shortest path distance for the box from its current location to its goal location using the pre-calculated distances on the full grid graph. This is the minimum number of pushes required for the box itself. If the goal is unreachable for the box on the grid, return a large heuristic value (infinity).
            ii. Determine the required location for the robot to perform the *first* push for this box. This location is adjacent to the box's current location, in the direction opposite to the first step of a shortest path for the box towards its goal.
            iii. Calculate the shortest path distance for the robot from its current location to this required robot position using the pre-calculated distances. If the required robot position is unreachable for the robot on the grid, return a large heuristic value (infinity).
            iv. Add the box's path distance (pushes) and the robot's positioning distance to the total heuristic value.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the graph,
        and pre-calculating shortest path distances.
        """
        # Call the base class constructor if necessary, depending on the framework
        # super().__init__(task)

        self.goals = task.goals
        self.static = task.static

        # 1. Parse goal conditions
        self.box_goals = {}
        # Goal is a frozenset of facts like '(at box1 loc_2_4)'
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # 2. Parse static 'adjacent' facts and build graph
        self.graph = {} # {loc: {neighbor: direction}}
        self.all_locations = set()
        self.opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

                if loc1 not in self.graph:
                    self.graph[loc1] = {}
                self.graph[loc1][loc2] = direction

                # Add the reverse adjacency for symmetric movement
                if loc2 not in self.graph:
                    self.graph[loc2] = {}
                reverse_direction = self.opposite_dir.get(direction)
                if reverse_direction: # Only add if opposite direction is defined
                    self.graph[loc2][loc1] = reverse_direction

        # Ensure all locations mentioned in the graph are in the set of all locations
        # (already covered by adding loc1 and loc2 above)
        # Ensure all locations are keys in the graph dictionary, even if they have no neighbors
        for loc in list(self.all_locations):
             if loc not in self.graph:
                 self.graph[loc] = {}


        # 3. Pre-calculate all-pairs shortest path distances using BFS
        self.distances = {} # {start_loc: {end_loc: distance}}
        for start_loc in self.all_locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Performs BFS from start_loc to find distances to all reachable locations."""
        dist = {loc: math.inf for loc in self.all_locations}
        dist[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            u = queue.popleft()
            # Ensure u is a valid key in the graph before accessing neighbors
            if u in self.graph:
                for v in self.graph[u]:
                    if dist[v] == math.inf:
                        dist[v] = dist[u] + 1
                        queue.append(v)
        return dist

    def _find_required_robot_pos(self, box_loc, goal_loc):
        """
        Finds the location adjacent to box_loc where the robot must be
        to push the box towards goal_loc along a shortest path on the grid.
        Returns the required location string or None if no such location exists
        (e.g., box is at goal, or no path to goal, or required adjacent spot is a wall).
        """
        # If box is at goal, no push needed from this location
        if box_loc == goal_loc:
            return None

        # Get distance from box_loc to goal_loc
        box_to_goal_dist = self.distances.get(box_loc, {}).get(goal_loc, math.inf)

        # If goal is unreachable or box is already at goal (dist 0), no push needed from here
        if box_to_goal_dist == math.inf or box_to_goal_dist == 0:
             return None

        # Find a neighbor of box_loc that is one step closer to the goal
        next_box_loc = None
        push_direction = None

        # Iterate through neighbors of box_loc
        if box_loc in self.graph:
            for neighbor_loc, direction in self.graph[box_loc].items():
                # Check if this neighbor is on a shortest path to the goal
                if self.distances.get(neighbor_loc, {}).get(goal_loc, math.inf) == box_to_goal_dist - 1:
                    next_box_loc = neighbor_loc
                    push_direction = direction
                    break # Found a valid first step

        # If no such neighbor found, it implies the box cannot move towards the goal
        # along a shortest path from its current location on the grid.
        if next_box_loc is None:
             return None

        # Find the location adjacent to box_loc in the opposite direction of the push
        required_dir = self.opposite_dir.get(push_direction)
        if required_dir is None:
             # Should not happen with standard directions, but defensive
             return None

        # Iterate through neighbors of box_loc again to find the one in the required direction
        if box_loc in self.graph:
            for neighbor_loc, direction in self.graph[box_loc].items():
                 if direction == required_dir:
                     return neighbor_loc # Found the required robot position

        # If no location is adjacent to box_loc in the required direction (e.g., wall),
        # the robot cannot get into position for this push.
        return None


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # State is a frozenset of fact strings

        # 1. Identify robot and box locations
        robot_loc = None
        box_locations = {} # {box_name: location_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Check if the object is a box relevant to the goal
                if obj in self.box_goals:
                     box_locations[obj] = loc

        # If robot location is unknown, state is likely malformed or unreachable
        if robot_loc is None:
             return math.inf # Cannot solve without robot

        total_heuristic = 0

        # 3. Calculate cost for each box that is not at its goal
        for box_name, current_box_loc in box_locations.items():
            goal_box_loc = self.box_goals.get(box_name)

            # If box is not in goals or already at goal, skip
            if goal_box_loc is None or current_box_loc == goal_box_loc:
                continue

            # Ensure current box location is part of the known graph
            if current_box_loc not in self.all_locations:
                 # Box is in an unknown location, likely unsolvable
                 return math.inf

            # i. Box path distance (minimum pushes)
            # Check if goal_box_loc is in the distances calculated from current_box_loc
            box_dist = self.distances.get(current_box_loc, {}).get(goal_box_loc, math.inf)

            if box_dist == math.inf:
                # Box cannot reach goal location on the grid
                return math.inf

            # ii. Find required robot position for the first push
            req_robot_loc = self._find_required_robot_pos(current_box_loc, goal_box_loc)

            if req_robot_loc is None:
                 # This can happen if the box is at the goal (handled above),
                 # or if the box is in a location from which it cannot be pushed towards the goal
                 # along any shortest path on the grid (e.g., cornered).
                 # This state is likely a dead end for this box.
                 return math.inf

            # iii. Robot positioning distance
            # Ensure robot_loc and req_robot_loc are part of the known graph
            if robot_loc not in self.all_locations or req_robot_loc not in self.all_locations:
                 # Robot or required position is in an unknown location, likely unsolvable
                 return math.inf

            robot_cost = self.distances.get(robot_loc, {}).get(req_robot_loc, math.inf)

            if robot_cost == math.inf:
                # Robot cannot reach the required push position
                return math.inf

            # iv. Add costs for this box
            # The cost is the number of pushes + the cost for the robot to get into position for the first push
            total_heuristic += box_dist + robot_cost

        return total_heuristic
