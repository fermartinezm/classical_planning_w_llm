from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential extra spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, unless args has wildcards
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def get_opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen for valid sokoban directions

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing:
    1. The shortest path distance for each box from its current location to its goal location.
    2. The shortest path distance for the robot from its current location to the required push position for the first step of the nearest box that needs moving.

    # Assumptions
    - The locations form a graph where adjacency is defined by the 'adjacent' facts.
    - The shortest path distance calculation considers only the static layout (walls/non-adjacent locations), ignoring dynamic obstacles (other boxes, robot).
    - The heuristic assumes the robot can always reach the required push position, even if it means moving around other objects (which is not explicitly modeled in the distance calculation).
    - The heuristic is non-admissible.
    - All locations mentioned in adjacent facts or initial/goal states are part of a single connected component relevant to the problem, or unreachable goals/push positions result in infinite heuristic.

    # Heuristic Initialization
    - Parses static facts to build an adjacency graph of locations, storing neighbors by direction.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Stores the goal location for each box.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and each box from the state.
    2. Initialize total heuristic `total_h` to 0 and minimum robot-to-push-position distance `min_robot_to_push_pos_dist` to infinity.
    3. Iterate through each box and its goal location:
       a. Get the box's current location.
       b. If the box is not at its goal location:
          i. Calculate the shortest path distance from the box's current location to its goal location using the precomputed distances. If unreachable, return infinity. Add this distance to `total_h` (representing the minimum number of pushes needed for this box).
          ii. Find a location adjacent to the box's current location that lies on a shortest path towards the goal. This is a potential `next_box_loc`. We find one such neighbor that is one step closer to the goal.
          iii. If such a `next_box_loc` is found, determine the required robot location (`push_pos`) to push the box from its current location (`box_loc`) to `next_box_loc`. According to the PDDL, the robot must be adjacent to `box_loc` in the *opposite* direction of the push.
          iv. If a valid `push_pos` is found and is reachable by the robot, calculate the shortest path distance from the robot's current location to `push_pos`. Update `min_robot_to_push_pos_dist` with the minimum distance found so far across all boxes needing movement.
    4. If there were any boxes needing movement and a reachable push position was found for at least one of them, add `min_robot_to_push_pos_dist` to `total_h`. If boxes needed moving but no reachable push position was found for any of them, return infinity.
    5. Return `total_h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the graph and computing distances."""
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency list graph: {location: {direction: neighbor_location}}
        self.adj_list = {}
        # Keep track of all locations mentioned in adjacent facts
        self.locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.adj_list:
                    self.adj_list[loc1] = {}
                if loc2 not in self.adj_list:
                    self.adj_list[loc2] = {}
                self.adj_list[loc1][direction] = loc2
                self.adj_list[loc2][get_opposite_direction(direction)] = loc1

        # Compute all-pairs shortest path distances
        # This distance is for the robot moving on the static graph
        all_relevant_locations = set(self.locations)
        # Add locations from goals and initial state that might not be in adjacent facts
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at':
                  all_relevant_locations.add(parts[2]) # goal location
        # Assuming initial state locations are also covered by adjacent facts or goals
        # If not, would need to iterate through initial_state as well.

        self.distances = self._compute_all_pairs_shortest_paths(all_relevant_locations)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def _compute_all_pairs_shortest_paths(self, all_nodes):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        for start_node in all_nodes:
            distances[start_node] = self._bfs(start_node, all_nodes)
        return distances

    def _bfs(self, start_node, all_nodes):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        dist = {node: math.inf for node in all_nodes}
        if start_node in dist: # Ensure start_node is one of the known nodes
            dist[start_node] = 0
            queue = deque([start_node])
        else:
             # Start node is not in the set of relevant locations, cannot start BFS
             return dist # All distances remain infinity

        while queue:
            u = queue.popleft()
            if u in self.adj_list: # Check if node has any neighbors in the graph
                for direction, v in self.adj_list[u].items():
                    if v in dist and dist[v] == math.inf: # Ensure neighbor is a relevant node
                        dist[v] = dist[u] + 1
                        queue.append(v)
        return dist

    def _get_location_in_direction(self, loc, direction):
        """Returns the location adjacent to loc in the given direction, or None."""
        return self.adj_list.get(loc, {}).get(direction)

    def _get_direction(self, loc1, loc2):
        """Returns the direction from loc1 to loc2, or None."""
        if loc1 in self.adj_list:
            for direction, neighbor in self.adj_list[loc1].items():
                if neighbor == loc2:
                    return direction
        return None

    def _get_push_position(self, box_loc, next_box_loc):
        """
        Returns the required robot location to push a box from box_loc to next_box_loc.
        This is the location adjacent to box_loc in the opposite direction of the push.
        Returns None if the push direction is invalid or no location exists in the opposite direction.
        """
        direction_of_push = self._get_direction(box_loc, next_box_loc)
        if not direction_of_push:
             # next_box_loc is not adjacent to box_loc based on the graph
             return None
        opposite_dir = get_opposite_direction(direction_of_push)
        return self._get_location_in_direction(box_loc, opposite_dir)


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state  # frozenset of facts

        # Find current robot location
        robot_location = None
        # Find current box locations
        current_box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_location = parts[1]
            elif parts[0] == 'at' and parts[1] in self.goal_locations:
                 current_box_locations[parts[1]] = parts[2]

        total_h = 0
        min_robot_to_push_pos_dist = math.inf
        any_box_needs_moving = False

        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # Ensure box location is known and it's not at the goal
            if current_location and current_location != goal_location:
                any_box_needs_moving = True

                # Add box-goal distance (minimum pushes)
                # Check if locations are in our distance map (should be if graph built correctly)
                if current_location not in self.distances or goal_location not in self.distances[current_location]:
                     # This implies a location from state/goal wasn't in the graph, problem setup issue?
                     # Or goal is unreachable from current location
                     return math.inf # Cannot calculate distance

                box_dist = self.distances[current_location][goal_location]

                if box_dist == math.inf:
                    # Box cannot reach goal from here (e.g., trapped in a disconnected component)
                    return math.inf

                total_h += box_dist

                # Find a location adjacent to current_location that is one step closer to the goal
                # This is a potential 'next_box_loc' on a shortest path for the box
                next_box_loc = None
                if current_location in self.adj_list:
                    for direction, neighbor_loc in self.adj_list[current_location].items():
                         # Check if neighbor is one step closer to the goal and is a relevant location
                         if neighbor_loc in self.distances and goal_location in self.distances[neighbor_loc] and \
                            self.distances[neighbor_loc][goal_location] == box_dist - 1:
                             next_box_loc = neighbor_loc
                             break # Found one such neighbor, use it

                if next_box_loc:
                    # Determine the required robot push position for this first step
                    push_pos = self._get_push_position(current_location, next_box_loc)

                    # Calculate robot distance to this push position if it exists and is reachable
                    if push_pos and robot_location in self.distances and push_pos in self.distances[robot_location]:
                         dist_robot_to_this_push_pos = self.distances[robot_location][push_pos]
                         if dist_robot_to_this_push_pos != math.inf:
                             min_robot_to_push_pos_dist = min(min_robot_to_push_pos_dist, dist_robot_to_this_push_pos)
                    # else: push_pos might be None (e.g., box against wall, needs push from outside grid)
                    # or push_pos might be unreachable from robot_location

        # Add the minimum robot distance to a push position, if any boxes need moving
        # If any_box_needs_moving is True, but min_robot_to_push_pos_dist is still inf,
        # it means the robot cannot reach a valid push position for any box needing movement.
        if any_box_needs_moving:
             if min_robot_to_push_pos_dist != math.inf:
                 total_h += min_robot_to_push_pos_dist
             else:
                 # Robot cannot reach any push position for any box that needs moving
                 return math.inf # State is likely unsolvable

        # If no boxes need moving (any_box_needs_moving is False), total_h is 0,
        # which is correct for a goal state.

        return total_h
