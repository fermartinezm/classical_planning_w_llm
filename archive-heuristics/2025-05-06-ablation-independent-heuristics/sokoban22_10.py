import heapq
import logging
import re
from collections import deque

from heuristics.heuristic_base import Heuristic
from task import Operator, Task # Assuming Task and Operator are available

# Define opposite directions
OPPOSITE_DIRECTION = {
    'up': 'down',
    'down': 'up',
    'left': 'right',
    'right': 'left',
}

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        two components:
        1. The sum of shortest path distances for each box from its current
           location to its assigned goal location. This estimates the minimum
           number of pushes required.
        2. The minimum shortest path distance from the robot's current location
           to any location from which it can make a "progress push" for any
           box that is not yet at its goal. A progress push is one that moves
           a box closer to its goal location. This estimates the robot's
           movement cost to get into a useful position.

    Assumptions:
        - The problem involves moving boxes to specific goal locations.
        - Goal facts are exclusively of the form `(at <box-name> <location-name>)`.
        - There is a one-to-one mapping between boxes and goal locations,
          determined by the `(at ...)` facts in the goal state. The mapping
          is established based on the order of appearance of boxes in the
          goal facts.
        - Locations are connected via `(adjacent <loc1> <loc2> <dir>)` facts,
          forming a graph.
        - Location names follow a structure that allows unique identification
          and graph representation.
        - The heuristic does not explicitly detect complex deadlocks (e.g.,
          boxes in corners they cannot leave), but unreachable goal locations
          or required push positions will result in an infinite heuristic value.
        - The heuristic is non-admissible.

    Heuristic Initialization:
        - Parses `adjacent` facts from `task.static` to build an adjacency
          graph of locations.
        - Computes all-pairs shortest path distances between all locations
          using Breadth-First Search (BFS).
        - Parses goal facts from `task.goals` to determine the target location
          for each box, storing this mapping.
        - Stores the mapping of directions to their opposites.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state. If yes, return 0.
        2. Identify the robot's current location.
        3. Identify the current location for each box.
        4. Calculate the "box-goal distance sum":
           Initialize sum = 0.
           For each box:
             Get its current location and its assigned goal location.
             If the box is not at its goal location:
               Add the precomputed shortest path distance from the box's
               current location to its goal location to the sum.
        5. Calculate the "robot-to-push-position distance":
           Initialize min_robot_dist = infinity.
           For each box that is not at its goal location:
             Get its current location (l_b) and its goal location (l_g).
             Find neighbors (v) of l_b in the location graph.
             For each neighbor v:
               If the distance from v to l_g is strictly less than the
               distance from l_b to l_g (i.e., moving the box to v makes progress):
                 Find the direction (dir) from l_b to v.
                 Find the location (push_pos) adjacent to l_b in the
                 opposite direction of dir.
                 If push_pos exists and is reachable from the robot's
                 current location (check precomputed distances):
                   Calculate the distance from the robot's current location
                   to push_pos.
                   Update min_robot_dist with the minimum distance found so far.
        6. If min_robot_dist is still infinity (meaning no progress push is
           possible for any box, or the required push position is unreachable),
           and the box-goal distance sum is greater than 0, the state is likely
           unsolvable or very difficult; return infinity.
        7. Otherwise, the heuristic value is the box-goal distance sum plus
           min_robot_dist (which might be 0 if all boxes are at goals or
           the robot is already in a push position).
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # 1. Build location graph from adjacent facts
        self.location_graph = self._build_location_graph()

        # 2. Compute all-pairs shortest paths
        self.distances = self._compute_all_pairs_shortest_paths()

        # 3. Map boxes to goals
        self.box_goals = self._map_boxes_to_goals()

        # 4. Opposite directions mapping
        self.opposite_direction = OPPOSITE_DIRECTION

        # Store all location names for easy access
        self.all_locations = list(self.location_graph.keys())


    def _build_location_graph(self):
        """Builds an adjacency list graph from adjacent facts."""
        graph = {}
        all_locations_set = set()

        # First pass to collect all locations mentioned in adjacent facts
        for fact in self.static:
            if fact.startswith('(adjacent '):
                parts = fact.strip('()').split()
                # Ensure fact has enough parts
                if len(parts) >= 4:
                    loc1 = parts[1]
                    loc2 = parts[2]
                    all_locations_set.add(loc1)
                    all_locations_set.add(loc2)

        # Initialize graph with all locations
        for loc in all_locations_set:
            graph[loc] = []

        # Second pass to add edges with directions
        for fact in self.static:
            if fact.startswith('(adjacent '):
                parts = fact.strip('()').split()
                 # Ensure fact has enough parts
                if len(parts) >= 4:
                    loc1 = parts[1]
                    loc2 = parts[2]
                    direction = parts[3]
                    # Add directed edge loc1 -> loc2 with direction
                    graph[loc1].append((loc2, direction))
                    # We assume the reverse edge is also present in static facts
                    # If not, we would need to infer it using OPPOSITE_DIRECTION
                    # and add it here. The example PDDL includes both directions.

        return graph

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        locations = list(self.location_graph.keys())

        for start_node in locations:
            distances[start_node] = {}
            # Initialize distances
            for loc in locations:
                distances[start_node][loc] = float('inf')
            distances[start_node][start_node] = 0

            # Use collections.deque for efficient queue operations
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                (current_node, dist) = queue.popleft()

                # Check neighbors
                if current_node in self.location_graph:
                    for neighbor, _ in self.location_graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_node][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
        return distances

    def _map_boxes_to_goals(self):
        """Maps each box object to its goal location based on goal facts."""
        box_goals = {}
        # Goal facts are like '(at box1 loc_2_4)'
        for goal_fact in self.goals:
            if goal_fact.startswith('(at '):
                parts = goal_fact.strip('()').split()
                # Ensure the fact has the expected structure (at box loc)
                if len(parts) == 3:
                    box_name = parts[1]
                    goal_loc = parts[2]
                    box_goals[box_name] = goal_loc
                else:
                    logging.warning(f"Unexpected goal fact format: {goal_fact}")
        return box_goals

    def _get_location_from_fact(self, fact_string):
        """Helper to extract location name from facts like '(at-robot loc)' or '(at box loc)'."""
        parts = fact_string.strip('()').split()
        if parts[0] == 'at-robot' and len(parts) == 2:
            return parts[1]
        elif parts[0] == 'at' and len(parts) == 3: # Ensure it's (at box loc)
             return parts[2]
        return None # Should not happen for relevant facts

    def _get_box_name_from_fact(self, fact_string):
         """Helper to extract box name from facts like '(at box loc)'."""
         parts = fact_string.strip('()').split()
         if parts[0] == 'at' and len(parts) == 3:
              return parts[1]
         return None # Should not happen for relevant facts

    def _get_direction(self, loc1, loc2):
        """Finds the direction from loc1 to loc2 based on the graph."""
        if loc1 in self.location_graph:
            for neighbor, direction in self.location_graph[loc1]:
                if neighbor == loc2:
                    return direction
        return None # Should not happen if loc2 is a neighbor of loc1 and loc1 is in graph

    def _get_location_in_direction(self, loc, direction):
        """Finds the location adjacent to loc in the given direction."""
        if loc in self.location_graph:
            for neighbor, dir in self.location_graph[loc]:
                if dir == direction:
                    return neighbor
        return None # No location in that direction

    def __call__(self, node):
        """Computes the heuristic value for the given state."""
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        robot_loc = None
        box_locations = {} # {box_name: location_name}

        # Extract robot and box locations from the current state
        for fact in state:
            if fact.startswith('(at-robot '):
                robot_loc = self._get_location_from_fact(fact)
            elif fact.startswith('(at '):
                box_name = self._get_box_name_from_fact(fact)
                box_loc = self._get_location_from_fact(fact)
                if box_name and box_loc:
                    box_locations[box_name] = box_loc

        if robot_loc is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             logging.error("Robot location not found in state!")
             return float('inf')

        box_goal_sum = 0
        boxes_not_at_goal = []

        # Calculate box-goal distance sum
        for box_name, current_loc in box_locations.items():
            goal_loc = self.box_goals.get(box_name)
            if goal_loc is None:
                 # This box doesn't have a goal specified? Should not happen in valid problems.
                 logging.warning(f"Goal location not found for box {box_name}")
                 # Treat this as an unsolvable state for this box?
                 return float('inf')

            # Ensure current_loc is a valid key in distances before lookup
            if current_loc not in self.distances:
                 logging.warning(f"Box location {current_loc} not found in distance map.")
                 return float('inf')

            if current_loc != goal_loc:
                # Check if goal_loc is in the distance map and reachable
                if goal_loc not in self.distances[current_loc] or self.distances[current_loc][goal_loc] == float('inf'):
                    logging.warning(f"Distance unknown or unreachable for box {box_name} from {current_loc} to {goal_loc}")
                    return float('inf') # Indicate unsolvable path for this box

                box_goal_sum += self.distances[current_loc][goal_loc]
                boxes_not_at_goal.append(box_name)

        # If all boxes are at goals, the sum is 0, return 0.
        # This check is technically redundant due to task.goal_reached(state)
        # at the beginning, but good for logic clarity.
        if box_goal_sum == 0:
             return 0

        # Calculate robot-to-push-position distance
        min_robot_dist = float('inf')

        # Only consider boxes that are not at their goal
        for box_name in boxes_not_at_goal:
            current_loc = box_locations[box_name]
            goal_loc = self.box_goals[box_name]
            current_box_goal_dist = self.distances[current_loc][goal_loc]

            # Find neighbors of the box's current location
            if current_loc in self.location_graph:
                for neighbor_loc, dir_to_neighbor in self.location_graph[current_loc]:
                    # Check if moving the box to neighbor_loc makes progress towards the goal
                    # Ensure neighbor_loc is a valid key in distances before lookup
                    if neighbor_loc in self.distances and goal_loc in self.distances[neighbor_loc]:
                         if self.distances[neighbor_loc][goal_loc] < current_box_goal_dist:
                            # This is a progress push direction (current_loc -> neighbor_loc)
                            # Find the required robot position (adjacent to current_loc in opposite direction)
                            push_dir = dir_to_neighbor
                            required_robot_dir = self.opposite_direction.get(push_dir)

                            if required_robot_dir:
                                push_pos = self._get_location_in_direction(current_loc, required_robot_dir)

                                # Check if push_pos exists and is reachable by the robot on the static graph
                                if push_pos and robot_loc in self.distances and push_pos in self.distances[robot_loc] and self.distances[robot_loc][push_pos] != float('inf'):
                                    dist_robot_to_push_pos = self.distances[robot_loc][push_pos]
                                    min_robot_dist = min(min_robot_dist, dist_robot_to_push_pos)
                                else:
                                     # Required push_pos doesn't exist or is unreachable by robot on static graph
                                     # This specific progress push is not currently possible or leads to unreachable state
                                     pass # Don't update min_robot_dist with infinity from here

        # If after checking all progress pushes for all boxes, we couldn't find
        # any reachable push position (on the static graph) for any progress push,
        # it means no progress is possible from the current robot position towards
        # getting any box closer to its goal. This state is likely a dead end
        # or requires moving other objects first (which this heuristic doesn't model).
        # If box_goal_sum > 0 but min_robot_dist is inf, return infinity
        if min_robot_dist == float('inf'):
             # If box_goal_sum is 0, we already returned 0.
             # If box_goal_sum > 0 and min_robot_dist is inf, it's likely unsolvable from here.
             return float('inf')


        # Total heuristic is sum of box distances + minimum robot distance to a useful push position
        return box_goal_sum + min_robot_dist
