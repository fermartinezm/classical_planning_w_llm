import math
from collections import deque
from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator classes are available in the environment
# from task import Task, Operator

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing two components:
    1. The sum of the shortest path distances for each box from its current location
       to its assigned goal location. The assignment is implicitly given by the goal state.
    2. The shortest path distance for the robot from its current location to the
       closest location adjacent to any box that is not yet at its goal location.

    Assumptions:
    - Location names follow the pattern 'loc_row_col'.
    - The 'adjacent' facts define a graph representing the valid movements between locations.
    - The goal state specifies a unique target location for each box using '(at box_name loc_name)' facts.
    - The heuristic relies on precomputed shortest path distances on the graph defined by 'adjacent' facts.

    Heuristic Initialization:
    The constructor performs the following steps:
    1. Extracts all unique location names mentioned in the task facts.
    2. Builds an adjacency list representation of the graph based on the 'adjacent' static facts.
    3. Precomputes the shortest path distances between all pairs of locations using Breadth-First Search (BFS) on the built graph.
    4. Parses the goal facts to create a mapping from each box name to its target goal location.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the current location of the robot by finding the fact '(at-robot ?l)' in the state.
    2. Identify the current location of each box by finding facts '(at ?b ?l)' in the state.
    3. Check if the current state satisfies all goal facts. If it does, the heuristic value is 0.
    4. If the goal is not reached, calculate the box-goal component:
       Iterate through each box specified in the goal state. Find its current location and its target goal location (from the precomputed map).
       Using the precomputed all-pairs shortest path distances, find the distance between the box's current location and its goal location.
       Sum these distances for all boxes. This sum is the box-goal heuristic component.
    5. Calculate the robot component:
       Identify the set of boxes that are not currently at their goal locations.
       Find all locations that are directly adjacent (one step away according to the adjacency graph) to any of these boxes.
       If there are no boxes needing a push (i.e., all boxes are at their goals), the robot component is 0.
       Otherwise, find the shortest path distance from the robot's current location to the closest location in the set of adjacent locations found in the previous step, using the precomputed distances. This distance is the robot heuristic component.
    6. The total heuristic value for the state is the sum of the box-goal component and the robot component.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self.all_locations = self._extract_all_locations(task.facts)
        self.adj_list = self._build_adjacency_list()
        self.distances = self._precompute_distances()
        self.box_goal_map = self._parse_goal_locations()

    def _extract_all_locations(self, all_possible_facts):
        """Extracts all location names from the set of all possible facts."""
        locations = set()
        for fact in all_possible_facts:
            # Remove parentheses and split by space
            parts = fact[1:-1].split()
            for part in parts:
                if part.startswith('loc_'):
                    locations.add(part)
        return list(locations)

    def _build_adjacency_list(self):
        """Builds the adjacency list graph from static adjacent facts."""
        adj_list = {loc: set() for loc in self.all_locations}
        for fact in self.static:
            if fact.startswith('(adjacent '):
                parts = fact[1:-1].split()
                # Fact is like '(adjacent loc_1_1 loc_1_2 right)'
                if len(parts) == 4:
                    l1 = parts[1]
                    l2 = parts[2]
                    if l1 in adj_list and l2 in adj_list:
                        adj_list[l1].add(l2)
                        adj_list[l2].add(l1) # Adjacency is symmetric
        return {loc: list(neighbors) for loc, neighbors in adj_list.items()}

    def _precompute_distances(self):
        """Precomputes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        for start_loc in self.all_locations:
            distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}

            while q:
                curr_loc, dist = q.popleft()
                distances[start_loc][curr_loc] = dist

                if curr_loc in self.adj_list:
                    for neighbor in self.adj_list[curr_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))
        return distances

    def _parse_goal_locations(self):
        """Parses the goal facts to map boxes to their target locations."""
        box_goal_map = {}
        for goal_fact in self.goals:
            if goal_fact.startswith('(at '):
                parts = goal_fact[1:-1].split()
                # Fact is like '(at box1 loc_2_4)'
                if len(parts) == 3:
                    box_name = parts[1]
                    goal_loc = parts[2]
                    box_goal_map[box_name] = goal_loc
        return box_goal_map

    def __call__(self, node):
        """Computes the heuristic value for the given state."""
        state = node.state

        # 3. Check if the goal state is reached
        if self.goals <= state:
            return 0

        robot_loc = None
        current_box_locs = {}

        # Extract current state information
        for fact in state:
            if fact.startswith('(at-robot '):
                parts = fact[1:-1].split()
                if len(parts) == 2:
                    robot_loc = parts[1]
            elif fact.startswith('(at '):
                parts = fact[1:-1].split()
                if len(parts) == 3:
                    box_name = parts[1]
                    box_loc = parts[2]
                    current_box_locs[box_name] = box_loc

        # Basic validation: robot must be present, all goal boxes must be present
        if robot_loc is None or len(current_box_locs) != len(self.box_goal_map):
             # This indicates a state representation issue or an unsolvable state
             # where a box is missing. Return infinity.
             return math.inf

        # 4. Calculate the box-goal distance component
        box_goal_h = 0
        boxes_needing_push = []
        for box_name, goal_loc in self.box_goal_map.items():
            if box_name in current_box_locs:
                current_loc = current_box_locs[box_name]
                if current_loc != goal_loc:
                    # Add distance from current box location to its goal location
                    if current_loc in self.distances and goal_loc in self.distances[current_loc]:
                         box_goal_h += self.distances[current_loc][goal_loc]
                         boxes_needing_push.append(box_name)
                    else:
                         # A location is unreachable or not in the precomputed graph
                         return math.inf
            else:
                 # Box mentioned in goal is not in current state - should not happen in valid states
                 return math.inf

        # 5. Calculate the robot distance component
        robot_h = 0
        if boxes_needing_push:
            push_adjacent_locs = set()
            for box_name in boxes_needing_push:
                box_loc = current_box_locs[box_name]
                # Find locations adjacent to this box location
                if box_loc in self.adj_list:
                    for neighbor_loc in self.adj_list[box_loc]:
                        # We need to be able to reach this neighbor location from the robot's current location
                        # and the neighbor must be a valid location in our distance map.
                        if neighbor_loc in self.distances and robot_loc in self.distances:
                             push_adjacent_locs.add(neighbor_loc)

            if push_adjacent_locs:
                 min_robot_dist = math.inf
                 # Find the minimum distance from the robot to any of these adjacent locations
                 if robot_loc in self.distances:
                     for adj_loc in push_adjacent_locs:
                         if adj_loc in self.distances[robot_loc]:
                             min_robot_dist = min(min_robot_dist, self.distances[robot_loc][adj_loc])

                 if min_robot_dist == math.inf:
                      # Robot cannot reach any location adjacent to a box needing push
                      return math.inf
                 robot_h = min_robot_dist
            else:
                 # This case happens if boxes_needing_push is not empty, but no adjacent locations
                 # were found for any of them in the graph. This might indicate a deadlock
                 # (box is in a corner with no adjacent free space) or a graph issue.
                 # Treat as unsolvable from this state.
                 if boxes_needing_push:
                      return math.inf
                 # If boxes_needing_push is empty, robot_h remains 0, which is correct.


        # 6. Total heuristic
        return box_goal_h + robot_h
