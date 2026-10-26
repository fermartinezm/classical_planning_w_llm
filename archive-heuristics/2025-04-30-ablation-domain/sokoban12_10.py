import collections
import fnmatch
import math
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though unlikely with planner input
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing two components:
    1. The sum of the shortest path distances for each misplaced box to its goal location.
    2. The shortest path distance from the robot's current location to any location adjacent
       to any of the misplaced boxes.

    This heuristic is non-admissible but aims to guide a greedy best-first search by
    prioritizing states where boxes are closer to their goals and the robot is closer
    to a position where it can start pushing a box.

    # Assumptions
    - The grid structure and adjacency relationships are defined by the `adjacent` predicates
      in the static facts.
    - All locations mentioned in the state and goals are part of the grid defined by `adjacent` facts.
    - The cost of moving the robot one step is 1.
    - The cost of pushing a box one step is 1 (this includes the robot moving into position and the push).
      The heuristic approximates this by summing box-goal distance (pushes) and robot-to-push-pos distance (robot moves).
    - Unreachable locations (for boxes to goals or robot to push positions) imply an infinite cost.

    # Heuristic Initialization
    - The heuristic parses the static facts to build a graph representing the grid, where nodes are
      locations and edges connect adjacent locations.
    - It parses the goal conditions to identify the target location for each box.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. Initialize `total_box_goal_distance` to 0.
    4. Initialize a list `misplaced_box_locations` to store the current locations of boxes not at their goals.
    5. Initialize a set `locations_adjacent_to_misplaced_boxes` to store all locations from which the robot could potentially push a misplaced box.
    6. Iterate through each box and its goal location (stored during initialization):
       - Get the box's current location from the state.
       - If the box's current location is not its goal location:
         - Add the box's current location to `misplaced_box_locations`.
         - Compute the shortest path distance from the box's current location to its goal location using BFS on the grid graph. This distance represents the minimum number of pushes required for this box.
         - If the goal is unreachable from the box's current location, the state is likely unsolvable; return infinity.
         - Add this distance to `total_box_goal_distance`.
         - Find all locations adjacent to the box's current location using the grid graph. These are potential positions for the robot to push the box. Add these adjacent locations to `locations_adjacent_to_misplaced_boxes`.
    7. If `misplaced_box_locations` is empty, all boxes are at their goals, so the heuristic is 0.
    8. If there are misplaced boxes, compute the shortest path distance from the robot's current location to each location in `locations_adjacent_to_misplaced_boxes` using BFS.
    9. Find the minimum of these distances. This represents the minimum number of robot moves required to get into a position to push *any* misplaced box.
    10. If the robot cannot reach any location adjacent to a misplaced box, the state is likely unsolvable; return infinity.
    11. The total heuristic value is `total_box_goal_distance` + `min_dist_robot_to_adj_box`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph and storing box goals.
        """
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = collections.defaultdict(list)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Add bidirectional edges
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Store goal locations for each box
        self.box_goals = {}
        # We can identify boxes by looking at the goal predicates
        self.boxes = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                box, location = parts[1], parts[2]
                self.box_goals[box] = location
                self.boxes.add(box) # Collect all box names

    def _bfs_distance(self, start, end):
        """
        Computes the shortest path distance between two locations in the grid graph
        using Breadth-First Search (BFS).

        Returns the distance or float('inf') if the end is unreachable from the start.
        """
        if start == end:
            return 0
        if start not in self.graph or end not in self.graph:
             # Handle cases where start or end might not be in the graph (e.g., walls)
             # If start is a wall, robot/box can't be there. If end is a wall, box can't go there.
             # This heuristic assumes valid locations are in the graph.
             # If a location is not in the graph, it means it has no adjacent locations
             # according to the PDDL, effectively making it a wall or isolated.
             # If start or end is not in the graph, they are unreachable from anywhere in the graph.
             return float('inf')


        queue = collections.deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Ensure current_loc is in the graph before accessing neighbors
            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # If the queue is empty and the end hasn't been reached, it's unreachable
        return float('inf')

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot':
                robot_loc = parts[1]
                break # Assuming only one robot

        if robot_loc is None:
             # Should not happen in a valid Sokoban state, but handle defensively
             return float('inf')

        # Find box locations
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is (at ?obj ?loc) and ?obj is one of the boxes we care about
            if parts and parts[0] == 'at' and parts[1] in self.boxes:
                 box_locations[parts[1]] = parts[2]

        total_box_goal_distance = 0
        misplaced_box_locations = []
        locations_adjacent_to_misplaced_boxes = set()

        # Calculate sum of box-goal distances and find push positions
        for box, goal_location in self.box_goals.items():
            current_location = box_locations.get(box) # Use .get for safety

            # If a box isn't in the state facts, something is wrong, treat as unsolvable
            if current_location is None:
                 return float('inf')

            if current_location != goal_location:
                misplaced_box_locations.append(current_location)

                # Distance from box to its goal
                dist_box_goal = self._bfs_distance(current_location, goal_location)
                if dist_box_goal == float('inf'):
                    # Box cannot reach its goal
                    return float('inf')
                total_box_goal_distance += dist_box_goal

                # Add adjacent locations as potential robot push positions
                if current_location in self.graph:
                    for neighbor in self.graph[current_location]:
                        locations_adjacent_to_misplaced_boxes.add(neighbor)

        # If all boxes are at their goals, heuristic is 0
        if not misplaced_box_locations:
            return 0

        # Calculate minimum distance from robot to a push position
        min_dist_robot_to_adj_box = float('inf')
        for adj_loc in locations_adjacent_to_misplaced_boxes:
             dist_robot_adj = self._bfs_distance(robot_loc, adj_loc)
             min_dist_robot_to_adj_box = min(min_dist_robot_to_adj_box, dist_robot_adj)

        # If robot cannot reach any push position
        if min_dist_robot_to_adj_box == float('inf'):
             return float('inf')

        # Total heuristic is sum of box-goal distances + robot distance to nearest push position
        return total_box_goal_distance + min_dist_robot_to_adj_box

