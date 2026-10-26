from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distances for each box to its goal location and adding the
    shortest path distance from the robot to the nearest box that is not yet
    at its goal.

    # Assumptions
    - The underlying location structure forms a graph defined by 'adjacent' facts.
    - Shortest path distances on this static graph are used, ignoring dynamic
      obstacles (like other boxes or the robot's current position for box paths,
      and 'clear' predicates for robot paths).
    - Each box mentioned in the goal state must reach its specified goal location.

    # Heuristic Initialization
    - Builds an undirected graph of locations based on 'adjacent' facts.
    - Stores the goal location for each box specified in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the robot's current location from the state.
    2. Identify the current location of each box that has a specified goal location from the state.
    3. Determine which of these boxes are not yet at their goal locations (ungoaled boxes).
    4. If there are no ungoaled boxes, the state is a goal state, and the heuristic is 0.
    5. If there are ungoaled boxes:
       a. Calculate the sum of shortest path distances for each ungoaled box from its current location to its goal location using BFS on the static location graph.
       b. Calculate the shortest path distance from the robot's current location to the current location of each ungoaled box using BFS on the static location graph. Find the minimum of these distances.
       c. The heuristic value is the sum of the total box-goal distance (from step 5a) and the minimum robot-box distance (from step 5b).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts ('adjacent' relationships).
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the undirected location graph from adjacent facts
        self.graph = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, 'adjacent', '*', '*', '*'):
                _, loc1, loc2, direction = get_parts(fact)
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Add reverse edge for undirected graph

        # Store goal locations for each box mentioned in the goal
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, 'at', '*', '*'):
                _, obj_name, loc_name = get_parts(goal)
                self.goal_locations[obj_name] = loc_name

    def bfs(self, start, end, graph):
        """
        Performs BFS on the location graph to find the shortest path distance.
        Ignores obstacles.
        Returns the distance (number of edges) or float('inf') if unreachable.
        """
        if start == end:
            return 0
        queue = collections.deque([(start, 0)])
        visited = {start}
        while queue:
            current_loc, dist = queue.popleft()

            # Ensure current_loc is a valid node in the graph
            if current_loc not in graph:
                 # This location is isolated or doesn't exist in the adjacent facts.
                 # Should ideally not happen in a well-formed problem instance.
                 continue

            for neighbor in graph[current_loc]:
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf') # End not reachable from start

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        robot_loc = None
        current_box_locations = {}

        # Parse state to find robot and box locations
        for fact in state:
            if match(fact, 'at-robot', '*'):
                _, robot_loc = get_parts(fact)
            elif match(fact, 'at', '*', '*'):
                _, obj_name, loc_name = get_parts(fact)
                # Only track boxes that are in the goal state
                if obj_name in self.goal_locations:
                     current_box_locations[obj_name] = loc_name
            # We don't need 'clear' facts for this simple heuristic's BFS

        # Check if robot location was found (should always be the case in valid states)
        if robot_loc is None:
             # Invalid state representation, return a high cost
             return float('inf')

        # Identify boxes that are not at their goal locations
        ungoaled_boxes = [
            box for box, goal_loc in self.goal_locations.items()
            if current_box_locations.get(box) != goal_loc
        ]

        # If all target boxes are at their goals, the heuristic is 0
        if not ungoaled_boxes:
            return 0

        # Calculate the sum of distances for each ungoaled box to its goal
        sum_box_dist = 0
        for box in ungoaled_boxes:
            box_loc = current_box_locations.get(box) # Use .get in case box isn't in state (unlikely)
            goal_loc = self.goal_locations[box]

            if box_loc is None:
                 # Box expected in goal is not in state, indicates problem or unreachable
                 return float('inf')

            d = self.bfs(box_loc, goal_loc, self.graph)
            if d == float('inf'):
                # A box cannot reach its goal on the static graph - likely unsolvable path
                return float('inf')
            sum_box_dist += d

        # Calculate the minimum distance from the robot to any ungoaled box
        min_robot_box_dist = float('inf')
        for box in ungoaled_boxes:
            box_loc = current_box_locations[box]
            d = self.bfs(robot_loc, box_loc, self.graph)
            # If robot cannot reach this box, d will be inf. We take the min over all reachable boxes.
            min_robot_box_dist = min(min_robot_box_dist, d)

        # If robot cannot reach any ungoaled box, the state is likely a dead end
        if min_robot_box_dist == float('inf'):
             return float('inf')

        # The heuristic is the sum of total box-goal distance and minimum robot-box distance
        return sum_box_dist + min_robot_box_dist
