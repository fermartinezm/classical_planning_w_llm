import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string before processing
    if not isinstance(fact, str):
        return []
    # Remove leading/trailing parentheses and split by spaces
    return fact.strip('()').split()

# Helper function to match PDDL facts (not strictly needed for this heuristic but good practice)
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#     - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
#     - `args`: The expected pattern (wildcards `*` allowed).
#     - Returns `True` if the fact matches the pattern, else `False`.
#     """
#     parts = get_parts(fact)
#     if len(parts) != len(args):
#         return False
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The minimum number of pushes required for each box to reach its goal location,
       calculated as the shortest path distance on the grid ignoring all obstacles.
    2. The minimum number of robot moves required to reach a location adjacent
       to *any* box that needs to be moved, considering current obstacles (other
       boxes and non-clear squares).

    # Assumptions
    - The locations form a grid-like structure defined by `adjacent` predicates.
    - A box can be pushed one step towards its goal if the robot is in the correct
      adjacent square and the target square is clear.
    - The cost of moving a box is primarily the number of grid steps it needs to take.
    - The robot must reach a push position for a box before it can be moved.

    # Heuristic Initialization
    - Build a graph representation of the locations based on `adjacent` facts.
    - Extract the goal location for each box from the task goals.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. Identify which boxes are not yet at their goal locations.
    3. If all boxes are at their goals, the heuristic is 0.
    4. Calculate the "box distance" component:
       - For each box not at its goal, find the shortest path distance from its
         current location to its goal location on the grid graph, *ignoring*
         all obstacles (other boxes, walls). This is a relaxed distance,
         representing the minimum number of pushes needed if the path were clear.
       - Sum these distances for all boxes not at their goals.
    5. Calculate the "robot accessibility" component:
       - Find the shortest path distance from the robot's current location to
         *any* location that is adjacent to *any* box that needs to be moved.
       - This shortest path must be calculated considering current obstacles:
         the robot can only move into locations that are currently `clear`.
       - Find the minimum such distance over all adjacent locations of all
         boxes that need moving.
    6. The total heuristic value is the sum of the "box distance" component
       and the "robot accessibility" component.

    This heuristic is non-admissible because:
    - The box distance ignores obstacles, which might make paths seem shorter than they are.
    - Summing box distances ignores interactions (e.g., one box blocking another).
    - The robot accessibility only considers reaching *any* push position for *any* box,
      not the sequence of robot movements needed to push all boxes.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - The grid graph from `adjacent` facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the graph from adjacent facts
        self.graph = self.build_graph(static_facts)

        # Store goal locations for each box
        self.goal_locations = self.extract_goal_locations(self.goals)

    def build_graph(self, static_facts):
        """
        Builds an adjacency list representation of the grid graph from static facts.
        The graph maps each location to a set of adjacent locations.
        """
        graph = collections.defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                graph[loc1].add(loc2)
                # Assuming adjacency is symmetric, although PDDL lists both directions
                # graph[loc2].add(loc1)
        return graph

    def extract_goal_locations(self, goals):
        """
        Extracts the goal location for each box from the task goals.
        Returns a dictionary mapping box name to its goal location.
        """
        goal_locations = {}
        # Goals are a frozenset of fact strings
        for goal_fact_str in goals:
            parts = get_parts(goal_fact_str)
            if parts[0] == "at" and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                # Assuming objects of type 'box' are the ones we care about in goals
                # In PDDL, we'd check type, but here we rely on the goal structure
                if obj_name.startswith('box'): # Simple check based on example names
                     goal_locations[obj_name] = loc_name
        return goal_locations

    def extract_state_info(self, state):
        """
        Extracts the current location of the robot and all boxes from the state.
        Returns robot_location (string) and box_locations (dict mapping box name to location).
        """
        robot_location = None
        box_locations = {}
        # State is a frozenset of fact strings
        for fact_str in state:
            parts = get_parts(fact_str)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == "at" and len(parts) == 3:
                 # Assuming objects of type 'box' start with 'box'
                 if parts[1].startswith('box'):
                    box_locations[parts[1]] = parts[2]
        return robot_location, box_locations

    def bfs_simple(self, start, end, graph):
        """
        Performs a simple BFS on the graph to find the shortest path distance,
        ignoring all potential obstacles (boxes, walls). Used for box-to-goal distance.
        Returns the distance or float('inf') if unreachable.
        """
        if start == end:
            return 0

        queue = collections.deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Check if current_loc exists in the graph keys
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # Target not reachable

    def bfs_robot(self, start, end, graph, state):
        """
        Performs BFS for the robot, considering locations occupied by boxes
        or walls as obstacles. The robot can only move into 'clear' locations.
        Returns the distance or float('inf') if unreachable.
        """
        if start == end:
            return 0

        queue = collections.deque([(start, 0)])
        visited = {start}

        # Determine traversable locations for the robot in the current state
        # Robot can move into any location that is currently 'clear'.
        # The robot's starting location is also traversable (from its neighbors).
        traversable_locations = {get_parts(fact)[1] for fact in state if fact.startswith('(clear ')}
        traversable_locations.add(start) # Robot's current location is traversable

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Check if current_loc exists in the graph keys
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    # Robot can move to neighbor if it's not visited AND it's traversable
                    # The target location 'end' must also be traversable for the robot to reach it.
                    if neighbor not in visited and (neighbor in traversable_locations or neighbor == end):
                         visited.add(neighbor)
                         queue.append((neighbor, dist + 1))

        return float('inf') # Target not reachable

    def calculate_total_box_distance(self, boxes_to_move, box_locations, goal_locations, graph):
        """
        Calculates the sum of simple BFS distances for all boxes not at their goal.
        """
        total_dist = 0
        for box in boxes_to_move:
            box_loc = box_locations.get(box)
            goal_loc = goal_locations.get(box)
            if box_loc and goal_loc: # Ensure locations are found
                dist = self.bfs_simple(box_loc, goal_loc, graph)
                if dist == float('inf'):
                    # If a box cannot reach its goal even on an empty grid, it's likely impossible
                    # Return a very large value to indicate a potentially unsolvable state
                    return float('inf')
                total_dist += dist
            else:
                 # Should not happen in valid states/goals, but handle defensively
                 return float('inf') # Indicate error or impossible state
        return total_dist

    def calculate_min_robot_distance(self, robot_loc, boxes_to_move, box_locations, graph, state):
        """
        Calculates the minimum robot distance to a location adjacent to any box that needs moving.
        """
        min_dist = float('inf')
        
        # Determine locations occupied by boxes to avoid planning robot path through them
        # This is implicitly handled by checking for '(clear neighbor)' in bfs_robot
        # box_obstacle_locations = {loc for box, loc in box_locations.items() if box in boxes_to_move}

        for box in boxes_to_move:
            box_loc = box_locations.get(box)
            if box_loc and box_loc in graph: # Ensure box location and its neighbors exist in graph
                # Find locations adjacent to the box
                adjacent_to_box = graph[box_loc]
                for adj_loc in adjacent_to_box:
                    # Calculate robot distance to this adjacent location
                    # The robot needs to reach a square *next* to the box.
                    # The BFS for the robot considers obstacles (non-clear squares).
                    dist = self.bfs_robot(robot_loc, adj_loc, graph, state)
                    min_dist = min(min_dist, dist)

        return min_dist


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings).

        # 1. Extract Relevant Information
        robot_loc, box_locations = self.extract_state_info(state)

        # 2. Identify boxes not at goal
        boxes_to_move = [
            box for box, goal_loc in self.goal_locations.items()
            if box_locations.get(box) != goal_loc # Use .get for safety
        ]

        # 3. If all boxes are at goals, heuristic is 0
        if not boxes_to_move:
            return 0

        # 4. Calculate the "box distance" component (sum of simple BFS distances)
        total_box_distance = self.calculate_total_box_distance(
            boxes_to_move, box_locations, self.goal_locations, self.graph
        )

        # If any box is unreachable even on an empty grid, return large value
        if total_box_distance == float('inf'):
             return 1000000 # Use a large integer value

        # 5. Calculate the "robot accessibility" component (min robot BFS distance)
        min_robot_to_adj_dist = self.calculate_min_robot_distance(
            robot_loc, boxes_to_move, box_locations, self.graph, state
        )

        # If the robot cannot reach any push position for any box, return large value
        if min_robot_to_adj_dist == float('inf'):
             return 1000000 # Use a large integer value

        # 6. Total heuristic is the sum
        # The heuristic is the sum of minimum pushes needed for all boxes
        # plus the cost for the robot to get to a position to start pushing.
        # This is non-admissible but aims to guide towards moving boxes.
        return total_box_distance + min_robot_to_adj_dist

