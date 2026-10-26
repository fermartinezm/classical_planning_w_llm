from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match_exact(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with an exact number of arguments.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing:
    1. The minimum number of pushes required for each box to reach its goal location (calculated as the shortest path distance on the undirected location graph, ignoring dynamic obstacles).
    2. The minimum number of robot moves required to reach any ungoaled box (calculated as the shortest path distance on the undirected location graph, considering locations occupied by *any* box as obstacles).

    # Assumptions
    - Each box mentioned in the goal has a unique goal location.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - The heuristic does not attempt to detect complex deadlocks (e.g., boxes in unpushable positions not on goals) beyond simple graph reachability.
    - The distance for a box to its goal is the simple undirected graph distance, ignoring dynamic obstacles.
    - The distance for the robot to a box considers all box locations as static obstacles.
    - Adjacency is symmetric for robot movement (if A is adjacent to B, B is adjacent to A).

    # Heuristic Initialization
    - Builds an undirected graph representation of the locations and their adjacencies based on the 'adjacent' static facts.
    - Extracts the goal location for each box from the task's goal conditions.
    - Defines a large value to represent infinity for unreachable locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify which boxes are not yet at their goal locations ('ungoaled_boxes').
    3. If there are no ungoaled boxes, the state is a goal state, and the heuristic is 0.
    4. Initialize the total heuristic value `h` to 0.
    5. Determine the set of locations currently occupied by *any* box (these are obstacles for the robot's movement BFS).
    6. Calculate the sum of minimum pushes for ungoaled boxes:
       - For each ungoaled box `b` at location `l_b` with goal `g_b`:
         - Compute the shortest path distance from `l_b` to `g_b` on the *undirected* location graph (using BFS). This estimates the minimum number of push actions needed for this box, ignoring all obstacles.
         - If the goal is unreachable for this box on the graph, the state might be unsolvable; return infinity.
         - Add this distance to `h`.
    7. Calculate the minimum robot distance to any ungoaled box:
       - Compute the shortest path distance from the robot's current location `l_r` to the location `l_b` of each ungoaled box `b` using BFS on the *undirected* graph. During this BFS, treat locations occupied by *any* box as obstacles.
       - Find the minimum of these distances over all ungoaled boxes.
       - If the robot cannot reach any ungoaled box location avoiding other boxes, return infinity.
    8. Add the minimum robot distance calculated in step 7 to `h`.
    9. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the undirected location graph
        and extracting goal locations.
        """
        self.infinity = float('inf')

        self.undirected_graph = {} # {loc: [neighbor_loc]}
        self.locations = set()

        # Build undirected graph from adjacent facts
        for fact in task.static:
            parts = get_parts(fact)
            # Assuming adjacent facts have 4 parts: (adjacent loc1 loc2 dir)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1:]
                self.locations.add(loc1)
                self.locations.add(loc2)

                # Build undirected graph
                if loc1 not in self.undirected_graph:
                    self.undirected_graph[loc1] = []
                if loc2 not in self.undirected_graph:
                    self.undirected_graph[loc2] = []
                # Add edges only if they don't exist to avoid duplicates
                if loc2 not in self.undirected_graph[loc1]:
                    self.undirected_graph[loc1].append(loc2)
                if loc1 not in self.undirected_graph[loc2]:
                    self.undirected_graph[loc2].append(loc1) # Assuming adjacency is symmetric for movement

        # Extract goal locations for each box
        self._parse_goals(task.goals)

    def _parse_goals(self, goals):
        """Parses goal conditions to extract box goal locations."""
        goal_facts = []
        if isinstance(goals, list):
            goal_facts = goals
        elif isinstance(goals, str):
            goals_str = goals.strip()
            if goals_str.startswith('(and '):
                # Simple parsing for (and (fact1) (fact2) ...)
                content = goals_str[5:-1].strip() # Remove (and )
                # Find all substrings that look like facts (start with '(' and end with ')')
                balance = 0
                start = -1
                for i, char in enumerate(content):
                    if char == '(':
                        if balance == 0:
                            start = i
                        balance += 1
                    elif char == ')':
                        balance -= 1
                        if balance == 0 and start != -1:
                            goal_facts.append(content[start : i + 1])
                            start = -1 # Reset start
                # Note: This simple parser might fail on complex nested PDDL structures,
                # but is sufficient for typical Sokoban goal formats.
            elif goals_str.startswith('(') and goals_str.endswith(')'):
                 # Single goal fact
                 goal_facts = [goals_str]
            # else: malformed goal?

        self.goal_locations = {}
        for fact_str in goal_facts:
            parts = get_parts(fact_str)
            # Match exactly 3 parts: predicate, obj, loc
            if match_exact(fact_str, "at", "*", "*"):
                # Assume any object in an 'at' goal is a box we need to move
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def BFS_distance(self, start, end, obstacles):
        """
        Calculates the shortest path distance between start and end locations
        on the undirected graph, avoiding specified obstacle locations.
        Returns infinity if end is unreachable.
        """
        if start == end:
            return 0
        # Start location cannot be an obstacle for the path starting there
        # if start in obstacles:
        #      return self.infinity # This check is only relevant if start is a box location and obstacles are other boxes

        queue = deque([(start, 0)])
        visited = {start}

        # Ensure obstacles set does not contain the start node itself
        effective_obstacles = obstacles - {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Ensure current_loc is a valid node in the graph
            if current_loc in self.undirected_graph:
                for neighbor_loc in self.undirected_graph[current_loc]:
                    if neighbor_loc not in effective_obstacles and neighbor_loc not in visited:
                        visited.add(neighbor_loc)
                        queue.append((neighbor_loc, dist + 1))
            # else: current_loc is not in the graph, cannot move from here

        return self.infinity # End not reachable

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # Find robot location and box locations
        robot_location = None
        box_locations = {} # {box_name: location_name}

        for fact in state:
            parts = get_parts(fact)
            if match_exact(fact, "at-robot", "*"):
                robot_location = parts[1]
            elif match_exact(fact, "at", "*", "*"):
                 # Assume any object in an 'at' fact in the state that is also in goal_locations is a box
                 obj_name, loc_name = parts[1], parts[2]
                 if obj_name in self.goal_locations: # Check if this object is one of the boxes we care about
                    box_locations[obj_name] = loc_name

        # Identify ungoaled boxes
        ungoaled_boxes = []
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            # Only consider boxes that exist in the current state and are not at their goal
            if current_loc is not None and current_loc != goal_loc:
                ungoaled_boxes.append(box)

        # If all boxes are at their goals, the heuristic is 0
        if not ungoaled_boxes:
            return 0

        total_heuristic = 0
        # Locations occupied by *any* box are obstacles for robot movement
        current_box_locations_set = set(box_locations.values())

        # Calculate sum of box-to-goal distances (minimum pushes)
        # This BFS ignores all obstacles (robot, other boxes) - simple graph distance
        for box in ungoaled_boxes:
            current_loc = box_locations[box]
            goal_loc = self.goal_locations[box]

            # Distance for the box path (minimum pushes), ignoring dynamic obstacles for simplicity
            # Use an empty set for obstacles as this is a relaxed distance for the box itself
            dist_box_goal = self.BFS_distance(current_loc, goal_loc, set())

            if dist_box_goal == self.infinity:
                # Box cannot reach its goal on the graph - likely unsolvable or trapped
                # Return infinity to prune this branch
                return self.infinity

            total_heuristic += dist_box_goal

        # Calculate minimum robot distance to any ungoaled box
        # Robot needs to reach a location adjacent to the box to push it.
        # A simple approximation is the distance to the box's square itself.
        min_dist_robot_to_box = self.infinity
        for box in ungoaled_boxes:
            box_loc = box_locations[box]
            # Distance for the robot to reach the box's location, avoiding other boxes
            # Obstacles for robot are locations occupied by *any* box.
            # The robot's own location is not an obstacle for its path.
            dist_robot_to_this_box = self.BFS_distance(robot_location, box_loc, current_box_locations_set)
            min_dist_robot_to_box = min(min_dist_robot_to_box, dist_robot_to_this_box)

        if min_dist_robot_to_box == self.infinity:
             # Robot cannot reach any ungoaled box location avoiding other boxes
             # This state might be a deadlock for the robot
             return self.infinity

        total_heuristic += min_dist_robot_to_box

        return total_heuristic
