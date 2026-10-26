from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Basic check for valid fact format
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return [] # Return empty for invalid format
    return fact[1:-1].split()

# Helper function for shortest path calculation
def shortest_path_distance(start, end, adj_list, obstacles):
    """
    Finds the shortest path distance between start and end locations
    on the grid defined by adj_list, avoiding obstacles.
    Returns float('inf') if no path exists.
    """
    if start == end:
        return 0
    # If start or end is in obstacles, path is impossible via this function
    # The caller should ensure start is a valid starting point not in obstacles
    # for the specific pathfinding context (e.g., robot's current location).
    # If end is an obstacle, we cannot reach it.
    if start in obstacles or end in obstacles:
         return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Get neighbors from adjacency list, handle locations not in adj_list (e.g., walls)
        for neighbor in adj_list.get(current_loc, []):
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It is calculated as the sum of the shortest path distances for each box to its
    assigned goal location (ignoring dynamic obstacles for box paths) plus the
    shortest path distance for the robot to reach a location adjacent to any
    box that needs to be moved (considering dynamic obstacles for the robot's path).

    # Assumptions
    - Each box has a unique goal location assigned in the problem file using '(at boxN loc_...)'.
    - The grid connectivity is defined by 'adjacent' predicates.
    - The robot can only move to 'clear' locations.
    - Boxes can only be pushed into 'clear' locations.
    - Box-goal distance is calculated based on grid connectivity only, ignoring
      other boxes or walls as obstacles for the box itself. This is an optimistic
      estimate for the number of pushes.
    - Robot-to-box-adjacent distance considers locations occupied by boxes or
      not marked as 'clear' as obstacles.

    # Heuristic Initialization
    - Parses 'adjacent' facts from static information to build an adjacency list
      representing the grid graph.
    - Parses 'at' facts in the goal state to map each box to its specific goal location.
    - Collects all unique location names from the adjacency list.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information:
       - Identify the current location of the robot.
       - Identify the current location of every box.
       - Identify all locations that are currently 'clear'.

    2. Identify Boxes to Move:
       - Compare the current location of each box with its assigned goal location.
       - Create a list of boxes that are not yet at their goal.

    3. Check for Goal State:
       - If the list of boxes to move is empty, the current state is a goal state,
         and the heuristic value is 0.

    4. Calculate Total Box-Goal Distance:
       - Initialize a sum `box_goal_dist_sum` to 0.
       - For each box in the list of boxes to move:
         - Get its current location and its goal location.
         - Calculate the shortest path distance between these two locations using
           BFS on the pre-computed adjacency list. Obstacles are ignored for this
           calculation, as we are estimating the minimum number of pushes on the
           underlying grid structure.
         - If a box cannot reach its goal even on the empty grid (distance is infinity),
           the problem might be unsolvable from this state; return infinity.
         - Add the calculated distance to `box_goal_dist_sum`. This sum is a lower
           bound on the total number of push actions required across all boxes.

    5. Calculate Minimum Robot-to-Box-Adjacent Distance:
       - Initialize `min_robot_to_box_adj_dist` to infinity.
       - Determine the set of locations that are unreachable by the robot in the
         current state. These are locations that are not 'clear', excluding the
         robot's current location (as the robot can start *from* its current location).
       - For each box in the list of boxes to move:
         - Get the box's current location.
         - Find all locations that are adjacent to the box's location according
           to the adjacency list.
         - For each adjacent location:
           - Calculate the shortest path distance from the robot's current location
             to this adjacent location using BFS. This BFS *does* consider the
             robot's unreachable locations as obstacles.
           - Update `min_robot_to_box_adj_dist` with the minimum distance found
             so far across all adjacent locations of all boxes that need moving.
       - If `min_robot_to_box_adj_dist` remains infinity after checking all relevant
         adjacent locations, it means the robot cannot reach any position from
         which it could potentially push a box; return infinity.

    6. Combine Distances:
       - The final heuristic value is `box_goal_dist_sum + min_robot_to_box_adj_dist`.
         This combines the estimated cost of moving the boxes themselves with the
         estimated cost of getting the robot into a position to perform the next push.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure and goal locations.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        self.static = task.static

        # Build adjacency list from 'adjacent' facts
        self.adj_list = {}
        self.all_locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1:]
                self.adj_list.setdefault(loc1, []).append(loc2)
                # Add reverse direction as well, assuming movement is bidirectional
                self.adj_list.setdefault(loc2, []).append(loc1)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Extract box goal locations from goal facts
        self.box_goals = {}
        # Assuming task.goals is a frozenset of goal facts
        if isinstance(self.goals, frozenset):
             goal_facts = self.goals
        else:
             # Handle unexpected goal format, though frozenset is standard
             print(f"Warning: task.goals is not a frozenset: {type(self.goals)}")
             goal_facts = set() # Default to empty set

        for goal_fact in goal_facts:
            parts = get_parts(goal_fact)
            # Check for '(at boxN loc_...)' facts in the goal
            if parts and parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box, location = parts[1:]
                self.box_goals[box] = location
            # Ignore other types of goal facts if any

        # Ensure all locations mentioned in adj_list are in all_locations
        # (redundant with how all_locations is built, but safe)
        for loc_list in self.adj_list.values():
             for loc in loc_list:
                  self.all_locations.add(loc)


    def shortest_path_distance(self, start, end, adj_list, obstacles):
        """
        Finds the shortest path distance between start and end locations
        on the grid defined by adj_list, avoiding obstacles.
        Returns float('inf') if no path exists.
        """
        if start == end:
            return 0
        # If start or end is in obstacles, path is impossible via this function
        # The caller should ensure start is a valid starting point not in obstacles
        # for the specific pathfinding context (e.g., robot's current location).
        # If end is an obstacle, we cannot reach it.
        if start in obstacles or end in obstacles:
             return float('inf')

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Get neighbors from adjacency list, handle locations not in adj_list (e.g., walls)
            for neighbor in adj_list.get(current_loc, []):
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # No path found


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # state is a frozenset of fact strings

        # Find robot and box locations and clear locations
        robot_loc = None
        box_locations = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts if any

            if parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box, location = parts[1:]
                box_locations[box] = location
            elif parts[0] == "clear" and len(parts) == 2:
                 clear_locations.add(parts[1])

        # Identify boxes not at goal
        boxes_to_move = [
            box for box, goal_loc in self.box_goals.items()
            if box_locations.get(box) != goal_loc # Use .get() in case a box is missing (invalid state?)
        ]

        # If all boxes are at goals, heuristic is 0
        if not boxes_to_move:
            return 0

        # Calculate sum of box-goal distances (ignoring obstacles for box path)
        box_goal_dist_sum = 0
        for box in boxes_to_move:
            current_loc = box_locations.get(box) # Use .get()
            goal_loc = self.box_goals[box]

            # If a box is not found in the state, it's an invalid state?
            if current_loc is None:
                 # This shouldn't happen in a valid Sokoban state representation
                 return float('inf') # Indicate unsolvable/invalid state

            dist = self.shortest_path_distance(current_loc, goal_loc, self.adj_list, set())
            if dist == float('inf'):
                 # Box cannot reach its goal even on an empty grid - likely unsolvable
                 return float('inf')
            box_goal_dist_sum += dist

        # Calculate minimum robot distance to a location adjacent to a box that needs moving
        min_robot_to_box_adj_dist = float('inf')

        # Locations the robot cannot move into (not clear, excluding robot's current spot)
        # A location is unreachable if it's not in the set of clear locations AND it's not the robot's current location.
        robot_unreachable_locations = {
            loc for loc in self.all_locations
            if loc != robot_loc and loc not in clear_locations
        }

        robot_is_reachable = False # Flag to check if robot can reach *any* adjacent spot

        for box in boxes_to_move:
            box_loc = box_locations.get(box) # Use .get()
            if box_loc is None: continue # Should not happen

            # Find locations adjacent to the box
            for adj_loc in self.adj_list.get(box_loc, []):
                 # Calculate distance from robot to this adjacent location, avoiding obstacles
                 # The start location for BFS is robot_loc. robot_loc is explicitly NOT in robot_unreachable_locations.
                 # The end location for BFS is adj_loc. If adj_loc is in robot_unreachable_locations, shortest_path_distance returns inf.
                 dist = self.shortest_path_distance(robot_loc, adj_loc, self.adj_list, robot_unreachable_locations)

                 if dist != float('inf'): # Only consider reachable adjacent locations
                     robot_is_reachable = True
                     if dist < min_robot_to_box_adj_dist:
                          min_robot_to_box_adj_dist = dist

        # If robot cannot reach any adjacent location to any box that needs moving, return infinity
        if not robot_is_reachable:
             return float('inf')

        # Total heuristic is sum of box distances and robot approach distance
        total_heuristic = box_goal_dist_sum + min_robot_to_box_adj_dist

        return total_heuristic
