import heapq
from collections import deque
import sys

# Assuming Heuristic and Task classes are available from the planner's core
# from heuristics.heuristic_base import Heuristic
# from task import Task

# Define dummy classes if not running within the planner environment
# This allows the code structure to be checked, but it won't run without the actual planner classes
try:
    from heuristics.heuristic_base import Heuristic
    from task import Task, Operator
except ImportError:
    print("Warning: Planner core classes not found. Using dummy classes.")
    class Heuristic:
        def __init__(self, task):
            self.task = task
            self.goals = task.goals

        def __call__(self, node):
            raise NotImplementedError

    class Operator:
         def __init__(self, name, preconditions, add_effects, del_effects):
             self.name = name
             self.preconditions = frozenset(preconditions)
             self.add_effects = frozenset(add_effects)
             self.del_effects = frozenset(del_effects)

    class Task:
        def __init__(self, name, facts, initial_state, goals, operators, static):
            self.name = name
            self.facts = facts
            self.initial_state = initial_state
            self.goals = goals
            self.operators = operators
            self.static = static
            # Dummy objects list for testing
            self.objects = []
            # Infer objects from facts if not provided
            all_args = set()
            for fact in list(initial_state) + list(goals) + list(static):
                 predicate, args = parse_fact(fact)
                 all_args.update(args)
            # Simple guess: anything not 'loc_R_C' or direction is an object
            for arg in all_args:
                 if not arg.startswith('loc_') and arg not in ['up', 'down', 'left', 'right']:
                      # Crude type inference based on examples
                      if arg.startswith('box'):
                           self.objects.append(f"{arg} - box")
                      elif arg.startswith('robot'): # Assuming robot name might appear
                           self.objects.append(f"{arg} - robot")
                      else:
                           self.objects.append(f"{arg} - object") # Generic fallback


def parse_fact(fact_str):
    """Helper function to parse a PDDL fact string."""
    # Remove leading/trailing parens and split by space
    parts = fact_str.strip('()').split()
    if not parts:
        return None, [] # Handle empty string case
    predicate = parts[0]
    args = parts[1:]
    return predicate, args


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing two main components:
    1. The total minimum number of pushes required for all boxes that are not yet at their goal locations. This is calculated as the sum of shortest path distances for each box from its current location to its goal location on a relaxed grid graph where only walls (absence of adjacent facts) are obstacles.
    2. The minimum number of robot moves required to reach a location adjacent to *any* box that needs pushing. This is calculated as the shortest path distance for the robot from its current location to the closest location adjacent to any goal-seeking box, considering other boxes and walls as obstacles for the robot.

    Assumptions:
    - Location names are in the format 'loc_R_C' where R and C are integers.
    - The grid structure is defined by 'adjacent' facts in the static information.
    - Goal facts are of the form '(at box_name loc_R_C)'.
    - The heuristic is non-admissible and designed for greedy best-first search.
    - The grid graph is connected enough for solvable problems.
    - All boxes that appear in goal facts are present in the initial state and subsequent states.

    Heuristic Initialization:
    - Parses 'adjacent' facts from static information to build a grid graph representation (mapping 'loc_R_C' strings to (R, C) tuples and vice versa, and storing adjacencies). Locations that cannot be parsed are warned and ignored.
    - Parses goal facts to store the target (R, C) location for each box. Warnings are issued for unparseable or unknown goal locations.
    - Identifies box names from task.objects or infers them from goal facts.
    - Precomputes all-pairs shortest paths on the *empty* grid graph (no obstacles) to quickly get box-to-goal distances.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current state's facts: robot location, box locations, clear locations. Parse their locations into (R, C) tuples. If parsing fails or location is not in the grid, return infinity.
    2. Store current box locations in a dictionary `box_locations_rc` and a set `occupied_rc` for obstacle checking.
    3. Initialize `total_box_distance = 0`.
    4. Initialize `min_robot_distance_to_any_box = float('inf')`.
    5. Initialize `any_box_needs_pushing = False`.
    6. Iterate over all known box names (`self.box_names`):
       a. Get the box's goal location `goal_rc` from `self.box_goal_locations_rc`. If the box does not have a goal specified, skip it.
       b. Get the box's current location `current_box_rc` from `box_locations_rc`. If the box is unexpectedly missing from the state, return infinity.
       c. If `current_box_rc` is the same as `goal_rc`, the box is at its goal; continue to the next box.
       d. Set `any_box_needs_pushing = True`.
       e. Calculate the minimum number of pushes required for this box: `box_dist = self.get_distance_empty_grid(current_box_rc, goal_rc)`. This uses the precomputed distances on the empty grid. If `box_dist` is infinity, the goal is unreachable for this box even without obstacles; return infinity for the heuristic.
       f. Add `box_dist` to `total_box_distance`.
       g. Calculate the minimum robot moves to reach this box: `robot_dist_to_this_box = self.get_distance_robot_grid(robot_rc, current_box_rc, occupied_rc)`. This uses BFS on the grid considering other boxes as obstacles.
       h. Update `min_robot_distance_to_any_box = min(min_robot_distance_to_any_box, robot_dist_to_this_box)`.
    7. After iterating through all boxes:
       a. If `any_box_needs_pushing` is False, all goal boxes are at their goals, so the state is a goal state. Return 0.
       b. If `min_robot_distance_to_any_box` is still infinity, it means the robot cannot reach *any* box that needs pushing in the current state. This state is likely a dead end. Return infinity.
    8. The total heuristic value is `total_box_distance + min_robot_distance_to_any_box`. Return this value.
    """

    def __init__(self, task):
        super().__init__(task)
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Precompute grid graph and location mappings from static facts
        self.loc_str_to_rc = {}
        self.rc_to_loc_str = {}
        self.grid_graph = {} # (r, c) -> [(r', c'), ...]
        self._build_grid_graph()

        # Precompute goal locations for each box
        self.box_goal_locations_rc = {} # box_name -> (r, c)
        self._parse_goal_locations()

        # Identify box names
        self.box_names = set()
        if hasattr(task, 'objects') and task.objects:
            for obj_str in task.objects:
                parts = obj_str.strip().split(' - ')
                if len(parts) == 2 and parts[1] == 'box':
                    self.box_names.add(parts[0])
        else:
            # Fallback: infer box names from goal facts
            self.box_names = set(box_name for box_name in self.box_goal_locations_rc.keys())

        # Precompute shortest paths on the empty grid (for box distances)
        # Only compute for locations present in the grid graph
        grid_nodes = list(self.grid_graph.keys())
        self.empty_grid_distances = self._precompute_all_pairs_shortest_paths(grid_nodes, self.grid_graph, obstacles=set())


    def _parse_location(self, loc_str):
        """Helper to parse 'loc_R_C' string into (R, C) tuple."""
        # Expecting 'loc_R_C'
        loc_str = loc_str.strip('()') # Ensure no outer parens from fact parsing
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            try:
                r = int(parts[1])
                c = int(parts[2])
                return (r, c)
            except ValueError:
                # print(f"Warning: Invalid row/column in location string: {loc_str}", file=sys.stderr)
                return None # Indicate parsing failed
        # print(f"Warning: Unexpected location format: {loc_str}", file=sys.stderr)
        return None # Indicate parsing failed

    def _build_grid_graph(self):
        """Builds the grid graph and location mappings from adjacent facts."""
        adjacent_facts = [parse_fact(f) for f in self.static if f.startswith('(adjacent')]

        parsed_locations = {} # loc_str -> (r, c) or None

        # First pass: parse all location strings from adjacent facts
        all_loc_strings = set()
        for _, args in adjacent_facts:
             if len(args) >= 2:
                all_loc_strings.add(args[0])
                all_loc_strings.add(args[1])
             else:
                 print(f"Warning: Skipping malformed adjacent fact args: {args}", file=sys.stderr)


        for loc_str in sorted(list(all_loc_strings)):
             rc = self._parse_location(loc_str)
             if rc is not None:
                 parsed_locations[loc_str] = rc
                 self.loc_str_to_rc[loc_str] = rc
                 self.rc_to_loc_str[rc] = loc_str
             else:
                 print(f"Warning: Could not parse location string '{loc_str}' from adjacent facts.", file=sys.stderr)

        # Build adjacency list using (r, c) tuples
        for _, args in adjacent_facts:
            if len(args) >= 2:
                loc1_str, loc2_str = args[:2]
                rc1 = parsed_locations.get(loc1_str)
                rc2 = parsed_locations.get(loc2_str)
                if rc1 is not None and rc2 is not None: # Only add if both locations were parsed
                    if rc1 not in self.grid_graph:
                        self.grid_graph[rc1] = []
                    self.grid_graph[rc1].append(rc2)
                else:
                     # Warning already printed during parsing pass if location was unparseable
                     pass # Skip if parsing failed for either location
            # Warning for malformed args already printed

    def _parse_goal_locations(self):
        """Parses goal facts to find target locations for boxes."""
        for goal_fact_str in self.goals:
            predicate, args = parse_fact(goal_fact_str)
            if predicate == 'at' and len(args) == 2:
                box_name, loc_str = args
                goal_rc = self._parse_location(loc_str)
                if goal_rc is not None:
                    if goal_rc in self.rc_to_loc_str: # Check if parsed location is in our grid
                        self.box_goal_locations_rc[box_name] = goal_rc
                    else:
                         print(f"Warning: Goal location '{loc_str}' for box '{box_name}' parsed but not found in grid graph.", file=sys.stderr)
                else:
                    print(f"Warning: Could not parse goal location string '{loc_str}' for box '{box_name}'.", file=sys.stderr)
            # Ignore non-'at' goal facts if any

    def _bfs(self, start_rc, graph, obstacles_rc):
        """
        Performs BFS from start_rc on the given graph, avoiding obstacles.
        Returns a dictionary mapping reachable (r, c) to distance from start.
        """
        distances = {}
        # Ensure start_rc is in the graph and not an obstacle
        if start_rc not in graph or start_rc in obstacles_rc:
             return distances # Cannot start from an obstacle or non-existent node

        queue = deque([(start_rc, 0)])
        visited = {start_rc}
        distances[start_rc] = 0

        while queue:
            current_rc, dist = queue.popleft()

            if current_rc in graph: # Check if current_rc is a valid node in the graph
                for neighbor_rc in graph[current_rc]:
                    if neighbor_rc not in visited and neighbor_rc not in obstacles_rc:
                        visited.add(neighbor_rc)
                        distances[neighbor_rc] = dist + 1
                        queue.append((neighbor_rc, dist + 1))
        return distances

    def _precompute_all_pairs_shortest_paths(self, nodes, graph, obstacles):
        """
        Precomputes shortest paths from a set of nodes to all other nodes
        on a static graph (e.g., the empty grid).
        Returns a dict: (start_rc, end_rc) -> distance
        """
        distances = {}
        for start_rc in nodes:
            if start_rc not in obstacles:
                # Perform BFS from this start node
                bfs_distances = self._bfs(start_rc, graph, obstacles)
                for end_rc, dist in bfs_distances.items():
                    distances[(start_rc, end_rc)] = dist
        return distances

    def get_distance_empty_grid(self, start_rc, end_rc):
        """Retrieves precomputed distance on the empty grid."""
        # Ensure both start and end are valid grid locations before lookup
        if start_rc not in self.rc_to_loc_str or end_rc not in self.rc_to_loc_str:
             return float('inf')
        return self.empty_grid_distances.get((start_rc, end_rc), float('inf'))

    def get_distance_robot_grid(self, start_rc, end_rc, state_occupied_rc):
        """
        Calculates distance for the robot, considering current box locations as obstacles.
        state_occupied_rc: set of (r, c) tuples representing locations occupied by boxes.
        """
        # Robot cannot move into locations occupied by boxes
        obstacles_rc = state_occupied_rc.copy() # Copy the set of box locations

        # Ensure start_rc is a valid grid location
        if start_rc not in self.rc_to_loc_str:
             return float('inf')

        # Perform BFS from robot's start location on the main grid graph
        bfs_distances = self._bfs(start_rc, self.grid_graph, obstacles_rc)
        return bfs_distances.get(end_rc, float('inf'))


    def __call__(self, node):
        """
        Computes the Sokoban heuristic for the given state.
        See class docstring for detailed explanation.
        """
        state = node.state

        # 1. Identify current state facts
        robot_rc = None
        box_locations_rc = {} # box_name -> (r, c)
        occupied_rc = set() # Set of (r, c) tuples occupied by boxes

        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'at-robot' and len(args) == 1:
                loc_str = args[0]
                robot_rc = self._parse_location(loc_str)
                if robot_rc is None or robot_rc not in self.rc_to_loc_str:
                     # print(f"Error: Robot location '{loc_str}' in state could not be parsed or is not in grid.", file=sys.stderr)
                     return float('inf') # Robot location invalid
            elif predicate == 'at' and len(args) == 2:
                box_name, loc_str = args
                box_rc = self._parse_location(loc_str)
                if box_rc is None or box_rc not in self.rc_to_loc_str:
                     # print(f"Error: Box '{box_name}' location '{loc_str}' in state could not be parsed or is not in grid.", file=sys.stderr)
                     return float('inf') # Box location invalid
                box_locations_rc[box_name] = box_rc
                occupied_rc.add(box_rc)
            # Ignore 'clear' facts for this heuristic calculation

        if robot_rc is None:
             # print("Error: Robot location not found in state.", file=sys.stderr)
             return float('inf') # Robot location must be in state

        # 3. Initialize total_box_distance
        total_box_distance = 0
        # 4. Initialize min_robot_distance_to_any_box
        min_robot_distance_to_any_box = float('inf')
        # 5. Initialize any_box_needs_pushing
        any_box_needs_pushing = False

        # 6. Iterate over all known box names
        for box_name in self.box_names:
            # 6a. Get goal location
            goal_rc = self.box_goal_locations_rc.get(box_name)
            if goal_rc is None:
                 continue # This box does not have a goal specified

            # 6b. Get current location
            current_box_rc = box_locations_rc.get(box_name)
            if current_box_rc is None:
                 # print(f"Error: Box '{box_name}' from task objects/goals not found in current state.", file=sys.stderr)
                 return float('inf') # Box missing from state

            # 6c. Check if at goal
            if current_box_rc == goal_rc:
                continue # Box is already at its goal

            # 6d. Mark that at least one box needs pushing
            any_box_needs_pushing = True

            # 6e. Calculate box distance on empty grid
            box_dist = self.get_distance_empty_grid(current_box_rc, goal_rc)
            if box_dist == float('inf'):
                # Goal is unreachable for this box even on the empty grid
                return float('inf')

            # 6f. Add box distance to total
            total_box_distance += box_dist

            # 6g. Calculate robot distance to this box
            # Obstacles for robot are other boxes (excluding the current box itself for the target location)
            robot_obstacles_rc = occupied_rc - {current_box_rc}
            robot_dist_to_this_box = self.get_distance_robot_grid(robot_rc, current_box_rc, robot_obstacles_rc)

            # 6h. Update minimum robot distance
            min_robot_distance_to_any_box = min(min_robot_distance_to_any_box, robot_dist_to_this_box)

        # 7a. Check if goal state
        if not any_box_needs_pushing:
             return 0 # All goal boxes are at their goals

        # 7b. Check if robot can reach any box
        if min_robot_distance_to_any_box == float('inf'):
             # Robot cannot reach any box that needs pushing
             return float('inf')

        # 8. Calculate total heuristic
        # Heuristic = Sum of box distances + Minimum robot distance to any box
        return total_box_distance + min_robot_distance_to_any_box

