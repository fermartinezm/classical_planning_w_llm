import collections
import math

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
    The heuristic estimates the cost to reach a goal state by summing two components:
    1. The sum of shortest path distances for each box from its current location
       to its goal location on the static grid graph (ignoring obstacles).
    2. The minimum shortest path distance for the robot from its current location
       to any location from which it can make a valid push on any box that is
       not yet at its goal location (considering current obstacles).

    Assumptions:
    - The PDDL domain follows the standard Sokoban structure with predicates
      at-robot, at (for boxes), clear, and adjacent.
    - Goal states are defined by the locations of boxes using the 'at' predicate.
      There is a one-to-one correspondence between boxes and goal locations specified.
    - Locations are connected via 'adjacent' predicates forming a graph.
    - Location names are consistent.
    - The goal facts are either a set of individual facts like `(at box1 locA)`
      or a single fact `(and (at box1 locA) (at box2 locB) ...)` within the goal set.

    Heuristic Initialization:
    1. Parses static 'adjacent' facts to build a graph representation of the locations.
    2. Computes all-pairs shortest path distances between all locations on this
       static graph using BFS. This is stored for efficient lookup.
    3. Parses goal facts to determine the target location for each box.

    Step-By-Step Thinking for Computing Heuristic:
    1. Extract the current location of the robot and each box from the state facts.
       Also, identify locations currently occupied by boxes.
    2. Check if all boxes are already at their designated goal locations. If so,
       the state is a goal state, and the heuristic returns 0.
    3. Calculate the 'box distance sum': For each box, retrieve its goal location
       (precomputed during initialization) and its current location. Add the
       precomputed static shortest path distance between these two locations
       to a running sum. If any box's goal is unreachable statically, return infinity
       (represented by a large integer).
    4. Identify potential 'valid push-from' locations for the robot: For each box
       that is not yet at its goal, iterate through possible push directions. A
       push is valid from location L_r to push a box at L_b to L_f if L_r is
       adjacent to L_b, L_b is adjacent to L_f in the same direction, and L_f
       is currently clear (not occupied by any box). Collect all such L_r locations.
    5. Perform a Breadth-First Search (BFS) starting from the robot's current
       location on the graph of locations reachable by the robot in the current
       state. The robot cannot move into locations occupied by boxes. This BFS
       computes the shortest distance from the robot to all reachable locations.
    6. Calculate the 'robot accessibility cost': Find the minimum distance from
       the robot's current location (obtained from the BFS in step 5) to any
       of the 'valid push-from' locations identified in step 4. If none are
       reachable, return infinity (represented by a large integer).
    7. The final heuristic value is the sum of the 'box distance sum' (step 3)
       and the 'robot accessibility cost' (step 6). A large integer is returned
       instead of math.inf if any component is infinite, indicating a likely
       unsolvable state or deadlock.
    """
    def __init__(self, task):
        self.task = task
        self.location_graph = collections.defaultdict(list)
        self.locations = set()
        self.box_goals = {}
        self.opposite_direction = {
            'up': 'down', 'down': 'up',
            'left': 'right', 'right': 'left'
        }
        self.LARGE_VALUE = 1_000_000 # Value to return for infinity

        # 5) The information from static facts is extracted into suitable data structures in the constructor.
        self._parse_static_info(task.static)
        self._parse_goal_info(task.goals)

        # Precompute all-pairs shortest paths on the static location graph
        self.static_distances = self._compute_all_pairs_shortest_paths()

    def _parse_static_info(self, static_facts):
        """
        Parses static facts like adjacent.
        """
        for fact in static_facts:
            # 1) The code for extracting objects from facts remembers to ignore the surrounding brackets.
            parts = fact.strip("()").split()
            predicate = parts[0]
            if predicate == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.location_graph[loc1].append((loc2, direction))
                self.locations.add(loc1)
                self.locations.add(loc2)

    def _parse_goal_info(self, goal_facts):
        """
        Parses goal facts to find box target locations.
        Assumes goal facts are of the form (at boxX locY) potentially wrapped in (and ...).
        """
        goal_facts_to_process = []
        # Check if the goal is a single fact wrapped in (and ...)
        if len(goal_facts) == 1:
            goal_str = list(goal_facts)[0].strip()
            if goal_str.startswith('(and '):
                 # Extract inner facts from (and (fact1) (fact2) ...)
                 # Remove '(and ' and the final ')'
                 inner_facts_str = goal_str[5:-1].strip()
                 # Split by ') (' to get individual fact strings
                 # Example: '(f1) (f2) (f3)' -> split(') (') -> ['(f1', 'f2', 'f3)']
                 fact_parts = inner_facts_str.split(') (')
                 reconstructed_facts = []
                 if fact_parts:
                     # First part needs trailing ')'
                     reconstructed_facts.append(fact_parts[0] + ')')
                     # Middle parts need leading '(' and trailing ')'
                     for part in fact_parts[1:-1]:
                          reconstructed_facts.append('(' + part + ')')
                     # Last part needs leading '(' if there was more than one part
                     if len(fact_parts) > 1:
                         reconstructed_facts.append('(' + fact_parts[-1])

                 goal_facts_to_process = reconstructed_facts
            else:
                 # Single goal fact not wrapped in (and)
                 goal_facts_to_process = list(goal_facts)
        else:
             # Goal facts are already provided as a set of individual facts
             goal_facts_to_process = list(goal_facts)

        for fact in goal_facts_to_process:
            parts = fact.strip("()").split()
            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3: # (at ?o - box ?l - location)
                box_name, goal_loc = parts[1], parts[2]
                self.box_goals[box_name] = goal_loc

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        on the static location graph using BFS.
        """
        distances = {}
        for start_loc in self.locations:
            distances[start_loc] = self._bfs_shortest_paths(start_loc, self.location_graph)
        return distances

    def _bfs_shortest_paths(self, start_loc, graph, occupied_locations=None):
        """
        Performs BFS from start_loc to find shortest paths.
        Can operate on the static graph (occupied_locations=None) or
        the dynamic robot-reachable graph (occupied_locations provided).
        """
        q = collections.deque([(start_loc, 0)])
        visited = {start_loc}
        dist = {start_loc: 0}

        while q:
            current_loc, current_dist = q.popleft()

            # Neighbors from the static graph
            neighbors_with_dir = graph.get(current_loc, [])

            for neighbor_loc, direction in neighbors_with_dir:
                # Check if neighbor is occupied by a box in the current state
                # This check is only relevant for the robot's movement (dynamic graph)
                if occupied_locations is not None and neighbor_loc in occupied_locations:
                    continue # Robot cannot move into an occupied location

                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    dist[neighbor_loc] = current_dist + 1
                    q.append((neighbor_loc, current_dist + 1))

        return dist

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            An integer heuristic value (or a large integer for infinity).
        """
        # Step-By-Step Thinking for Computing Heuristic:

        # 1. Extract current robot and box locations and occupied locations.
        current_box_locations = {}
        robot_loc = None
        occupied_by_box = set()
        clear_locations = set() # Need clear locations to check valid pushes
        for fact in state:
            parts = fact.strip("()").split()
            if parts[0] == 'at' and len(parts) == 3:
                box_name, loc = parts[1], parts[2]
                current_box_locations[box_name] = loc
                occupied_by_box.add(loc)
            elif parts[0] == 'at-robot' and len(parts) == 2:
                 robot_loc = parts[1]
            elif parts[0] == 'clear' and len(parts) == 2:
                 clear_locations.add(parts[1])


        # Handle case where robot_loc is not found (should not happen in valid states)
        if robot_loc is None:
             return self.LARGE_VALUE

        # 2. Check for goal state.
        all_boxes_at_goal = True
        for box_name, goal_loc in self.box_goals.items():
            if current_box_locations.get(box_name) != goal_loc:
                all_boxes_at_goal = False
                break

        # 2) The heuristic is 0 only for goal states.
        if all_boxes_at_goal:
             return 0

        # 3. Calculate the sum of static distances for each box to its goal.
        box_dist_sum = 0
        for box_name, goal_loc in self.box_goals.items():
            current_loc = current_box_locations.get(box_name)
            if current_loc is None:
                 # Box not found in state - indicates a problem
                 return self.LARGE_VALUE

            if current_loc in self.static_distances and goal_loc in self.static_distances[current_loc]:
                 box_dist_sum += self.static_distances[current_loc][goal_loc]
            else:
                 # Goal location is statically unreachable from box location
                 return self.LARGE_VALUE

        # 4. Identify valid push-from locations for the robot.
        # A location L_r is a valid push-from location if:
        # - There is a box B at L_b (where L_b is adjacent to L_r) that is not at its goal.
        # - Pushing B from L_b in the direction from L_r to L_b leads to L_f (L_b adjacent to L_f in the same direction).
        # - L_f is currently clear (not occupied by any box).
        valid_push_from_locations = set()

        for box_name, current_box_loc in current_box_locations.items():
            if current_box_loc != self.box_goals.get(box_name): # Box needs moving
                # Find locations adjacent to the box's current location
                for loc_adj_to_box, dir_from_box_to_adj in self.location_graph.get(current_box_loc, []):
                    # loc_adj_to_box is a potential push-from location (L_r)
                    # current_box_loc is L_b
                    # dir_from_box_to_adj is the direction from L_b to L_r.
                    # The push direction is the opposite: from L_r to L_b.
                    push_dir = self.opposite_direction.get(dir_from_box_to_adj)

                    if push_dir:
                        # Find the location L_f that is adjacent to L_b in the push_dir
                        for loc_target_box, dir_from_box_to_target in self.location_graph.get(current_box_loc, []):
                            if dir_from_box_to_target == push_dir:
                                # loc_target_box is L_f
                                # Check if L_f is clear
                                if loc_target_box in clear_locations:
                                    # Found a valid push: robot at loc_adj_to_box, box at current_box_loc, target loc_target_box is clear
                                    valid_push_from_locations.add(loc_adj_to_box)
                                # Note: A location is also clear if it's the robot's current location
                                # (the robot will move out of the way during the push).
                                # However, the 'clear' predicate usually means empty. Let's stick to 'clear' predicate for simplicity.
                                # If the robot is at loc_target_box, it's not clear.

        # If no valid push is currently possible for any non-goal box,
        # the robot cannot make progress towards pushing boxes.
        if not valid_push_from_locations:
             # This state might be a deadlock or require complex setup moves.
             # A simple heuristic might struggle here. Return a large value.
             return self.LARGE_VALUE

        # 5. Compute robot's reachable locations and distances in the current state.
        # The robot cannot move into locations occupied by boxes.
        robot_distances = self._bfs_shortest_paths(robot_loc, self.location_graph, occupied_locations=occupied_by_box)

        # 6. Find the minimum distance from the robot to any valid push-from location.
        robot_accessibility_cost = math.inf
        for target_loc in valid_push_from_locations:
            if target_loc in robot_distances:
                robot_accessibility_cost = min(robot_accessibility_cost, robot_distances[target_loc])

        # If the robot cannot reach any valid push-from location, return infinity.
        if robot_accessibility_cost == math.inf:
             return self.LARGE_VALUE

        # 7. Calculate the final heuristic value.
        # The heuristic is the sum of box distances and robot accessibility.
        heuristic_value = box_dist_sum + robot_accessibility_cost

        # 3) The heuristic value is finite for solvable states.
        # If a state is solvable, there must be a path for the robot to reach
        # positions to push boxes towards goals. The BFS should find finite distances.
        # If it returns infinity, it suggests unsolvability from this state.

        return heuristic_value

