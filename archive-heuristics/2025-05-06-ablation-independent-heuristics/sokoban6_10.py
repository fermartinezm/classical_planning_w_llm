import re
import collections
import math
from heuristics.heuristic_base import Heuristic
from task import Task

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the number of actions required to reach a goal state
        by summing two components:
        1. The total shortest path distance for all boxes from their current
           locations to their assigned goal locations on the grid graph.
        2. The minimum shortest path distance for the robot from its current
           location to any location from which it can perform a valid push action
           on any box towards a currently clear adjacent location.

    Assumptions:
        - The grid structure is defined by the 'adjacent' static facts.
        - Actions (move and push) have unit cost.
        - Each box is assigned to a specific goal location, as defined in the
          problem's goal facts (e.g., (at box1 loc_A), (at box2 loc_B)). The
          number of boxes equals the number of goal locations.
        - The 'clear' predicate in a state means the location is not occupied
          by the robot or a box.
        - The heuristic is non-admissible, designed for greedy best-first search.
        - The PDDL definition of the 'push' action's effects might contain
          an anomaly ((clear ?bloc)), but the heuristic relies on the state
          representation where 'clear' seems to mean unoccupied.

    Heuristic Initialization:
        1. Parses all 'adjacent' facts from the static information to identify
           all locations in the grid and build an adjacency list representation
           of the grid graph. It also stores adjacency information indexed by
           direction for efficient lookup of push positions.
        2. Computes all-pairs shortest path distances between all locations
           on the grid graph using Breadth-First Search (BFS) starting from
           each location. These distances are stored for quick lookup.
        3. Parses the goal facts to determine the specific goal location for
           each box.
        4. Parses the initial state facts to identify the names of all boxes
           in the problem.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Check if the state satisfies all goal facts. If yes, the heuristic
           value is 0.
        2. Extract the current location of the robot, the current location of
           each box, and the set of currently clear locations from the state.
        3. Calculate the box-goal distance component:
           - Initialize the total distance to 0.
           - For each box, find its current location and its assigned goal location.
           - Look up the precomputed shortest path distance between these two
             locations.
           - Add this distance to the total.
           - If any box's current location is unreachable from its goal location
             (distance is infinity), the total box-goal distance is infinity.
        4. Calculate the robot-to-push distance component:
           - Initialize the minimum robot distance to infinity.
           - For each box at its current location `L_box`:
             - Consider all four possible push directions (up, down, left, right).
             - For a given direction `dir_box_moves`:
               - Find the potential destination location `L_front` if the box
                 were pushed in `dir_box_moves`.
               - If `L_front` exists (i.e., is adjacent to `L_box` in `dir_box_moves`)
                 and is currently clear in the state:
                 - Find the required robot location `L_push` from which to push
                   the box in `dir_box_moves`. `L_push` is adjacent to `L_box`
                   in `dir_box_moves`. This means `L_box` is adjacent to `L_push`
                   in the opposite direction.
                 - If `L_push` exists:
                   - Look up the precomputed shortest path distance from the
                     robot's current location to `L_push`.
                   - Update the minimum robot distance found so far.
           - If no valid push is possible from any reachable location for any box,
             the minimum robot distance remains infinity.
        5. The final heuristic value is the sum of the box-goal distance component
           and the robot-to-push distance component. If either component is
           infinity, the total heuristic is infinity.
    """

    def __init__(self, task: Task):
        super().__init__(task)
        self.goals = task.goals
        self.static = task.static

        # Data structures for grid and distances
        self.locations = set()
        self.adj_map = {}  # loc -> list of adjacent locs (for BFS graph)
        self.adj_by_dir = {} # loc -> dir -> adjacent loc (for push logic)
        self.distances = {} # (loc1, loc2) -> shortest_path_distance

        # Data structures for boxes and goals
        self.box_names = set()
        self.goal_locations = {} # box_name -> goal_loc_str

        # Define opposite directions for push logic
        self.opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}


        # Parse static facts to build grid graph
        self._parse_static(task.static)

        # Compute all-pairs shortest paths
        self._compute_all_pairs_shortest_paths()

        # Parse goal facts to get box-goal mapping
        self._parse_goals(task.goals)

        # Parse initial state to get box names (they are constant throughout the problem)
        self._parse_initial_state(task.initial_state)


    def _parse_fact(self, fact_str: str):
        """Helper to parse a PDDL fact string."""
        # Removes surrounding brackets and splits by space
        parts = fact_str[1:-1].split()
        if not parts: # Handle empty fact string, though unlikely
            return None, []
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _parse_static(self, static_facts: frozenset[str]):
        """Parses static facts to build grid graph structures."""
        for fact_str in static_facts:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'adjacent' and len(args) == 3:
                l1, l2, direction = args
                self.locations.add(l1)
                self.locations.add(l2)
                # Build adjacency map for BFS (undirected graph assuming bidirectionality)
                # Add edge l1 -> l2
                self.adj_map.setdefault(l1, []).append(l2)
                # Add edge l2 -> l1 (assuming bidirectionality for movement)
                self.adj_map.setdefault(l2, []).append(l1)

                # Build adjacency by direction map (directed)
                self.adj_by_dir.setdefault(l1, {})[direction] = l2

        # Ensure all locations mentioned in adj_map keys are also in locations set
        for loc in self.adj_map:
             self.locations.add(loc)


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations."""
        for start_loc in self.locations:
            self.distances.update(self._bfs(start_loc))

    def _bfs(self, start_loc: str):
        """Performs BFS from a start location to find distances to all others."""
        distances_from_start = {loc: float('inf') for loc in self.locations}
        distances_from_start[start_loc] = 0
        queue = collections.deque([start_loc])

        while queue:
            curr_loc = queue.popleft()
            dist = distances_from_start[curr_loc]

            # Check if curr_loc has any outgoing edges in the adj_map
            if curr_loc in self.adj_map:
                for neighbor_loc in self.adj_map[curr_loc]:
                    if distances_from_start[neighbor_loc] == float('inf'):
                        distances_from_start[neighbor_loc] = dist + 1
                        queue.append(neighbor_loc)

        # Store distances in the main distances dictionary
        pair_distances = {}
        for end_loc in self.locations:
             pair_distances[(start_loc, end_loc)] = distances_from_start[end_loc]
        return pair_distances


    def _parse_goals(self, goal_facts: frozenset[str]):
        """Parses goal facts to get the target location for each box."""
        for goal_fact_str in goal_facts:
            predicate, args = self._parse_fact(goal_fact_str)
            # Goal facts are typically (at boxN loc_M)
            if predicate == 'at' and len(args) == 2 and args[0].startswith('box'):
                box_name, loc_str = args
                self.goal_locations[box_name] = loc_str

    def _parse_initial_state(self, initial_state_facts: frozenset[str]):
        """Parses initial state to identify box names."""
        for fact_str in initial_state_facts:
             predicate, args = self._parse_fact(fact_str)
             # Box locations are given by (at boxN loc_M) in the initial state
             if predicate == 'at' and len(args) == 2 and args[0].startswith('box'):
                 self.box_names.add(args[0])


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated cost (heuristic value) as an integer or float('inf').
        """
        state = node.state

        # 1. Check if goal state
        if self.goals <= state:
            return 0

        # 2. Extract current state information
        current_box_locs_map = {} # box_name -> current_loc_str
        current_clear_locs = set()
        robot_loc = None

        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'at' and len(args) == 2 and args[0] in self.box_names:
                current_box_locs_map[args[0]] = args[1]
            elif predicate == 'at-robot' and len(args) == 1:
                robot_loc = args[0]
            elif predicate == 'clear' and len(args) == 1:
                current_clear_locs.add(args[0])

        # Basic validation (should hold for valid states in typical Sokoban)
        if robot_loc is None or len(current_box_locs_map) != len(self.box_names):
             # State is missing robot or boxes, likely invalid or unsolvable
             return float('inf')


        # 3. Calculate box-goal distance component (fixed assignment)
        box_goal_dist_sum = 0
        for box_name in self.box_names:
            current_loc = current_box_locs_map.get(box_name)
            goal_loc = self.goal_locations.get(box_name)

            # Should not happen if initialization and state parsing are correct
            if current_loc is None or goal_loc is None:
                 return float('inf') # Should indicate a problem with problem definition or parsing

            # Get precomputed shortest path distance
            dist = self.distances.get((current_loc, goal_loc), float('inf'))

            if dist == float('inf'):
                box_goal_dist_sum = float('inf')
                break # This box cannot reach its goal

            box_goal_dist_sum += dist

        # If box_goal_dist_sum is already infinity, the problem is unsolvable from here
        if box_goal_dist_sum == float('inf'):
            return float('inf')


        # 4. Calculate robot-to-push distance component
        min_robot_dist = float('inf')

        # Iterate through all boxes to find a pushable one
        for box_name in self.box_names:
            L_box_str = current_box_locs_map[box_name]

            # Iterate through all 4 possible push directions for this box
            for dir_box_moves in ['up', 'down', 'left', 'right']:
                # Find the location the box would move into if pushed in this direction
                L_front_str = self.adj_by_dir.get(L_box_str, {}).get(dir_box_moves)

                # Check if L_front_str exists (is a valid location adjacent to box)
                # AND is currently clear in the state (precondition for push)
                if L_front_str is not None and L_front_str in current_clear_locs:
                    # Found a potential pushable spot for the box.
                    # Now find the required robot location L_push to perform this push.
                    # Robot must be adjacent to L_box_str in the direction opposite to dir_box_moves.
                    # PDDL: adjacent(?rloc ?bloc ?dir) and adjacent(?bloc ?floc ?dir)
                    # ?rloc is L_push_str, ?bloc is L_box_str, ?floc is L_front_str, ?dir is dir_box_moves
                    # So, adjacent(L_push_str, L_box_str, dir_box_moves) must hold.
                    # This means L_box_str is adjacent to L_push_str in opposite_dir[dir_box_moves].
                    dir_from_box_to_push = self.opposite_dir.get(dir_box_moves)

                    if dir_from_box_to_push is not None:
                         L_push_str = self.adj_by_dir.get(L_box_str, {}).get(dir_from_box_to_push)

                         # Check if L_push_str exists (is a valid location adjacent to box)
                         if L_push_str is not None:
                             # Found a valid push: robot at L_push_str pushes box at L_box_str to L_front_str
                             # Calculate distance from current robot location to this required push location L_push_str
                             dist = self.distances.get((robot_loc, L_push_str), float('inf'))
                             min_robot_dist = min(min_robot_dist, dist)

        # 5. Return the sum of the two components
        # If min_robot_dist is still infinity, it means no valid push is currently possible
        # from any reachable location for any box. This state is likely a dead end
        # or requires complex maneuvers not captured by this simple robot component.
        if min_robot_dist == float('inf'):
            return float('inf')
        else:
            # The heuristic is the sum of total box movement cost (pushes)
            # and the cost for the robot to get to the first push position.
            # This is a non-admissible estimate.
            return box_goal_dist_sum + min_robot_dist
