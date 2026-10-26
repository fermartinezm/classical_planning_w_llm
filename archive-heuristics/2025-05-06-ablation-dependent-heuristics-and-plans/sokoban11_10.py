import collections
# Assume heuristics.heuristic_base.Heuristic is available
# from heuristics.heuristic_base import Heuristic

# Utility function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper class for graph operations
class BFS:
    """Helper class to compute shortest path distances on the grid graph and find pushing locations."""
    def __init__(self, adjacent_facts):
        self.graph = collections.defaultdict(set)
        self.directional_graph = {} # Map loc1 -> dir -> loc2 where adjacent(loc1, loc2, dir)
        self.reverse_directional_graph = {} # Map loc2 -> dir -> loc1 where adjacent(loc1, loc2, dir)

        for fact in adjacent_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.graph[l1].add(l2)

                if l1 not in self.directional_graph:
                    self.directional_graph[l1] = {}
                self.directional_graph[l1][direction] = l2

                if l2 not in self.reverse_directional_graph:
                    self.reverse_directional_graph[l2] = {}
                self.reverse_directional_graph[l2][direction] = l1


    def shortest_path_distance(self, start_loc, end_loc):
        """Computes the shortest path distance between two locations using BFS."""
        if start_loc == end_loc:
            return 0

        # Check if locations exist in the graph (at least one outgoing or incoming edge)
        # This check is important if the graph is sparse or locations exist without adjacencies
        if start_loc not in self.graph and start_loc not in self.reverse_directional_graph:
             return float('inf')
        if end_loc not in self.graph and end_loc not in self.reverse_directional_graph and start_loc != end_loc:
             return float('inf')


        queue = collections.deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Explore neighbors from the graph (all adjacent locations)
            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # No path found

    def get_pushing_location(self, box_loc, push_direction):
        """
        Given a box location (box_loc) and the direction of the push (push_direction,
        which is the direction from box_loc to loc_next), return the required robot location.
        Robot must be at loc_r such that adjacent(loc_r, box_loc, push_direction).
        This means loc_r is adjacent to box_loc in the *same* direction as the push.
        We use the reverse_directional_graph to find loc_r.
        """
        if box_loc in self.reverse_directional_graph and push_direction in self.reverse_directional_graph[box_loc]:
             return self.reverse_directional_graph[box_loc][push_direction]
        return None # No location found where pushing in that direction leads to box_loc

    def get_direction(self, loc1, loc2):
        """
        Given two adjacent locations loc1 and loc2, return the direction from loc1 to loc2.
        Assumes loc1 and loc2 are adjacent and the directional link exists.
        """
        if loc1 in self.directional_graph:
            for direction, neighbor in self.directional_graph[loc1].items():
                if neighbor == loc2:
                    return direction
        return None # Not adjacent or direction not found (shouldn't happen if loc2 is in graph[loc1])


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It sums the shortest path distance for each box
    to its goal (minimum pushes) and the shortest path distance for the robot
    to reach a position from which it can make the first push towards the goal
    for each box.

    # Assumptions
    - Each box has a unique goal location specified by an '(at box_name loc_name)' predicate in the goals.
    - The grid connectivity is fully defined by '(adjacent l1 l2 dir)' facts in the static information.
    - All locations and objects referenced in the state and goals are valid and exist in the graph defined by adjacent facts.
    - The state representation includes '(at-robot loc)' and '(at box loc)' facts for all relevant objects.

    # Heuristic Initialization
    - Extract goal locations for each box from `task.goals`.
    - Build the adjacency graph and directional graphs (forward and reverse) from `task.static` facts
      to enable shortest path calculations and finding required robot pushing locations efficiently.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic value to 0.
    2. Identify the robot's current location and the current location of each box from the state.
    3. For each box that has a goal location specified and is not yet at its goal:
       a. Get the box's current location (`loc_b`) and its goal location (`loc_g`).
       b. Calculate the shortest path distance from `loc_b` to `loc_g` using BFS. This is the minimum number of pushes required for this box. Add this distance to the total heuristic. If the distance is infinite, the state is likely unsolvable, return infinity.
       c. Identify potential locations (`loc_next`) adjacent to `loc_b` such that moving the box to `loc_next` brings it strictly closer to `loc_g` (i.e., `shortest_path_distance(loc_next, loc_g) < shortest_path_distance(loc_b, loc_g)`).
       d. For each such `loc_next`, determine the required robot location (`loc_r`) to perform the push from `loc_b` to `loc_next`. Based on the PDDL `push` action, the robot must be adjacent to `loc_b` in the *same* direction as the push from `loc_b` to `loc_next`. Use the reverse directional graph to find `loc_r`. Collect all such valid `loc_r` into a set `potential_push_locs`.
       e. Calculate the shortest path distance from the robot's current location to the *nearest* location in `potential_push_locs`. This estimates the robot's cost to get into position for the first useful push for this box. If `potential_push_locs` is empty (meaning the box cannot be pushed towards the goal from its current location towards the goal), or if the robot cannot reach any of these locations, the state is likely a dead end for this box, return infinity.
       f. Add this minimum robot distance to the total heuristic.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each box
        self.box_to_goal_location = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically '(at obj loc)'
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                # Assuming objects in 'at' goals are the boxes we need to move
                self.box_to_goal_location[obj_name] = loc_name

        # 2. Build the adjacency graph and directional graphs
        # Filter for only 'adjacent' facts
        adjacent_facts = [fact for fact in static_facts if get_parts(fact) and get_parts(fact)[0] == 'adjacent']
        self.bfs_helper = BFS(adjacent_facts)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 2. Parse current state
        robot_location = None
        box_locations = {} # box_name -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 # Only track locations for boxes that are in our goals
                 if obj_name in self.box_to_goal_location:
                    box_locations[obj_name] = loc_name

        # If robot location is unknown, state is invalid/unreachable
        if robot_location is None:
             return float('inf')

        total_heuristic = 0

        # 3. Calculate heuristic for each box not at its goal
        for box_name, goal_location in self.box_to_goal_location.items():
            current_box_location = box_locations.get(box_name)

            # If a goal box is not found in the state's 'at' facts, it's an invalid state.
            # This shouldn't happen in a well-formed problem instance and state representation.
            # Return infinity as a safeguard.
            if current_box_location is None:
                 return float('inf')

            if current_box_location == goal_location:
                continue # Box is already at its goal

            # b. Calculate box movement cost (minimum pushes)
            box_movement_cost = self.bfs_helper.shortest_path_distance(current_box_location, goal_location)

            # If box cannot reach goal (e.g., trapped), distance is inf. This state is likely a dead end.
            if box_movement_cost == float('inf'):
                 return float('inf')

            total_heuristic += box_movement_cost

            # c, d. Identify potential first pushing locations
            potential_push_locs = set()
            current_dist_to_goal = self.bfs_helper.shortest_path_distance(current_box_location, goal_location)

            # If box is at goal, current_dist_to_goal is 0, this loop is skipped.
            # If box_movement_cost was inf, we already returned inf.
            # If box_movement_cost > 0, current_dist_to_goal should also be > 0.

            # Iterate through neighbors of the box's current location
            if current_box_location in self.bfs_helper.graph:
                for loc_next in self.bfs_helper.graph[current_box_location]:
                    # Check if moving to loc_next gets strictly closer to the goal
                    dist_next_to_goal = self.bfs_helper.shortest_path_distance(loc_next, goal_location)

                    # Check if dist_next_to_goal is finite before comparing
                    if dist_next_to_goal != float('inf') and dist_next_to_goal < current_dist_to_goal:
                        # This loc_next is a step towards the goal.
                        # Find the direction from current_box_location to loc_next
                        push_dir = self.bfs_helper.get_direction(current_box_location, loc_next)
                        if push_dir:
                            # Find the required robot location (loc_r) for this push.
                            # Robot must be adjacent to current_box_location in the *same* direction as push_dir.
                            loc_r = self.bfs_helper.get_pushing_location(current_box_location, push_dir)
                            if loc_r:
                                potential_push_locs.add(loc_r)

            # e. Calculate robot cost for the first push
            robot_first_push_cost = float('inf')
            if potential_push_locs:
                # Find the minimum distance from the robot to any of the potential pushing locations
                min_robot_dist = float('inf')
                for push_loc in potential_push_locs:
                    dist = self.bfs_helper.shortest_path_distance(robot_location, push_loc)
                    min_robot_dist = min(min_robot_dist, dist)
                robot_first_push_cost = min_robot_dist

            # If no potential pushing locations were found (box stuck) or robot cannot reach any
            # Note: If box_movement_cost was already inf, we returned early.
            # If box_movement_cost > 0 but potential_push_locs is empty, it means the box is at a local minimum
            # w.r.t. distance to goal, which is a dead end unless other boxes are moved.
            # Returning inf here is appropriate for a non-admissible heuristic.
            if robot_first_push_cost == float('inf'):
                 return float('inf')

            total_heuristic += robot_first_push_cost

        # 4. Return total heuristic
        return total_heuristic
