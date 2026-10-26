from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_distance(start, end, graph):
    """
    Computes the shortest path distance between two locations using BFS
    on the provided graph (adjacency list).
    Returns float('inf') if no path exists.
    Assumes start and end are valid nodes in the graph.
    """
    if start == end:
        return 0
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Get neighbors from the provided graph
        # Ensure current_loc is a valid key in the graph
        neighbors = graph.get(current_loc, [])

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to reach a goal state.
    It is calculated as the sum of:
    1. The minimum number of pushes required for each misplaced box to reach its goal
       (shortest path distance on the location graph).
    2. The minimum number of robot moves required to reach a valid pushing position
       for *any* of the misplaced boxes.

    # Assumptions
    - The grid structure and adjacency are defined by the 'adjacent' facts.
    - Goal is defined by 'at' facts for boxes.
    - The heuristic does not detect complex deadlocks (e.g., boxes in corners
      with no path to a goal) beyond simple graph reachability.
    - Robot can only move to 'clear' locations. A location is clear if the
      '(clear loc)' predicate is true in the current state.

    # Heuristic Initialization
    - Builds an adjacency graph of all locations from 'adjacent' facts.
    - Stores the goal location for each box from the goal state.
    - Builds a mapping from (location1, location2) pair to the direction
      from location1 to location2, based on 'adjacent' facts.
    - Identifies all possible locations in the domain.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to find the robot's location, box locations,
       and the set of 'clear' locations (based on the '(clear loc)' predicate).
    2. Identify which boxes are not at their goal locations (misplaced boxes).
       Only consider boxes that are mentioned in the goal state.
    3. If no boxes are misplaced, the heuristic is 0.
    4. Calculate the total minimum pushes required for all misplaced boxes:
       - For each misplaced box, find the shortest path distance from its
         current location to its goal location using BFS on the full location graph.
       - Sum these distances. If any box is unreachable from its goal on the
         full graph, return a large value (e.g., 1,000,000) indicating a likely
         unsolvable state or a state far from the goal that requires complex maneuvers.
    5. Calculate the minimum robot moves required to reach a valid pushing position:
       - A location R is a valid pushing position for a box at L_b if R is
         adjacent to L_b, and pushing the box from L_b in the direction from
         R to L_b moves the box to a location L_next, and L_next is currently clear
         (i.e., '(clear L_next)' is true in the state).
       - Determine the set of locations the robot can move to. These are all
         locations for which '(clear loc)' is true in the state, plus the
         robot's current location (as it can start BFS from there).
       - Build a robot movement graph where nodes are the robot-reachable
         locations and edges connect adjacent locations if the destination is
         robot-reachable.
       - For each misplaced box:
         - Find all locations R adjacent to the box's current location L_b.
         - For each R, determine the location L_next the box would move to if
           pushed from L_b in the direction from R to L_b.
         - Check if '(clear L_next)' is in the current state.
         - If L_next is clear, R is a valid pushing position. Calculate the
           shortest path distance from the robot's current location to R
           using BFS on the robot movement graph.
       - The minimum robot distance is the minimum of these distances over all
         misplaced boxes and all their valid pushing positions. If no valid
         pushing position is reachable for any misplaced box (meaning the robot
         cannot get into position to push any box towards *any* valid clear spot),
         return a large value.
    6. The total heuristic value is the sum of the total minimum pushes for
       all boxes and the minimum robot moves to reach an initial pushing position.
       This represents the estimated total number of actions (robot moves + pushes).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting graph structure and goals."""
        self.goals = task.goals
        static_facts = task.static
        self.all_locations = set()
        self.graph = {} # Adjacency list: location -> list of adjacent locations
        self.dir_map = {} # Map: (loc1, loc2) -> direction from loc1 to loc2

        # Build graph and direction map from adjacent facts
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                self.graph[loc1].append(loc2)
                self.dir_map[(loc1, loc2)] = direction

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Ensure all locations mentioned in goals are in our set of locations
        for loc in self.goal_locations.values():
             self.all_locations.add(loc)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        robot_loc = None
        box_locs = {} # box -> location
        clear_facts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and parts[1].startswith("box"):
                box, loc = parts[1], parts[2]
                box_locs[box] = loc
            elif parts[0] == "clear":
                 clear_facts.add(fact)

        # Identify misplaced boxes (only consider boxes that have a goal location)
        misplaced_boxes = {
            box for box in self.goal_locations.keys()
            if box in box_locs and box_locs[box] != self.goal_locations[box]
        }

        # If all goal boxes are in place, heuristic is 0
        if not misplaced_boxes:
            return 0

        # --- Calculate total minimum pushes for boxes ---
        total_box_dist = 0
        for box in misplaced_boxes:
            start_loc = box_locs[box]
            goal_loc = self.goal_locations[box]
            # Box moves on the full graph (ignoring robot/other boxes for this distance)
            dist = bfs_distance(start_loc, goal_loc, self.graph)
            if dist == float('inf'):
                # Box cannot reach its goal - likely a deadlock or requires complex setup
                # Return a large value to indicate a bad state
                return 1000000 # Use a large integer instead of inf

            total_box_dist += dist

        # --- Calculate minimum robot moves to reach a valid pushing position ---
        min_robot_dist_to_push = float('inf')

        # Determine locations the robot can move to (clear locations + robot's current spot)
        robot_clear_locations = {loc for fact in clear_facts for loc in get_parts(fact)[1:]}
        robot_reachable_locations = robot_clear_locations | {robot_loc} # Set of nodes robot can be on or move to

        # Build the robot movement graph
        robot_movement_graph = {}
        for u in robot_reachable_locations:
            robot_movement_graph[u] = []
            # Iterate through neighbors in the full graph
            for v in self.graph.get(u, []):
                # Robot can move to v if v is clear
                if v in robot_clear_locations:
                    robot_movement_graph[u].append(v)

        # If robot_loc is not in the robot_movement_graph (e.g., no clear locations at all),
        # BFS from robot_loc will handle it, but we need to ensure robot_loc is a valid start node.
        # It is added to robot_reachable_locations, so it will be a key in robot_movement_graph.


        for box in misplaced_boxes:
            box_loc = box_locs[box]

            # Find locations adjacent to the box's current location in the full graph
            adjacent_to_box = self.graph.get(box_loc, [])

            for potential_push_pos in adjacent_to_box:
                # Check if potential_push_pos is a location the robot can reach or is currently at
                # It must be a node in the robot_movement_graph
                if potential_push_pos not in robot_reachable_locations:
                     continue

                # Determine the direction from potential_push_pos to box_loc
                push_dir = self.dir_map.get((potential_push_pos, box_loc))
                if push_dir is None:
                    continue # Should not happen if graph is built correctly

                # Find L_next such that adjacent(L_b, L_next, push_dir)
                l_next = None
                # Iterate through neighbors of box_loc in the full graph
                for neighbor_of_box in self.graph.get(box_loc, []):
                     if self.dir_map.get((box_loc, neighbor_of_box)) == push_dir:
                         l_next = neighbor_of_box
                         break

                if l_next is None:
                    continue # Box cannot be pushed in this direction (e.g., wall)

                # Check if L_next is clear (i.e., (clear L_next) is true in the state)
                l_next_is_clear = f'(clear {l_next})' in state

                if l_next_is_clear:
                    # potential_push_pos (R) is a valid position to push this box from
                    # Calculate robot distance from robot_loc to R using the robot movement graph
                    dist = bfs_distance(robot_loc, potential_push_pos, robot_movement_graph)
                    min_robot_dist_to_push = min(min_robot_dist_to_push, dist)


        # If robot cannot reach any valid pushing position for any box
        if min_robot_dist_to_push == float('inf'):
             # This state might be a deadlock for the robot
             return 1000000 # Use a large integer

        # Total heuristic is sum of box distances and robot distance to first push
        # total_box_dist is the number of pushes. min_robot_dist_to_push is the number of robot moves
        # to get into position for the first push.
        # The total estimated actions are robot moves + pushes.
        return total_box_dist + min_robot_dist_to_push
