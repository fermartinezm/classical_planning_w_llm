from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function
def bfs_from_single_source(graph, all_nodes, start_node, end_node=None):
    """
    Computes shortest path distances from start_node to all reachable nodes
    in the graph using BFS.

    Args:
        graph: Adjacency list where graph[u] is a list of (v, direction) tuples.
        all_nodes: A set of all possible nodes in the graph.
        start_node: The node to start BFS from.
        end_node: Optional. If specified, returns only the distance to this node.
                  If None, returns a dictionary of distances to all reachable nodes.

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from start_node
        if end_node is None, or the distance (float('inf') if unreachable) to end_node
        if end_node is specified.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
        # Start node is not a known location, cannot start BFS.
        # If end_node is specified, distance is inf. If returning dict, return empty.
        return float('inf') if end_node is not None else {}

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        u = queue.popleft()

        # If we found the end_node and it was specified, we can stop early
        if end_node is not None and u == end_node:
            break

        # Check if u has neighbors defined in the graph
        if u in graph:
            for v, _ in graph[u]: # Ignore direction for distance calculation
                # Ensure neighbor v is a known node
                if v in all_nodes and distances[v] == float('inf'):
                    distances[v] = distances[u] + 1
                    queue.append(v)

    # Return distances based on whether end_node was specified
    if end_node is not None:
        return distances.get(end_node, float('inf'))
    else:
        # Return distances only for reachable nodes
        return {node: dist for node, dist in distances.items() if dist != float('inf')}


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing,
    for each box not at its goal, the minimum number of pushes required
    to move the box to its goal plus the minimum number of robot moves
    required to reach the position from which the first push towards the
    goal can be made along a shortest path.

    # Assumptions
    - The grid structure is defined by the 'adjacent' facts.
    - The heuristic assumes that each box can be moved independently towards
      its goal along a shortest path, ignoring potential blockages by other
      boxes or the robot, and ignoring the need to clear target locations.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - The required robot position to push a box from location L1 to L2
      is the location L0 adjacent to L1 in the direction opposite L1->L2.
    - If a box's goal is unreachable, or the required first push position
      is unreachable by the robot, the state is considered unsolvable
      within this heuristic's model, and returns infinity.

    # Heuristic Initialization
    - Parses 'adjacent' facts to build an undirected graph representing
      the grid connectivity. Stores adjacency information including directions.
    - Creates a mapping from directions to their opposites.
    - Extracts goal locations for each box from the task's goal conditions.
    - Collects all unique location names mentioned in adjacent facts, initial
      state, and goals to ensure all potential nodes are included in graph
      operations (like BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. Check if all boxes are already at their goal locations. If yes, the state is a goal state, return 0.
    4. Initialize the total heuristic cost to 0.
    5. Compute shortest path distances from the robot's current location to all reachable locations using BFS.
    6. For each box specified in the goal:
       a. Determine its current location and its goal location.
       b. If the box is already at its goal location, add 0 to the total cost for this box and continue to the next box.
       c. If the box is not at its goal:
          i. Compute the shortest path distance from the box's current location to its goal location on the grid graph using BFS (`box_dist`). If the goal is unreachable, return infinity.
          ii. Find a location `next_loc_b` adjacent to the box's current location that lies on a shortest path towards the goal (i.e., `distance(next_loc_b, goal) == distance(current_loc, goal) - 1`).
          iii. Determine the required robot position (`push_pos`) needed to push the box from its current location towards `next_loc_b`. This `push_pos` is the location adjacent to the box's current location in the direction opposite to the direction from the box's current location to `next_loc_b`. If no such `push_pos` exists (e.g., no space behind the box), return infinity.
          iv. Get the shortest path distance from the robot's current location to this `push_pos` from the precomputed distances (`robot_dist`). If `push_pos` is unreachable by the robot, return infinity.
          v. The estimated cost for this box is `box_dist + robot_dist`.
          vi. Add the estimated cost for this box to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure and goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph (adjacency list) from adjacent facts
        # graph[loc] = [(neighbor, direction), ...]
        self.adj_list = {}
        self.all_locations = set()

        # Add locations from adjacent facts
        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.adj_list.setdefault(loc1, []).append((loc2, direction))
                self.adj_list.setdefault(loc2, []).append((loc1, self._get_opposite_direction(direction))) # Add reverse edge

                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Add locations from initial state and goals that might not be in adjacent facts
        for fact in task.initial_state:
             if match(fact, "at-robot", "*"):
                 self.all_locations.add(get_parts(fact)[1])
             elif match(fact, "at", "*", "*"):
                 self.all_locations.add(get_parts(fact)[2])
        for goal in task.goals:
             if match(goal, "at", "*", "*"):
                 self.all_locations.add(get_parts(goal)[2])

        # Ensure all_locations are keys in adj_list, even if they have no neighbors initially
        # This is important for BFS when the start node might be isolated.
        for loc in self.all_locations:
             self.adj_list.setdefault(loc, [])


        # Map directions to their opposites
        self.opposite_direction = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left',
        }

        # Store goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if match(goal, "at", "*", "*"):
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

    def _get_opposite_direction(self, direction):
         """Helper to get opposite direction."""
         return self.opposite_direction.get(direction, None)

    def _get_required_push_pos(self, current_box_loc, goal_box_loc, dist_to_goal):
        """
        Finds the required robot location to push the box from current_box_loc
        towards goal_box_loc along a shortest path.

        Args:
            current_box_loc: The current location of the box.
            goal_box_loc: The goal location of the box.
            dist_to_goal: Dictionary of distances from all nodes *to* goal_box_loc.

        Returns:
            The required robot location (string) or None if no valid push position
            towards the goal along a shortest path is found (e.g., goal unreachable,
            or box is at goal, or no space behind the box in the required direction).
        """
        if current_box_loc == goal_box_loc:
            return None # Box is already at goal

        # If box location is unreachable from goal (or vice versa), return None
        if current_box_loc not in dist_to_goal or dist_to_goal[current_box_loc] == float('inf'):
             return None # Problem likely unsolvable or graph issue

        box_dist = dist_to_goal[current_box_loc]

        # If box_dist is 0, it should have been caught by the first check, but double-check
        if box_dist == 0:
             return None

        # Find a neighbor that is one step closer to the goal (on a shortest path)
        next_loc_b = None
        push_direction = None # Direction from current_box_loc to next_loc_b

        # Iterate through neighbors of current_box_loc
        if current_box_loc in self.adj_list:
            for neighbor, direction in self.adj_list[current_box_loc]:
                if neighbor in dist_to_goal and dist_to_goal[neighbor] == box_dist - 1:
                    next_loc_b = neighbor
                    push_direction = direction
                    break # Found one shortest path step, that's enough

        if next_loc_b is None:
             # This can happen if current_box_loc is not reachable from goal_box_loc
             # or if the graph is weirdly disconnected.
             return None # Cannot find a step towards the goal

        # The required robot position is adjacent to current_box_loc
        # in the opposite direction of the push
        required_robot_direction = self._get_opposite_direction(push_direction)

        # Find the neighbor of current_box_loc in the required_robot_direction
        push_pos = None
        if current_box_loc in self.adj_list:
            for neighbor, direction in self.adj_list[current_box_loc]:
                if direction == required_robot_direction:
                    push_pos = neighbor
                    break

        return push_pos


    def __call__(self, node):
        """
        Estimate the required number of actions to reach a goal state.
        """
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        if robot_loc is None:
             # Robot location not found, this shouldn't happen in a valid state
             return float('inf') # Indicate invalid state


        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                box, loc = get_parts(fact)[1], get_parts(fact)[2]
                box_locations[box] = loc

        # Check if goal is reached (heuristic is 0)
        # A state is a goal state if all goal facts are in the state.
        # For Sokoban, goals are typically (at box goal_loc).
        # We check if all boxes specified in the goal are at their goal locations.
        all_boxes_at_goal = True
        for box, goal_loc in self.box_goals.items():
             # Check if the box exists in the current state and is at the goal location
             if box not in box_locations or box_locations[box] != goal_loc:
                  all_boxes_at_goal = False
                  break

        if all_boxes_at_goal:
             # The PDDL goal only specifies box locations.
             # If all required boxes are at their goals, the state is a goal state.
             return 0


        total_heuristic = 0

        # Compute robot distances from robot_loc once
        robot_distances_from_current_pos = bfs_from_single_source(self.adj_list, self.all_locations, robot_loc, end_node=None)

        if not robot_distances_from_current_pos: # Robot is isolated or not found
             return float('inf') # Robot cannot move anywhere


        # Compute heuristic for each box not at its goal
        for box, goal_loc in self.box_goals.items():
            # If a box required by the goal is not present in the state, something is wrong.
            if box not in box_locations:
                 return float('inf') # Indicate invalid state

            current_box_loc = box_locations[box]

            if current_box_loc == goal_loc:
                continue # This box is already at its goal

            # Compute distances from all nodes *to* the box's goal
            box_dist_to_goal = bfs_from_single_source(self.adj_list, self.all_locations, goal_loc, end_node=None)

            if current_box_loc not in box_dist_to_goal or box_dist_to_goal[current_box_loc] == float('inf'):
                 # Box goal is unreachable from its current location
                 return float('inf') # Problem is unsolvable from this state

            box_dist = box_dist_to_goal[current_box_loc]

            # Find the required robot position for the first push towards the goal
            push_pos = self._get_required_push_pos(current_box_loc, goal_loc, box_dist_to_goal)

            if push_pos is None:
                 # This can happen if the box is at the goal (handled above)
                 # or if the box location is unreachable from the goal (handled by box_dist check)
                 # or if no neighbor is on a shortest path (shouldn't happen if box_dist > 0 and reachable)
                 # or if the required push_pos doesn't exist (e.g., pushing from an edge where there's no space behind)
                 # If push_pos is None, it means we can't make the first push towards the goal along a shortest path
                 # using the simple model. This state might be a dead end for this heuristic.
                 return float('inf') # Indicate unsolvable/dead end state


            # Get robot distance to the required push position from the precomputed distances
            robot_dist = robot_distances_from_current_pos.get(push_pos, float('inf'))

            if robot_dist == float('inf'):
                # Required push position is unreachable by the robot
                return float('inf') # Problem is unsolvable from this state

            # Heuristic contribution for this box:
            # box_dist pushes + robot_dist moves to get into position for the first push.
            # This is a simplification; subsequent pushes also require robot movement,
            # but this captures the main costs.
            total_heuristic += box_dist + robot_dist

        return total_heuristic
