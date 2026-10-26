from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def bfs_distances(start_node, graph):
    """
    Computes shortest path distances from start_node to all reachable nodes
    in an unweighted graph using BFS.

    Args:
        start_node: The starting node.
        graph: Adjacency list representation {node: set of neighbors}.

    Returns:
        A dictionary mapping reachable nodes to their distance from start_node.
    """
    distances = {start_node: 0}
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        dist = distances[current_node]

        for neighbor in graph.get(current_node, set()):
            if neighbor not in distances:
                distances[neighbor] = dist + 1
                queue.append(neighbor)
    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing,
    for each box not at its goal, the minimum number of pushes required
    to move the box to its goal, plus the minimum number of robot moves
    required to get into position to make the first push for that box
    towards its goal.

    # Assumptions
    - The grid is defined by the 'adjacent' predicates.
    - Each box has a unique goal location specified in the task goals.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1 (after the robot is in position).
    - The heuristic calculates distances on the static grid graph, ignoring
      dynamic obstacles (other boxes, robot).
    - The heuristic assumes that if a path exists between two locations,
      the required pushing positions also exist (ignores complex deadlocks
      like boxes in corners unless the graph structure explicitly prevents
      the required adjacency).

    # Heuristic Initialization
    - Builds a graph representation of the grid from 'adjacent' facts,
      including simple adjacency for BFS and directional mappings.
    - Computes all-pairs shortest paths on the grid graph using BFS.
    - Extracts goal locations for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box.
    2. Initialize the total heuristic cost to 0.
    3. For each box that is not currently at its assigned goal location:
        a. Find the box's current location (`box_loc`) and its goal location (`goal_loc`).
        b. Calculate the shortest path distance from `box_loc` to `goal_loc` on the grid graph. This distance represents the minimum number of 'push' actions required for this box if the path were clear and the robot always ready. Add this distance to the total cost.
        c. If the box cannot reach its goal location (distance is infinity), the state is likely unsolvable; return infinity.
        d. If the box is not yet at its goal (`dist_box_goal > 0`):
            i. Identify a location `next_loc` adjacent to `box_loc` that is on a shortest path from `box_loc` to `goal_loc`. (There might be multiple such locations; pick any one).
            ii. Determine the direction (`push_dir`) from `box_loc` to `next_loc` using the precomputed direction map.
            iii. Determine the required robot position (`push_loc`) to push the box from `box_loc` to `next_loc`. Based on the PDDL, this is the location `p` such that `(adjacent p box_loc push_dir)` is true. Use the precomputed reverse direction map `rev_dir_map[(box_loc, push_dir)]` to find `push_loc`.
            iv. If the required `push_loc` does not exist (e.g., box is in a corner), the box cannot be pushed in that direction; return infinity as a potential deadlock indicator.
            v. Calculate the shortest path distance from the robot's current location (`robot_loc`) to the required pushing position (`push_loc`). This distance represents the minimum number of 'move' actions the robot needs to make to get ready for the first push of this box towards its goal. Add this distance to the total cost.
            vi. If the robot cannot reach the required `push_loc` (distance is infinity), the state is unsolvable; return infinity.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph, computing shortest paths,
        and extracting goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        self.adj_graph = defaultdict(set)
        self.dir_map = {} # (loc1, loc2) -> dir if (adjacent loc1 loc2 dir)
        self.rev_dir_map = {} # (loc2, dir) -> loc1 if (adjacent loc1 loc2 dir)
        self.all_locations = set()

        # Build graph and direction maps from adjacent facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, dir = parts[1], parts[2], parts[3]
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
                self.adj_graph[loc1].add(loc2)
                self.adj_graph[loc2].add(loc1) # Adjacency is symmetric for simple movement distance
                self.dir_map[(loc1, loc2)] = dir
                # rev_dir_map[(loc2, dir)] = loc1 means loc1 is the location you come from
                # to reach loc2 by moving in direction dir.
                # In a push (rloc bloc floc dir), (adjacent rloc bloc dir) must hold.
                # So, rloc is the location you come from to reach bloc in direction dir.
                # Thus, rloc = rev_dir_map[(bloc, dir)].
                self.rev_dir_map[(loc2, dir)] = loc1

        # Compute all-pairs shortest paths
        self.shortest_paths = {}
        for start_loc in self.all_locations:
            self.shortest_paths[start_loc] = bfs_distances(start_loc, self.adj_graph)

        # Extract goal locations for boxes
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get current robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
                break
        if robot_loc is None:
             # Should not happen in valid Sokoban states
             return float('inf')

        # Get current box locations
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                box_locations[box] = location

        total_cost = 0

        # Consider each box that is not at its goal
        for box, goal_loc in self.goal_locations.items():
            box_loc = box_locations.get(box)

            # If box is not in the state or not a box we care about (shouldn't happen)
            if box_loc is None:
                 return float('inf')

            # If box is already at goal, no cost for this box
            if box_loc == goal_loc:
                continue

            # Calculate min pushes for the box
            dist_box_goal = self.shortest_paths.get(box_loc, {}).get(goal_loc, float('inf'))

            # If box cannot reach goal, state is unsolvable
            if dist_box_goal == float('inf'):
                return float('inf')

            total_cost += dist_box_goal

            # If box needs to move (dist_box_goal > 0)
            if dist_box_goal > 0:
                # Find a next_loc on a shortest path towards the goal
                next_loc = None
                # Iterate through neighbors of box_loc
                for neighbor_loc in self.adj_graph.get(box_loc, set()):
                     # Check if neighbor is reachable from box_loc and is on a shortest path to goal
                     if neighbor_loc in self.shortest_paths and goal_loc in self.shortest_paths[neighbor_loc] and self.shortest_paths[neighbor_loc][goal_loc] == dist_box_goal - 1:
                         next_loc = neighbor_loc
                         break # Found one, pick it

                # If no next_loc found (implies box_loc is not adjacent to any location closer to goal)
                # This should not happen if dist_box_goal > 0 and a path exists in the graph,
                # but acts as a safeguard or potential deadlock indicator.
                if next_loc is None:
                     return float('inf')

                # Find the direction of the push from box_loc to next_loc
                push_dir = self.dir_map.get((box_loc, next_loc))
                if push_dir is None:
                     # Should not happen if next_loc is a valid neighbor found via adj_graph and dir_map is built correctly
                     return float('inf')

                # Find the required robot position (push_loc) to push box_loc in push_dir
                # push_loc is the location such that (adjacent push_loc box_loc push_dir) is true
                # This is the location you come from to reach box_loc in direction push_dir
                push_loc = self.rev_dir_map.get((box_loc, push_dir))

                if push_loc is None:
                     # This indicates the box is in a position where it cannot be pushed
                     # in the required direction (e.g., against a wall/corner not in graph)
                     # This is a form of deadlock.
                     return float('inf')

                # Calculate min robot moves to reach the push_loc
                dist_robot_push = self.shortest_paths.get(robot_loc, {}).get(push_loc, float('inf'))

                if dist_robot_push == float('inf'):
                    return float('inf') # Robot cannot reach the push position

                total_cost += dist_robot_push

        return total_cost
