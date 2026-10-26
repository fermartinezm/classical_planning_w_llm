from collections import deque

# Define dummy Heuristic class if not available (for testing/standalone)
# In the actual planner environment, this should be imported from heuristics.heuristic_base
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError
        def __str__(self):
            return self.__class__.__name__
        def __repr__(self):
            return f"<{self.__class__.__name__}>"


# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes parentheses, splits by space, and filters out empty strings."""
    return [part for part in fact[1:-1].split() if part]

# Helper function to get opposite direction
def opposite_dir(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen in valid PDDL

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Calculates the sum of minimum costs for each box to reach its specific
    goal location independently, considering the robot's movement and
    treating other boxes as fixed obstacles.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static facts and goals.

        Heuristic Initialization:
        - Builds an adjacency graph representing the grid connections based on
          'adjacent' static facts. This graph is used for robot movement and
          determining valid push directions. The graph maps each location to
          a dictionary of neighboring locations and the direction to reach them.
        - Identifies all valid locations present in the adjacency graph.
        - Stores the specific goal location for each box from the task goals.
        """
        # Access task attributes
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency graph: {location: {neighbor_location: direction}}
        self.adj_graph = {}
        self.all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                if l1 not in self.adj_graph:
                    self.adj_graph[l1] = {}
                if l2 not in self.adj_graph:
                    self.adj_graph[l2] = {}
                self.adj_graph[l1][l2] = direction
                self.all_locations.add(l1)
                self.all_locations.add(l2)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal format is typically (at box location) in Sokoban
            if parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location
            # Note: This heuristic assumes each box has a unique goal location.

        # Define a large cost for unreachable goals in the subproblem BFS
        self.UNREACHABLE_COST = 1000000 # Use a large integer to represent effective infinity

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Summary:
        The heuristic estimates the total effort required to move all boxes
        to their goal locations. It does this by summing up the minimum costs
        for each box to reach its individual goal, calculated independently
        using a Breadth-First Search (BFS).

        Assumptions:
        - The grid defined by 'adjacent' facts is connected for relevant areas.
        - Each box has a unique goal location specified by an '(at box location)'
          fact in the task goals.
        - The heuristic is non-admissible and designed for greedy best-first search.
        - Locations not in the adjacency graph are considered walls.
        - Other boxes are considered fixed obstacles during the BFS for a single box.

        Step-By-Step Thinking for Computing Heuristic:
        1.  Parse the current state to find the robot's location and the
            current location of each box.
        2.  Identify the locations occupied by boxes other than the one
            currently being considered in the independent subproblem.
        3.  Initialize the total heuristic cost to 0.
        4.  For each box specified in the task goals:
            a.  Get the box's current location from the state. If the box
                is not found in the state (should not happen in valid states),
                add a large penalty and continue.
            b.  If the box is already at its goal location, its cost for this
                box is 0. Continue to the next box.
            c.  If the box is not at its goal, perform a BFS to find the minimum
                number of actions (robot moves + box pushes) required to move
                *only this box* from its current location to its goal location.
                - The state in the BFS is a tuple: (robot_location, box_location).
                - The initial state is (current_robot_location, current_box_location).
                - The goal condition is reaching any state where the box's location
                  matches its target goal location.
                - Robot moves: The robot can move from its current location `r`
                  to an adjacent location `r_prime` if `r_prime` is connected
                  in the adjacency graph (not a wall), and `r_prime` is not
                  occupied by the target box or any other box. Cost: 1.
                - Box pushes: The target box can be pushed from its current
                  location `b_loc` to an adjacent location `b_prime` if `b_prime`
                  is connected in the adjacency graph (not a wall), `b_prime`
                  is not occupied by any *other* box, AND the robot is currently
                  positioned at the location adjacent to `b_loc` on the opposite
                  side of the push direction (from `b_loc` to `b_prime`). Cost: 1.
                - The BFS explores the state space (robot_location, box_location)
                  until the box reaches its goal location.
            d.  If the BFS finds a path, add the path length (minimum actions)
                to the total heuristic cost.
            e.  If the BFS fails to find a path (queue becomes empty before
                reaching the goal), it means this box cannot reach its goal
                given the fixed positions of other boxes. Add a large penalty
                cost (self.UNREACHABLE_COST) to the total heuristic.
        5.  Return the total heuristic cost.
        """
        state = node.state

        # Parse current state
        robot_at = None
        box_locations = {}
        # clear_locations = set() # Not used in this heuristic logic
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_at = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                box, location = parts[1], parts[2]
                box_locations[box] = location
            # elif parts[0] == 'clear':
            #      clear_locations.add(parts[1])

        total_heuristic = 0

        # Calculate cost for each box independently
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            if current_box_loc is None:
                 # Should not happen in a valid state representation
                 # If a box is missing, it's likely an error or a state
                 # where the box was somehow removed. Treat as very high cost.
                 total_heuristic += self.UNREACHABLE_COST
                 continue

            if current_box_loc == goal_loc:
                # Box is already at its goal
                continue

            # Calculate cost for this box using BFS
            cost_this_box = self._bfs_cost_for_box(robot_at, box, current_box_loc, goal_loc, box_locations)

            total_heuristic += cost_this_box

        return total_heuristic

    def _bfs_cost_for_box(self, initial_robot_loc, target_box, initial_box_loc, goal_box_loc, all_box_locations):
        """
        Performs BFS to find the minimum actions to move target_box to goal_box_loc.

        State: (robot_location, target_box_location)
        Obstacles: Walls (locations not in adj_graph) and other boxes (fixed).
        """
        q = deque([(initial_robot_loc, initial_box_loc)])
        visited = {(initial_robot_loc, initial_box_loc)}
        dist = {(initial_robot_loc, initial_box_loc): 0}

        # Locations of other boxes (fixed obstacles for robot and target_box)
        other_boxes_locations = {loc for box, loc in all_box_locations.items() if box != target_box}

        while q:
            current_robot_loc, current_box_loc = q.popleft()
            current_dist = dist[(current_robot_loc, current_box_loc)]

            # Goal check for this box
            if current_box_loc == goal_box_loc:
                return current_dist

            # --- Generate neighbors ---

            # 1. Robot Moves
            # Robot can move to any adjacent location that is not a wall
            # and is not occupied by the target box or any other box.
            if current_robot_loc in self.adj_graph:
                for next_robot_loc in self.adj_graph[current_robot_loc]:
                    # Check if next_robot_loc is a valid move target
                    # A location is a "wall" if it's not a key in the adj_graph,
                    # meaning it has no defined adjacencies.
                    is_wall = next_robot_loc not in self.adj_graph
                    is_occupied_by_target_box = (next_robot_loc == current_box_loc)
                    is_occupied_by_other_box = (next_robot_loc in other_boxes_locations)

                    if not is_wall and not is_occupied_by_target_box and not is_occupied_by_other_box:
                        next_state = (next_robot_loc, current_box_loc)
                        if next_state not in visited:
                            visited.add(next_state)
                            dist[next_state] = current_dist + 1
                            q.append(next_state)

            # 2. Box Pushes (for target_box)
            # The target box can be pushed from current_box_loc to next_box_loc
            # if next_box_loc is adjacent, clear (not wall, not other box),
            # and the robot is in the correct position behind the box.
            if current_box_loc in self.adj_graph:
                 for next_box_loc, push_direction in self.adj_graph[current_box_loc].items():
                    # Find the required robot location for this push
                    # The robot must be adjacent to current_box_loc in the opposite direction of the push.
                    required_robot_loc = None
                    opposite_push_dir = opposite_dir(push_direction)

                    # Find the location adjacent to current_box_loc in the opposite direction
                    if current_box_loc in self.adj_graph: # Redundant check, but safe
                        for neighbor_of_box, dir_from_box in self.adj_graph[current_box_loc].items():
                             if dir_from_box == opposite_push_dir:
                                 required_robot_loc = neighbor_of_box
                                 break # Found the location behind the box

                    # Check if robot is in the required position and the push target is valid
                    if required_robot_loc is not None and current_robot_loc == required_robot_loc:
                        # Check if next_box_loc is a valid push target location
                        is_wall = next_box_loc not in self.adj_graph
                        is_occupied_by_other_box = (next_box_loc in other_boxes_locations)

                        if not is_wall and not is_occupied_by_other_box:
                            # A push action moves the box and the robot swaps places with the box
                            next_state = (current_box_loc, next_box_loc)
                            if next_state not in visited:
                                visited.add(next_state)
                                dist[next_state] = current_dist + 1
                                q.append(next_state)

        # If BFS completes without reaching the goal for this box
        return self.UNREACHABLE_COST
