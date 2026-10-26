# -*- coding: utf-8 -*-

from heuristics.heuristic_base import Heuristic
import collections # Used for BFS queue

def get_parts(fact):
    """Removes surrounding parentheses and splits by space."""
    return fact[1:-1].split()

def bfs_distance(start, end, adj_list):
    """
    Calculates the shortest path distance between two locations using BFS
    on the full grid graph (ignoring dynamic obstacles).

    Args:
        start (str): The starting location.
        end (str): The target location.
        adj_list (dict): Adjacency list representation of the graph.

    Returns:
        int or float('inf'): The shortest distance, or infinity if no path exists.
    """
    if start == end:
        return 0

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        for neighbor in adj_list.get(current_loc, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # Path not found

def bfs_distance_available(start, end, adj_list, available_locations):
    """
    Calculates the shortest path distance between two locations using BFS,
    only traversing through locations in the available_locations set.

    Args:
        start (str): The starting location.
        end (str): The target location.
        adj_list (dict): Adjacency list representation of the graph.
        available_locations (set): Set of locations the path is allowed to traverse.

    Returns:
        int or float('inf'): The shortest distance, or infinity if no path exists.
    """
    if start not in available_locations:
        return float('inf') # Cannot start in an unavailable location

    if start == end:
        return 0

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        for neighbor in adj_list.get(current_loc, []):
            if neighbor in available_locations and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # Path not found


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated costs for each box that is not yet at its goal location.
        The estimated cost for a single box is the sum of:
        1. The minimum number of push actions required to move the box from its
           current location to its goal location on the grid graph (ignoring
           other boxes as obstacles for the box's path). This is the shortest
           path distance for the box.
        2. The minimum number of robot movement actions required to move the
           robot from its current location to a position adjacent to the box
           from which the first push towards the goal can be made. The robot's
           path considers other boxes and non-clear locations as obstacles.

    Assumptions:
        - The grid structure is defined by the 'adjacent' facts.
        - Locations are named consistently (e.g., loc_R_C).
        - The shortest path for a box on the grid graph (ignoring other boxes)
          is a reasonable estimate of the minimum pushes required.
        - The primary costs are moving the robot to a push position and performing
          the push action itself. Costs for clearing paths (moving other boxes)
          or subsequent robot movements after the first push are partially or
          fully ignored, making the heuristic non-admissible but potentially
          effective for greedy search.
        - Dead-end states (where a box is pushed into a corner it cannot leave)
          are not explicitly detected, but may result in infinite heuristic values
          if the goal becomes unreachable.

    Heuristic Initialization:
        - Parses 'adjacent' facts to build an adjacency list representation of
          the grid graph (`self.adj_list`) for general movement (robot or box
          on empty grid).
        - Parses 'adjacent' facts to build a directional adjacency map
          (`self.location_neighbors_by_dir`) to quickly find neighbors in a
          specific direction and the location required for the robot to push
          from a given direction.
        - Stores a mapping of directions to their opposites (`self.opposite_direction`).
        - Parses goal facts to map each box object to its target goal location
          (`self.goal_locations`).

    Step-By-Step Thinking for Computing Heuristic:
        1.  Get the current state, which is a frozenset of facts.
        2.  Extract the current robot location, the current location of each box,
            and the set of clear locations from the state facts.
        3.  Initialize the total heuristic value to 0.
        4.  Identify locations occupied by any box (`all_box_locations`).
        5.  Determine locations available for robot movement: these are locations
            that are currently clear, plus the robot's current location, excluding
            any location occupied by a box.
        6.  For each box specified in the goal:
            a.  If the box is already at its goal location in the current state,
                continue to the next box.
            b.  Calculate the shortest path distance for the box from its current
                location to its goal location on the grid graph (`box_dist`).
                This is done using BFS on the precomputed adjacency list, ignoring
                other boxes as obstacles for the box's own path. If the goal is
                unreachable for the box, return infinity for the total heuristic.
            c.  Initialize `min_robot_dist_to_push` to infinity. This will store
                the minimum distance for the robot to reach a valid push position
                for the *first* step towards the goal.
            d.  Find potential next locations for the box: Iterate through the
                neighbors of the box's current location. A neighbor is a potential
                next location if moving the box there would reduce its shortest
                path distance to the goal (`dist_neighbor_to_goal < box_dist`).
            e.  For each potential next location (`next_box_loc`):
                i.  Determine the direction (`push_dir`) from the box's current
                    location to `next_box_loc`.
                ii. Determine the required robot location (`required_robot_loc`)
                    needed to push the box in that direction. This location is
                    adjacent to the box's current location in the direction
                    opposite to `push_dir`.
                iii. Check if `required_robot_loc` is available for the robot
                     to move to (i.e., it's in the set of `robot_available_locations`).
                     If it is not available (e.g., occupied by another box or not clear),
                     this specific push position is not currently viable, so skip it.
                iv. If `required_robot_loc` is available, calculate the shortest
                    path distance for the robot from its current location to
                    `required_robot_loc`. This BFS only traverses locations in
                    `robot_available_locations`.
                v.  Update `min_robot_dist_to_push` with the minimum distance found
                    so far among all viable push positions for this box.
            f.  If, after checking all potential next locations, `min_robot_dist_to_push`
                is still infinity, it means the robot cannot reach any position
                to push this box towards the goal (e.g., the box is surrounded,
                or all required robot positions are blocked or not clear). In this case,
                return infinity for the total heuristic, indicating a likely
                unsolvable state.
            g.  Add the box's distance (`box_dist`) and the minimum robot distance
                to a push position (`min_robot_dist_to_push`) to the total heuristic.
        7.  Return the total heuristic value.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.adj_list = {}
        self.locations = set()
        self.location_neighbors_by_dir = {}

        # Parse adjacent facts to build the graph and directional neighbors
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.locations.add(l1)
                self.locations.add(l2)

                # Build general adjacency list (bidirectional for movement)
                if l1 not in self.adj_list:
                    self.adj_list[l1] = []
                if l2 not in self.adj_list:
                    self.adj_list[l2] = []
                self.adj_list[l1].append(l2)
                self.adj_list[l2].append(l1)

                # Store directional adjacency for push logic
                if l1 not in self.location_neighbors_by_dir:
                     self.location_neighbors_by_dir[l1] = {}
                self.location_neighbors_by_dir[l1][direction] = l2

        # Remove duplicates from adj_list (due to bidirectional adding)
        for loc in self.adj_list:
            self.adj_list[loc] = list(set(self.adj_list[loc]))

        # Mapping from direction string to its opposite
        self.opposite_direction = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }

        # Parse goal facts to map boxes to goal locations
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goal facts are always (at box_name goal_loc)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box_name, goal_loc = parts[1], parts[2]
                self.goal_locations[box_name] = goal_loc

    def __call__(self, node):
        state = node.state

        current_robot_loc = None
        current_box_locations = {}
        current_clear_locations = set()

        # Extract dynamic information from the state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                current_robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                current_box_locations[parts[1]] = parts[2]
            elif parts[0] == 'clear' and len(parts) == 2:
                current_clear_locations.add(parts[1])

        total_heuristic = 0

        # Locations occupied by any box
        all_box_locations = set(current_box_locations.values())

        # Locations available for the robot path: clear locations + robot's current location,
        # excluding any location occupied by a box.
        # The robot can move to any clear location. It is currently at current_robot_loc.
        # It cannot move onto a box location.
        robot_available_locations = (current_clear_locations | {current_robot_loc}) - all_box_locations


        # Iterate through each box that has a goal location
        for box_name, goal_loc in self.goal_locations.items():
            current_box_loc = current_box_locations.get(box_name)

            # If box is not found in the state, something is wrong, or it's not relevant
            # for this heuristic (assuming all goal boxes are present in all states)
            if current_box_loc is None:
                 # This case should ideally not happen in a valid state representation
                 # where goal objects persist. Let's assume it doesn't happen.
                 continue

            # If the box is already at its goal, it contributes 0 to the heuristic
            if current_box_loc == goal_loc:
                continue

            # --- Calculate estimated cost for this box ---

            # 1. Minimum pushes required for the box (shortest path on empty grid)
            # This ignores other boxes as obstacles for the box's own path.
            box_dist = bfs_distance(current_box_loc, goal_loc, self.adj_list)

            # If the box cannot reach the goal even on an empty grid, it's unsolvable
            if box_dist == float('inf'):
                return float('inf')

            # 2. Minimum robot moves to get into a push position for the first step

            min_robot_dist_to_push = float('inf')

            # Find potential next locations for the box that are strictly closer to the goal
            potential_box_next_locs = []
            # We already calculated box_dist = bfs_distance(current_box_loc, goal_loc, self.adj_list)
            # Use this pre-calculated distance

            # Iterate through neighbors of the current box location
            for neighbor_loc in self.adj_list.get(current_box_loc, []):
                # Check if moving box to neighbor_loc gets it strictly closer to goal
                # Need distance from neighbor_loc to goal_loc on the empty grid graph
                dist_neighbor_to_goal = bfs_distance(neighbor_loc, goal_loc, self.adj_list)

                # Note: dist_neighbor_to_goal could be inf if neighbor_loc cannot reach goal,
                # but if current_box_loc can reach goal, at least one neighbor on a shortest
                # path must also be able to reach the goal with distance box_dist - 1.
                # The check `dist_neighbor_to_goal < box_dist` handles this correctly.
                if dist_neighbor_to_goal < box_dist:
                    potential_box_next_locs.append(neighbor_loc)

            # For each potential next location for the box, find the required robot location
            for next_box_loc in potential_box_next_locs:
                # Find the direction from current_box_loc to next_box_loc
                push_dir = None
                # Iterate through directional neighbors of current_box_loc
                for direction, loc in self.location_neighbors_by_dir.get(current_box_loc, {}).items():
                    if loc == next_box_loc:
                        push_dir = direction
                        break

                # If we found the direction
                if push_dir:
                    # Find the required robot location to push from current_box_loc to next_box_loc
                    # This is the location adjacent to current_box_loc in the opposite direction
                    opposite_dir = self.opposite_direction.get(push_dir)
                    required_robot_loc = self.location_neighbors_by_dir.get(current_box_loc, {}).get(opposite_dir)

                    # If such a location exists in the grid
                    if required_robot_loc:
                        # Check if the required robot location is available for the robot to move to.
                        # It must be clear and not occupied by another box.
                        # This is equivalent to checking if required_robot_loc is in robot_available_locations.
                        if required_robot_loc in robot_available_locations:
                             # Calculate robot distance from current_robot_loc to required_robot_loc
                             # The robot path must only traverse available locations.
                             robot_dist = bfs_distance_available(current_robot_loc, required_robot_loc, self.adj_list, available_locations=robot_available_locations)

                             # Update the minimum robot distance found so far
                             min_robot_dist_to_push = min(min_robot_dist_to_push, robot_dist)

            # If no viable push position was found for the first step towards the goal
            # (e.g., box is surrounded, or all required robot positions are blocked or not clear)
            if min_robot_dist_to_push == float('inf'):
                # This box cannot be moved towards the goal currently without unblocking.
                # Return infinity to indicate a potentially unsolvable state or one requiring complex unblocking.
                return float('inf')

            # Add the estimated cost for this box to the total heuristic
            # Cost = (minimum pushes) + (minimum robot moves to get ready for the first push)
            total_heuristic += box_dist + min_robot_dist_to_push

        # Return the sum of estimated costs for all boxes not at their goals
        return total_heuristic
