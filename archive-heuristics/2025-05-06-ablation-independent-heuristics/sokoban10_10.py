# Need to import Heuristic from the base module
from heuristics.heuristic_base import Heuristic
from collections import deque
import re
# import logging # Optional: for debugging

# Configure logging (optional)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class sokobanHeuristic(Heuristic):
    """
    Summary:
        Domain-dependent heuristic for the Sokoban domain.
        Estimates the cost to reach the goal state by summing, for each box not at its goal:
        1. The shortest path distance for the box from its current location to its goal location on the grid graph.
        2. The minimum shortest path distance for the robot to reach a location adjacent to the box from which it can push the box one step strictly closer to its goal, considering clear locations.

    Assumptions:
        - The problem is defined on a grid-like structure where connectivity is given by 'adjacent' facts.
        - Goal states consist of 'at' predicates for boxes at specific locations.
        - Locations are named consistently (e.g., loc_X_Y).
        - The grid is assumed to be connected for robot and box movement paths unless explicitly blocked by state.
        - The heuristic implicitly handles some simple dead ends (e.g., box in a corner where no push towards goal is possible) by assigning a high cost. It does not detect complex deadlocks involving multiple boxes.

    Heuristic Initialization:
        1. Parses 'adjacent' facts from the static information to build an undirected adjacency graph (`adj_graph`) for distance calculations and a directional graph (`dir_adj_graph`) mapping location+direction to adjacent location.
        2. Collects all unique location names from static, initial, and goal facts.
        3. Computes all-pairs shortest path distances between all locations on the undirected grid graph using BFS. Stores results in `self.all_pairs_dist`.
        4. Identifies the goal location for each box from the task's goal facts and stores them in `self.box_goals`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state using `self.task.goal_reached(state)`. If yes, return 0.
        2. Parse the current state (`node.state`) to find the robot's location (`robot_l`), the location of each box (`box_locations`), and determine the set of currently clear locations (`current_clear_locs`) by subtracting occupied locations (robot + boxes) from all known locations.
        3. Initialize the total heuristic value `h` to 0.
        4. Iterate through each box (`box_name`) and its assigned goal location (`goal_loc`) stored in `self.box_goals`.
            a. Get the box's current location (`box_l`) from `box_locations`.
            b. If the box is not at its goal location (`box_l is not None` and `box_l != goal_loc`):
                # Handle case where box might be missing from state (invalid state?)
                if box_l is None:
                     # logger.error(f"Box {box_name} location not found in state.")
                     return float('inf')

                # Check if box_l or goal_loc are known locations
                if box_l not in self.locations or goal_loc not in self.locations:
                     # logger.error(f"Unknown location for box {box_name} ({box_l}) or goal ({goal_loc}).")
                     return float('inf')

                # Box-Goal Distance
                # Check if distance is computable (locations are connected)
                if goal_loc not in self.all_pairs_dist.get(box_l, {}):
                     # logger.warning(f"Box location {box_l} not connected to goal {goal_loc}.")
                     return float('inf') # Box cannot reach goal

                box_dist = self.all_pairs_dist[box_l][goal_loc]
                h += box_dist

                # Robot-Box Access Cost
                min_robot_dist = float('inf')
                directions = ['up', 'down', 'left', 'right'] # Assuming these are the only directions

                for push_dir in directions:
                    # Find potential next box location if pushed in this direction
                    next_l_candidate = self.dir_adj_graph.get(box_l, {}).get(push_dir)

                    # Check if next location exists and is clear
                    if next_l_candidate and next_l_candidate in current_clear_locs:
                        # Check if this push moves the box strictly closer to the goal
                        # Ensure next_l_candidate and goal_loc are in the distance map
                        if next_l_candidate in self.all_pairs_dist and goal_loc in self.all_pairs_dist.get(next_l_candidate, {}):
                             dist_next_to_goal = self.all_pairs_dist[next_l_candidate][goal_loc]

                             # Use the pre-calculated box_dist for comparison
                             if dist_next_to_goal < box_dist:
                                 # Find the required robot location
                                 # Robot must be at r_loc_candidate such that box_l is push_dir from r_loc_candidate
                                 # This means r_loc_candidate is reverse(push_dir) from box_l
                                 rev_push_dir = self._reverse_direction_map.get(push_dir)
                                 r_loc_candidate = self.dir_adj_graph.get(box_l, {}).get(rev_push_dir) if rev_push_dir else None

                                 # Check if robot candidate location exists and is clear
                                 if r_loc_candidate and r_loc_candidate in current_clear_locs:
                                     # Calculate robot's travel distance to r_loc_candidate
                                     # Ensure robot_l and r_loc_candidate are in the distance map
                                     if robot_l in self.all_pairs_dist and r_loc_candidate in self.all_pairs_dist.get(robot_l, {}):
                                         robot_dist = self.all_pairs_dist[robot_l][r_loc_candidate]
                                         min_robot_dist = min(min_robot_dist, robot_dist)
                                     # else: robot_l not connected to r_loc_candidate (indicates unsolvable or very hard)
                                     # logger.debug(f"Robot {robot_l} not connected to push position {r_loc_candidate} for box {box_name}.")
                                     # min_robot_dist remains inf, handled below
                                 # else: r_loc_candidate does not exist or is not clear
                                 # logger.debug(f"Robot push position {r_loc_candidate} for box {box_name} (dir {push_dir}) is not valid or clear.")
                             # else: push does not move box strictly closer to goal
                             # logger.debug(f"Pushing box {box_name} in {push_dir} to {next_l_candidate} does not get closer to goal {goal_loc}.")
                         # else: next_l_candidate not connected to goal_loc (should not happen if box_l is connected)
                         # logger.warning(f"Next box location {next_l_candidate} not connected to goal {goal_loc}.")
                         # min_robot_dist remains inf, handled below
                    # else: next_l_candidate does not exist or is not clear
                    # logger.debug(f"Next box location {next_l_candidate} for box {box_name} (dir {push_dir}) is not valid or clear.")


                # Add the minimum robot distance for this box
                # If no valid push position towards the goal was found, min_robot_dist remains infinity
                h += min_robot_dist

        # 5. Return total heuristic value
        # logger.debug(f"Heuristic value: {h}")
        return h


    def _parse_fact(self, fact_string):
        """Helper to parse PDDL fact strings."""
        match = re.match(r'\((\S+)(?:\s+(.*))?\)', fact_string)
        if not match:
            # logger.warning(f"Failed to parse fact string: {fact_string}")
            return None, []
        predicate = match.group(1)
        args_str = match.group(2)
        args = args_str.split() if args_str else []
        return predicate, args

    def _bfs(self, graph, start_node):
        """Helper to compute shortest paths from a start node using BFS."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # logger.warning(f"BFS started from unknown location: {start_node}")
             return distances # Cannot compute distances from unknown node

        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            # Check if current_node is in graph keys before iterating
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

