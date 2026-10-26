import collections
import math
from heuristics.heuristic_base import Heuristic
from task import Operator, Task


# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple of strings.
    e.g., '(at rover1 waypoint1)' -> ('at', 'rover1', 'waypoint1')
    """
    # Removes outer parentheses and splits by space
    parts = fact_string.strip('()').split()
    return tuple(parts)

# Helper function for BFS
def bfs(graph, start_node):
    """
    Performs Breadth-First Search on a graph to find shortest distances
    from a start node.

    Args:
        graph: A dictionary representing the graph (node -> list of neighbors).
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
        Nodes not in the graph or unreachable will have distance infinity.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph, no paths possible from here
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()
        # Ensure current_node is a valid key before accessing graph[current_node]
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Helper function to compute all-pairs shortest paths
def compute_all_pairs_shortest_paths(graphs):
    """
    Computes shortest path distances between all pairs of nodes for each graph.

    Args:
        graphs: A dictionary mapping graph identifiers (e.g., rover names)
                to graph representations (node -> list of neighbors).

    Returns:
        A dictionary mapping graph identifiers -> start_node -> end_node -> distance.
    """
    distances = {}
    for graph_id, graph in graphs.items():
        distances[graph_id] = {}
        for start_wp in graph:
            distances[graph_id][start_wp] = bfs(graph, start_wp)
    return distances


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing the
    estimated costs for each unachieved goal fact. The cost for an unachieved
    goal fact is estimated as the minimum number of actions required to achieve
    it from the current state, considering navigation costs (estimated by
    shortest path on the rover's traverse graph) and action costs (each action
    costs 1). It considers which rovers are capable of performing necessary
    tasks (sampling, imaging) and the current state of resources (stores,
    calibration). It is designed for greedy best-first search and is not
    admissible.

    Assumptions:
    - Each action has a cost of 1.
    - Navigation cost between waypoints for a rover is the shortest path
      distance in the graph defined by the rover's 'can_traverse' facts.
    - Dropped samples/images are not recoverable for goal achievement.
    - Rovers have at most one store.
    - The lander's position is static.
    - The 'can_traverse' relation is static.
    - Camera properties (on_board, supports, calibration_target) are static.
    - Objective visibility ('visible_from') is static.
    - Rover equipment ('equipped_for_...') is static.
    - Object types (rover, waypoint, etc.) can be inferred from predicate
      arguments in initial state, goals, and static facts.

    Heuristic Initialization:
    The constructor precomputes static information from the task definition:
    - Identifies all objects (rovers, waypoints, etc.) by inferring types
      from predicates in static facts, initial state, and goals.
    - Determines the lander location and identifies communication waypoints
      (waypoints visible from the lander).
    - Records which rovers are equipped for soil analysis, rock analysis,
      and imaging.
    - Maps each rover to its store.
    - Maps each camera to the rover it's on board, the modes it supports,
      and its calibration target objective.
    - Maps each objective to the waypoints from which it is visible.
    - Builds navigation graphs for each rover based on 'can_traverse' facts
      and computes all-pairs shortest path distances for each rover.
    - Computes the minimum navigation cost for each rover from any waypoint
      to any communication waypoint.
    These precomputed structures are stored as member variables for efficient
    lookup during heuristic calculation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Initialize the total heuristic value `h_value` to 0.
    2.  Parse the current state (`node.state`) to extract dynamic information:
        -   The current location of each rover (`at ?r ?w`).
        -   The waypoints where soil samples are located (`at_soil_sample ?w`).
        -   The waypoints where rock samples are located (`at_rock_sample ?w`).
        -   Which rovers have which soil analyses (`have_soil_analysis ?r ?w`).
        -   Which rovers have which rock analyses (`have_rock_analysis ?r ?w`).
        -   Which rovers have which images (`have_image ?r ?o ?m`).
        -   Which cameras are currently calibrated on which rovers (`calibrated ?c ?r`).
        -   Which stores are currently empty (`empty ?s`).
    3.  Iterate through each goal fact (`goal_str`) in `self.goals`.
    4.  If `goal_str` is already present in the current state, its contribution
        to the heuristic is 0. Continue to the next goal.
    5.  If `goal_str` is not in the state, estimate the minimum cost to achieve it:
        a.  For a `(communicated_soil_data ?w)` goal:
            -   Initialize the minimum cost for this goal (`goal_cost`) to infinity.
            -   Iterate through each rover `?r`.
            -   Get the rover's current location (`current_wp`). If the rover's location
                is unknown (should not happen in valid states), skip this rover.
            -   Calculate the cost for this rover to achieve the goal:
                -   If `?r` has `(have_soil_analysis ?r ?w)` in the state:
                    -   The remaining task is to navigate to a communication waypoint
                        and communicate. Cost is `self.get_min_nav_to_comm_cost(rover, current_wp) + 1`
                        (where +1 is for the `communicate_soil_data` action).
                -   Else if `(at_soil_sample ?w)` is in the state AND `?r` is equipped
                    for soil analysis (static) AND `?r` has an empty store (state):
                    -   The task is to navigate to `?w`, sample, navigate to a
                        communication waypoint, and communicate. Cost is
                        `self.get_nav_cost(rover, current_wp, ?w) + 1` (sample) +
                        `self.get_min_nav_to_comm_cost(rover, ?w) + 1` (communicate).
                -   Otherwise (sample taken but not on rover, or no suitable rover,
                    or no sample at waypoint): This rover cannot achieve this goal
                    via sampling from this waypoint in this state. Cost is infinity
                    for this rover for this goal.
            -   Update `goal_cost` with the minimum cost found among all rovers.
        b.  For a `(communicated_rock_data ?w)` goal:
            -   Follow a similar logic as for soil data, using rock-specific
                predicates (`have_rock_analysis`, `at_rock_sample`, `equipped_for_rock_analysis`)
                and the `communicate_rock_data` action.
        c.  For a `(communicated_image_data ?o ?m)` goal:
            -   Initialize the minimum cost for this goal (`goal_cost`) to infinity.
            -   Identify all suitable (camera, rover) pairs where the camera is
                on board the rover, the rover is equipped for imaging, and the
                camera supports the required mode `?m`. If no such pair exists,
                the goal is unachievable (cost infinity).
            -   For each suitable (camera `?i`, rover `?r`) pair:
                -   Get the rover's current location (`current_wp`).
                -   Find waypoints `?p` visible from objective `?o` and waypoints `?w`
                    visible from the calibration target `?t` for camera `?i`. If no
                    such waypoints exist, this pair cannot achieve the goal (cost infinity).
                -   Calculate the cost for this rover/camera/waypoints combination:
                    -   If `?r` has `(have_image ?r ?o ?m)` in the state:
                        -   Cost is `self.get_min_nav_to_comm_cost(rover, current_wp) + 1` (communicate).
                    -   Otherwise (need to take the image):
                        -   Initialize cost for this path to infinity.
                        -   Iterate over all possible view waypoints `view_wp` from `view_waypoints`.
                        -   Iterate over all possible calibration waypoints `calib_wp` from `calib_waypoints`.
                        -   Calculate the cost for this specific `view_wp` and `calib_wp`:
                            -   Cost from having the image at `view_wp`: `cost_from_have = self.get_min_nav_to_comm_cost(rover, view_wp)`
                                # Need to add +1 for the communicate action *after* getting to the comm waypoint
                                if cost_from_have != float('inf'):
                                     cost_from_have += 1
                                else:
                                     continue # Cannot communicate from view_wp, this path is impossible

                            -   Cost to take image (includes calibration if needed):
                                -   `take_image_path_cost = float('inf')`
                                -   `wp_after_calib = current_wp` # Default if no calibration needed

                                -   if camera in rover_calibrated_cams.get(rover, set()):
                                    # Already calibrated, just need to navigate to view_wp and take image
                                    nav_to_view_cost = self.get_nav_cost(rover, current_wp, view_wp)
                                    if nav_to_view_cost != float('inf'):
                                        take_image_path_cost = nav_to_view_cost + 1 # +1 for take_image
                                else:
                                    # Need to calibrate, then navigate to view_wp, then take image
                                    nav_to_calib_cost = self.get_nav_cost(rover, current_wp, calib_wp)
                                    if nav_to_calib_cost != float('inf'):
                                        wp_after_calib = calib_wp # Rover is at calib_wp after calibrating
                                        nav_calib_to_view_cost = self.get_nav_cost(rover, wp_after_calib, view_wp)
                                        if nav_calib_to_view_cost != float('inf'):
                                            take_image_path_cost = nav_to_calib_cost + 1 + nav_calib_to_view_cost + 1 # +1 calibrate, +1 take_image

                            # Total cost for this specific path (`current_wp` -> `calib_wp` -> `view_wp` -> comm_wp):
                            # This is the cost to get the image (take_image_path_cost) + the cost to communicate it (cost_from_have)
                            if take_image_path_cost != float('inf'):
                                 path_cost = take_image_path_cost + cost_from_have
                                 min_path_cost_for_achiever = min(min_path_cost_for_achiever, path_cost)

                        rover_cost = min_path_cost_for_achiever # This is the cost for this rover/camera combo

                        # Update the minimum cost for the goal across all suitable achievers
                        min_achiever_cost = min(min_achiever_cost, rover_cost)

                    goal_cost = min_achiever_cost # Minimum cost over all suitable achievers

            # Add cost for this goal to total heuristic
            if goal_cost == float('inf'):
                 # If any unachieved goal is impossible, the whole state is likely a dead end
                 # Return infinity for the whole heuristic.
                 return float('inf')
            h_value += goal_cost

        return h_value
