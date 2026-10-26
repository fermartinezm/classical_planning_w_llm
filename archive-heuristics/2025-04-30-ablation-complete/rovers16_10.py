from fnmatch import fnmatch
import collections
from heuristics.heuristic_base import Heuristic

# Define a large cost for unreachable goals to guide the search away
UNREACHABLE_COST = 1000

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, unless the last arg is a wildcard
    if len(parts) != len(args) and (not args or args[-1] != '*'):
         return False
    # Use zip to handle cases where parts might be shorter than args (if args doesn't end with *)
    # or args might be shorter than parts (if args ends with *).
    # fnmatch handles the wildcard matching.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_waypoints, target_waypoints, rover, can_traverse_graph):
    """
    Find the shortest path distance for a rover from a set of start waypoints
    to a set of target waypoints using the can_traverse graph.
    Returns infinity if no path exists.
    """
    if not start_waypoints or not target_waypoints:
        return float('inf')

    # Filter start waypoints to only include those the rover can actually traverse from
    # and which are present in the graph for this rover.
    valid_start_waypoints = {w for w in start_waypoints if w in can_traverse_graph.get(rover, {})}

    if not valid_start_waypoints:
        return float('inf')

    q = collections.deque([(w, 0) for w in valid_start_waypoints])
    visited = set(valid_start_waypoints)

    while q:
        current_w, dist = q.popleft()

        if current_w in target_waypoints:
            return dist

        # Check if current_w is in the graph for this rover
        if rover in can_traverse_graph and current_w in can_traverse_graph[rover]:
            for next_w in can_traverse_graph[rover][current_w]:
                if next_w not in visited:
                    visited.add(next_w)
                    q.append((next_w, dist + 1))

    return float('inf') # No path found

def get_rover_location(state, rover):
    """Find the current waypoint of a specific rover in the state."""
    for fact in state:
        parts = get_parts(fact)
        if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] == rover:
            return parts[2]
    return None # Should not happen in a valid state, but handle defensively

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions. It decomposes the problem into achieving each individual goal predicate (communicating soil data, rock data, or image data) and sums the estimated costs. For each unachieved communication goal, it estimates the cost to first obtain the necessary data/image (if not already held) and then the cost to move to a communication location and communicate.

    # Assumptions
    - Actions have a cost of 1.
    - The primary goal is communication of data/images.
    - Obtaining data/images (sampling, imaging) and communicating are sequential steps.
    - Movement costs are estimated using Breadth-First Search (BFS) on the rover's traversable waypoint graph.
    - Calibration, if needed for imaging, is treated as a step that requires moving to a calibration target waypoint and performing the calibrate action. The cost is added to the image acquisition cost.
    - Dropping a sample from a full store, if needed before sampling, adds a cost of 1.
    - The heuristic sums the costs for each unachieved goal independently, ignoring potential synergies (e.g., collecting multiple samples/images on one trip, communicating multiple items at once).
    - If a required sample is no longer at its initial location and no rover holds the data, or if required waypoints (sample, image, calibration, communication) are unreachable for all suitable rovers, the cost for that goal is considered infinite (represented by a large constant).

    # Heuristic Initialization
    - Extracts static facts from the task:
        - Lander locations.
        - Rover traversability graph (`can_traverse`).
        - Waypoint visibility graph (`visible`).
        - Objective visibility from waypoints (`visible_from`).
        - Camera information (on which rover, supported modes, calibration target).
        - Store ownership per rover.
        - Rover capabilities (soil, rock, imaging).
    - Precomputes the set of waypoints from which communication with a lander is possible (`lander_comm_waypoints`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the sum of estimated costs for each goal predicate that is not yet satisfied:

    1.  **Initialize total heuristic cost to 0.**
    2.  **Identify unachieved goal predicates** from `task.goals` that are not present in the current `state`.
    3.  **For each unachieved goal predicate G:**
        *   Initialize the estimated cost for this goal (`goal_cost`) to infinity.
        *   **If G is `(communicated_soil_data ?w)`:**
            *   Extract waypoint `w`.
            *   Check if any rover `r` already has `(have_soil_analysis ?r ?w)`.
            *   If yes, calculate the minimum cost for such a rover `r` to move from its current location to any `lander_comm_waypoint` and communicate (BFS distance + 1). Update `goal_cost` with the minimum found cost.
            *   If no rover has the data, *and* if `(at_soil_sample ?w)` is still in the state:
                *   Find rovers `r` equipped for soil analysis.
                *   For each equipped rover `r`, calculate the cost sequence:
                    *   Find rover's current location. If unknown, skip rover.
                    *   Move from current location to `?w` (BFS distance).
                    *   If reachable, add distance + 1 (sample) to cost.
                    *   If `r`'s store is full, add 1 for `drop`.
                    *   Move from `?w` to any `lander_comm_waypoint` (BFS distance).
                    *   If reachable, add distance + 1 (communicate) to cost.
                *   Update `goal_cost` with the minimum total cost found over all equipped rovers.
        *   **If G is `(communicated_rock_data ?w)`:** Follow the same logic as soil data, substituting rock-specific predicates and capabilities.
        *   **If G is `(communicated_image_data ?o ?m)`:**
            *   Extract objective `o` and mode `m`.
            *   Check if any rover `r` already has `(have_image ?r ?o ?m)`.
            *   If yes, calculate the minimum cost for such a rover `r` to move from its current location to any `lander_comm_waypoint` and communicate (BFS distance + 1). Update `goal_cost` with the minimum found cost.
            *   If no rover has the image:
                *   Find rovers `r` equipped for imaging with a camera `i` supporting mode `m`.
                *   For each suitable pair `(r, i)`, calculate the cost sequence:
                    *   Initialize sequence cost to 0.
                    *   Find rover's current location. If unknown, skip pair.
                    *   Needs_calibration = f'(calibrated {camera_i} {r})' not in state

                    # Step 1: Calibrate if needed
                    if Needs_calibration:
                        cal_target = self.camera_info.get(camera_i, {}).get('cal_target')
                        if cal_target:
                            cal_w_set = self.objective_visibility.get(cal_target, set())
                            dist_to_cal_w = bfs({current_w}, cal_w_set, r_equip, self.can_traverse_graph)
                            if dist_to_cal_w != float('inf'):
                                cost_sequence += dist_to_cal_w + 1 # move + calibrate
                            else:
                                continue # Cannot calibrate, skip this pair
                        else:
                            continue # No cal target, skip this pair

                    # Step 2: Move to image waypoint and take image
                    image_w_set = self.objective_visibility.get(objective_o, set())
                    dist_to_image_w = bfs({current_w}, image_w_set, r_equip, self.can_traverse_graph) # Start from current_w

                    if dist_to_image_w != float('inf'):
                         cost_sequence += dist_to_image_w + 1 # move + take_image

                         # Step 3: Move to communication waypoint and communicate
                         # BFS from image_w_set to lander_comm_waypoints
                         dist_to_comm = bfs(image_w_set, self.lander_comm_waypoints, r_equip, self.can_traverse_graph)

                         if dist_to_comm != float('inf'):
                             cost_sequence += dist_to_comm + 1 # move + communicate
                             min_get_image_and_comm_cost = min(min_get_image_and_comm_cost, cost_sequence)

                goal_cost = min(goal_cost, min_get_image_and_comm_cost)

            # Add the cost for this goal to the total
            if goal_cost == float('inf'):
                total_cost += UNREACHABLE_COST
            else:
                total_cost += goal_cost

        return total_cost
