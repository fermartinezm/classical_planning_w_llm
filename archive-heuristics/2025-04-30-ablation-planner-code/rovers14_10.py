from fnmatch import fnmatch
from collections import deque, defaultdict
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
# Replace this with the actual import in the target environment
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs BFS on a graph to find shortest distances from a start node.
    graph: dict {node: set(neighbor_nodes)}
    start_node: The starting node.
    Returns: dict {node: distance}
    """
    # Initialize distances for all nodes present in the graph keys
    distances = {node: float('inf') for node in graph}

    # If the start node is not in the graph, no paths are possible from it
    if start_node not in graph:
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is also a valid node in the graph (should be if graph is built correctly)
                if neighbor in distances and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve each
    unmet goal fact independently and sums these estimates. The estimate for
    each goal considers the necessary steps (navigation, sampling/imaging,
    communication) and chooses the cheapest available rover/camera combination.
    Navigation costs are estimated using precomputed shortest paths on the
    rover-specific traversal graphs.

    # Assumptions
    - The heuristic assumes that any required sample exists initially at the
      specified waypoint if a 'communicated_soil_data' or 'communicated_rock_data'
      goal exists for that waypoint.
    - The heuristic assumes that necessary calibration targets and image
      visibility points exist if 'communicated_image_data' goals exist.
    - The heuristic assumes that a rover can always drop a full store if needed
      to collect a new sample.
    - The heuristic assumes that solvable instances have paths allowing rovers
      to reach necessary waypoints (sample sites, calibration sites, image sites,
      communication sites). Unreachable waypoints result in infinite navigation cost.
    - The heuristic sums the costs for each goal independently, ignoring potential
      synergies (e.g., one navigation action fulfilling needs for multiple goals)
      or conflicts (e.g., multiple rovers needing the same resource). This makes
      it non-admissible but potentially effective for greedy search.

    # Heuristic Initialization
    - Parses goal facts to identify required communications.
    - Parses static facts to build:
        - Rover traversal graphs and precompute all-pairs shortest paths.
        - Sets of equipped rovers for each task type (soil, rock, imaging).
        - Mapping from rover to its store.
        - Mapping from camera to supported modes.
        - Mapping from objective to waypoints it's visible from.
        - Mapping from camera to its calibration target.
        - Mapping from rover to cameras on board.
        - Set of lander locations.
        - Set of communication waypoints (visible from any lander location).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1. Initialize total heuristic cost `h = 0`.
    2. Identify the current location of each rover from the state.
    3. For each goal fact `g` in the initial goals:
        a. If `g` is already present in the current state, add 0 to `h` and continue.
        b. If `g` is `(communicated_soil_data ?w)`:
            i. Find the minimum cost among all soil-equipped rovers `r`.
            ii. For a rover `r`:
                - Start with cost 0 and the rover's current location.
                - If `(have_soil_analysis ?r ?w)` is NOT in the state:
                    - Add navigation cost from the rover's current location to waypoint `w`.
                    - Add 1 for the `sample_soil` action.
                    - If the rover's store is full, add 1 for the `drop` action.
                    - Update the rover's effective location to `w`.
                - Add navigation cost from the rover's effective location to the nearest communication waypoint.
                - Add 1 for the `communicate_soil_data` action.
            iii. Add the minimum cost found for this goal to `h`.
        c. If `g` is `(communicated_rock_data ?w)`: Similar logic as for soil data, using rock-equipped rovers.
        d. If `g` is `(communicated_image_data ?o ?m)`:
            i. Find the minimum cost among all imaging-equipped rovers `r` with a camera `i` that supports mode `m` and is on board `r`.
            ii. For a rover `r` and camera `i`:
                - Start with cost 0 and the rover's current location.
                - If `(have_image ?r ?o ?m)` is NOT in the state:
                    - If `(calibrated ?i ?r)` is NOT in the state:
                        - Find the calibration target `t` for camera `i`.
                        - Find waypoints `cal_wps` visible from `t`.
                        - Add navigation cost from the rover's current location to the nearest waypoint in `cal_wps`.
                        - Add 1 for the `calibrate` action.
                        - Update the rover's effective location to the chosen calibration waypoint.
                    - Find waypoints `img_wps` visible from objective `o`.
                    - Add navigation cost from the rover's effective location to the nearest waypoint in `img_wps`.
                    - Add 1 for the `take_image` action.
                    - Update the rover's effective location to the chosen image waypoint.
                - Add navigation cost from the rover's effective location to the nearest communication waypoint.
                - Add 1 for the `communicate_image_data` action.
            iii. Add the minimum cost found for this goal to `h`.
    4. Return the total cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Navigation Graphs and Distances ---
        self.rover_traversal_graphs = defaultdict(lambda: defaultdict(set))
        self.rover_distances = {} # {rover: {start_wp: {end_wp: dist}}}
        all_waypoints = set() # Collect all waypoints mentioned in can_traverse

        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, rover, wp1, wp2 = get_parts(fact)
                self.rover_traversal_graphs[rover][wp1].add(wp2)
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)

        # Ensure all waypoints mentioned in can_traverse are keys in the graph
        # even if they have no outgoing edges. This is important so BFS initializes
        # distances for all relevant nodes for this rover.
        for rover in self.rover_traversal_graphs:
             for wp in all_waypoints:
                 if wp not in self.rover_traversal_graphs[rover]:
                     self.rover_traversal_graphs[rover][wp] = set()


        for rover, graph in self.rover_traversal_graphs.items():
            self.rover_distances[rover] = {}
            # BFS from every waypoint to every other waypoint for this rover
            for start_wp in graph:
                 self.rover_distances[rover][start_wp] = bfs(graph, start_wp)

        # --- Extract Other Static Information ---
        self.equipped_rovers = defaultdict(set)
        self.rover_stores = {} # {rover: store}
        self.camera_supports = defaultdict(set) # {camera: {mode}}
        self.objective_visible_from = defaultdict(set) # {objective: {waypoint}}
        self.camera_calibration_target = {} # {camera: objective}
        self.rover_cameras = defaultdict(set) # {rover: {camera}}
        self.lander_locations = set()
        self.communication_waypoints = set()

        visible_graph = defaultdict(set) # For finding communication points

        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_rovers['soil'].add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_rovers['rock'].add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_rovers['imaging'].add(get_parts(fact)[1])
            elif match(fact, "store_of", "*", "*"):
                _, store, rover = get_parts(fact)
                self.rover_stores[rover] = store
            elif match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                self.camera_supports[camera].add(mode)
            elif match(fact, "visible_from", "*", "*"):
                _, objective, waypoint = get_parts(fact)
                self.objective_visible_from[objective].add(waypoint)
            elif match(fact, "calibration_target", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.camera_calibration_target[camera] = objective
            elif match(fact, "on_board", "*", "*"):
                _, camera, rover = get_parts(fact)
                self.rover_cameras[rover].add(camera)
            elif match(fact, "at_lander", "*", "*"):
                self.lander_locations.add(get_parts(fact)[2])
            elif match(fact, "visible", "*", "*"):
                 _, wp1, wp2 = get_parts(fact)
                 visible_graph[wp1].add(wp2)
                 visible_graph[wp2].add(wp1) # Visibility is bidirectional

        # Determine communication waypoints: any waypoint visible from any lander location
        for lander_wp in self.lander_locations:
            if lander_wp in visible_graph:
                self.communication_waypoints.update(visible_graph[lander_wp])
            # A rover might need to go *to* the lander waypoint itself if reachable
            self.communication_waypoints.add(lander_wp)


    def get_rover_location(self, state, rover_name):
        """Find the current waypoint of a specific rover in the state."""
        for fact in state:
            if match(fact, "at", rover_name, "*"):
                return get_parts(fact)[2]
        return None # Should not happen in a valid state

    def get_distance(self, rover_name, start_wp, end_wp):
        """Get the precomputed shortest distance for a rover between two waypoints."""
        # Handle case where start or end waypoint might not be in the rover's graph
        # or if the rover_name is not in the precomputed distances (e.g., rover has no can_traverse facts)
        if rover_name not in self.rover_distances or \
           start_wp not in self.rover_distances[rover_name] or \
           end_wp not in self.rover_distances[rover_name][start_wp]:
             return float('inf')
        return self.rover_distances[rover_name][start_wp][end_wp]

    def get_min_nav_cost_and_wp(self, rover_name, start_wp, target_wps):
        """
        Find the minimum navigation cost from start_wp to any waypoint in target_wps
        for the given rover, and return the cost and the target waypoint.
        Returns (float('inf'), None) if target_wps is empty or no path exists.
        """
        min_cost = float('inf')
        closest_wp = None
        if not target_wps:
            return min_cost, closest_wp

        for target_wp in target_wps:
            cost = self.get_distance(rover_name, start_wp, target_wp)
            if cost < min_cost:
                min_cost = cost
                closest_wp = target_wp
        return min_cost, closest_wp

    def get_min_nav_cost_to_comm_wp(self, rover_name, start_wp):
        """
        Find the minimum navigation cost from start_wp to any communication waypoint
        for the given rover.
        Returns float('inf') if no communication waypoint is reachable.
        """
        min_cost, _ = self.get_min_nav_cost_and_wp(rover_name, start_wp, self.communication_waypoints)
        return min_cost


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Get current rover locations {rover_name: waypoint}
        rover_locations = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in state if match(fact, "at", "*", "*")
        }

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            predicate, *args = get_parts(goal)

            if predicate == 'communicated_soil_data':
                waypoint = args[0]
                min_goal_cost = float('inf')

                for rover in self.equipped_rovers['soil']:
                    if rover not in rover_locations:
                        # This rover is not in the current state (e.g., not created yet? Or problem definition issue)
                        # Skip this rover as it cannot perform actions.
                        continue
                    rover_current_wp = rover_locations[rover]
                    cost = 0
                    current_wp_after_step = rover_current_wp

                    # Check if soil analysis is needed
                    have_soil = f'(have_soil_analysis {rover} {waypoint})' in state

                    if not have_soil:
                        # Need to navigate to waypoint, sample, maybe drop
                        nav_cost_to_sample = self.get_distance(rover, rover_current_wp, waypoint)
                        if nav_cost_to_sample == float('inf'):
                            continue # Cannot reach sample waypoint with this rover

                        cost += nav_cost_to_sample
                        cost += 1 # sample_soil

                        rover_store = self.rover_stores.get(rover)
                        # Check if the specific store for this rover is full
                        if rover_store and f'(full {rover_store})' in state:
                            cost += 1 # drop

                        current_wp_after_step = waypoint # Rover is now at sample waypoint

                    # Need to navigate to communication waypoint and communicate
                    nav_cost_to_comm = self.get_min_nav_cost_to_comm_wp(rover, current_wp_after_step)
                    if nav_cost_to_comm == float('inf'):
                         continue # Cannot reach any communication waypoint with this rover

                    cost += nav_cost_to_comm
                    cost += 1 # communicate_soil_data

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost
                else:
                    # If no equipped rover can achieve this goal, add a large penalty
                    # or infinity. Adding infinity makes it a dead end.
                    h += float('inf')


            elif predicate == 'communicated_rock_data':
                waypoint = args[0]
                min_goal_cost = float('inf')

                for rover in self.equipped_rovers['rock']:
                    if rover not in rover_locations: continue
                    rover_current_wp = rover_locations[rover]
                    cost = 0
                    current_wp_after_step = rover_current_wp

                    # Check if rock analysis is needed
                    have_rock = f'(have_rock_analysis {rover} {waypoint})' in state

                    if not have_rock:
                        # Need to navigate to waypoint, sample, maybe drop
                        nav_cost_to_sample = self.get_distance(rover, rover_current_wp, waypoint)
                        if nav_cost_to_sample == float('inf'):
                            continue # Cannot reach sample waypoint

                        cost += nav_cost_to_sample
                        cost += 1 # sample_rock

                        rover_store = self.rover_stores.get(rover)
                        if rover_store and f'(full {rover_store})' in state:
                            cost += 1 # drop

                        current_wp_after_step = waypoint # Rover is now at sample waypoint

                    # Need to navigate to communication waypoint and communicate
                    nav_cost_to_comm = self.get_min_nav_cost_to_comm_wp(rover, current_wp_after_step)
                    if nav_cost_to_comm == float('inf'):
                         continue # Cannot reach any communication waypoint

                    cost += nav_cost_to_comm
                    cost += 1 # communicate_rock_data

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost
                else:
                    h += float('inf')


            elif predicate == 'communicated_image_data':
                objective, mode = args
                min_goal_cost = float('inf')

                # Iterate through rovers equipped for imaging
                for rover in self.equipped_rovers['imaging']:
                    if rover not in rover_locations: continue
                    rover_current_wp = rover_locations[rover]

                    # Iterate through cameras on board this rover
                    for camera in self.rover_cameras.get(rover, set()):
                        # Check if camera supports the required mode
                        if mode not in self.camera_supports.get(camera, set()):
                            continue # Camera doesn't support this mode

                        cost = 0
                        current_wp_after_step = rover_current_wp

                        # Check if image is needed
                        have_image = f'(have_image {rover} {objective} {mode})' in state

                        if not have_image:
                            # Need to calibrate and take image
                            calibrated = f'(calibrated {camera} {rover})' in state

                            if not calibrated:
                                # Need to navigate to calibration waypoint and calibrate
                                cal_target = self.camera_calibration_target.get(camera)
                                if not cal_target:
                                     # Camera has no calibration target defined, cannot calibrate
                                     continue

                                cal_wps = self.objective_visible_from.get(cal_target, set())
                                min_cal_nav, closest_cal_wp = self.get_min_nav_cost_and_wp(rover, rover_current_wp, cal_wps)

                                if min_cal_nav == float('inf'):
                                    continue # Cannot reach any calibration waypoint

                                cost += min_cal_nav
                                cost += 1 # calibrate
                                current_wp_after_step = closest_cal_wp # Rover is now at calibration WP

                            # Need to navigate to image waypoint and take image
                            img_wps = self.objective_visible_from.get(objective, set())
                            min_img_nav, closest_img_wp = self.get_min_nav_cost_and_wp(rover, current_wp_after_step, img_wps)

                            if min_img_nav == float('inf'):
                                continue # Cannot reach any image waypoint

                            cost += min_img_nav
                            cost += 1 # take_image
                            current_wp_after_step = closest_img_wp # Rover is now at image WP

                        # Need to navigate to communication waypoint and communicate
                        nav_cost_to_comm = self.get_min_nav_cost_to_comm_wp(rover, current_wp_after_step)
                        if nav_cost_to_comm == float('inf'):
                             continue # Cannot reach any communication waypoint

                        cost += nav_cost_to_comm
                        cost += 1 # communicate_image_data

                        min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost
                else:
                    h += float('inf') # No suitable rover/camera combination can achieve this goal


        # Ensure heuristic is 0 only at the goal state
        # If h is 0, it means all goals were in the state.
        # If h is infinity, it means at least one goal is unreachable.
        # If h is finite and > 0, it means some goals are not met but reachable.
        # This structure naturally satisfies the h=0 iff goal state property.

        return h
