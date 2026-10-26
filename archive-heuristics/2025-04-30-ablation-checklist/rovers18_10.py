from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic
import math # Use math.inf for unreachable states

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest paths from a start node
    in an unweighted graph.

    Args:
        graph: Adjacency list representation {node: [neighbor1, neighbor2, ...]}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
        Unreachable nodes are not included.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Handle nodes that might not have any edges
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    goal conditions. It breaks down the problem into achieving each
    individual goal fact (communicating data) and sums the estimated costs.
    The cost for each goal fact is estimated based on the minimum actions
    needed to acquire the necessary data (sample or image) and then
    communicate it, considering rover capabilities and locations. Navigation
    costs are estimated using precomputed shortest paths.

    # Assumptions (for non-admissibility and efficiency):
    - The cost of achieving different goal facts is summed independently,
      ignoring potential synergies (e.g., one navigation covering multiple
      sampling/imaging locations) or conflicts (e.g., store capacity,
      camera calibration state after taking an image).
    - Navigation cost is the shortest path distance (number of `navigate` actions).
    - Calibration is assumed to be needed before taking an image if the camera
      is not currently calibrated. The cost includes navigation to a calibration
      target waypoint and the calibrate action. This cost is added *per image goal*
      if the camera isn't calibrated, which is a simplification.
    - Dropping a sample is assumed to be needed if a rover's store is full
      and it needs to take a new sample.
    - The heuristic finds the minimum cost over all suitable rovers and waypoints
      for each step (acquiring data, communicating data).

    # Heuristic Initialization
    - Parses static facts to identify rover capabilities, camera properties,
      objective visibility, lander location, and waypoint connectivity.
    - Precomputes shortest path distances between all pairs of waypoints
      for each rover using BFS on the navigation graph.
    - Identifies waypoints suitable for communication (visible from the lander).

    # Step-by-Step Thinking for Computing Heuristic (`__call__`)
    1. Initialize total heuristic cost `h = 0`.
    2. Extract current state information: rover locations, `have_X` facts,
       `empty` stores, `calibrated` cameras, `at_soil_sample`, `at_rock_sample`.
    3. Iterate through each goal fact specified in `task.goals`.
    4. If a goal fact is already true in the current state, its cost is 0, continue.
    5. If the goal fact is `(communicated_soil_data wp)`:
       - Estimate cost `h_goal = infinity`.
       - Find rovers equipped for soil analysis.
       - For each equipped rover `r`:
         - Estimate cost to acquire soil data from `wp` (`h_acquire`):
           - If `(have_soil_analysis r wp)` is true in state: `h_acquire = 0`.
           - Else if `(at_soil_sample wp)` is true in state:
             - `r_pos = current_location(r)`.
             - `nav_to_sample = shortest_path(r, r_pos, wp)`.
             - `h_store = 0` if rover's store is empty, `1` (for drop action) if full.
             - `h_acquire = nav_to_sample + h_store + 1` (sample action).
           - Else (`at_soil_sample wp` is false and no rover has data): `h_acquire = infinity` (sample is gone).
         - Estimate cost to communicate data (`h_communicate`):
           - `r_pos_for_comm`: If data was just acquired, assume rover is at `wp`. If data was already held, assume rover is at its current location. (Simplification: use current location `r_pos` always).
           - Find minimum navigation cost from `r_pos_for_comm` to any communication waypoint (`min_nav_to_comm`).
           - `h_communicate = min_nav_to_comm + 1` (communicate action).
         - Update `h_goal = min(h_goal, h_acquire + h_communicate)`.
       - Add `h_goal` to total `h`.
    6. If the goal fact is `(communicated_rock_data wp)`: Similar logic as soil data.
    7. If the goal fact is `(communicated_image_data o m)`:
       - Estimate cost `h_goal = infinity`.
       - Find rovers `r` equipped for imaging with camera `c` on board supporting mode `m`.
       - For each such `(r, c)` pair:
         - Estimate cost to acquire image data `(o, m)` (`h_acquire`):
           - If `(have_image r o m)` is true in state: `h_acquire = 0`.
           - Else:
             - `r_pos = current_location(r)`.
             - Find waypoints `image_wps` where `visible_from(o, image_wp)`.
             - Find nearest `image_wp` from `r_pos`: `min_nav_to_image = min(shortest_path(r, r_pos, image_wp))` over `image_wps`.
             - Estimate calibration cost (`h_calibrate`):
               - If `(calibrated c r)` is true in state: `h_calibrate = 0`.
               - Else:
                 - Find calibration target `t` for camera `c`.
                 - Find waypoints `cal_wps` where `visible_from(t, cal_wp)`.
                 - Find nearest `cal_wp` from `r_pos`: `min_nav_to_cal = min(shortest_path(r, r_pos, cal_wp))` over `cal_wps`.
                 - `h_calibrate = min_nav_to_cal + 1` (calibrate action).
             - `h_acquire = min_nav_to_image + h_calibrate + 1` (take_image action). (This simplifies the sequence Nav->Cal->Nav->Image into Nav+CalCost+ImageCost, which is non-admissible).
           - Estimate cost to communicate data (`h_communicate`):
             - `r_pos_for_comm`: If data was just acquired, assume rover is at the image site. If data was already held, assume rover is at its current location. (Simplification: use current location `r_pos` always).
             - Find minimum navigation cost from `r_pos_for_comm` to any communication waypoint (`min_nav_to_comm`).
             - `h_communicate = min_nav_to_comm + 1` (communicate action).
           - Update `h_goal = min(h_goal, h_acquire + h_communicate)`.
       - Add `h_goal` to total `h`.
    8. Return `h`. If `h` is infinity, the goal is unreachable from this state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.initial_state = task.initial_state # Keep initial state to check if samples existed
        static_facts = task.static

        # --- Parse Static Facts ---
        self.lander_location = {} # lander -> waypoint
        self.rover_equipment = {} # rover -> set of equipment types (soil, rock, imaging)
        self.rover_store_map = {} # rover -> store
        self.rover_camera_map = {} # rover -> set of cameras
        self.camera_mode_map = {} # camera -> set of modes
        self.camera_cal_target_map = {} # camera -> objective
        self.obj_visible_from_map = {} # objective -> set of waypoints
        self.waypoint_visibility_map = {} # waypoint -> set of visible waypoints
        self.rover_can_traverse_map = {} # rover -> waypoint -> set of reachable waypoints

        # Collect all objects by type (useful for graph building)
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        all_facts = set(static_facts) | set(task.initial_state) # Include initial state to find all objects

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]

            # Collect objects by type (heuristic doesn't strictly need all objects, but good practice)
            if predicate in ['at', 'at_lander', 'can_traverse', 'visible', 'at_soil_sample', 'at_rock_sample', 'visible_from']:
                 for p in parts[1:]:
                     if p.startswith('rover'): self.rovers.add(p)
                     elif p.startswith('waypoint'): self.waypoints.add(p)
                     elif p.startswith('lander'): self.landers.add(p)
                     elif p.startswith('objective'): self.objectives.add(p)
            elif predicate in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']:
                 self.rovers.add(parts[1])
            elif predicate in ['empty', 'full', 'store_of']:
                 self.stores.add(parts[1])
                 if len(parts) > 2: self.rovers.add(parts[2]) # store_of
            elif predicate in ['calibrated', 'supports', 'calibration_target', 'on_board']:
                 self.cameras.add(parts[1])
                 if len(parts) > 2:
                     if parts[2].startswith('rover'): self.rovers.add(parts[2]) # calibrated, on_board
                     elif parts[2].startswith('mode'): self.modes.add(parts[2]) # supports
                     elif parts[2].startswith('objective'): self.objectives.add(parts[2]) # calibration_target
            elif predicate in ['have_rock_analysis', 'have_soil_analysis']:
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
            elif predicate == 'have_image':
                 self.rovers.add(parts[1])
                 self.objectives.add(parts[2])
                 self.modes.add(parts[3])
            elif predicate in ['communicated_soil_data', 'communicated_rock_data']:
                 self.waypoints.add(parts[1])
            elif predicate == 'communicated_image_data':
                 self.objectives.add(parts[1])
                 self.modes.add(parts[2])


        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'at_lander':
                lander, waypoint = parts[1], parts[2]
                self.lander_location[lander] = waypoint
            elif predicate.startswith('equipped_for_'):
                rover, equipment_type = parts[1], predicate[len('equipped_for_'):]
                self.rover_equipment.setdefault(rover, set()).add(equipment_type)
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_store_map[rover] = store
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_camera_map.setdefault(rover, set()).add(camera)
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_mode_map.setdefault(camera, set()).add(mode)
            elif predicate == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_cal_target_map[camera] = objective
            elif predicate == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.obj_visible_from_map.setdefault(objective, set()).add(waypoint)
            elif predicate == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_visibility_map.setdefault(wp1, set()).add(wp2)
                self.waypoint_visibility_map.setdefault(wp2, set()).add(wp1) # Visibility is symmetric
            elif predicate == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rover_can_traverse_map.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)


        # --- Precompute Navigation Costs (Shortest Paths) ---
        self.rover_paths = {} # rover -> start_wp -> end_wp -> distance
        for rover in self.rovers:
            self.rover_paths[rover] = {}
            # Build the actual graph for this rover considering both can_traverse and visible
            rover_graph = {}
            for wp1 in self.waypoints:
                rover_graph[wp1] = []
                if wp1 in self.rover_can_traverse_map.get(rover, {}):
                     for wp2 in self.rover_can_traverse_map[rover][wp1]:
                         # Check if wp1 and wp2 are mutually visible for navigation precondition
                         if wp2 in self.waypoint_visibility_map.get(wp1, set()):
                             rover_graph[wp1].append(wp2)

            # Run BFS from each waypoint to find distances
            for start_wp in self.waypoints:
                 self.rover_paths[rover][start_wp] = bfs(rover_graph, start_wp)

        # --- Identify Communication Waypoints ---
        self.comm_wps = set()
        # Assuming there's at least one lander and its location is static
        if self.lander_location:
            lander_wp = list(self.lander_location.values())[0] # Get the waypoint of the first lander
            self.comm_wps = self.waypoint_visibility_map.get(lander_wp, set())
            # Add the lander waypoint itself if it's a valid waypoint for rovers
            if lander_wp in self.waypoints:
                 self.comm_wps.add(lander_wp)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0 # Total heuristic cost

        # --- Extract Current State Information ---
        rover_at = {} # rover -> current_waypoint
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        empty_stores = set() # store
        calibrated_cameras = set() # (camera, rover)
        at_soil_sample = set() # waypoint
        at_rock_sample = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'at' and parts[1].startswith('rover'):
                rover_at[parts[1]] = parts[2]
            elif predicate == 'have_soil_analysis':
                have_soil.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis':
                have_rock.add((parts[1], parts[2]))
            elif predicate == 'have_image':
                have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == 'empty':
                empty_stores.add(parts[1])
            elif predicate == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2]))
            elif predicate == 'at_soil_sample':
                at_soil_sample.add(parts[1])
            elif predicate == 'at_rock_sample':
                at_rock_sample.add(parts[1])

        # --- Estimate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts or len(parts) < 2: continue # Skip malformed goals

            predicate = parts[0]
            h_goal = math.inf # Cost for this specific goal fact

            if predicate == 'communicated_soil_data':
                wp = parts[1]
                # Check if sample ever existed at this waypoint (based on initial state)
                if f'(at_soil_sample {wp})' not in self.initial_state:
                     continue # Cannot achieve this goal if sample never existed

                # Find a rover that can achieve this goal
                for rover in self.rovers:
                    if 'soil_analysis' not in self.rover_equipment.get(rover, set()):
                        continue # Rover not equipped

                    r_pos = rover_at.get(rover)
                    if r_pos is None: continue # Rover location unknown (shouldn't happen in valid states)

                    h_acquire = 0
                    # Check if rover already has the data
                    if (rover, wp) not in have_soil:
                        # Need to acquire the sample
                        if wp not in at_soil_sample:
                            h_acquire = math.inf # Sample is gone, cannot acquire from here
                        else:
                            nav_to_sample = self.rover_paths[rover].get(r_pos, {}).get(wp, math.inf)
                            if nav_to_sample == math.inf:
                                h_acquire = math.inf # Cannot reach sample location
                            else:
                                store = self.rover_store_map.get(rover)
                                h_store = 0
                                if store and store not in empty_stores:
                                    h_store = 1 # Need to drop
                                h_acquire = nav_to_sample + h_store + 1 # Nav + Drop (if needed) + Sample

                    if h_acquire == math.inf:
                        continue # Cannot acquire data with this rover

                    # Estimate cost to communicate
                    # Assume rover is at sample location if it just acquired, else its current location
                    r_pos_for_comm = wp if (rover, wp) not in have_soil and wp in at_soil_sample else r_pos

                    min_nav_to_comm = math.inf
                    if self.comm_wps:
                        for comm_wp in self.comm_wps:
                            nav_cost = self.rover_paths[rover].get(r_pos_for_comm, {}).get(comm_wp, math.inf)
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                    h_communicate = math.inf
                    if min_nav_to_comm != math.inf:
                         h_communicate = min_nav_to_comm + 1 # Nav + Communicate

                    if h_communicate != math.inf:
                        h_goal = min(h_goal, h_acquire + h_communicate)

            elif predicate == 'communicated_rock_data':
                wp = parts[1]
                 # Check if sample ever existed at this waypoint (based on initial state)
                if f'(at_rock_sample {wp})' not in self.initial_state:
                     continue # Cannot achieve this goal if sample never existed

                # Find a rover that can achieve this goal
                for rover in self.rovers:
                    if 'rock_analysis' not in self.rover_equipment.get(rover, set()):
                        continue # Rover not equipped

                    r_pos = rover_at.get(rover)
                    if r_pos is None: continue # Rover location unknown

                    h_acquire = 0
                    # Check if rover already has the data
                    if (rover, wp) not in have_rock:
                        # Need to acquire the sample
                        if wp not in at_rock_sample:
                            h_acquire = math.inf # Sample is gone
                        else:
                            nav_to_sample = self.rover_paths[rover].get(r_pos, {}).get(wp, math.inf)
                            if nav_to_sample == math.inf:
                                h_acquire = math.inf # Cannot reach sample location
                            else:
                                store = self.rover_store_map.get(rover)
                                h_store = 0
                                if store and store not in empty_stores:
                                    h_store = 1 # Need to drop
                                h_acquire = nav_to_sample + h_store + 1 # Nav + Drop (if needed) + Sample

                    if h_acquire == math.inf:
                        continue # Cannot acquire data with this rover

                    # Estimate cost to communicate
                    r_pos_for_comm = wp if (rover, wp) not in have_rock and wp in at_rock_sample else r_pos

                    min_nav_to_comm = math.inf
                    if self.comm_wps:
                        for comm_wp in self.comm_wps:
                            nav_cost = self.rover_paths[rover].get(r_pos_for_comm, {}).get(comm_wp, math.inf)
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                    h_communicate = math.inf
                    if min_nav_to_comm != math.inf:
                         h_communicate = min_nav_to_comm + 1 # Nav + Communicate

                    if h_communicate != math.inf:
                        h_goal = min(h_goal, h_acquire + h_communicate)


            elif predicate == 'communicated_image_data':
                o, m = parts[1], parts[2]

                # Find a rover/camera pair that can achieve this goal
                for rover in self.rovers:
                    if 'imaging' not in self.rover_equipment.get(rover, set()):
                        continue # Rover not equipped for imaging

                    r_pos = rover_at.get(rover)
                    if r_pos is None: continue # Rover location unknown

                    for camera in self.rover_camera_map.get(rover, set()):
                        if m not in self.camera_mode_map.get(camera, set()):
                            continue # Camera doesn't support mode

                        h_acquire = 0
                        # Check if rover already has the image
                        if (rover, o, m) not in have_image:
                            # Need to acquire the image
                            image_wps = self.obj_visible_from_map.get(o, set())
                            if not image_wps:
                                h_acquire = math.inf # Objective not visible from anywhere
                            else:
                                min_nav_to_image = math.inf
                                for image_wp in image_wps:
                                    nav_cost = self.rover_paths[rover].get(r_pos, {}).get(image_wp, math.inf)
                                    min_nav_to_image = min(min_nav_to_image, nav_cost)

                                if min_nav_to_image == math.inf:
                                     h_acquire = math.inf # Cannot reach any image location
                                else:
                                    h_calibrate = 0
                                    # Check if camera needs calibration
                                    if (camera, rover) not in calibrated_cameras:
                                        cal_target = self.camera_cal_target_map.get(camera)
                                        if cal_target is None:
                                            h_acquire = math.inf # Camera has no calibration target
                                        else:
                                            cal_wps = self.obj_visible_from_map.get(cal_target, set())
                                            if not cal_wps:
                                                h_acquire = math.inf # Calibration target not visible from anywhere
                                            else:
                                                min_nav_to_cal = math.inf
                                                for cal_wp in cal_wps:
                                                    nav_cost = self.rover_paths[rover].get(r_pos, {}).get(cal_wp, math.inf)
                                                    min_nav_to_cal = min(min_nav_to_cal, nav_cost)

                                                if min_nav_to_cal == math.inf:
                                                    h_acquire = math.inf # Cannot reach any calibration location
                                                else:
                                                    h_calibrate = min_nav_to_cal + 1 # Nav + Calibrate

                                    if h_acquire != math.inf: # Check if previous steps failed
                                        # Simplified acquisition cost: Nav to image site + Calibrate cost + Take Image
                                        # This assumes Nav to cal site and Nav to image site are separate costs,
                                        # which is a non-admissible simplification.
                                        h_acquire = min_nav_to_image + h_calibrate + 1 # Nav + Calibrate (if needed) + Take Image

                        if h_acquire == math.inf:
                            continue # Cannot acquire image with this rover/camera

                        # Estimate cost to communicate
                        # Assume rover is at image location if it just acquired, else its current location
                        # Simplification: use current location r_pos always
                        r_pos_for_comm = r_pos # Simplified assumption

                        min_nav_to_comm = math.inf
                        if self.comm_wps:
                            for comm_wp in self.comm_wps:
                                nav_cost = self.rover_paths[rover].get(r_pos_for_comm, {}).get(comm_wp, math.inf)
                                min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                        h_communicate = math.inf
                        if min_nav_to_comm != math.inf:
                             h_communicate = min_nav_to_comm + 1 # Nav + Communicate

                        if h_communicate != math.inf:
                            h_goal = min(h_goal, h_acquire + h_communicate)

            # Add the minimum cost for this goal to the total heuristic
            h += h_goal

        return h

