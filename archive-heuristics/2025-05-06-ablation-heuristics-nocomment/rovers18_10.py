from fnmatch import fnmatch
from collections import deque
import sys

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy base class for standalone execution/testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static
        # Assuming task object might have other relevant attributes like 'objects'
        # but relying on static_facts and goals for object parsing as per examples.

    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """
    Removes surrounding parentheses from a PDDL fact string and splits it into parts.
    e.g., '(at rover1 waypoint1)' -> ['at', 'rover1', 'waypoint1']
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Checks if the parts of a fact match a sequence of arguments,
    supporting fnmatch wildcards.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Computes shortest distances from start_node to all reachable nodes in the graph
    using Breadth-First Search.
    Returns a dictionary {node: distance}. Unreachable nodes have distance float('inf').
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not a known waypoint in the graph
         return distances # All distances remain inf

    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Ensure current_node is in graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    Estimates the cost to reach the goal by summing up the estimated costs
    for each unachieved goal fact. The cost for each goal fact is estimated
    based on the necessary actions (navigate, sample/image/calibrate,
    communicate) and the current state. Navigation costs are estimated
    using precomputed shortest paths on the waypoint graph for each rover.
    Calibration cost for image goals is added if the required camera is
    not calibrated, estimated as 1 action plus navigation to a calibration
    waypoint. Store capacity is simplified to a binary full/empty check,
    adding a cost of 1 if a drop action is needed before sampling.
    The heuristic is non-admissible as it sums costs for potentially
    interdependent goals and simplifies action sequences (e.g., calibration
    cost is added in parallel).

    Assumptions:
    - The problem is solvable.
    - Each rover has exactly one store.
    - Calibration targets and image objectives are visible from at least
      one waypoint reachable by a suitable rover.
    - Lander is at a fixed location visible from at least one waypoint
      reachable by a suitable rover.
    - The waypoint graph defined by (can_traverse ?r ?y ?z) and (visible ?y ?z)
      is connected enough for solvable instances.

    Heuristic Initialization:
    - Parses static facts and goal facts to identify all objects (rovers, waypoints,
      landers, cameras, objectives, stores, modes) by inferring types from predicate
      argument positions.
    - Populates static data structures: lander location, rover capabilities
      (soil, rock, imaging), rover-store mapping, visible graph (symmetric),
      rover-specific traversable graphs (can_traverse + visible), camera properties
      (on-board, supported modes, calibration target), and objective visibility.
    - Builds a navigation graph for each rover based on (can_traverse) and
      (visible) predicates. Ensures all identified waypoints are nodes in the graph.
    - Precomputes all-pairs shortest paths for each rover on its navigation
      graph using BFS.
    - Identifies sets of waypoints relevant for communication (visible from lander),
      imaging (visible from objectives), and calibration (visible from calibration
      targets).
    - Precomputes minimum navigation costs for each rover from any waypoint
      to any of these goal-relevant waypoint sets.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `h` to 0.
    2. Parse the current state to determine dynamic facts: rover locations,
       store statuses, which rovers have which data/images, camera calibration
       statuses, and remaining samples at waypoints.
    3. Iterate through each goal fact defined in the task.
    4. If a goal fact is already true in the current state, its contribution
       to the heuristic is 0. Continue to the next goal.
    5. If the goal is `(communicated_soil_data ?w)`:
       - Find the minimum estimated cost to achieve this goal.
       - This cost is the minimum over all rovers `r`:
         - If rover `r` already has `(have_soil_analysis ?r ?w)`:
           Cost = (Min navigation cost for `r` from its current location
                   to any waypoint visible from the lander) + 1 (communicate action).
         - If rover `r` is equipped for soil analysis and needs to sample:
           Cost = (Navigation cost for `r` from its current location to `?w`)
                  + (1 if `r`'s store is full else 0) + 1 (sample action)
                  + (Min navigation cost for `r` from `?w` to any waypoint
                     visible from the lander) + 1 (communicate action).
         - If rover `r` is not suitable (not equipped), its cost is infinity.
       - Add the minimum cost found (over all suitable rovers) to `h`.
    6. If the goal is `(communicated_rock_data ?w)`:
       - Similar logic as for soil data, using rock analysis equipment,
         rock samples, and `have_rock_analysis` predicate.
    7. If the goal is `(communicated_image_data ?o ?m)`:
       - Find the minimum estimated cost to achieve this goal.
       - This cost is the minimum over all rovers `r`, cameras `i` on board `r`
         supporting mode `m`, and waypoints `p` visible from objective `o`:
         - If rover `r` already has `(have_image ?r ?o ?m)`:
           Cost = (Min navigation cost for `r` from its current location
                   to any waypoint visible from the lander) + 1 (communicate action).
         - If rover `r` is equipped for imaging and needs to take the image:
           Cost = (Navigation cost for `r` from its current location to `p`)
                  + 1 (take_image action)
                  + (Min navigation cost for `r` from `p` to any waypoint
                     visible from the lander) + 1 (communicate action)
                  + (Cost to calibrate camera `i` for rover `r`).
           - Cost to calibrate camera `i` for rover `r`:
             - If `(calibrated ?i ?r)` is true: 0.
             - If `(calibrated ?i ?r)` is false: (Min navigation cost for `r`
               from its current location to any waypoint visible from `i`'s
               calibration target) + 1 (calibrate action).
             (Note: This calibration cost is added in parallel and simplifies
              the actual sequence where calibration must precede taking the image.
              It estimates the cost of the calibration *task* from the current state).
         - If rover `r` or camera `i` is not suitable, its cost is infinity.
       - Add the minimum cost found (over all suitable rover/camera/waypoint
         combinations) to `h`.
    8. Return the total heuristic value `h`. If `h` is infinity (meaning at least one goal component is unreachable by this model), return infinity.
    """
    def __init__(self, task):
        super().__init__(task)

        # --- Parse Static Facts and Goals to Identify Objects ---
        # Infer types based on predicate positions in static facts and goals
        potential_rovers = set()
        potential_waypoints = set()
        potential_landers = set()
        potential_stores = set()
        potential_cameras = set()
        potential_modes = set()
        potential_objectives = set()

        # Collect objects based on predicate positions in static facts and goals
        for fact in self.static_facts | self.goals:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at' and len(parts) == 3: potential_rovers.add(parts[1]); potential_waypoints.add(parts[2])
            elif pred == 'at_lander' and len(parts) == 3: potential_landers.add(parts[1]); potential_waypoints.add(parts[2])
            elif pred == 'can_traverse' and len(parts) == 4: potential_rovers.add(parts[1]); potential_waypoints.add(parts[2]); potential_waypoints.add(parts[3])
            elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'] and len(parts) == 2: potential_rovers.add(parts[1])
            elif pred in ['empty', 'full'] and len(parts) == 2: potential_stores.add(parts[1])
            elif pred in ['have_rock_analysis', 'have_soil_analysis'] and len(parts) == 3: potential_rovers.add(parts[1]); potential_waypoints.add(parts[2])
            elif pred == 'calibrated' and len(parts) == 3: potential_cameras.add(parts[1]); potential_rovers.add(parts[2])
            elif pred == 'supports' and len(parts) == 3: potential_cameras.add(parts[1]); potential_modes.add(parts[2])
            elif pred == 'visible' and len(parts) == 3: potential_waypoints.add(parts[1]); potential_waypoints.add(parts[2])
            elif pred == 'have_image' and len(parts) == 5: potential_rovers.add(parts[1]); potential_objectives.add(parts[2]); potential_modes.add(parts[3])
            elif pred in ['communicated_soil_data', 'communicated_rock_data'] and len(parts) == 2: potential_waypoints.add(parts[1])
            elif pred == 'communicated_image_data' and len(parts) == 3: potential_objectives.add(parts[1]); potential_modes.add(parts[2])
            elif pred in ['at_soil_sample', 'at_rock_sample'] and len(parts) == 2: potential_waypoints.add(parts[1])
            elif pred == 'visible_from' and len(parts) == 3: potential_objectives.add(parts[1]); potential_waypoints.add(parts[2])
            elif pred == 'store_of' and len(parts) == 3: potential_stores.add(parts[1]); potential_rovers.add(parts[2])
            elif pred == 'calibration_target' and len(parts) == 3: potential_cameras.add(parts[1]); potential_objectives.add(parts[2])
            elif pred == 'on_board' and len(parts) == 3: potential_cameras.add(parts[1]); potential_rovers.add(parts[2])

        self.rovers = potential_rovers
        self.waypoints = potential_waypoints
        self.landers = potential_landers
        self.stores = potential_stores
        self.cameras = potential_cameras
        self.modes = potential_modes
        self.objectives = potential_objectives

        # --- Populate Static Data Structures ---
        self.lander_location = {} # {lander: waypoint}
        self.equipped_soil = set() # {rover1, rover2, ...}
        self.equipped_rock = set() # {rover1, rover2, ...}
        self.equipped_imaging = set() # {rover1, rover2, ...}
        self.rover_stores = {} # {rover: store}
        self.visible_graph = {} # {wp: {neighbor1, ...}} (undirected)
        self.rover_graphs = {} # {rover: {wp: {neighbor1, ...}}} (directed)
        self.rover_cameras_onboard = {} # {rover: {cam1, ...}}
        self.camera_modes_supported = {} # {cam: {mode1, ...}}
        self.camera_calibration_targets = {} # {cam: obj}
        self.objective_visibility = {} # {obj: {wp1, ...}}

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if match(fact, 'at_lander', '*', '*'):
                self.lander_location[parts[1]] = parts[2]
            elif match(fact, 'equipped_for_soil_analysis', '*'):
                self.equipped_soil.add(parts[1])
            elif match(fact, 'equipped_for_rock_analysis', '*'):
                self.equipped_rock.add(parts[1])
            elif match(fact, 'equipped_for_imaging', '*'):
                self.equipped_imaging.add(parts[1])
            elif match(fact, 'store_of', '*', '*'):
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif match(fact, 'visible', '*', '*'):
                wp1, wp2 = parts[1], parts[2]
                self.visible_graph.setdefault(wp1, set()).add(wp2)
                self.visible_graph.setdefault(wp2, set()).add(wp1) # visible is symmetric
            elif match(fact, 'can_traverse', '*', '*', '*'):
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                # Build rover-specific graph considering both can_traverse and visible
                if wp_to in self.visible_graph.get(wp_from, set()):
                     self.rover_graphs.setdefault(rover, {}).setdefault(wp_from, set()).add(wp_to)

            elif match(fact, 'on_board', '*', '*'):
                cam, rover = parts[1], parts[2]
                self.rover_cameras_onboard.setdefault(rover, set()).add(cam)
            elif match(fact, 'supports', '*', '*'):
                cam, mode = parts[1], parts[2]
                self.camera_modes_supported.setdefault(cam, set()).add(mode)
            elif match(fact, 'calibration_target', '*', '*'):
                cam, obj = parts[1], parts[2]
                self.camera_calibration_targets[cam] = obj
            elif match(fact, 'visible_from', '*', '*'):
                obj, wp = parts[1], parts[2]
                self.objective_visibility.setdefault(obj, set()).add(wp)

        # Ensure all waypoints are keys in rover_graphs, even if no edges
        for rover in self.rovers:
             self.rover_graphs.setdefault(rover, {})
             for wp in self.waypoints:
                  self.rover_graphs[rover].setdefault(wp, set())

        # Precompute shortest paths for each rover
        self.rover_shortest_paths = {} # {rover: {from_wp: {to_wp: dist}}}
        for rover in self.rovers:
            self.rover_shortest_paths[rover] = {}
            for start_wp in self.waypoints:
                self.rover_shortest_paths[rover][start_wp] = bfs(self.rover_graphs.get(rover, {}), start_wp)

        # Precompute goal-relevant waypoint sets
        self.lander_comm_wps = {} # {lander: {wp1, ...}}
        for lander, l_loc in self.lander_location.items():
             self.lander_comm_wps[lander] = self.visible_graph.get(l_loc, set()) # Waypoints visible from lander's location

        self.camera_cal_wps = {} # {camera: {wp1, ...}}
        for cam, cal_target in self.camera_calibration_targets.items():
             self.camera_cal_wps[cam] = self.objective_visibility.get(cal_target, set()) # Waypoints visible from cal target

        # Precompute minimum navigation costs to goal-relevant sets
        self.min_dist_to_comm_wp = {} # {rover: {from_wp: min_dist}}
        self.min_dist_to_image_wp = {} # {rover: {objective: {from_wp: min_dist}}}
        self.min_dist_to_cal_wp = {} # {rover: {camera: {from_wp: min_dist}}}

        lander_comm_wp_set = set()
        # Assuming only one lander for simplicity based on examples
        if self.landers:
             # Take the first lander found
             first_lander = next(iter(self.landers))
             lander_comm_wp_set = self.lander_comm_wps.get(first_lander, set())

        for rover in self.rovers:
            self.min_dist_to_comm_wp[rover] = {}
            self.min_dist_to_image_wp[rover] = {}
            self.min_dist_to_cal_wp[rover] = {}

            for from_wp in self.waypoints:
                # Min dist to communication waypoint
                min_dist = float('inf')
                for comm_wp in lander_comm_wp_set:
                     dist = self.rover_shortest_paths.get(rover, {}).get(from_wp, {}).get(comm_wp, float('inf'))
                     min_dist = min(min_dist, dist)
                self.min_dist_to_comm_wp[rover][from_wp] = min_dist

                # Min dist to image waypoints for each objective
                for obj in self.objectives:
                     self.min_dist_to_image_wp[rover].setdefault(obj, {})
                     image_wp_set = self.objective_visibility.get(obj, set())
                     min_dist = float('inf')
                     for image_wp in image_wp_set:
                          dist = self.rover_shortest_paths.get(rover, {}).get(from_wp, {}).get(image_wp, float('inf'))
                          min_dist = min(min_dist, dist)
                     self.min_dist_to_image_wp[rover][obj][from_wp] = min_dist

                # Min dist to calibration waypoints for each camera
                for cam in self.cameras:
                     self.min_dist_to_cal_wp[rover].setdefault(cam, {})
                     cal_wp_set = self.camera_cal_wps.get(cam, set())
                     min_dist = float('inf')
                     for cal_wp in cal_wp_set:
                          dist = self.rover_shortest_paths.get(rover, {}).get(from_wp, {}).get(cal_wp, float('inf'))
                          min_dist = min(min_dist, dist)
                     self.min_dist_to_cal_wp[rover][cam][from_wp] = min_dist


    def __call__(self, node):
        state = node.state
        h = 0

        # --- Parse State Facts ---
        rover_locations = {} # {rover: waypoint}
        store_status = {} # {store: 'empty' or 'full'}
        rover_soil_data = {} # {rover: {wp1, ...}}
        rover_rock_data = {} # {rover: {wp1, ...}}
        rover_image_data = {} # {rover: {(o, m), ...}}
        camera_calibration_status = {} # {(camera, rover): True} (default False)
        soil_samples_remaining = set() # {wp1, ...}
        rock_samples_remaining = set() # {wp1, ...}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if match(fact, 'at', '*', '*'):
                rover_locations[parts[1]] = parts[2]
            elif match(fact, 'empty', '*'):
                store_status[parts[1]] = 'empty'
            elif match(fact, 'full', '*'):
                store_status[parts[1]] = 'full'
            elif match(fact, 'have_soil_analysis', '*', '*'):
                rover, wp = parts[1], parts[2]
                rover_soil_data.setdefault(rover, set()).add(wp)
            elif match(fact, 'have_rock_analysis', '*', '*'):
                rover, wp = parts[1], parts[2]
                rover_rock_data.setdefault(rover, set()).add(wp)
            elif match(fact, 'have_image', '*', '*', '*'):
                rover, obj, mode = parts[1], parts[2], parts[3]
                rover_image_data.setdefault(rover, set()).add((obj, mode))
            elif match(fact, 'calibrated', '*', '*'):
                cam, rover = parts[1], parts[2]
                camera_calibration_status[(cam, rover)] = True
            elif match(fact, 'at_soil_sample', '*'):
                soil_samples_remaining.add(parts[1])
            elif match(fact, 'at_rock_sample', '*'):
                rock_samples_remaining.add(parts[1])

        # --- Compute Heuristic ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            pred = parts[0]

            if pred == 'communicated_soil_data' and len(parts) == 2:
                # Goal: (communicated_soil_data ?w)
                w = parts[1]
                min_goal_cost = float('inf')

                # Option 1: Data already collected by some rover
                rover_with_data_found = False
                for r in self.rovers:
                     if w in rover_soil_data.get(r, set()):
                          rover_with_data_found = True
                          # Cost = Nav(curr_loc, comm_wp) + communicate
                          current_loc = rover_locations.get(r)
                          if current_loc:
                               nav_cost = self.min_dist_to_comm_wp.get(r, {}).get(current_loc, float('inf'))
                               if nav_cost != float('inf'):
                                    min_goal_cost = min(min_goal_cost, nav_cost + 1)

                # Option 2: Need to sample and communicate
                if not rover_with_data_found:
                     # Need to sample at waypoint w
                     if w in soil_samples_remaining: # Sample must exist
                         for r in self.rovers:
                             if r in self.equipped_soil: # Rover must be equipped
                                 # Cost = Nav(curr_loc, w) + (store) + sample + Nav(w, comm_wp) + communicate
                                 current_loc = rover_locations.get(r)
                                 store = self.rover_stores.get(r)
                                 # Cost to drop if store is full. Assume 1 action.
                                 store_c = 1 if store and store_status.get(store) == 'full' else 0

                                 if current_loc:
                                     nav_to_sample = self.rover_shortest_paths.get(r, {}).get(current_loc, {}).get(w, float('inf'))
                                     nav_to_comm_after_sample = self.min_dist_to_comm_wp.get(r, {}).get(w, float('inf'))

                                     if nav_to_sample != float('inf') and nav_to_comm_after_sample != float('inf'):
                                         cost = nav_to_sample + store_c + 1 + nav_to_comm_after_sample + 1
                                         min_goal_cost = min(min_goal_cost, cost)
                     # else: sample already taken, but no rover has data? Impossible state in solvable problem?

                if min_goal_cost != float('inf'):
                    h += min_goal_cost
                else:
                    # Goal seems unreachable from this state based on this heuristic model
                    h += float('inf')


            elif pred == 'communicated_rock_data' and len(parts) == 2:
                # Goal: (communicated_rock_data ?w)
                w = parts[1]
                min_goal_cost = float('inf')

                # Option 1: Data already collected by some rover
                rover_with_data_found = False
                for r in self.rovers:
                     if w in rover_rock_data.get(r, set()):
                          rover_with_data_found = True
                          # Cost = Nav(curr_loc, comm_wp) + communicate
                          current_loc = rover_locations.get(r)
                          if current_loc:
                               nav_cost = self.min_dist_to_comm_wp.get(r, {}).get(current_loc, float('inf'))
                               if nav_cost != float('inf'):
                                    min_goal_cost = min(min_goal_cost, nav_cost + 1)

                # Option 2: Need to sample and communicate
                if not rover_with_data_found:
                     # Need to sample at waypoint w
                     if w in rock_samples_remaining: # Sample must exist
                         for r in self.rovers:
                             if r in self.equipped_rock: # Rover must be equipped
                                 # Cost = Nav(curr_loc, w) + (store) + sample + Nav(w, comm_wp) + communicate
                                 current_loc = rover_locations.get(r)
                                 store = self.rover_stores.get(r)
                                 # Cost to drop if store is full. Assume 1 action.
                                 store_c = 1 if store and store_status.get(store) == 'full' else 0

                                 if current_loc:
                                     nav_to_sample = self.rover_shortest_paths.get(r, {}).get(current_loc, {}).get(w, float('inf'))
                                     nav_to_comm_after_sample = self.min_dist_to_comm_wp.get(r, {}).get(w, float('inf'))

                                     if nav_to_sample != float('inf') and nav_to_comm_after_sample != float('inf'):
                                         cost = nav_to_sample + store_c + 1 + nav_to_comm_after_sample + 1
                                         min_goal_cost = min(min_goal_cost, cost)
                         # else: sample already taken, but no rover has data? Impossible state?

                if min_goal_cost != float('inf'):
                    h += min_goal_cost
                else:
                    h += float('inf')


            elif pred == 'communicated_image_data' and len(parts) == 3:
                # Goal: (communicated_image_data ?o ?m)
                o, m = parts[1], parts[2]
                min_goal_cost = float('inf')

                # Option 1: Image already collected by some rover
                rover_has_image_found = False
                for r in self.rovers:
                     if (o, m) in rover_image_data.get(r, set()):
                          rover_has_image_found = True
                          # Cost = Nav(curr_loc, comm_wp) + communicate
                          current_loc = rover_locations.get(r)
                          if current_loc:
                               nav_cost = self.min_dist_to_comm_wp.get(r, {}).get(current_loc, float('inf'))
                               if nav_cost != float('inf'):
                                    min_goal_cost = min(min_goal_cost, nav_cost + 1)

                # Option 2: Need to take image and communicate
                if not rover_has_image_found:
                     # Need to take image of o in mode m
                     # Find suitable rover, camera, image waypoint
                     for r in self.rovers:
                         if r in self.equipped_imaging: # Rover must be equipped
                             for i in self.rover_cameras_onboard.get(r, set()): # Camera must be on board
                                 if m in self.camera_modes_supported.get(i, set()): # Camera must support mode
                                     # Find best image waypoint for this objective
                                     image_wp_set = self.objective_visibility.get(o, set())
                                     if not image_wp_set: continue # Cannot see objective

                                     current_loc = rover_locations.get(r)
                                     if not current_loc: continue # Rover location unknown?

                                     # Cost = Nav(curr_loc, image_wp) + take_image + Nav(image_wp, comm_wp) + communicate + (calibrate)
                                     # Find best image_wp 'p' for this rover
                                     min_nav_to_image_wp = float('inf')
                                     best_image_wp = None
                                     for p in image_wp_set:
                                          nav_cost = self.rover_shortest_paths.get(r, {}).get(current_loc, {}).get(p, float('inf'))
                                          if nav_cost != float('inf') and nav_cost < min_nav_to_image_wp:
                                               min_nav_to_image_wp = nav_cost
                                               best_image_wp = p

                                     if best_image_wp:
                                          nav_to_comm_after_image = self.min_dist_to_comm_wp.get(r, {}).get(best_image_wp, float('inf'))

                                          if nav_to_comm_after_image != float('inf'):
                                              # Calibration cost (added in parallel)
                                              cal_cost = 0
                                              if not camera_calibration_status.get((i, r), False):
                                                  # Need to calibrate camera i on rover r
                                                  # Cost = Nav(curr_loc, cal_wp) + calibrate
                                                  nav_to_cal = self.min_dist_to_cal_wp.get(r, {}).get(i, {}).get(current_loc, float('inf'))
                                                  if nav_to_cal != float('inf'):
                                                       cal_cost = nav_to_cal + 1
                                                  else:
                                                       # Cannot reach calibration waypoint for this camera/rover
                                                       cal_cost = float('inf') # This rover/camera combo cannot calibrate

                                              if cal_cost != float('inf'):
                                                  # Total cost for this rover/camera/image_wp combo
                                                  cost = min_nav_to_image_wp + 1 + nav_to_comm_after_image + 1 + cal_cost
                                                  min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost != float('inf'):
                    h += min_goal_cost
                else:
                    h += float('inf')

        # If total heuristic is infinity, return infinity
        return h

