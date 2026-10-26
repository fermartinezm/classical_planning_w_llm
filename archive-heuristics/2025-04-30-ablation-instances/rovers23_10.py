from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys # For float('inf') comparison

# Helper functions get_parts and match (adapted from Logistics example)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
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
    # Ensure the number of parts is at least the number of args
    if len(parts) < len(args):
         return False
    # Check if parts match args, allowing '*' wildcard
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy each
    individual goal condition that is not yet met in the current state.
    The total heuristic value is the sum of these individual goal costs.
    For data communication goals (soil, rock, image), the heuristic estimates
    the minimum actions needed to acquire the data (if not already acquired)
    and then navigate to a communication point and communicate. Navigation
    costs are estimated using shortest paths on the waypoint graph.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Store state (full/empty) is considered for sampling actions.
    - Camera calibration is required before taking an image and is consumed by the action.
    - For a given goal, any available equipped rover/camera can be used. The heuristic
      calculates the minimum cost over all suitable resources.
    - The heuristic ignores potential negative interactions between goals (e.g.,
      resource contention, store becoming full unexpectedly) and positive interactions
      (e.g., one navigation covering distance for multiple goals). It is additive.
    - Samples (`at_soil_sample`, `at_rock_sample`) are assumed to exist at their
      initial locations until collected. If a sample is not `at_soil_sample`/`at_rock_sample`
      and no rover `have_soil_analysis`/`have_rock_analysis` for that waypoint, the
      heuristic assumes the goal is unreachable from this state (assigns infinity cost).
    - Similarly, if required visible_from waypoints for imaging/calibration or
      communication waypoints are unreachable or non-existent, the goal is assumed
      unreachable.

    # Heuristic Initialization
    The constructor precomputes static information from the task:
    - The navigation graph between waypoints based on `can_traverse` and `visible` facts.
    - Shortest path distances between all pairs of waypoints using BFS.
    - The lander's waypoint.
    - The set of waypoints visible from the lander (communication points).
    - For each objective, the set of waypoints from which it is visible.
    - For each camera, its calibration target.
    - For each calibration target objective, the set of waypoints from which it is visible.
    - Information about each rover (equipped tools, stores, cameras).
    - Information about each camera (on board rover, supported modes, calibration target).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic `h(s)` is calculated as follows:
    1. Initialize total cost `h(s) = 0`.
    2. Extract relevant dynamic information from the state: rover locations, full stores,
       collected data (`have_soil_analysis`, `have_rock_analysis`, `have_image`),
       communicated data, and remaining samples (`at_soil_sample`, `at_rock_sample`).
    3. For each goal predicate in the task's goals:
        a. If the goal predicate is already true in the current state, add 0 to the total cost.
        b. If the goal is `(communicated_soil_data ?w)` and it's not in the state:
            i. Check if `(have_soil_analysis ?r ?w)` is true for any rover `r` in the state.
            ii. If yes: The cost is the minimum navigation cost for such a rover `r` from its current location to any communication waypoint, plus 1 (for the `communicate_soil_data` action). Minimize this over all rovers that have the data.
            iii. If no: The data needs to be sampled and then communicated. Check if a sample exists at `?w`. If not, the goal is assumed unreachable (infinity cost). If yes, find soil-equipped rovers. For each such rover `r`, calculate the cost: navigate from `at(r)` to `w`, plus 1 (sample), plus 1 (drop) if `r`'s store is full, plus navigate from `w` to the nearest communication waypoint, plus 1 (communicate). The cost for this goal is the minimum of this calculation over all soil-equipped rovers.
        c. If the goal is `(communicated_rock_data ?w)` and it's not in the state:
            i. Similar logic to soil data, using rock-specific predicates and equipment and checking `at_rock_sample`.
        d. If the goal is `(communicated_image_data ?o ?m)` and it's not in the state:
            i. Check if `(have_image ?r ?o ?m)` is true for any rover `r` in the state.
            ii. If yes: The cost is the minimum navigation cost for such a rover `r` from its current location to any communication waypoint, plus 1 (for the `communicate_image_data` action). Minimize this over all rovers that have the image.
            iii. If no: The image needs to be taken and then communicated. Find imaging-equipped rovers `r` with cameras `i` supporting mode `m` and having calibration target `t`. For each such combination `(r, i, t)`, calculate the cost: navigate from `at(r)` to a calibration waypoint for `i` (visible from `t`), plus 1 (calibrate), plus navigate from the calibration waypoint to an image waypoint for `o` (visible from `o`), plus 1 (take image), plus navigate from the image waypoint to a communication waypoint, plus 1 (communicate). The navigation costs are shortest paths, minimizing over all valid intermediate waypoints. The cost for this goal is the minimum of this calculation over all suitable `(r, i, t)` combinations and optimal intermediate waypoints.
    4. The total heuristic value `h(s)` is the sum of the costs calculated for each unsatisfied goal. If any unsatisfied goal has an infinite cost (deemed unreachable by the heuristic), the total cost is infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Navigation Graph and Distances ---
        self.waypoint_graph = {}
        waypoints = set()
        # Collect all waypoints first
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] in ('at_lander', 'at_soil_sample', 'at_rock_sample', 'visible', 'can_traverse', 'visible_from'):
                  for part in parts[1:]:
                       # Simple check for waypoint type - assumes waypoint names start with 'waypoint'
                       if isinstance(part, str) and part.startswith('waypoint'):
                            waypoints.add(part)

        # Build graph edges based on can_traverse and visible
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'can_traverse' and len(parts) == 4:
                _, r, w1, w2 = parts
                # Add edge if visible (precondition for navigate)
                # Assuming visible is symmetric or checked both ways in problem files
                if f'(visible {w1} {w2})' in static_facts:
                     self.waypoint_graph.setdefault(w1, []).append(w2)
                if f'(visible {w2} {w1})' in static_facts:
                     self.waypoint_graph.setdefault(w2, []).append(w1)

        # Ensure all waypoints are keys in the graph, even if they have no edges
        for wp in waypoints:
             self.waypoint_graph.setdefault(wp, [])

        self.dist = {}
        for wp in self.waypoint_graph:
            self.dist[wp] = self._bfs(wp, self.waypoint_graph)

        # --- Precompute Other Static Information ---
        self.lander_waypoint = None
        for fact in static_facts:
            if match(fact, 'at_lander', '*', '*'):
                self.lander_waypoint = get_parts(fact)[2]
                break # Assuming only one lander

        self.comm_waypoints = set()
        if self.lander_waypoint:
             for fact in static_facts:
                 if match(fact, 'visible', '*', self.lander_waypoint):
                     self.comm_waypoints.add(get_parts(fact)[1])

        self.obj_visible_waypoints = {}
        for fact in static_facts:
            if match(fact, 'visible_from', '*', '*'):
                _, obj, wp = get_parts(fact)
                self.obj_visible_waypoints.setdefault(obj, set()).add(wp)

        self.cam_cal_target = {}
        for fact in static_facts:
            if match(fact, 'calibration_target', '*', '*'):
                _, cam, target = get_parts(fact)
                self.cam_cal_target[cam] = target

        self.cal_target_visible_waypoints = {}
        # Populate this based on obj_visible_waypoints and cam_cal_target
        for cam, target in self.cam_cal_target.items():
             if target in self.obj_visible_waypoints:
                  self.cal_target_visible_waypoints[target] = self.obj_visible_waypoints[target]
             else:
                  self.cal_target_visible_waypoints[target] = set() # Should not happen in valid problems?

        self.rover_info = {}
        self.camera_info = {}
        rovers = set()
        cameras = set()

        # Collect all rovers and cameras first
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] in ('equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'):
                  if len(parts) > 1: rovers.add(parts[1])
             elif parts[0] == 'store_of':
                  if len(parts) > 2: rovers.add(parts[2])
             elif parts[0] == 'on_board':
                  if len(parts) > 2: cameras.add(parts[1]); rovers.add(parts[2])
             elif parts[0] == 'supports':
                  if len(parts) > 1: cameras.add(parts[1])
             elif parts[0] == 'calibration_target':
                  if len(parts) > 1: cameras.add(parts[1])

        # Initialize info dictionaries
        for r in rovers: self.rover_info.setdefault(r, {'stores': set(), 'cameras': set()})
        for c in cameras: self.camera_info.setdefault(c, {'supports': set()})

        # Populate info dictionaries
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'store_of' and len(parts) == 3:
                  _, store, rover = parts
                  self.rover_info[rover]['stores'].add(store)
             elif parts[0] == 'equipped_for_soil_analysis' and len(parts) == 2:
                  _, rover = parts
                  self.rover_info[rover]['soil'] = True
             elif parts[0] == 'equipped_for_rock_analysis' and len(parts) == 2:
                  _, rover = parts
                  self.rover_info[rover]['rock'] = True
             elif parts[0] == 'equipped_for_imaging' and len(parts) == 2:
                  _, rover = parts
                  self.rover_info[rover]['imaging'] = True
             elif parts[0] == 'on_board' and len(parts) == 3:
                  _, camera, rover = parts
                  self.rover_info[rover]['cameras'].add(camera)
                  self.camera_info[camera]['on_board'] = rover
             elif parts[0] == 'supports' and len(parts) == 3:
                  _, camera, mode = parts
                  self.camera_info[camera]['supports'].add(mode)
             elif parts[0] == 'calibration_target' and len(parts) == 3:
                  _, camera, target = parts
                  self.camera_info[camera]['cal_target'] = target


    def _bfs(self, start_node, graph):
        """Helper to compute shortest path distances from a start node using BFS."""
        distances = {node: float('inf') for node in graph}
        if start_node not in graph: # Handle isolated waypoints if any
             return distances
        distances[start_node] = 0
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            # Ensure current_node is in graph keys before accessing neighbors
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, w1, w2):
        """Helper to get shortest distance between two waypoints."""
        if w1 is None or w2 is None: return float('inf')
        return self.dist.get(w1, {}).get(w2, float('inf'))

    def get_min_distance_to_set(self, start_w, target_ws):
        """Helper to get minimum shortest distance from a waypoint to a set of waypoints."""
        if start_w is None or not target_ws: return float('inf')
        min_d = float('inf')
        for target_w in target_ws:
            min_d = min(min_d, self.get_distance(start_w, target_w))
        return min_d

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract Relevant State Information ---
        rover_locations = {}
        rover_stores_full = set()
        rover_have_soil = set() # (rover, waypoint)
        rover_have_rock = set() # (rover, waypoint)
        rover_have_image = set() # (rover, objective, mode)
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)
        at_soil_sample = set() # waypoint
        at_rock_sample = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'full' and len(parts) == 2 and parts[1].endswith('store'):
                 store = parts[1]
                 # Find which rover this store belongs to
                 for r, info in self.rover_info.items():
                      if store in info.get('stores', set()):
                           rover_stores_full.add(r)
                           break
            elif parts[0] == 'have_soil_analysis' and len(parts) == 3:
                rover_have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis' and len(parts) == 3:
                rover_have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image' and len(parts) == 4:
                rover_have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'communicated_soil_data' and len(parts) == 2:
                communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data' and len(parts) == 2:
                communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data' and len(parts) == 3:
                communicated_image.add((parts[1], parts[2]))
            elif parts[0] == 'at_soil_sample' and len(parts) == 2:
                 at_soil_sample.add(parts[1])
            elif parts[0] == 'at_rock_sample' and len(parts) == 2:
                 at_rock_sample.add(parts[1])


        total_cost = 0

        # --- Calculate Cost for Unsatisfied Goals ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts: continue # Skip empty goals if any
            predicate = parts[0]

            if predicate == 'communicated_soil_data' and len(parts) == 2:
                w = parts[1]
                goal_cost = float('inf')

                # Check if data is already collected by any rover
                rover_with_data = None
                for r, wp in rover_have_soil:
                    if wp == w:
                        rover_with_data = r
                        break

                if rover_with_data:
                    # Data exists, just need to communicate
                    curr_w = rover_locations.get(rover_with_data)
                    cost = self.get_min_distance_to_set(curr_w, self.comm_waypoints) # Navigation
                    if cost != float('inf'):
                         cost += 1 # +1 for communicate
                         goal_cost = min(goal_cost, cost)
                else:
                    # Need to sample and communicate
                    # Check if sample exists at waypoint w
                    if w not in at_soil_sample:
                         # Sample is gone and no one has it -> unreachable from this state
                         pass # goal_cost remains inf
                    else:
                        # Find suitable rovers
                        soil_rovers = [r for r, info in self.rover_info.items() if info.get('soil')]
                        for r in soil_rovers:
                            curr_w = rover_locations.get(r)
                            if curr_w is None: continue # Rover location unknown

                            # Cost to get sample: navigate to w + sample + (drop if store full)
                            cost_get_sample_nav = self.get_distance(curr_w, w) # Navigation
                            if cost_get_sample_nav == float('inf'): continue # Cannot reach sample

                            cost_get_sample = cost_get_sample_nav + 1 # +1 for sample
                            if r in rover_stores_full:
                                cost_get_sample += 1 # +1 for drop

                            # Cost to communicate: navigate from w to comm waypoint + communicate
                            cost_communicate_nav = self.get_min_distance_to_set(w, self.comm_waypoints) # Navigation
                            if cost_communicate_nav == float('inf'): continue # Cannot reach comm point from sample point

                            cost_communicate = cost_communicate_nav + 1 # +1 for communicate

                            cost = cost_get_sample + cost_communicate
                            goal_cost = min(goal_cost, cost)

                if goal_cost != float('inf'):
                    total_cost += goal_cost
                else:
                    # If any goal is unreachable, the whole plan is likely impossible
                    # Assign a large value or infinity. Using infinity.
                    return float('inf')


            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                w = parts[1]
                goal_cost = float('inf')

                # Check if data is already collected by any rover
                rover_with_data = None
                for r, wp in rover_have_rock:
                    if wp == w:
                        rover_with_data = r
                        break

                if rover_with_data:
                    # Data exists, just need to communicate
                    curr_w = rover_locations.get(rover_with_data)
                    cost = self.get_min_distance_to_set(curr_w, self.comm_waypoints) # Navigation
                    if cost != float('inf'):
                         cost += 1 # +1 for communicate
                         goal_cost = min(goal_cost, cost)
                else:
                    # Need to sample and communicate
                    # Check if sample exists at waypoint w
                    if w not in at_rock_sample:
                         pass # goal_cost remains inf
                    else:
                        # Find suitable rovers
                        rock_rovers = [r for r, info in self.rover_info.items() if info.get('rock')]
                        for r in rock_rovers:
                            curr_w = rover_locations.get(r)
                            if curr_w is None: continue # Rover location unknown

                            # Cost to get sample: navigate to w + sample + (drop if store full)
                            cost_get_sample_nav = self.get_distance(curr_w, w) # Navigation
                            if cost_get_sample_nav == float('inf'): continue # Cannot reach sample

                            cost_get_sample = cost_get_sample_nav + 1 # +1 for sample
                            if r in rover_stores_full:
                                cost_get_sample += 1 # +1 for drop

                            # Cost to communicate: navigate from w to comm waypoint + communicate
                            cost_communicate_nav = self.get_min_distance_to_set(w, self.comm_waypoints) # Navigation
                            if cost_communicate_nav == float('inf'): continue # Cannot reach comm point from sample point

                            cost_communicate = cost_communicate_nav + 1 # +1 for communicate

                            cost = cost_get_sample + cost_communicate
                            goal_cost = min(goal_cost, cost)

                if goal_cost != float('inf'):
                    total_cost += goal_cost
                else:
                    # If any goal is unreachable, the whole plan is likely impossible
                    return float('inf')


            elif predicate == 'communicated_image_data' and len(parts) == 3:
                o = parts[1]
                m = parts[2]
                goal_cost = float('inf')

                # Check if image is already taken by any rover
                rover_with_image = None
                for r, obj, mode in rover_have_image:
                    if obj == o and mode == m:
                        rover_with_image = r
                        break

                if rover_with_image:
                    # Image exists, just need to communicate
                    curr_w = rover_locations.get(rover_with_image)
                    cost = self.get_min_distance_to_set(curr_w, self.comm_waypoints) # Navigation
                    if cost != float('inf'):
                         cost += 1 # +1 for communicate
                         goal_cost = min(goal_cost, cost)
                else:
                    # Need to take image and communicate
                    # Find suitable (rover, camera, target) combinations
                    image_combos = []
                    for r, r_info in self.rover_info.items():
                         if r_info.get('imaging'):
                              for cam in r_info.get('cameras', set()):
                                   cam_info = self.camera_info.get(cam, {})
                                   if m in cam_info.get('supports', set()):
                                        cal_target = cam_info.get('cal_target')
                                        if cal_target: # Must have a calibration target
                                             image_combos.append((r, cam, cal_target))

                    for r, i, t in image_combos:
                        curr_w = rover_locations.get(r)
                        if curr_w is None: continue # Rover location unknown

                        cal_ws = self.cal_target_visible_waypoints.get(t, set())
                        img_ws = self.obj_visible_waypoints.get(o, set())
                        comm_ws = self.comm_waypoints

                        if not cal_ws or not img_ws or not comm_ws:
                            continue # Cannot achieve this goal with this combo (missing visible points)

                        # Calculate min path cost: curr_w -> cal_w -> img_w -> comm_w
                        # min_{cw in cal_ws} ( dist(curr_w, cw) + min_{iw in img_ws} ( dist(cw, iw) + min_{cmw in comm_ws} dist(iw, cmw) ) )
                        min_path_nav_cost = float('inf')

                        for cw in cal_ws:
                             dist_curr_to_cw = self.get_distance(curr_w, cw)
                             if dist_curr_to_cw == float('inf'): continue

                             min_cw_to_comm = float('inf')
                             for iw in img_ws:
                                  dist_cw_to_iw = self.get_distance(cw, iw)
                                  if dist_cw_to_iw == float('inf'): continue

                                  dist_iw_to_comm_set = self.get_min_distance_to_set(iw, comm_ws)
                                  if dist_iw_to_comm_set == float('inf'): continue

                                  min_cw_to_comm = min(min_cw_to_comm, dist_cw_to_iw + dist_iw_to_comm_set)

                             if min_cw_to_comm != float('inf'):
                                  min_path_nav_cost = min(min_path_nav_cost, dist_curr_to_cw + min_cw_to_comm)


                        if min_path_nav_cost != float('inf'):
                             # Cost = Navigation + Calibrate + TakeImage + Communicate
                             cost = min_path_nav_cost + 3 # +1 calibrate, +1 take_image, +1 communicate
                             goal_cost = min(goal_cost, cost)

                if goal_cost != float('inf'):
                    total_cost += goal_cost
                else:
                    # If any goal is unreachable, the whole plan is likely impossible
                    return float('inf')

        return total_cost
