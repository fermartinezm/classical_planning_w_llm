from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the minimum number of actions required to achieve
    each uncommunicated goal (soil data, rock data, image data). The total
    heuristic value is the sum of these minimum costs. For each unachieved
    goal, it finds the minimum cost for a single suitable rover to perform
    the necessary steps (collect sample/take image, then communicate),
    including movement costs estimated by shortest path distance.

    # Assumptions
    - Each rover has a single store.
    - A rover can only carry one sample at a time (implied by empty/full store).
    - A rover can only take one image at a time (not explicitly stated, but reasonable).
    - The same rover that collects/images data is used to communicate it.
    - Dropping a sample (if store is full before taking a new one) costs 1 action and 0 movement (simplification).
    - Unreachable goals (e.g., no equipped rover, objective not visible) have infinite cost.

    # Heuristic Initialization
    - Extracts goal conditions.
    - Extracts static facts: lander location, rover capabilities, store mapping,
      sample locations, camera information (on_board, supports, calibration_target),
      objective visibility, and rover traversal graph.
    - Precomputes shortest path distances between all pairs of waypoints for each rover
      using BFS on the `can_traverse` graph.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total heuristic cost to 0.
    2.  Parse the current state to extract dynamic facts (rover locations,
        have_*, calibrated, communicated_*).
    3.  Identify all goal facts that are not yet true in the current state.
    4.  For each unachieved goal fact:
        a.  If the goal is `(communicated_soil_data ?w)`:
            - If `(communicated_soil_data ?w)` is in the state, cost is 0.
            - Otherwise, find the minimum cost for *any* soil-equipped rover to:
                - If no rover has `(have_soil_analysis ?r ?w)`: move to `?w`, take sample (1 action), potentially drop sample first (1 action if store full).
                - Move to the lander location, communicate data (1 action).
                - The cost includes movement (shortest path distance) and actions. Find the minimum cost over all suitable rovers. Add this minimum cost to the total heuristic.
        b.  If the goal is `(communicated_rock_data ?w)`:
            - Similar logic to soil data, using rock-specific predicates and rovers.
        c.  If the goal is `(communicated_image_data ?o ?m)`:
            - If `(communicated_image_data ?o ?m)` is in the state, cost is 0.
            - Otherwise, find the minimum cost for *any* imaging-equipped rover with a suitable camera to:
                - If no rover has `(have_image ?r ?o ?m)`: potentially calibrate camera (1 action + move to calibration waypoint), move to image waypoint, take image (1 action).
                - Move to the lander location, communicate data (1 action).
                - The cost includes movement (shortest path distance) and actions. Find the minimum cost over all suitable rovers and combinations of calibration/image waypoints. Add this minimum cost to the total heuristic.
    5.  Return the total heuristic cost. If any unachieved goal is determined to be unreachable (e.g., no equipped rover, objective not visible), a large value (infinity) is returned.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.lander_loc = None
        self.rover_capabilities = {} # {rover: {soil: bool, rock: bool, imaging: bool}}
        self.rover_stores = {} # {rover: store}
        self.soil_sample_locs = set() # {waypoint}
        self.rock_sample_locs = set() # {waypoint}
        self.camera_info = {} # {camera: {on_board: rover, supports: {mode}, calibration_target: objective}}
        self.objective_visibility = {} # {objective: {waypoint}}
        self.rovers = set()
        self.waypoint_objects = set() # Store waypoint object names

        # Build traversal graph for each rover
        rover_graph = {} # {rover: {waypoint: {neighbor_waypoint}}}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_loc = parts[2]
            elif parts[0] == 'rover':
                 self.rovers.add(parts[1])
            elif parts[0] == 'waypoint':
                 self.waypoint_objects.add(parts[1])
            elif parts[0] == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, {}).update({'soil': True})
            elif parts[0] == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, {}).update({'rock': True})
            elif parts[0] == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, {}).update({'imaging': True})
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif parts[0] == 'at_soil_sample':
                self.soil_sample_locs.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                self.rock_sample_locs.add(parts[1])
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).update({'on_board': rover})
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('supports', set()).add(mode)
            elif parts[0] == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).update({'calibration_target': objective})
            elif parts[0] == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility.setdefault(objective, set()).add(waypoint)
            elif parts[0] == 'can_traverse':
                rover, w1, w2 = parts[1], parts[2], parts[3]
                rover_graph.setdefault(rover, {}).setdefault(w1, set()).add(w2)
                # Ensure all waypoints mentioned in can_traverse are known
                self.waypoint_objects.add(w1)
                self.waypoint_objects.add(w2)
            # Ensure all rovers mentioned in can_traverse are known
            elif parts[0] == 'rover' and parts[1] not in self.rovers:
                 self.rovers.add(parts[1])


        # Fill in default capabilities for rovers not explicitly listed
        for rover in list(self.rovers): # Iterate over a copy if adding
             self.rover_capabilities.setdefault(rover, {}).update({'soil': False, 'rock': False, 'imaging': False})

        # Precompute shortest paths for each rover
        self.dist = {} # {rover: {start_w: {end_w: distance}}}
        for rover in self.rovers:
            self.dist[rover] = {}
            graph = rover_graph.get(rover, {})
            for start_w in self.waypoint_objects:
                self.dist[rover][start_w] = {}
                q = deque([(start_w, 0)])
                visited = {start_w}
                self.dist[rover][start_w][start_w] = 0 # Distance to self is 0

                while q:
                    current_w, d = q.popleft()

                    for neighbor_w in graph.get(current_w, set()):
                        if neighbor_w not in visited:
                            visited.add(neighbor_w)
                            self.dist[rover][start_w][neighbor_w] = d + 1
                            q.append((neighbor_w, d + 1))

                # Mark unreachable waypoints with infinity
                for end_w in self.waypoint_objects:
                    if end_w not in self.dist[rover][start_w]:
                        self.dist[rover][start_w][end_w] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract dynamic information from the current state
        current_at = {} # {rover: waypoint}
        current_empty_stores = set() # {store}
        current_full_stores = set() # {store}
        current_calibrated = set() # {(camera, rover)}
        current_have_soil = set() # {(rover, waypoint)}
        current_have_rock = set() # {(rover, waypoint)}
        current_have_image = set() # {(rover, objective, mode)}
        current_communicated_soil = set() # {waypoint}
        current_communicated_rock = set() # {waypoint}
        current_communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.rovers: # Ensure it's a rover
                current_at[parts[1]] = parts[2]
            elif parts[0] == 'empty':
                current_empty_stores.add(parts[1])
            elif parts[0] == 'full':
                current_full_stores.add(parts[1])
            elif parts[0] == 'calibrated':
                current_calibrated.add((parts[1], parts[2]))
            elif parts[0] == 'have_soil_analysis':
                current_have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis':
                current_have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image':
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'communicated_soil_data':
                current_communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                current_communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                current_communicated_image.add((parts[1], parts[2]))

        total_cost = 0

        # Calculate cost for unachieved goals
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint not in current_communicated_soil:
                    cost = self._calculate_soil_goal_cost(waypoint, current_at, current_have_soil, current_full_stores)
                    if cost == float('inf'): return float('inf') # Unreachable goal
                    total_cost += cost

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint not in current_communicated_rock:
                    cost = self._calculate_rock_goal_cost(waypoint, current_at, current_have_rock, current_full_stores)
                    if cost == float('inf'): return float('inf') # Unreachable goal
                    total_cost += cost

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                if (objective, mode) not in current_communicated_image:
                    cost = self._calculate_image_goal_cost(objective, mode, current_at, current_have_image, current_calibrated)
                    if cost == float('inf'): return float('inf') # Unreachable goal
                    total_cost += cost

        # The heuristic should be 0 only at the goal state.
        # If total_cost is 0, it implies all goal facts are achieved.
        return total_cost

    def _calculate_soil_goal_cost(self, waypoint, current_at, current_have_soil, current_full_stores):
        """Estimates cost for a single communicated_soil_data goal."""
        min_total_cost = float('inf')
        lander_loc = self.lander_loc

        if waypoint not in self.soil_sample_locs:
             return float('inf') # No sample at this waypoint

        equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps.get('soil')]
        if not equipped_rovers:
            return float('inf') # No rover can collect soil

        for rover in equipped_rovers:
            if rover not in current_at: continue # Rover location unknown (shouldn't happen in valid states)
            current_loc = current_at[rover]

            rover_cost = 0
            temp_loc = current_loc

            has_sample = (rover, waypoint) in current_have_soil

            if not has_sample:
                # Need to take sample
                store = self.rover_stores.get(rover)
                is_store_full = (store is not None) and (store in current_full_stores)

                if is_store_full:
                    rover_cost += 1 # drop_sample action (simplified, no move cost)

                dist_to_sample = self.dist[rover][temp_loc].get(waypoint, float('inf'))
                if dist_to_sample == float('inf'):
                    continue # Cannot reach sample waypoint

                rover_cost += dist_to_sample # move to w
                rover_cost += 1 # take_soil_sample
                temp_loc = waypoint # Assume rover is now at w

            # Now communicate
            dist_to_lander = self.dist[rover][temp_loc].get(lander_loc, float('inf'))
            if dist_to_lander == float('inf'):
                 continue # Cannot reach lander

            rover_cost += dist_to_lander # move to lander
            rover_cost += 1 # communicate_data

            min_total_cost = min(min_total_cost, rover_cost)

        return min_total_cost

    def _calculate_rock_goal_cost(self, waypoint, current_at, current_have_rock, current_full_stores):
        """Estimates cost for a single communicated_rock_data goal."""
        min_total_cost = float('inf')
        lander_loc = self.lander_loc

        if waypoint not in self.rock_sample_locs:
             return float('inf') # No sample at this waypoint

        equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps.get('rock')]
        if not equipped_rovers:
            return float('inf') # No rover can collect rock

        for rover in equipped_rovers:
            if rover not in current_at: continue
            current_loc = current_at[rover]

            rover_cost = 0
            temp_loc = current_loc

            has_sample = (rover, waypoint) in current_have_rock

            if not has_sample:
                # Need to take sample
                store = self.rover_stores.get(rover)
                is_store_full = (store is not None) and (store in current_full_stores)

                if is_store_full:
                    rover_cost += 1 # drop_sample action (simplified, no move cost)

                dist_to_sample = self.dist[rover][temp_loc].get(waypoint, float('inf'))
                if dist_to_sample == float('inf'):
                    continue # Cannot reach sample waypoint

                rover_cost += dist_to_sample # move to w
                rover_cost += 1 # take_rock_sample
                temp_loc = waypoint # Assume rover is now at w

            # Now communicate
            dist_to_lander = self.dist[rover][temp_loc].get(lander_loc, float('inf'))
            if dist_to_lander == float('inf'):
                 continue # Cannot reach lander

            rover_cost += dist_to_lander # move to lander
            rover_cost += 1 # communicate_data

            min_total_cost = min(min_total_cost, rover_cost)

        return min_total_cost

    def _calculate_image_goal_cost(self, objective, mode, current_at, current_have_image, current_calibrated):
        """Estimates cost for a single communicated_image_data goal."""
        min_total_cost = float('inf')
        lander_loc = self.lander_loc

        image_ws = self.objective_visibility.get(objective, set())
        if not image_ws:
            return float('inf') # Objective not visible from anywhere

        equipped_rovers = [r for r, caps in self.rover_capabilities.items() if caps.get('imaging')]
        if not equipped_rovers:
            return float('inf') # No rover can take images

        for rover in equipped_rovers:
            if rover not in current_at: continue
            current_loc = current_at[rover]

            # Find a camera on this rover that supports the mode
            suitable_cameras = [
                cam for cam, info in self.camera_info.items()
                if info.get('on_board') == rover and mode in info.get('supports', set())
            ]
            if not suitable_cameras:
                continue # No suitable camera on this rover

            # For simplicity, just pick the first suitable camera
            camera = suitable_cameras[0]

            cal_o = self.camera_info[camera].get('calibration_target')
            if not cal_o:
                continue # Camera has no calibration target

            cal_ws = self.objective_visibility.get(cal_o, set())
            if not cal_ws:
                continue # Calibration target not visible from anywhere

            has_image = (rover, objective, mode) in current_have_image
            is_calibrated = (camera, rover) in current_calibrated

            rover_cost = 0
            temp_loc = current_loc
            best_image_w_for_rover = None # Track the waypoint that gives min cost for imaging

            if not has_image:
                # Need to take image
                min_image_path_cost = float('inf')

                for cal_w in cal_ws:
                    for image_w in image_ws:
                        path_cost = 0
                        current_path_loc = temp_loc

                        if not is_calibrated:
                            dist_to_cal = self.dist[rover][current_path_loc].get(cal_w, float('inf'))
                            if dist_to_cal == float('inf'): continue
                            path_cost += dist_to_cal
                            path_cost += 1 # calibrate
                            current_path_loc = cal_w
                        # else: current_path_loc is already temp_loc

                        dist_to_image = self.dist[rover][current_path_loc].get(image_w, float('inf'))
                        if dist_to_image == float('inf'): continue

                        path_cost += dist_to_image
                        path_cost += 1 # take_image

                        if path_cost < min_image_path_cost:
                             min_image_path_cost = path_cost
                             best_image_w_for_rover = image_w

                if min_image_path_cost == float('inf'):
                    rover_cost = float('inf') # Cannot get image with this rover/camera
                else:
                    rover_cost += min_image_path_cost
                    temp_loc = best_image_w_for_rover # Assume rover is now at the best image waypoint
            # else: already has image, rover_cost is 0, temp_loc remains current_loc

            # Now communicate
            if rover_cost != float('inf'):
                 dist_to_lander = self.dist[rover][temp_loc].get(lander_loc, float('inf'))
                 if dist_to_lander == float('inf'):
                     rover_cost = float('inf') # Cannot reach lander
                 else:
                    rover_cost += dist_to_lander
                    rover_cost += 1 # communicate_data

            min_total_cost = min(min_total_cost, rover_cost)

        return min_total_cost
