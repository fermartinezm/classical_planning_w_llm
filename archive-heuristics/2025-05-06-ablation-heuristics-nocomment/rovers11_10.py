from fnmatch import fnmatch
from collections import deque

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Removes surrounding parentheses from a PDDL fact string and splits it into parts."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if the parts of a PDDL fact string match a sequence of pattern arguments."""
    parts = get_parts(fact)
    # Ensure the fact has at least as many parts as the arguments provided
    if len(parts) < len(args):
        return False
    # Use fnmatch for flexible matching (e.g., '*' wildcard)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming a base class like 'Heuristic' exists with __init__(self, task) and __call__(self, node)
# If a specific base class is required, this class should inherit from it.
# For this response, we provide the class definition with the expected methods.

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        estimated costs for each unachieved goal fact. For each unachieved
        goal, it finds the minimum cost among all capable rovers to perform
        the necessary sequence of actions (navigate, sample/image, communicate).
        Navigation costs are estimated using precomputed shortest path distances
        on the rover-specific traversability graphs. Other actions (sample,
        drop, calibrate, take_image, communicate) are assigned a cost of 1.
        The heuristic is non-admissible as it sums costs for potentially
        interdependent goals and simplifies action preconditions/effects
        (e.g., simplified calibration cost, simplified store management).

    Assumptions:
        - The PDDL task and state are provided in the expected format (frozensets of strings).
        - The 'Heuristic' base class (if used) provides the expected structure (__init__, __call__).
        - Rover capabilities, camera properties, objective visibility, and
          waypoint traversability are static and provided in task.static.
        - Lander location(s) are static.
        - Samples at waypoints are consumed upon sampling.
        - Stores are either empty or full and belong to a specific rover.
        - Cameras are on board specific rovers and have calibration targets.
        - The waypoint graph for each rover is connected enough to reach relevant locations (or unreachable goals are penalized).

    Heuristic Initialization:
        - Parses static facts to build data structures:
            - Lander location(s).
            - Rover capabilities (equipped for soil, rock, imaging).
            - Store ownership.
            - Camera properties (on board, supported modes, calibration target).
            - Objective visibility from waypoints.
            - Rover-specific waypoint traversability graphs.
        - Precomputes all-pairs shortest path distances for each rover's
          traversability graph using BFS.
        - Identifies waypoints visible from lander location(s).

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize heuristic value `h = 0`.
        2. Parse the current state to determine:
           - Current location of each rover.
           - Locations of remaining soil/rock samples.
           - Data currently held by each rover (soil, rock, image).
           - Status of each store (empty/full).
           - Calibration status of each camera.
           - Which goal facts have already been communicated.
        3. Identify the set of waypoints visible from any lander location (`lander_visible_wps`).
        4. Iterate through each goal fact specified in `task.goals`:
           a. If the goal fact is already present in the current state (i.e., already communicated), continue to the next goal.
           b. If the goal is `(communicated_soil_data ?w)`:
              - Find the minimum estimated cost among all soil-equipped rovers to achieve this goal.
              - For a rover `r`:
                - If `r` already has `(have_soil_analysis r ?w)`:
                  - Cost is navigation from `r`'s current location to the nearest `lander_visible_wp` + 1 (for communicate action).
                - If `r` does NOT have the data and `(at_soil_sample ?w)` is still in the state:
                  - Cost is navigation from `r`'s current location to `?w` + (1 if store is full for drop action) + 1 (for sample action) + navigation from `?w` to the nearest `lander_visible_wp` + 1 (for communicate action).
                - If `r` does NOT have the data and `(at_soil_sample ?w)` is NOT in the state (sample taken by someone else or gone): This rover cannot sample. Skip this rover for this goal.
              - Take the minimum cost over all capable rovers. If no rover can achieve it (e.g., unreachable waypoints), assign a large penalty.
           c. If the goal is `(communicated_rock_data ?w)`:
              - Similar logic as soil data, using rock-specific predicates and actions.
           d. If the goal is `(communicated_image_data ?o ?m)`:
              - Find the minimum estimated cost among all imaging-equipped rovers with a suitable camera to achieve this goal.
              - For a rover `r` and suitable camera `i`:
                - If `r` already has `(have_image r ?o ?m)`:
                  - Cost is navigation from `r`'s current location to the nearest `lander_visible_wp` + 1 (for communicate action).
                - If `r` does NOT have the image:
                  - Cost involves navigation to a waypoint visible from `?o` (`img_wps`), calibration if needed, taking the image, and communication.
                  - If camera `i` is NOT calibrated:
                    - Cost is navigation from `r`'s current location to the nearest waypoint visible from `i`'s calibration target (`cal_wps`) + 1 (for calibrate action) + navigation from a `cal_wps` to a `img_wps` + 1 (for take_image action) + navigation from a `img_wps` to a `lander_visible_wp` + 1 (for communicate action).
                  - If camera `i` IS calibrated:
                    - Cost is navigation from `r`'s current location to the nearest `img_wps` + 1 (for take_image action) + navigation from a `img_wps` to a `lander_visible_wp` + 1 (for communicate action).
                  - Handle cases where necessary waypoints (calibration, image, lander) are unreachable for the rover.
              - Take the minimum cost over all capable rover/camera combinations. If no combination can achieve it, assign a large penalty.
           e. Add the minimum estimated cost for the current goal (or penalty) to `h`.
        5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Data structures for static information
        self.lander_location = {} # {lander: waypoint}
        self.equipped_soil = set() # {rover}
        self.equipped_rock = set() # {rover}
        self.equipped_imaging = set() # {rover}
        self.store_owner = {} # {store: rover}
        self.rover_cameras = {} # {rover: {camera}}
        self.camera_modes = {} # {camera: {mode}}
        self.camera_calibration_target = {} # {camera: objective}
        self.waypoint_graph = {} # {rover: {waypoint: {waypoint}}} (adjacency list)
        self.objective_visible_wps = {} # {objective: {waypoint}}
        self.lander_visible_wps = set() # {waypoint} visible from lander

        # Collect all relevant objects/types mentioned in static facts
        all_rovers = set()
        all_waypoints = set()
        all_objectives = set()
        all_cameras = set()
        all_modes = set()
        all_stores = set()
        all_landers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'at_lander' and len(parts) == 3:
                lander, wp = parts[1], parts[2]
                self.lander_location[lander] = wp
                all_landers.add(lander)
                all_waypoints.add(wp)
            elif predicate == 'equipped_for_soil_analysis' and len(parts) == 2:
                self.equipped_soil.add(parts[1])
                all_rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis' and len(parts) == 2:
                self.equipped_rock.add(parts[1])
                all_rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging' and len(parts) == 2:
                self.equipped_imaging.add(parts[1])
                all_rovers.add(parts[1])
            elif predicate == 'store_of' and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.store_owner[store] = rover
                all_stores.add(store)
                all_rovers.add(rover)
            elif predicate == 'on_board' and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
                all_cameras.add(camera)
                all_rovers.add(rover)
            elif predicate == 'supports' and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
                all_cameras.add(camera)
                all_modes.add(mode)
            elif predicate == 'calibration_target' and len(parts) == 3:
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective
                all_cameras.add(camera)
                all_objectives.add(objective)
            elif predicate == 'visible_from' and len(parts) == 3:
                objective, wp = parts[1], parts[2]
                self.objective_visible_wps.setdefault(objective, set()).add(wp)
                all_objectives.add(objective)
                all_waypoints.add(wp)
            elif predicate == 'can_traverse' and len(parts) == 4:
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.waypoint_graph.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
                all_rovers.add(rover)
                all_waypoints.add(wp1)
                all_waypoints.add(wp2)
            elif predicate == 'visible' and len(parts) == 3:
                 # Visible facts define the base graph, but can_traverse is rover specific.
                 # We only need visible to find lander_visible_wps
                 all_waypoints.add(parts[1])
                 all_waypoints.add(parts[2])

        # Ensure all rovers/waypoints mentioned are in the graph structure keys for BFS
        for rover in all_rovers:
            self.waypoint_graph.setdefault(rover, {})
            for wp in all_waypoints:
                 self.waypoint_graph[rover].setdefault(wp, set())

        # Find waypoints visible from lander location(s)
        lander_locations = set(self.lander_location.values())
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2]
                if wp1 in lander_locations:
                    self.lander_visible_wps.add(wp2)
                if wp2 in lander_locations:
                    self.lander_visible_wps.add(wp1)
        # Lander location itself is visible from itself
        self.lander_visible_wps.update(lander_locations)


        # Precompute shortest paths for each rover
        self.dist = {} # {rover: {start_wp: {end_wp: distance}}}
        for rover in all_rovers:
            self.dist[rover] = {}
            graph = self.waypoint_graph.get(rover, {})
            # Ensure all waypoints are keys in the graph dict for BFS
            for wp in all_waypoints:
                 graph.setdefault(wp, set())

            for start_wp in graph:
                self.dist[rover][start_wp] = self._bfs(graph, start_wp)

        # Map rover to its store for quick lookup
        self._rover_to_store = {v: k for k, v in self.store_owner.items()}


    def _bfs(self, graph, start_node):
        """Helper to perform BFS and return distances."""
        distances = {node: float('inf') for node in graph}
        if start_node not in graph:
             # Start node is not in the graph (e.g., rover at a waypoint not in any can_traverse)
             # This shouldn't happen in valid problems, but handle defensively.
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def _min_dist(self, rover, start_wps, target_wps):
        """Helper to find min distance from a set of start wps to a set of target wps for a rover."""
        min_d = float('inf')
        # Ensure start_wps is iterable
        if isinstance(start_wps, str):
            start_wps = {start_wps}
        if isinstance(target_wps, str):
             target_wps = {target_wps}

        # Handle cases where start or target sets are empty
        if not start_wps or not target_wps:
             return float('inf')

        for sw in start_wps:
            if rover in self.dist and sw in self.dist[rover]:
                for tw in target_wps:
                    min_d = min(min_d, self.dist[rover][sw].get(tw, float('inf')))
        return min_d

    def __call__(self, node):
        state = node.state

        # Parse current state facts
        rover_locations = {} # {rover: waypoint}
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}
        rover_soil_data = set() # {(rover, waypoint)}
        rover_rock_data = set() # {(rover, waypoint)}
        rover_image_data = set() # {(rover, objective, mode)}
        empty_stores = set() # {store}
        calibrated_cameras = set() # {(camera, rover)}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3 and parts[1].startswith('rover'): # Assuming only rovers are 'at' waypoints in state
                rover_locations[parts[1]] = parts[2]
            elif predicate == 'at_soil_sample' and len(parts) == 2:
                soil_samples_at.add(parts[1])
            elif predicate == 'at_rock_sample' and len(parts) == 2:
                rock_samples_at.add(parts[1])
            elif predicate == 'have_soil_analysis' and len(parts) == 3:
                rover_soil_data.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis' and len(parts) == 3:
                rover_rock_data.add((parts[1], parts[2]))
            elif predicate == 'have_image' and len(parts) == 4:
                rover_image_data.add((parts[1], parts[2], parts[3]))
            elif predicate == 'empty' and len(parts) == 2:
                empty_stores.add(parts[1])
            elif predicate == 'calibrated' and len(parts) == 3:
                calibrated_cameras.add((parts[1], parts[2]))
            elif predicate == 'communicated_soil_data' and len(parts) == 2:
                communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data' and len(parts) == 3:
                communicated_image.add((parts[1], parts[2]))

        # Calculate heuristic
        h = 0
        UNREACHABLE_PENALTY = 1000 # Penalty for goals that seem unreachable

        for goal in self.goals:
            goal_parts = get_parts(goal)
            if not goal_parts: continue # Skip empty goals

            goal_predicate = goal_parts[0]

            if goal_predicate == 'communicated_soil_data' and len(goal_parts) == 2:
                wp = goal_parts[1]
                if wp in communicated_soil:
                    continue # Goal already achieved

                min_goal_cost = float('inf')

                for r in self.equipped_soil:
                    curr_wp = rover_locations.get(r)
                    if curr_wp is None: continue # Rover not in state? Should not happen.

                    cost = 0
                    needs_sample = (r, wp) not in rover_soil_data
                    can_sample = needs_sample and wp in soil_samples_at

                    if needs_sample and not can_sample:
                        # Sample is gone from waypoint, and this rover doesn't have it.
                        # This path for this rover is blocked for this goal.
                        continue

                    if needs_sample:
                        # Need to navigate to sample location, sample, then navigate to lander
                        nav_to_sample = self._min_dist(r, curr_wp, {wp})
                        if nav_to_sample == float('inf'): continue # Cannot reach sample wp

                        cost += nav_to_sample

                        # Check store status
                        rover_store = self._rover_to_store.get(r)
                        if rover_store and rover_store not in empty_stores:
                            cost += 1 # Drop action needed

                        cost += 1 # Sample action

                        # Now rover is at wp with data (estimated location)
                        nav_to_communicate = self._min_dist(r, {wp}, self.lander_visible_wps)
                        if nav_to_communicate == float('inf'): continue # Cannot reach lander wp from sample wp

                        cost += nav_to_communicate
                        cost += 1 # Communicate action

                    else: # Rover already has (have_soil_analysis r wp)
                        # Need to navigate to lander location
                        nav_to_communicate = self._min_dist(r, curr_wp, self.lander_visible_wps)
                        if nav_to_communicate == float('inf'): continue # Cannot reach lander wp

                        cost += nav_to_communicate
                        cost += 1 # Communicate action

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == float('inf'):
                    h += UNREACHABLE_PENALTY
                else:
                    h += min_goal_cost

            elif goal_predicate == 'communicated_rock_data' and len(goal_parts) == 2:
                wp = goal_parts[1]
                if wp in communicated_rock:
                    continue # Goal already achieved

                min_goal_cost = float('inf')

                for r in self.equipped_rock:
                    curr_wp = rover_locations.get(r)
                    if curr_wp is None: continue

                    cost = 0
                    needs_sample = (r, wp) not in rover_rock_data
                    can_sample = needs_sample and wp in rock_samples_at

                    if needs_sample and not can_sample:
                         continue # Sample gone, rover doesn't have it

                    if needs_sample:
                        nav_to_sample = self._min_dist(r, curr_wp, {wp})
                        if nav_to_sample == float('inf'): continue

                        cost += nav_to_sample

                        rover_store = self._rover_to_store.get(r)
                        if rover_store and rover_store not in empty_stores:
                            cost += 1 # Drop action needed

                        cost += 1 # Sample action

                        nav_to_communicate = self._min_dist(r, {wp}, self.lander_visible_wps)
                        if nav_to_communicate == float('inf'): continue

                        cost += nav_to_communicate
                        cost += 1 # Communicate action

                    else: # Rover already has (have_rock_analysis r wp)
                        nav_to_communicate = self._min_dist(r, curr_wp, self.lander_visible_wps)
                        if nav_to_communicate == float('inf'): continue

                        cost += nav_to_communicate
                        cost += 1 # Communicate action

                    min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == float('inf'):
                    h += UNREACHABLE_PENALTY
                else:
                    h += min_goal_cost

            elif goal_predicate == 'communicated_image_data' and len(goal_parts) == 3:
                obj, mode = goal_parts[1], goal_parts[2]
                if (obj, mode) in communicated_image:
                    continue # Goal already achieved

                min_goal_cost = float('inf')

                for r in self.equipped_imaging:
                    curr_wp = rover_locations.get(r)
                    if curr_wp is None: continue

                    suitable_cameras = {i for i in self.rover_cameras.get(r, set()) if mode in self.camera_modes.get(i, set())}

                    for i in suitable_cameras:
                        cost = 0
                        needs_image = (r, obj, mode) not in rover_image_data

                        if needs_image:
                            img_wps = self.objective_visible_wps.get(obj, set())
                            if not img_wps: continue # Cannot image this objective

                            needs_calibration = (i, r) not in calibrated_cameras

                            if needs_calibration:
                                cal_target = self.camera_calibration_target.get(i)
                                if cal_target is None: continue # Camera cannot be calibrated

                                cal_wps = self.objective_visible_wps.get(cal_target, set())
                                if not cal_wps: continue # Cannot calibrate this camera

                                # Nav to cal wp
                                nav1 = self._min_dist(r, curr_wp, cal_wps)
                                if nav1 == float('inf'): continue
                                cost += nav1 + 1 # +1 for calibrate action

                                # Nav from cal wp to image wp
                                nav2 = self._min_dist(r, cal_wps, img_wps)
                                if nav2 == float('inf'): continue
                                cost += nav2

                            else: # Already calibrated
                                # Nav to image wp
                                nav1 = self._min_dist(r, curr_wp, img_wps)
                                if nav1 == float('inf'): continue
                                cost += nav1

                            cost += 1 # Take image action

                            # Now at an image waypoint with data (estimated location)
                            nav_to_communicate = self._min_dist(r, img_wps, self.lander_visible_wps)
                            if nav_to_communicate == float('inf'): continue

                            cost += nav_to_communicate
                            cost += 1 # Communicate action

                        else: # Rover already has (have_image r obj mode)
                            # Need to navigate to lander location
                            nav_to_communicate = self._min_dist(r, curr_wp, self.lander_visible_wps)
                            if nav_to_communicate == float('inf'): continue

                            cost += nav_to_communicate
                            cost += 1 # Communicate action

                        min_goal_cost = min(min_goal_cost, cost)

                if min_goal_cost == float('inf'):
                    h += UNREACHABLE_PENALTY
                else:
                    h += min_goal_cost
            # Note: Other goal predicates are not expected based on the domain file.

        return h
