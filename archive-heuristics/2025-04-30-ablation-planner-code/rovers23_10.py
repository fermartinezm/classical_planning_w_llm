# from heuristics.heuristic_base import Heuristic # Assuming this is provided externally

import collections
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    Graph is represented as an adjacency dictionary {node: set_of_neighbors}.
    Returns a dictionary {node: distance}. Unreachable nodes have infinite distance.
    """
    # Collect all nodes mentioned in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    distances = {node: float('inf') for node in all_nodes}

    # Only proceed if the start_node is a valid node in the graph
    if start_node not in all_nodes:
        # If the start node is not in the graph (e.g., a waypoint with no can_traverse edges),
        # it's isolated. Distance to itself is 0, to others is inf.
        # Add it to the distances map.
        distances[start_node] = 0
    else:
         distances[start_node] = 0 # Set distance for the start node

    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is in graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


# Assuming Heuristic base class is provided externally, e.g., in heuristics.heuristic_base
# If Heuristic base class is NOT provided externally, define a dummy one:
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError("Heuristic subclass must implement __call__")


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions.
    It sums up the estimated cost for each unachieved goal predicate.
    The cost for each goal is estimated based on the required steps (sampling/imaging, calibration, communication)
    and includes an estimate for navigation based on precomputed shortest paths between waypoints for each rover.

    # Assumptions
    - The problem is solvable (e.g., required equipment exists, samples/objectives exist, communication points are visible).
    - Navigation costs are estimated using shortest paths in the static `can_traverse` graph for each rover.
    - The heuristic assumes that any equipped rover can be used for a task, and any suitable waypoint can be used for navigation/actions. It takes the minimum navigation cost over available rovers and suitable waypoints.
    - Taking an image requires calibration, and calibration is lost after taking an image. The heuristic accounts for this by adding a calibration cost if the image goal is not met and no relevant camera is currently calibrated.
    - Dropping a sample is needed if an equipped rover's store is full and a sample needs to be collected.

    # Heuristic Initialization
    - Parse static facts to build:
        - Navigation graphs (`can_traverse`) for each rover.
        - Shortest path distances between all waypoints for each rover using BFS.
        - Mapping of lander to its location.
        - Mapping of stores to rovers.
        - Sets of equipped rovers for each task type (soil, rock, imaging).
        - Mapping of cameras to rovers, supported modes, and calibration targets.
        - Mapping of objectives to waypoints they are visible from (`visible_from`).
        - Set of waypoints visible from the lander location (`visible`).
    - Store the goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal predicates that are *not* currently true in the state.
    2. For each unachieved goal predicate, estimate the cost to achieve it:
        - **For `(communicated_soil_data ?w)`:**
            - Base cost: 1 (communicate).
            - Add navigation cost: minimum distance from any rover's current location to any waypoint visible from the lander.
            - If `(have_soil_analysis ?r ?w)` is not true for any equipped rover `r`:
                - Add 1 (sample).
                - Add navigation cost: minimum distance from any rover's current location to waypoint `w`.
                - Check if *any* equipped rover has an empty store. If not, add 1 (drop).
        - **For `(communicated_rock_data ?w)`:**
            - Base cost: 1 (communicate).
            - Add navigation cost: minimum distance from any rover's current location to any waypoint visible from the lander.
            - If `(have_rock_analysis ?r ?w)` is not true for any equipped rover `r`:
                - Add 1 (sample).
                - Add navigation cost: minimum distance from any rover's current location to waypoint `w`.
                - Check if *any* equipped rover has an empty store. If not, add 1 (drop).
        - **For `(communicated_image_data ?o ?m)`:**
            - Base cost: 1 (communicate).
            - Add navigation cost: minimum distance from any rover's current location to any waypoint visible from the lander.
            - If `(have_image ?r ?o ?m)` is not true for any equipped rover `r` with a camera `i` supporting `m`:
                - Add 1 (take_image).
                - Find waypoints visible from `o`. Add navigation cost: minimum distance from any rover's current location to any of these waypoints.
                - Check if *any* equipped rover `r` with a camera `i` supporting `m` is currently calibrated. If not:
                    - Add 1 (calibrate).
                    - Find camera `i`'s calibration target `t`. Find waypoints visible from `t`. Add navigation cost: minimum distance from any rover's current location to any of these waypoints.
    3. The total heuristic value is the sum of the estimated costs for all unachieved goals.
    4. If any required resource (equipped rover, visible waypoint for action/communication) is missing for an unachieved goal, the goal is considered unreachable, and the heuristic returns infinity (a large number).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # self.task = task # Store task if needed later, but goals and static are extracted
        self.goals = task.goals
        static_facts = task.static

        # Data structures to store static information
        self.lander_location = None
        self.store_of = {} # store -> rover
        self.equipped_soil = set() # rovers
        self.equipped_rock = set() # rovers
        self.equipped_imaging = set() # rovers
        self.on_board = {} # camera -> rover
        self.supports = collections.defaultdict(set) # camera -> set of modes
        self.calibration_target = {} # camera -> objective
        self.visible_from = collections.defaultdict(set) # objective -> set of waypoints
        self.visible_waypoints = collections.defaultdict(set) # waypoint -> set of visible waypoints
        self.can_traverse_graph = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> waypoint -> set of reachable waypoints

        # Sets to hold all objects of each type found in static facts or goals
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_landers = set()
        self.all_stores = set()

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at_lander':
                self.lander_location = parts[2] # Assuming only one lander
                self.all_landers.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif predicate == 'store_of':
                self.store_of[parts[1]] = parts[2]
                self.all_stores.add(parts[1])
                self.all_rovers.add(parts[2])
            elif predicate == 'equipped_for_soil_analysis':
                self.equipped_soil.add(parts[1])
                self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                self.equipped_rock.add(parts[1])
                self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging':
                self.equipped_imaging.add(parts[1])
                self.all_rovers.add(parts[1])
            elif predicate == 'on_board':
                self.on_board[parts[1]] = parts[2]
                self.all_cameras.add(parts[1])
                self.all_rovers.add(parts[2])
            elif predicate == 'supports':
                self.supports[parts[1]].add(parts[2])
                self.all_cameras.add(parts[1])
                self.all_modes.add(parts[2])
            elif predicate == 'calibration_target':
                self.calibration_target[parts[1]] = parts[2]
                self.all_cameras.add(parts[1])
                self.all_objectives.add(parts[2])
            elif predicate == 'visible_from':
                self.visible_from[parts[1]].add(parts[2])
                self.all_objectives.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif predicate == 'visible':
                self.visible_waypoints[parts[1]].add(parts[2])
                self.visible_waypoints[parts[2]].add(parts[1]) # Visibility is symmetric
                self.all_waypoints.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif predicate == 'can_traverse':
                rover, w1, w2 = parts[1], parts[2], parts[3]
                self.can_traverse_graph[rover][w1].add(w2)
                self.all_rovers.add(rover)
                self.all_waypoints.add(w1)
                self.all_waypoints.add(w2)
            # Add other types/objects found in static facts (less critical for heuristic logic but good for completeness)
            # Example: '(rover rover1)', '(waypoint waypoint1)' etc.
            # The parsing above already adds objects as they appear in predicates.

        # Add any objects mentioned in goals that weren't in static facts (e.g., goal waypoints, objectives, modes)
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'communicated_soil_data':
                 self.all_waypoints.add(parts[1])
             elif predicate == 'communicated_rock_data':
                 self.all_waypoints.add(parts[1])
             elif predicate == 'communicated_image_data':
                 self.all_objectives.add(parts[1])
                 self.all_modes.add(parts[2])


        # Precompute shortest paths for each rover
        self.dist = collections.defaultdict(dict) # rover -> start_wp -> end_wp -> distance
        for rover in self.all_rovers:
            rover_graph = self.can_traverse_graph.get(rover, {})
            # Ensure all waypoints are considered as potential start/end nodes in BFS
            # Add waypoints that might not have explicit can_traverse edges in static facts
            # but exist in the problem (e.g., initial location, goal location, sample location)
            all_nodes_for_bfs = set(self.all_waypoints)
            for wp in all_nodes_for_bfs:
                 if wp not in rover_graph:
                     rover_graph[wp] = set() # Add isolated waypoints to the graph structure

            for start_wp in all_nodes_for_bfs:
                 self.dist[rover][start_wp] = bfs(rover_graph, start_wp)

        # Find communication waypoints visible from the lander
        self.comm_waypoints = set()
        if self.lander_location and self.lander_location in self.visible_waypoints:
             self.comm_waypoints = self.visible_waypoints[self.lander_location]

        # Store goal predicates by type for easier processing
        self.soil_goals = set() # waypoint
        self.rock_goals = set() # waypoint
        self.image_goals = set() # (objective, mode)

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'communicated_soil_data':
                self.soil_goals.add(parts[1])
            elif predicate == 'communicated_rock_data':
                self.rock_goals.add(parts[1])
            elif predicate == 'communicated_image_data':
                self.image_goals.add((parts[1], parts[2]))

    def get_min_dist_from_any_rover(self, state, target_waypoints):
        """
        Finds the minimum shortest distance from any rover's current location
        to any waypoint in the target_waypoints set.
        Returns float('inf') if no path exists or no rovers are present at known locations.
        """
        if not target_waypoints:
            return float('inf')

        min_d = float('inf')
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                # Ensure the object is a known rover type before adding location
                if parts[1] in self.all_rovers:
                    rover_locations[parts[1]] = parts[2]

        if not rover_locations: # No rovers currently located? Should not happen in valid states.
             return float('inf')

        for rover, current_wp in rover_locations.items():
            # Check if we have precomputed distances for this rover and its current location
            if rover in self.dist and current_wp in self.dist[rover]:
                for target_wp in target_waypoints:
                    min_d = min(min_d, self.dist[rover][current_wp].get(target_wp, float('inf')))
            # else: This rover might be at an isolated waypoint not in the can_traverse graph
            # The BFS should handle this if the waypoint was added to the graph nodes.
            # The BFS fix ensures all self.all_waypoints are in the graph nodes.

        return min_d

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if it's a goal state
        if self.goals.issubset(state):
            return 0

        total_cost = 0

        # Extract current state information needed for heuristic calculation
        current_rover_locations = {} # rover -> waypoint
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated = set() # (camera, rover)
        current_empty_stores = set() # store
        current_full_stores = set() # store
        # We don't need current_soil_samples or current_rock_samples from state
        # because the goal is about communicating *collected* samples, not sampling *available* ones.
        # The presence of (at_soil_sample ?w) is a precondition for sample_soil,
        # but the heuristic assumes solvability, meaning samples exist if needed for a goal.

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at':
                if parts[1] in self.all_rovers: # Ensure the object is a rover
                    current_rover_locations[parts[1]] = parts[2]
            elif predicate == 'have_soil_analysis':
                current_have_soil.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis':
                current_have_rock.add((parts[1], parts[2]))
            elif predicate == 'have_image':
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == 'calibrated':
                current_calibrated.add((parts[1], parts[2]))
            elif predicate == 'empty':
                current_empty_stores.add(parts[1])
            elif predicate == 'full':
                current_full_stores.add(parts[1])


        # Check reachability of communication points
        if not self.comm_waypoints:
             # If lander location is unknown or has no visible waypoints, communication is impossible
             return float('inf')

        min_dist_to_comm = self.get_min_dist_from_any_rover(state, self.comm_waypoints)
        if min_dist_to_comm == float('inf'):
             # Cannot reach any communication point from any current rover location
             return float('inf')


        # Estimate cost for soil goals
        for w in self.soil_goals:
            if f'(communicated_soil_data {w})' not in state:
                # Goal not achieved
                cost = 0

                # 1. Communication step
                cost += 1 # communicate action
                cost += min_dist_to_comm # Navigation to communication point

                # 2. Sample collection step (if needed)
                sample_needed = True
                # Check if any equipped rover already has the sample from waypoint w
                for rover in self.equipped_soil:
                    if (rover, w) in current_have_soil:
                        sample_needed = False
                        break # Found a rover that already has the sample

                if sample_needed:
                    # Need to sample
                    cost += 1 # sample_soil action

                    # Navigation to sample location
                    min_dist_to_sample = self.get_min_dist_from_any_rover(state, {w})
                    if min_dist_to_sample == float('inf'):
                         return float('inf') # Cannot reach sample location w from any rover
                    cost += min_dist_to_sample

                    # Check if drop is needed before sampling
                    # A drop is needed if *any* equipped rover needs to sample AND all equipped rovers with stores have full stores.
                    # Simplified: If there's at least one equipped rover with a store, check if *any* store belonging to an equipped rover is empty. If not, assume a drop is needed.
                    need_drop = False
                    equipped_rovers_with_stores = {r for r in self.equipped_soil if any(store in self.store_of and self.store_of[store] == r for store in self.all_stores)}

                    if equipped_rovers_with_stores:
                         has_empty_store = False
                         for store, rover in self.store_of.items():
                              if rover in equipped_rovers_with_stores and store in current_empty_stores:
                                   has_empty_store = True
                                   break
                         if not has_empty_store:
                              need_drop = True
                    else:
                         # No equipped rover has a store? This problem might be unsolvable.
                         # Assuming solvable problems, this case implies an issue with problem definition or static facts.
                         # Treat as unreachable.
                         return float('inf') # Cannot sample without a store

                    if need_drop:
                         cost += 1 # drop action
                         # Navigation for drop is implicitly covered by navigation to sample/comm

                total_cost += cost


        # Estimate cost for rock goals
        for w in self.rock_goals:
            if f'(communicated_rock_data {w})' not in state:
                # Goal not achieved
                cost = 0

                # 1. Communication step
                cost += 1 # communicate action
                cost += min_dist_to_comm # Navigation to communication point

                # 2. Sample collection step (if needed)
                sample_needed = True
                # Check if any equipped rover already has the sample from waypoint w
                for rover in self.equipped_rock:
                    if (rover, w) in current_have_rock:
                        sample_needed = False
                        break # Found a rover that already has the sample

                if sample_needed:
                    # Need to sample
                    cost += 1 # sample_rock action

                    # Navigation to sample location
                    min_dist_to_sample = self.get_min_dist_from_any_rover(state, {w})
                    if min_dist_to_sample == float('inf'):
                         return float('inf') # Cannot reach sample location w from any rover
                    cost += min_dist_to_sample

                    # Check if drop is needed before sampling
                    need_drop = False
                    equipped_rovers_with_stores = {r for r in self.equipped_rock if any(store in self.store_of and self.store_of[store] == r for store in self.all_stores)}

                    if equipped_rovers_with_stores:
                         has_empty_store = False
                         for store, rover in self.store_of.items():
                              if rover in equipped_rovers_with_stores and store in current_empty_stores:
                                   has_empty_store = True
                                   break
                         if not has_empty_store:
                              need_drop = True
                    else:
                         return float('inf') # Cannot sample without a store

                    if need_drop:
                         cost += 1 # drop action
                         # Navigation for drop is implicitly covered

                total_cost += cost


        # Estimate cost for image goals
        for o, m in self.image_goals:
            if f'(communicated_image_data {o} {m})' not in state:
                # Goal not achieved
                cost = 0

                # 1. Communication step
                cost += 1 # communicate action
                cost += min_dist_to_comm # Navigation to communication point

                # 2. Image capture step (if needed)
                image_needed = True
                # Check if any equipped rover with a supporting camera already has the image
                for rover in self.equipped_imaging:
                    # Find cameras on this rover supporting the mode
                    cameras_on_rover = {cam for cam, r in self.on_board.items() if r == rover}
                    supporting_cameras = {cam for cam in cameras_on_rover if m in self.supports.get(cam, set())}
                    if any((rover, o, m) in current_have_image for cam in supporting_cameras):
                         image_needed = False
                         break # Found a rover/camera that already has the image

                if image_needed:
                    # Need to take image
                    cost += 1 # take_image action

                    # Find suitable rovers/cameras (equipped for imaging, camera on board, supports mode)
                    suitable_rovers_cameras = [] # list of (rover, camera)
                    for rover in self.equipped_imaging:
                         cameras_on_rover = {cam for cam, r in self.on_board.items() if r == rover}
                         for cam in cameras_on_rover:
                              if m in self.supports.get(cam, set()):
                                   suitable_rovers_cameras.append((rover, cam))

                    if not suitable_rovers_cameras:
                         return float('inf') # No suitable rover/camera combination exists

                    # Navigation to image location
                    img_waypoints = self.visible_from.get(o, set())
                    if not img_waypoints:
                         return float('inf') # Objective not visible from anywhere
                    min_dist_to_img_wp = self.get_min_dist_from_any_rover(state, img_waypoints)
                    if min_dist_to_img_wp == float('inf'):
                         return float('inf') # Cannot reach any image waypoint from any rover
                    cost += min_dist_to_img_wp

                    # Check if calibration is needed before taking image
                    # Need *a* suitable rover/camera that is currently calibrated.
                    calibration_needed = True
                    for rover, cam in suitable_rovers_cameras:
                         if (cam, rover) in current_calibrated:
                              calibration_needed = False
                              break # Found a calibrated camera on a suitable rover

                    if calibration_needed:
                         cost += 1 # calibrate action

                         # Navigation to calibration location
                         # Find calibration target for *any* suitable camera
                         cal_targets = {self.calibration_target[cam] for rover, cam in suitable_rovers_cameras if cam in self.calibration_target}
                         if not cal_targets:
                              # This means no suitable camera has a calibration target defined.
                              # Assuming solvable problems, this shouldn't happen if calibration is needed.
                              return float('inf') # No calibration target for suitable cameras

                         # Find waypoints visible from *any* of these calibration targets
                         cal_waypoints = set()
                         for target in cal_targets:
                              cal_waypoints.update(self.visible_from.get(target, set()))

                         if not cal_waypoints:
                              # Calibration target(s) not visible from anywhere.
                              return float('inf') # Calibration target not visible from anywhere
                         min_dist_to_cal_wp = self.get_min_dist_from_any_rover(state, cal_waypoints)
                         if min_dist_to_cal_wp == float('inf'):
                              return float('inf') # Cannot reach any calibration waypoint from any rover
                         cost += min_dist_to_cal_wp

                total_cost += cost

        return total_cost
