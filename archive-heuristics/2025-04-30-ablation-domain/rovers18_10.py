import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact strings gracefully
    if not fact or fact.strip() == '()':
        return []
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts is at least the number of args for zip
    if len(parts) < len(args):
         return False
    # Use zip to handle cases where parts might be longer than args (e.g., extra arguments in fact)
    # fnmatch handles '*' correctly.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function for shortest path calculation
def bfs(graph, start):
    """
    Performs Breadth-First Search on a graph to find shortest distances
    from a start node to all other reachable nodes.

    Args:
        graph: A dictionary representing the graph where keys are nodes
               and values are sets of neighbor nodes.
        start: The starting node for the BFS.

    Returns:
        A dictionary mapping each reachable node to its shortest distance
        from the start node. Unreachable nodes have distance infinity.
    """
    distances = {node: float('inf') for node in graph}
    if start in graph:
        distances[start] = 0
        queue = collections.deque([start])

        while queue:
            current = queue.popleft()
            # Ensure current node is in graph keys before accessing neighbors
            if current in graph:
                for neighbor in graph[current]:
                    if distances.get(neighbor, float('inf')) == float('inf'): # Use get with default for safety
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
    # Ensure all nodes in the graph have an entry in distances, even if unreachable from start
    for node in graph:
        distances.setdefault(node, float('inf'))

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal
    conditions. It sums the estimated cost for each unsatisfied goal independently.
    The cost for each goal includes the necessary actions (take sample/image,
    calibrate, communicate) and an estimate of the navigation cost. Navigation
    cost is estimated based on the minimum number of steps required to move
    between relevant waypoints (sample/image/calibration locations and the
    lander location) using precomputed shortest paths for each rover. The
    minimum cost across all capable rovers is used for each goal.

    # Assumptions
    - Each goal (communicating a specific piece of data) can be pursued independently.
    - Negative effects (like dropping samples, consuming images) are ignored for simplicity
      when estimating the *minimum* steps to achieve a goal predicate like `have_`.
      The heuristic only checks if `have_` is true or false, not how many times it
      might need to be re-achieved.
    - The analysis actions (`analyse_soil`, `analyse_rock`) are not required goal
      predicates for communication in the goal definition and are not explicitly
      modeled in the heuristic cost beyond achieving the `have_` predicate
      (which is an effect of `take_` actions). Communication actions directly
      require the `have_` predicates and a full store for samples. The heuristic
      simplifies sample communication by requiring `have_soil_analysis` or
      `have_rock_analysis` and a full store, and implicitly assumes the sample is
      in the store if `have_` is true (or can be retaken if needed). An extra cost
      is added if the rover's store is currently full when a sample needs to be taken.
    - Solvable instances have reachable required waypoints (soil samples, rock samples,
      image/calibration visibility waypoints) from rover starting locations and
      lander locations via the `can_traverse` predicates for at least one capable rover.
      Unreachable waypoints result in infinite distance costs, making the goal cost infinite.
    - Action costs are uniform (1).

    # Heuristic Initialization
    The heuristic precomputes static information from the task:
    - The location of the lander.
    - Which equipment each rover has (`equipped_for_soil_analysis`, etc.).
    - Which store belongs to which rover (`store_of`).
    - Information about cameras (`on_board`, `supports`, `calibration_target`):
      which rover they are on, which modes they support, and their calibration target.
    - Which objectives are visible from which waypoints (`visible_from`).
    - The traversal graph for each rover based on `can_traverse` predicates.
    - All-pairs shortest paths (minimum navigation steps) between all waypoints
      for each rover using BFS on their traversal graph. This is stored in
      `self.min_dist[rover][from_wp][to_wp]`.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Check if the state is a goal state by verifying if all goal predicates
        are present in the state facts. If yes, return 0.
    2.  Initialize `total_cost` to 0.
    3.  Extract relevant dynamic information from the current state:
        - Current location of each rover (`at`).
        - Which samples/images/calibrations the rovers `have_`.
        - Which cameras are `calibrated`.
        - Which rover stores are `full`.
        - Which waypoints have `at_soil_sample` or `at_rock_sample`.
    4.  Iterate through each goal predicate defined in the task.
    5.  If a goal predicate is already true in the current state, skip it.
    6.  For each unsatisfied goal predicate:
        a.  Determine the type of goal (soil, rock, or image communication)
            and its parameters (waypoint, objective, mode).
        b.  Find the lander location (communication point). If lander location
            is unknown, the goal is impossible (infinite cost).
        c.  Initialize `min_goal_cost` for this specific goal instance to infinity.
        d.  Iterate through all rovers that are capable of achieving this type of goal
            (e.g., equipped for soil analysis for a soil goal).
        e.  For each capable rover, calculate the estimated cost to achieve this
            specific goal instance:
            i.  Start with a base cost of 1 for the final communication action.
            ii. If the required `have_` predicate (`have_soil_analysis`,
                `have_rock_analysis`, or `have_image`) is not true for this rover:
                - Add 1 for the `take_` action (take sample or take image).
                - If it's a sample goal and the rover's store is currently full,
                  add 1 for a `drop_sample` action needed before taking the new sample.
                - If it's an image goal and the camera needed is not calibrated:
                    - Add 1 for the `calibrate` action.
                    - Add the minimum navigation cost from the rover's current
                      location to a suitable calibration waypoint, plus the minimum
                      navigation cost from the rover's current location to a
                      suitable image waypoint, plus the minimum navigation cost
                      from the rover's current location to the lander location.
                      (Navigation cost is the sum of shortest path distances from
                      the rover's current location to *any* waypoint of the required
                      type).
                - If it's an image goal and the camera is calibrated but the image
                  is not taken:
                    - Add the minimum navigation cost from the rover's current
                      location to a suitable image waypoint, plus the minimum
                      navigation cost from the rover's current location to the
                      lander location.
                - If it's a sample goal and the sample is not taken:
                    - Add the minimum navigation cost from the rover's current
                      location to the sample waypoint, plus the minimum navigation
                      cost from the sample waypoint to the lander location.
                    - (Navigation cost is the sum of shortest path distances for
                      the sequential path: rover's current location -> sample waypoint -> lander location).
            iii. If the required `have_` predicate is true for this rover:
                - Add the minimum navigation cost from the rover's current location
                  to the lander location.
            iv. If any required waypoint is unreachable by the rover, the navigation
                cost will be infinite, making the rover's cost for this goal infinite.
            v.  Update `min_goal_cost` with the minimum of the current `min_goal_cost`
                and the calculated cost for this rover.
        f.  Add the final `min_goal_cost` for this goal instance to the `total_cost`.
    7.  If the `total_cost` is infinite (meaning at least one unsatisfied goal
        is unreachable by any capable rover), return infinity. Otherwise, return
        the calculated `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing
        waypoint distances.
        """
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        self.rover_equipment = collections.defaultdict(set)
        self.rover_store = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': r, 'modes': set(), 'target': t}
        self.objective_visibility = collections.defaultdict(set) # objective -> set of waypoints
        self.rover_traversal_graph = collections.defaultdict(lambda: collections.defaultdict(set))
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()

        # 1. Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]

            if predicate == 'at_lander':
                # (at_lander lander waypoint)
                if len(parts) > 2:
                    self.lander_location = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                # (equipped_for_soil_analysis rover)
                if len(parts) > 1:
                    self.rover_equipment[parts[1]].add('soil')
                    self.rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                # (equipped_for_rock_analysis rover)
                if len(parts) > 1:
                    self.rover_equipment[parts[1]].add('rock')
                    self.rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging':
                # (equipped_for_imaging rover)
                if len(parts) > 1:
                    self.rover_equipment[parts[1]].add('imaging')
                    self.rovers.add(parts[1])
            elif predicate == 'store_of':
                # (store_of store rover)
                if len(parts) > 2:
                    self.rover_store[parts[2]] = parts[1] # Map rover to store
            elif predicate == 'on_board':
                # (on_board camera rover)
                if len(parts) > 2:
                    camera, rover = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).update({'rover': rover})
                    self.cameras.add(camera)
                    self.rovers.add(rover)
            elif predicate == 'supports':
                # (supports camera mode)
                if len(parts) > 2:
                    camera, mode = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
                    self.modes.add(mode)
                    self.cameras.add(camera)
            elif predicate == 'calibration_target':
                # (calibration_target camera objective)
                if len(parts) > 2:
                    camera, objective = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).update({'target': objective})
                    self.cameras.add(camera)
                    self.objectives.add(objective)
            elif predicate == 'visible_from':
                # (visible_from objective waypoint)
                if len(parts) > 2:
                    objective, waypoint = parts[1], parts[2]
                    self.objective_visibility[objective].add(waypoint)
                    self.objectives.add(objective)
                    self.waypoints.add(waypoint)
            elif predicate == 'can_traverse':
                # (can_traverse rover waypoint1 waypoint2)
                if len(parts) > 3:
                    rover, wp1, wp2 = parts[1], parts[2], parts[3]
                    self.rover_traversal_graph[rover][wp1].add(wp2)
                    self.rovers.add(rover)
                    self.waypoints.add(wp1)
                    self.waypoints.add(wp2)
            # Collect all waypoints mentioned in static facts
            elif predicate in ('visible', 'at_rock_sample', 'at_soil_sample'):
                 if len(parts) > 1:
                     if predicate in ('at_rock_sample', 'at_soil_sample'):
                         self.waypoints.add(parts[1])
                     elif predicate == 'visible' and len(parts) > 2:
                         self.waypoints.add(parts[1])
                         self.waypoints.add(parts[2])
            # Collect all objects by type mentioned in static facts
            # This helps ensure all relevant objects are considered even if not in specific predicates
            elif len(parts) > 1:
                 obj_name = parts[1]
                 # Check object type based on common naming conventions or domain knowledge
                 if obj_name.startswith('rover'): self.rovers.add(obj_name)
                 elif obj_name.startswith('waypoint'): self.waypoints.add(obj_name)
                 elif obj_name.startswith('camera'): self.cameras.add(obj_name)
                 elif obj_name.startswith('objective'): self.objectives.add(obj_name)
                 elif obj_name.startswith('mode'): self.modes.add(obj_name)
                 # lander and store objects are typically identified via specific predicates


        # Ensure all known waypoints are included in graph nodes for BFS
        for rover in self.rovers:
             self.rover_traversal_graph.setdefault(rover, collections.defaultdict(set))
             for wp in self.waypoints:
                 self.rover_traversal_graph[rover].setdefault(wp, set())


        # 2. Compute all-pairs shortest paths for each rover
        self.min_dist = {}
        for rover in self.rovers:
            self.min_dist[rover] = {}
            graph = self.rover_traversal_graph.get(rover, {})
            # Ensure all waypoints are keys in the graph dict, even if they have no edges
            for wp in self.waypoints:
                 graph.setdefault(wp, set())

            for start_wp in graph:
                self.min_dist[rover][start_wp] = bfs(graph, start_wp)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        total_cost = 0
        current_rover_locs = {}
        current_have_soil = {} # (rover, wp) -> True
        current_have_rock = {} # (rover, wp) -> True
        current_have_image = {} # (rover, obj, mode) -> True
        current_calibrated = {} # (camera, rover) -> True
        current_full_stores = set() # store -> True
        current_soil_samples = set() # waypoint -> True
        current_rock_samples = set() # waypoint -> True


        # Extract relevant state information
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at':
                if len(parts) > 2: current_rover_locs[parts[1]] = parts[2]
            elif predicate == 'have_soil_analysis':
                if len(parts) > 2: current_have_soil[(parts[1], parts[2])] = True
            elif predicate == 'have_rock_analysis':
                if len(parts) > 2: current_have_rock[(parts[1], parts[2])] = True
            elif predicate == 'have_image':
                if len(parts) > 3: current_have_image[(parts[1], parts[2], parts[3])] = True
            elif predicate == 'calibrated':
                if len(parts) > 2: current_calibrated[(parts[1], parts[2])] = True
            elif predicate == 'full':
                 if len(parts) > 1: current_full_stores.add(parts[1])
            elif predicate == 'at_soil_sample':
                 if len(parts) > 1: current_soil_samples.add(parts[1])
            elif predicate == 'at_rock_sample':
                 if len(parts) > 1: current_rock_samples.add(parts[1])


        lander_loc = self.lander_location
        # If lander location is unknown but communication goals exist, they are impossible
        lander_missing = (lander_loc is None)

        # Iterate through goals and sum costs for unsatisfied ones
        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                # (communicated_soil_data waypoint)
                if len(parts) < 2: continue # Malformed goal
                sample_wp = parts[1]

                if lander_missing:
                    min_goal_cost = float('inf')
                # Check if sample exists at waypoint
                elif sample_wp not in current_soil_samples:
                    # Cannot achieve this goal if sample is gone from its original location
                    min_goal_cost = float('inf')
                else:
                    min_goal_cost = float('inf')
                    # Find best rover
                    for rover in self.rovers:
                        if 'soil' in self.rover_equipment.get(rover, set()):
                            r_loc = current_rover_locs.get(rover)
                            if r_loc is None: continue # Rover location unknown or rover doesn't exist

                            cost = 1 # communicate action

                            # Check if sample analysis is already done by this rover
                            have_soil = current_have_soil.get((rover, sample_wp), False)

                            if not have_soil:
                                cost += 1 # take_soil_sample action
                                # Nav cost: r_loc -> sample_wp -> lander_loc
                                dist_r_to_sample = self.min_dist.get(rover, {}).get(r_loc, {}).get(sample_wp, float('inf'))
                                dist_sample_to_lander = self.min_dist.get(rover, {}).get(sample_wp, {}).get(lander_loc, float('inf'))

                                # If either leg is unreachable, this path is infinite
                                if dist_r_to_sample == float('inf') or dist_sample_to_lander == float('inf'):
                                     nav_cost = float('inf')
                                else:
                                     nav_cost = dist_r_to_sample + dist_sample_to_lander

                                cost += nav_cost

                                # Check if store is full - might need drop action before taking sample
                                rover_store = self.rover_store.get(rover)
                                if rover_store and rover_store in current_full_stores:
                                     cost += 1 # drop_sample action

                            else: # have_soil is true
                                # Nav cost: r_loc -> lander_loc
                                dist_r_to_lander = self.min_dist.get(rover, {}).get(r_loc, {}).get(lander_loc, float('inf'))
                                cost += dist_r_to_lander

                            min_goal_cost = min(min_goal_cost, cost)

                total_cost += min_goal_cost

            elif predicate == 'communicated_rock_data':
                # (communicated_rock_data waypoint)
                if len(parts) < 2: continue # Malformed goal
                sample_wp = parts[1]

                if lander_missing:
                    min_goal_cost = float('inf')
                # Check if sample exists at waypoint
                elif sample_wp not in current_rock_samples:
                    # Cannot achieve this goal if sample is gone from its original location
                    min_goal_cost = float('inf')
                else:
                    min_goal_cost = float('inf')
                    # Find best rover
                    for rover in self.rovers:
                        if 'rock' in self.rover_equipment.get(rover, set()):
                            r_loc = current_rover_locs.get(rover)
                            if r_loc is None: continue # Rover location unknown or rover doesn't exist

                            cost = 1 # communicate action

                            # Check if sample analysis is already done by this rover
                            have_rock = current_have_rock.get((rover, sample_wp), False)

                            if not have_rock:
                                cost += 1 # take_rock_sample action
                                # Nav cost: r_loc -> sample_wp -> lander_loc
                                dist_r_to_sample = self.min_dist.get(rover, {}).get(r_loc, {}).get(sample_wp, float('inf'))
                                dist_sample_to_lander = self.min_dist.get(rover, {}).get(sample_wp, {}).get(lander_loc, float('inf'))

                                # If either leg is unreachable, this path is infinite
                                if dist_r_to_sample == float('inf') or dist_sample_to_lander == float('inf'):
                                     nav_cost = float('inf')
                                else:
                                     nav_cost = dist_r_to_sample + dist_sample_to_lander

                                cost += nav_cost

                                # Check if store is full - might need drop action before taking sample
                                rover_store = self.rover_store.get(rover)
                                if rover_store and rover_store in current_full_stores:
                                     cost += 1 # drop_sample action

                            else: # have_rock is true
                                # Nav cost: r_loc -> lander_loc
                                dist_r_to_lander = self.min_dist.get(rover, {}).get(r_loc, {}).get(lander_loc, float('inf'))
                                cost += dist_r_to_lander

                            min_goal_cost = min(min_goal_cost, cost)

                total_cost += min_goal_cost

            elif predicate == 'communicated_image_data':
                # (communicated_image_data objective mode)
                if len(parts) < 3: continue # Malformed goal
                objective, mode = parts[1], parts[2]

                if lander_missing:
                    min_goal_cost = float('inf')
                else:
                    min_goal_cost = float('inf')

                    # Find waypoints where objective is visible
                    image_wps = self.objective_visibility.get(objective, set())
                    if not image_wps:
                        # Cannot achieve this goal if objective is not visible from anywhere
                        min_goal_cost = float('inf')
                    else:
                        # Find best rover/camera combination
                        for rover in self.rovers:
                            if 'imaging' in self.rover_equipment.get(rover, set()):
                                r_loc = current_rover_locs.get(rover)
                                if r_loc is None: continue # Rover location unknown or rover doesn't exist

                                # Find camera on this rover supporting this mode
                                suitable_camera = None
                                cal_target = None
                                for cam, info in self.camera_info.items():
                                    if info.get('rover') == rover and mode in info.get('modes', set()):
                                        suitable_camera = cam
                                        cal_target = info.get('target')
                                        break # Assume one suitable camera per rover/mode

                                if suitable_camera is None: continue # No suitable camera on this rover

                                # Find waypoints for calibration target
                                cal_wps = self.objective_visibility.get(cal_target, set())
                                # If calibration is needed but target is not visible anywhere, goal is impossible
                                calibrated = current_calibrated.get((suitable_camera, rover), False)
                                if not calibrated and not cal_wps:
                                    continue # Cannot calibrate

                                cost = 1 # communicate action

                                # Check if image is already taken by this rover/camera/mode
                                have_image = current_have_image.get((rover, objective, mode), False)

                                if not have_image:
                                    cost += 1 # take_image action

                                    if not calibrated:
                                        cost += 1 # calibrate action
                                        # Nav cost: r_loc -> w_cal -> w_img -> lander_loc
                                        # Simplified Nav: MinDist(r_loc, any w_cal) + MinDist(r_loc, any w_img) + MinDist(r_loc, lander_loc)
                                        min_dist_r_to_w_cal = min((self.min_dist.get(rover, {}).get(r_loc, {}).get(w, float('inf')) for w in cal_wps), default=float('inf'))
                                        min_dist_r_to_w_img = min((self.min_dist.get(rover, {}).get(r_loc, {}).get(w, float('inf')) for w in image_wps), default=float('inf'))
                                        min_dist_r_to_lander = self.min_dist.get(rover, {}).get(r_loc, {}).get(lander_loc, float('inf'))

                                        # Summing up minimum distances from r_loc to each required point type
                                        nav_cost = min_dist_r_to_w_cal + min_dist_r_to_w_img + min_dist_r_to_lander

                                    else: # calibrated
                                        # Nav cost: r_loc -> w_img -> lander_loc
                                        # Simplified Nav: MinDist(r_loc, any w_img) + MinDist(r_loc, lander_loc)
                                        min_dist_r_to_w_img = min((self.min_dist.get(rover, {}).get(r_loc, {}).get(w, float('inf')) for w in image_wps), default=float('inf'))
                                        min_dist_r_to_lander = self.min_dist.get(rover, {}).get(r_loc, {}).get(lander_loc, float('inf'))

                                        # Summing up minimum distances from r_loc to each required point type
                                        nav_cost = min_dist_r_to_w_img + min_dist_r_to_lander

                                    cost += nav_cost

                                else: # have_image is true
                                    # Nav cost: r_loc -> lander_loc
                                    dist_r_to_lander = self.min_dist.get(rover, {}).get(r_loc, {}).get(lander_loc, float('inf'))
                                    cost += dist_r_to_lander

                                min_goal_cost = min(min_goal_cost, cost)

                total_cost += min_goal_cost

            # Add other goal types if any (rock analysis, etc.) - handled above

        # If any goal was impossible, total_cost will be infinity
        if total_cost == float('inf'):
             return float('inf')

        return total_cost
