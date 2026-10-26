# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes leading/trailing parentheses and splits by space."""
    return fact[1:-1].split()

from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        Estimates the cost to reach the goal state by summing up the estimated
        costs for each unachieved goal predicate. The cost for each goal
        predicate is estimated based on whether the required intermediate
        conditions (like having a sample or an image) are met, and includes
        fixed costs for necessary actions (sampling, imaging, communicating,
        calibrating, dropping) and simplified navigation steps. It considers
        rover capabilities but does not perform detailed pathfinding or
        resource allocation optimization.

    Assumptions:
        - The problem instance is solvable. Impossible goals (e.g., needing
          a soil sample but no rover is equipped for soil analysis) are
          assigned a large penalty, assuming they indicate an unsolvable path
          or state.
        - Navigation between any two relevant waypoints (sample/image location,
          calibration target location, communication location) takes a fixed
          number of actions (e.g., 1 per leg). This ignores actual graph
          distances and rover-specific traversal capabilities for simplicity
          and efficiency.
        - Store capacity is 1 (a store is either empty or full).
        - A rover needs an empty store *only* for sampling soil or rock.
        - A camera needs calibration *before* taking an image, and taking an
          image uncalibrates it. Calibration requires being at a waypoint
          visible from the calibration target.
        - Communication requires being at a waypoint visible from the lander.
        - The heuristic does not attempt to assign tasks to specific rovers
          optimally; it just checks if *any* capable rover could potentially
          achieve the goal and adds the estimated cost.

    Heuristic Initialization:
        The constructor processes the static facts from the task definition
        to precompute information needed for the heuristic calculation. This
        includes:
        - `rover_capabilities`: Mapping rovers to the set of capabilities
          (soil, rock, imaging) they possess.
        - `rover_stores`: Mapping rovers to their associated store objects.
        - `camera_details`: Mapping cameras to their onboard rover, supported
          modes, and calibration target objective.
        - `objective_visibility_waypoints`: Mapping objectives to the set of
          waypoints from which they are visible.
        - `lander_locations`: Set of waypoints where landers are located.
        - `initial_soil_samples`: Set of waypoints that initially have soil samples.
        - `initial_rock_samples`: Set of waypoints that initially have rock samples.
        - `lander_comm_waypoints`: Set of waypoints visible from any lander location.
          (Computed based on lander locations and visible predicates).
        - `rover_camera_modes`: Mapping rovers to cameras and the modes supported
          by those cameras on that rover.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state to extract dynamic information: rover locations,
           store statuses, held samples/images, calibrated cameras, and
           communicated data.
        2. Initialize the heuristic value `h` to 0.
        3. Iterate through each goal predicate defined in the task.
        4. For each goal predicate not yet satisfied in the current state:
            a. If the goal is `(communicated_soil_data ?w)`:
                - Check if `(have_soil_analysis ?r ?w)` is true for any rover `r`
                  equipped for soil analysis.
                - If yes (sample is held): Add 2 to `h` (estimated cost for
                  navigation to communication point + communication action).
                - If no (sample needs to be collected): Check if any rover is
                  equipped for soil analysis. If not, add a large penalty (1000).
                  Otherwise, add 5 to `h` (estimated cost for navigation to sample
                  point + sample action + drop action (if needed) + navigation
                  to communication point + communication action).
            b. If the goal is `(communicated_rock_data ?w)`:
                - Similar logic as for soil data, checking for rock analysis
                  capability and `(have_rock_analysis ?r ?w)`. Add 2 if sample
                  is held, 5 if sample needs to be collected (plus penalty if
                  no capable rover).
            c. If the goal is `(communicated_image_data ?o ?m)`:
                - Check if `(have_image ?r ?o ?m)` is true for any rover `r`
                  equipped for imaging and having a camera supporting mode `m`.
                - If yes (image is held): Add 2 to `h` (estimated cost for
                  navigation to communication point + communication action).
                - If no (image needs to be taken): Check if any rover is equipped
                  for imaging and has a camera supporting mode `m`. If not, add
                  a large penalty (1000). Otherwise, add 6 to `h` (estimated cost
                  for navigation to calibration point + calibration action +
                  navigation to image point + take image action + navigation to
                  communication point + communication action).
        5. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state for sample locations

        # Precompute static information
        self.rover_capabilities = defaultdict(set)
        self.rover_stores = {} # {rover: store}
        self.camera_details = defaultdict(lambda: {'rover': None, 'modes': set(), 'cal_target': None}) # {camera: {rover, modes, cal_target}}
        self.objective_visibility_waypoints = defaultdict(set) # {objective: {waypoint}}
        self.lander_locations = set() # {waypoint}
        self.visible_map = defaultdict(set) # {waypoint_from: {waypoint_to}}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
            elif parts[0] == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
            elif parts[0] == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # store_of ?s ?r -> rover_stores[r] = s
            elif parts[0] == 'on_board':
                self.camera_details[parts[1]]['rover'] = parts[2] # on_board ?i ?r -> camera_details[i]['rover'] = r
            elif parts[0] == 'supports':
                self.camera_details[parts[1]]['modes'].add(parts[2]) # supports ?c ?m -> camera_details[c]['modes'].add(m)
            elif parts[0] == 'calibration_target':
                self.camera_details[parts[1]]['cal_target'] = parts[2] # calibration_target ?i ?o -> camera_details[i]['cal_target'] = o
            elif parts[0] == 'visible_from':
                self.objective_visibility_waypoints[parts[1]].add(parts[2]) # visible_from ?o ?w -> objective_visibility_waypoints[o].add(w)
            elif parts[0] == 'at_lander':
                self.lander_locations.add(parts[2]) # at_lander ?l ?y -> lander_locations.add(y)
            elif parts[0] == 'visible':
                 self.visible_map[parts[1]].add(parts[2]) # visible ?w ?p -> visible_map[w].add(p)

        # Compute waypoints visible from any lander location
        self.lander_comm_waypoints = set()
        for lander_loc in self.lander_locations:
             # A rover at waypoint X can communicate if X is visible from the lander location Y.
             # So we need waypoints X such that (visible X Y) is true.
             # The visible map stores (visible W P), meaning P is visible from W.
             # We need W such that (visible W lander_loc) is true.
             # This means lander_loc must be in the set of waypoints visible *from* W.
             for w_from, w_tos in self.visible_map.items():
                 if lander_loc in w_tos:
                     self.lander_comm_waypoints.add(w_from)

        # Initial samples are part of the initial state, not static facts
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at_soil_sample':
                self.initial_soil_samples.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                self.initial_rock_samples.add(parts[1])

        # Precompute which rovers have which cameras supporting which modes
        self.rover_camera_modes = defaultdict(lambda: defaultdict(set)) # {rover: {camera: {mode}}}
        for cam, details in self.camera_details.items():
            rover = details['rover']
            if rover: # Camera must be on a rover
                self.rover_camera_modes[rover][cam].update(details['modes'])


    def __call__(self, node):
        state = node.state

        # Parse dynamic state information
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'have_soil_analysis':
                have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis':
                have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image':
                have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))

        h = 0

        # Estimate cost for each unachieved goal
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint not in communicated_soil:
                    # Goal: communicate soil data from waypoint
                    # Check if any rover already has the sample
                    # We only care if *any* capable rover has it.
                    has_sample = any((r, waypoint) in have_soil for r in self.rover_capabilities if 'soil' in self.rover_capabilities[r])

                    if has_sample:
                        # Need to communicate
                        # Cost: Navigate to comm point (1) + Communicate (1) = 2
                        h += 2
                    else:
                        # Need to sample and communicate
                        # Check if any rover is capable
                        if not any('soil' in caps for caps in self.rover_capabilities.values()):
                            h += 1000 # Impossible goal
                            continue

                        # Cost: Navigate to sample point (1) + Sample (1) + Drop (maybe 1) + Navigate to comm point (1) + Communicate (1) = 5
                        h += 5

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint not in communicated_rock:
                    # Goal: communicate rock data from waypoint
                    # Check if any rover already has the sample
                    has_sample = any((r, waypoint) in have_rock for r in self.rover_capabilities if 'rock' in self.rover_capabilities[r])

                    if has_sample:
                        # Need to communicate
                        # Cost: Navigate to comm point (1) + Communicate (1) = 2
                        h += 2
                    else:
                        # Need to sample and communicate
                        # Check if any rover is capable
                        if not any('rock' in caps for caps in self.rover_capabilities.values()):
                            h += 1000 # Impossible goal
                            continue

                        # Cost: Navigate to sample point (1) + Sample (1) + Drop (maybe 1) + Navigate to comm point (1) + Communicate (1) = 5
                        h += 5

            elif predicate == 'communicated_image_data':
                objective = parts[1]
                mode = parts[2]
                if (objective, mode) not in communicated_image:
                    # Goal: communicate image data of objective in mode
                    # Check if any rover already has the image
                    has_image = any((r, objective, mode) in have_image for r in self.rover_capabilities if 'imaging' in self.rover_capabilities[r])

                    if has_image:
                        # Need to communicate
                        # Cost: Navigate to comm point (1) + Communicate (1) = 2
                        h += 2
                    else:
                        # Need to take image and communicate
                        # Check if any rover has a camera supporting this mode and is equipped for imaging
                        capable_rover_camera_exists = False
                        for r, caps in self.rover_capabilities.items():
                            if 'imaging' in caps:
                                # Check cameras on this rover
                                for cam, modes in self.rover_camera_modes.get(r, {}).items():
                                    if mode in modes:
                                        capable_rover_camera_exists = True
                                        break
                            if capable_rover_camera_exists:
                                break

                        if not capable_rover_camera_exists:
                            h += 1000 # Impossible goal
                            continue

                        # Cost: Navigate to cal point (1) + Calibrate (1) + Navigate to image point (1) + Take Image (1) + Navigate to comm point (1) + Communicate (1) = 6
                        h += 6
            # Ignore other predicates in goals if any, as per PDDL standard goals are conjunctions of positive literals

        return h
