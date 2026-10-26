from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Use fnmatch for wildcard matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    unmet goal conditions. It breaks down each goal (communicating data)
    into the necessary steps (sampling/imaging, calibration, navigation,
    dropping samples) and sums the estimated cost for each step that hasn't
    been completed yet, for each unachieved goal.

    # Assumptions
    - Each unachieved goal is considered independently (ignoring potential
      synergies or conflicts between goals or rovers).
    - Navigation cost is simplified: a fixed cost of 1 is added for each
      *type* of location visit required by a goal's sub-plan (e.g., sample
      location, image location, calibration location, lander-visible location)
      if no rover is currently at *any* suitable waypoint for that location type.
    - Dropping a sample is only considered if a rover needs to sample but
      all equipped rovers' stores are full.
    - Any equipped and capable rover can perform the necessary tasks for a goal.

    # Heuristic Initialization
    - Precomputes static information from the task, such as lander locations,
      lander-visible waypoints, rover equipment, store ownership, camera info
      (modes, calibration targets, on-board status), and objective visibility.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each goal condition that is not yet satisfied in the current state:
    1. Add 1 for the final communication action.
    2. If the required data (soil sample, rock sample, or image) is not
       currently held by any rover:
       a. Add 1 for the action to acquire the data (sample_soil, sample_rock, or take_image).
       b. If sampling (soil/rock):
          - Add 1 for the drop action if any equipped rover's store is full.
          - Add 1 for navigation to the sample location if no rover is currently
            at the sample waypoint.
       c. If imaging:
          - Add 1 for the calibrate action.
          - Add 1 for navigation to a suitable calibration location if no rover
            is currently at any waypoint visible from a calibration target for
            a suitable camera.
          - Add 1 for navigation to a suitable image location if no rover is
            currently at any waypoint visible from the objective.
    3. Add 1 for navigation to a lander-visible location if no rover is currently
       at any waypoint visible from a lander.

    The total heuristic value is the sum of these costs for all unsatisfied goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Precompute static information for efficiency
        self.all_rovers = set()
        self.lander_locations = set()
        self.lander_visible_wps = set()
        self.rover_store_map = {} # rover -> store
        self.rover_equipped_soil = set()
        self.rover_equipped_rock = set()
        self.rover_equipped_imaging = set()
        self.camera_info = {} # camera -> {'rover': r, 'modes': {m}, 'cal_target': t}
        self.objective_visible_wps = {} # objective -> {wp}
        self.calibration_target_wps = {} # cal_target_objective -> {wp visible from target}

        # First pass to collect basic info
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at_lander":
                self.lander_locations.add(parts[2])
            elif predicate == "store_of":
                self.rover_store_map[parts[2]] = parts[1]
            elif predicate == "equipped_for_soil_analysis":
                self.rover_equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.rover_equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.rover_equipped_imaging.add(parts[1])
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': rover, 'modes': set(), 'cal_target': None}
                else:
                     self.camera_info[camera]['rover'] = rover # Update rover if needed (should be static)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['modes'].add(mode)
            elif predicate == "calibration_target":
                camera, target = parts[1], parts[2]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'rover': None, 'modes': set(), 'cal_target': None}
                self.camera_info[camera]['cal_target'] = target
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visible_wps:
                    self.objective_visible_wps[objective] = set()
                self.objective_visible_wps[objective].add(waypoint)
            elif predicate == "at": # Rovers' initial locations are static in instance file
                 if parts[1].startswith('rover'):
                     self.all_rovers.add(parts[1])

        # Second pass for derived static info (lander_visible_wps, calibration_target_wps)
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]

             if predicate == "visible":
                 wp1, wp2 = parts[1], parts[2]
                 if wp2 in self.lander_locations:
                     self.lander_visible_wps.add(wp1)
                 if wp1 in self.lander_locations: # visible is symmetric
                     self.lander_visible_wps.add(wp2)

             # Map calibration targets to waypoints visible from them
             if predicate == "visible_from":
                 objective, waypoint = parts[1], parts[2]
                 # Check if this objective is a calibration target for any camera
                 is_cal_target = any(info['cal_target'] == objective for info in self.camera_info.values())
                 if is_cal_target:
                     if objective not in self.calibration_target_wps:
                         self.calibration_target_wps[objective] = set()
                     self.calibration_target_wps[objective].add(waypoint)

        # Ensure all rovers mentioned in goals or init are in self.all_rovers
        # (Important for instances where not all rovers are equipped)
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             # Goals are communication, which involves a rover implicitly
             # We assume any rover can potentially be used if equipped.
             # The set of rovers is usually defined by the objects section,
             # but parsing init facts is a reliable way to find them.
             pass # all_rovers should be populated from init facts in the first pass

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Extract relevant dynamic information from the current state
        current_rover_locations = {} # rover -> waypoint
        current_full_stores = set() # {store}
        current_have_soil = set() # {(rover, wp)}
        current_have_rock = set() # {(rover, wp)}
        current_have_image = set() # {(rover, obj, mode)}
        current_communicated_soil = set() # {wp}
        current_communicated_rock = set() # {wp}
        current_communicated_image = set() # {(obj, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == "at" and parts[1].startswith('rover'):
                current_rover_locations[parts[1]] = parts[2]
            elif predicate == "full":
                current_full_stores.add(parts[1])
            elif predicate == "have_soil_analysis":
                current_have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                current_have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "communicated_soil_data":
                current_communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                current_communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                current_communicated_image.add((parts[1], parts[2]))

        total_cost = 0  # Initialize heuristic cost

        # Iterate through each goal condition
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                w = parts[1]
                # Check if goal is already met
                if w in current_communicated_soil:
                    continue

                # Goal not met, estimate cost
                total_cost += 1 # Cost for communicate_soil_data action

                # Check if the required sample data is already collected by any rover
                has_sample = any((r, w) in current_have_soil for r in self.all_rovers)

                if not has_sample:
                    # Need to sample soil
                    total_cost += 1 # Cost for sample_soil action

                    # Check if drop is needed before sampling (if any equipped rover's store is full)
                    drop_needed = False
                    for r in self.rover_equipped_soil:
                        store = self.rover_store_map.get(r)
                        if store and store in current_full_stores:
                            drop_needed = True
                            break # Only need one instance of a full store on an equipped rover
                    if drop_needed:
                         total_cost += 1 # Cost for drop action

                    # Need to navigate to the sample location
                    # Check if any rover is already at the sample waypoint
                    any_rover_at_sample_spot = any(loc == w for loc in current_rover_locations.values())
                    if not any_rover_at_sample_spot:
                        total_cost += 1 # Cost for navigate to sample spot

                # Need to navigate to a lander-visible location for communication
                # Check if any rover is already at a lander-visible waypoint
                any_rover_at_comm_spot = any(loc in self.lander_visible_wps for loc in current_rover_locations.values())
                if not any_rover_at_comm_spot:
                    total_cost += 1 # Cost for navigate to lander-visible spot

            elif predicate == "communicated_rock_data":
                w = parts[1]
                # Check if goal is already met
                if w in current_communicated_rock:
                    continue

                # Goal not met, estimate cost
                total_cost += 1 # Cost for communicate_rock_data action

                # Check if the required sample data is already collected by any rover
                has_sample = any((r, w) in current_have_rock for r in self.all_rovers)

                if not has_sample:
                    # Need to sample rock
                    total_cost += 1 # Cost for sample_rock action

                    # Check if drop is needed before sampling (if any equipped rover's store is full)
                    drop_needed = False
                    for r in self.rover_equipped_rock:
                        store = self.rover_store_map.get(r)
                        if store and store in current_full_stores:
                            drop_needed = True
                            break # Only need one instance of a full store on an equipped rover
                    if drop_needed:
                         total_cost += 1 # Cost for drop action

                    # Need to navigate to the sample location
                    # Check if any rover is already at the sample waypoint
                    any_rover_at_sample_spot = any(loc == w for loc in current_rover_locations.values())
                    if not any_rover_at_sample_spot:
                        total_cost += 1 # Cost for navigate to sample spot

                # Need to navigate to a lander-visible location for communication
                # Check if any rover is already at a lander-visible waypoint
                any_rover_at_comm_spot = any(loc in self.lander_visible_wps for loc in current_rover_locations.values())
                if not any_rover_at_comm_spot:
                    total_cost += 1 # Cost for navigate to lander-visible spot

            elif predicate == "communicated_image_data":
                o, m = parts[1], parts[2]
                # Check if goal is already met
                if (o, m) in current_communicated_image:
                    continue

                # Goal not met, estimate cost
                total_cost += 1 # Cost for communicate_image_data action

                # Check if the required image data is already collected by any rover
                has_image = any((r, o, m) in current_have_image for r in self.all_rovers)

                if not has_image:
                    # Need to take image
                    total_cost += 1 # Cost for take_image action
                    total_cost += 1 # Cost for calibrate action

                    # Need to navigate to a suitable calibration location
                    # Find suitable calibration waypoints for any camera supporting this mode and objective
                    suitable_cal_wps = set()
                    for cam, info in self.camera_info.items():
                        if m in info['modes'] and info['cal_target']:
                            cal_target = info['cal_target']
                            suitable_cal_wps.update(self.calibration_target_wps.get(cal_target, set()))

                    # Check if any rover is already at a suitable calibration waypoint
                    any_rover_at_cal_spot = any(loc in suitable_cal_wps for loc in current_rover_locations.values())
                    if not any_rover_at_cal_spot:
                        total_cost += 1 # Cost for navigate to calibration spot

                    # Need to navigate to a suitable image location
                    # Find suitable image waypoints for the objective
                    suitable_img_wps = self.objective_visible_wps.get(o, set())

                    # Check if any rover is already at a suitable image waypoint
                    any_rover_at_img_spot = any(loc in suitable_img_wps for loc in current_rover_locations.values())
                    if not any_rover_at_img_spot:
                        total_cost += 1 # Cost for navigate to image spot

                # Need to navigate to a lander-visible location for communication
                # Check if any rover is already at a lander-visible waypoint
                any_rover_at_comm_spot = any(loc in self.lander_visible_wps for loc in current_rover_locations.values())
                if not any_rover_at_comm_spot:
                    total_cost += 1 # Cost for navigate to lander-visible spot

        return total_cost

