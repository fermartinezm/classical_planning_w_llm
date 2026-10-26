from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions, which involve communicating soil, rock, and image data. It
    estimates the cost for each uncommunicated goal item independently, summing
    up the estimated costs. The cost for each item is based on whether the
    required data has been collected and adds a fixed cost for necessary
    actions like sampling, imaging, calibrating, and navigating.

    # Assumptions
    - All goal conditions involve communicating data (soil, rock, or image).
    - Each uncommunicated goal item is considered independently.
    - Resource constraints (like store capacity or camera calibration state after use)
      and specific rover capabilities/locations are simplified: we assume a
      suitable rover exists and can reach necessary locations.
    - Navigation cost to a required location (sample site, image viewpoint,
      calibration target viewpoint, or lander-visible waypoint) is a fixed cost of 1
      action if the required data is not yet collected, or if the data is
      collected but the rover is not at a communication location.

    # Heuristic Initialization
    - Extracts the set of goal facts.
    - Identifies lander locations and precomputes the set of waypoints visible
      from any lander location. This set is where a rover must be to communicate data.
    - Maps objectives to the set of waypoints from which they are visible. This
      is where a rover must be to take an image of that objective.
    - Maps cameras to the set of waypoints visible from their calibration targets.
      This is where a rover must be to calibrate that camera.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1. Initialize the total heuristic cost to 0.
    2. Identify the set of soil waypoints, rock waypoints, and (objective, mode)
       pairs for which data has already been communicated in the current state.
    3. Identify the set of soil waypoints, rock waypoints, and (objective, mode)
       pairs for which data has already been collected (i.e., the rover has the
       sample/image) in the current state.
    4. Iterate through each goal fact specified in the task:
       - If the goal is `(communicated_soil_data ?w)` and `?w` is not in the set
         of communicated soil data waypoints:
         - Add 1 to the total cost (for the `communicate_soil_data` action).
         - If `?w` is not in the set of waypoints for which soil data has been
           collected by any rover:
           - Add 1 to the total cost (for the `sample_soil` action).
           - Add 1 to the total cost (for navigation to waypoint `?w`).
         - Else (some rover has the soil data for `?w`):
           - Add 1 to the total cost (for navigation to a lander-visible waypoint).
       - If the goal is `(communicated_rock_data ?w)` and `?w` is not in the set
         of communicated rock data waypoints:
         - Add 1 to the total cost (for the `communicate_rock_data` action).
         - If `?w` is not in the set of waypoints for which rock data has been
           collected by any rover:
           - Add 1 to the total cost (for the `sample_rock` action).
           - Add 1 to the total cost (for navigation to waypoint `?w`).
         - Else (some rover has the rock data for `?w`):
           - Add 1 to the total cost (for navigation to a lander-visible waypoint).
       - If the goal is `(communicated_image_data ?o ?m)` and `(?o, ?m)` is not
         in the set of communicated image data pairs:
         - Add 1 to the total cost (for the `communicate_image_data` action).
         - If `(?o, ?m)` is not in the set of (objective, mode) pairs for which
           image data has been collected by any rover:
           - Add 1 to the total cost (for the `take_image` action).
           - Add 1 to the total cost (for navigation to a waypoint visible from `?o`).
           - Add 1 to the total cost (for the `calibrate` action).
           - Add 1 to the total cost (for navigation to a waypoint visible from
             the camera's calibration target).
         - Else (some rover has the image data for `(?o, ?m)`):
           - Add 1 to the total cost (for navigation to a lander-visible waypoint).
    5. Return the total cost.

    This heuristic is not admissible because it ignores negative interactions
    (like camera uncalibration or store usage) and assumes navigation is always
    a fixed cost of 1 when needed, regardless of actual distance or rover location.
    However, it provides a simple estimate of remaining tasks.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goal_facts = task.goals

        # Precompute lander locations and lander-visible waypoints
        lander_locations = set()
        for fact in task.static:
            if match(fact, "at_lander", "*", "*"):
                lander_locations.add(get_parts(fact)[2])

        self.lander_visible_waypoints = set()
        for fact in task.static:
            if match(fact, "visible", "*", "*"):
                wp1, wp2 = get_parts(fact)[1:]
                if wp2 in lander_locations:
                    self.lander_visible_waypoints.add(wp1)

        # Precompute objective visibility waypoints
        self.objective_visible_waypoints = {}
        for fact in task.static:
            if match(fact, "visible_from", "*", "*"):
                obj, wp = get_parts(fact)[1:]
                if obj not in self.objective_visible_waypoints:
                    self.objective_visible_waypoints[obj] = set()
                self.objective_visible_waypoints[obj].add(wp)

        # Precompute camera calibration waypoints
        # Need calibration_target and visible_from the target
        calibration_targets = {}
        for fact in task.static:
             if match(fact, "calibration_target", "*", "*"):
                 camera, target = get_parts(fact)[1:]
                 calibration_targets[camera] = target

        self.camera_calibration_waypoints = {}
        for camera, target in calibration_targets.items():
            if target in self.objective_visible_waypoints:
                 self.camera_calibration_waypoints[camera] = self.objective_visible_waypoints[target]
            else:
                 # Should not happen in valid problems, but handle defensively
                 self.camera_calibration_waypoints[camera] = set()


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state

        # Extract relevant dynamic facts from the current state
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set() # Stores (objective, mode) tuples

        have_soil_wps = set() # Stores waypoints for which soil data is held by any rover
        have_rock_wps = set() # Stores waypoints for which rock data is held by any rover
        have_image_oms = set() # Stores (objective, mode) tuples for which image data is held by any rover

        # We don't strictly need rover locations, sample locations, calibration,
        # or store status for this simplified additive heuristic, but parsing
        # them is good practice if we were to refine the heuristic later.
        # For this version, we only need the 'communicated' and 'have' facts.

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue # Skip empty or malformed facts

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))
            elif predicate == 'have_soil_analysis':
                have_soil_wps.add(parts[2]) # parts[1] is rover, parts[2] is waypoint
            elif predicate == 'have_rock_analysis':
                have_rock_wps.add(parts[2]) # parts[1] is rover, parts[2] is waypoint
            elif predicate == 'have_image':
                have_image_oms.add((parts[2], parts[3])) # parts[1] is rover, parts[2] is objective, parts[3] is mode

        total_cost = 0

        # Iterate through goal facts and add cost for unachieved ones
        for goal in self.goal_facts:
            parts = get_parts(goal)
            if not parts:
                continue

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint not in communicated_soil:
                    total_cost += 1 # Cost for communicate action
                    if waypoint not in have_soil_wps:
                        total_cost += 1 # Cost for sample action
                        total_cost += 1 # Cost for navigation to sample waypoint
                    else:
                        total_cost += 1 # Cost for navigation to lander-visible waypoint

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint not in communicated_rock:
                    total_cost += 1 # Cost for communicate action
                    if waypoint not in have_rock_wps:
                        total_cost += 1 # Cost for sample action
                        total_cost += 1 # Cost for navigation to sample waypoint
                    else:
                        total_cost += 1 # Cost for navigation to lander-visible waypoint

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1:]
                if (objective, mode) not in communicated_image:
                    total_cost += 1 # Cost for communicate action
                    if (objective, mode) not in have_image_oms:
                        total_cost += 1 # Cost for take_image action
                        total_cost += 1 # Cost for navigation to image waypoint (visible from objective)
                        total_cost += 1 # Cost for calibrate action
                        total_cost += 1 # Cost for navigation to calibration waypoint

                    else:
                        total_cost += 1 # Cost for navigation to lander-visible waypoint

        return total_cost

