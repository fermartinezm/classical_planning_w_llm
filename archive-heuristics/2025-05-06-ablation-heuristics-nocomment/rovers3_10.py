from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Extract predicate and arguments from a PDDL fact string."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

# Helper function to match a fact string against a pattern
def match(fact, *args):
    """
    Check if a fact string matches a pattern using fnmatch.
    Args are predicate and arguments, can include wildcards (*).
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing up
        estimated costs for each unsatisfied goal fact. It uses a relaxed
        approach, assigning fixed costs based on the "stage" of achieving
        the goal (e.g., data not collected, data collected but not communicated,
        image not taken, image taken but not communicated). It simplifies
        navigation costs and ignores resource constraints like store capacity
        beyond the initial sample action cost estimate. It also simplifies
        camera calibration, assuming any calibrated camera of the right type
        can be used, and ignoring the fact that taking an image uncalibrates it.

    Assumptions:
        - The problem is solvable.
        - There is at least one lander.
        - For image goals, there exists at least one rover with imaging
          equipment and an onboard camera supporting the required mode.
        - For soil/rock goals, there exists at least one rover with the
          corresponding analysis equipment.
        - The specific location of the rover or lander is abstracted away
          in the cost estimates for navigation/communication steps.
        - Store capacity (beyond needing an empty store for sampling) and
          the need to drop samples are simplified or ignored in cost estimates.
        - Camera calibration state after taking an image is ignored.

    Heuristic Initialization:
        The constructor processes static facts from the task definition to
        pre-compute information about rover capabilities, camera properties
        (on-board status, supported modes, calibration targets), store ownership,
        and lander location. This information is stored in dictionaries and sets
        for efficient lookup during heuristic computation. Specifically, it
        identifies which (camera, rover) pairs are capable of taking images
        in specific modes.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize total heuristic cost to 0.
        2. Iterate through each goal fact defined in the task.
        3. For each goal fact:
            a. Check if the goal fact is already present in the current state. If yes, this goal is satisfied, add 0 cost for this goal, and continue to the next goal.
            b. If the goal fact is not in the current state, determine its type (soil, rock, or image communication).
            c. If the goal is `(communicated_soil_data ?w)`:
                i. Check if any rover `r` currently holds the soil analysis data from waypoint `w` (i.e., `(have_soil_analysis ?r ?w)` is in the state for any `r`). This check uses the `match` helper with a wildcard for the rover.
                ii. If yes, the data is collected. The remaining steps are navigation to a lander and communication. Add a fixed cost of 2 (representing navigate + communicate).
                iii. If no, the data needs to be collected. This requires navigating to the sample location `w`, sampling, navigating to a lander, and communicating. Add a fixed cost of 4 (representing navigate to sample + sample + navigate to lander + communicate). This simplifies the process and ignores the need for an empty store or the drop action. The heuristic assumes the sample is available if the data is not held and the goal is not met.
            d. If the goal is `(communicated_rock_data ?w)`:
                i. Similar logic to soil data. Check if any rover `r` holds the rock analysis data from waypoint `w` (i.e., `(have_rock_analysis ?r ?w)` is in the state for any `r`). This check uses the `match` helper with a wildcard for the rover.
                ii. If yes, add a fixed cost of 2.
                iii. If no, add a fixed cost of 4. The heuristic assumes the sample is available if the data is not held and the goal is not met.
            e. If the goal is `(communicated_image_data ?o ?m)`:
                i. Check if any rover `r` currently holds the image data for objective `o` in mode `m` (i.e., `(have_image ?r ?o ?m)` is in the state for any `r`). This check uses the `match` helper with a wildcard for the rover.
                ii. If yes, the image is taken. The remaining steps are navigation to a lander and communication. Add a fixed cost of 2.
                iii. If no, the image needs to be taken. This requires a rover with imaging equipment and a camera supporting mode `m`. Find if *any* such camera `i` on *any* such rover `r` is currently calibrated (i.e., `(calibrated ?i ?r)` is in the state for any suitable (i, r) pair). Suitable pairs are pre-computed in `__init__` based on static facts (`on_board`, `equipped_for_imaging`, `supports`).
                iv. If a suitable camera is calibrated, the remaining steps are navigating to an image viewpoint for `o`, taking the image, navigating to a lander, and communicating. Add a fixed cost of 4 (representing navigate to image waypoint + take image + navigate to lander + communicate). This ignores the fact that taking the image uncalibrates the camera.
                v. If no suitable camera is calibrated, calibration is also needed. The remaining steps are navigating to the calibration target for camera `i`, calibrating, navigating to an image viewpoint for `o`, taking the image, navigating to a lander, and communicating. Add a fixed cost of 6 (representing navigate to cal target + calibrate + navigate to image waypoint + take image + navigate to lander + communicate). If no capable camera/rover pair exists for the mode (checked using `self.imaging_capabilities`), the same cost of 6 is added, assuming solvability implies capability.
        4. The total heuristic cost is the sum of costs for all unsatisfied goals.
        5. If the total cost is 0, it means all goals are satisfied, which corresponds to a goal state.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Data structures to store pre-processed static information
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_stores = {} # rover -> store
        self.camera_on_rover = {} # camera -> rover
        self.camera_supports = {} # camera -> set of modes
        self.camera_cal_target = {} # camera -> objective
        self.lander_location = None # Assuming one lander

        # Pre-process static facts
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'equipped_for_soil_analysis':
                self.equipped_soil.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                self.equipped_rock.add(parts[1])
            elif predicate == 'equipped_for_imaging':
                self.equipped_imaging.add(parts[1])
            elif predicate == 'store_of':
                # store_of ?s ?r
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif predicate == 'on_board':
                # on_board ?i ?r
                self.camera_on_rover[parts[1]] = parts[2] # camera -> rover
            elif predicate == 'supports':
                # supports ?c ?m
                camera, mode = parts[1], parts[2]
                self.camera_supports.setdefault(camera, set()).add(mode)
            elif predicate == 'calibration_target':
                # calibration_target ?i ?t
                self.camera_cal_target[parts[1]] = parts[2] # camera -> objective
            elif predicate == 'at_lander':
                # at_lander ?x ?y
                self.lander_location = parts[2] # Assuming one lander

        # Pre-compute imaging capabilities per mode
        # Maps mode -> set of (camera, rover) pairs capable of imaging in that mode
        self.imaging_capabilities = {}
        for camera, rover in self.camera_on_rover.items():
            if rover in self.equipped_imaging:
                for mode in self.camera_supports.get(camera, set()):
                    self.imaging_capabilities.setdefault(mode, set()).add((camera, rover))


    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Convert state to a set for faster lookups
        state_set = set(state)

        for goal in self.goals:
            # Check if the goal is already satisfied in the current state
            if goal in state_set:
                continue # Goal already satisfied, cost is 0 for this goal

            # Goal is not satisfied, calculate its contribution to the heuristic
            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                # Check if any rover has the soil analysis data from this waypoint
                # Use match with wildcard for rover
                has_data = any(match(fact, 'have_soil_analysis', '*', waypoint) for fact in state_set)

                if has_data:
                    # Data collected, need to navigate to lander and communicate
                    # Estimated cost: navigate (1) + communicate (1) = 2
                    total_cost += 2
                else:
                    # Data not collected. Need to sample, navigate, communicate.
                    # Estimated cost: navigate_to_sample (1) + sample (1) + navigate_to_lander (1) + communicate (1) = 4
                    # This assumes the sample is available if the data is not held.
                    total_cost += 4

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                # Check if any rover has the rock analysis data from this waypoint
                # Use match with wildcard for rover
                has_data = any(match(fact, 'have_rock_analysis', '*', waypoint) for fact in state_set)

                if has_data:
                    # Data collected, need to navigate to lander and communicate
                    # Estimated cost: navigate (1) + communicate (1) = 2
                    total_cost += 2
                else:
                    # Data not collected. Need to sample, navigate, communicate.
                    # Estimated cost: navigate_to_sample (1) + sample (1) + navigate_to_lander (1) + communicate (1) = 4
                    # This assumes the sample is available if the data is not held.
                    total_cost += 4

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                # Check if any rover has the image data for this objective and mode
                # Use match with wildcard for rover
                has_image = any(match(fact, 'have_image', '*', objective, mode) for fact in state_set)

                if has_image:
                    # Image taken, need to navigate to lander and communicate
                    # Estimated cost: navigate (1) + communicate (1) = 2
                    total_cost += 2
                else:
                    # Image not taken. Need to take image, navigate, communicate.
                    # First, find capable camera/rover pairs for this mode
                    capable_pairs = self.imaging_capabilities.get(mode, set())

                    # Check if any capable camera is currently calibrated
                    # Iterate through pre-computed capable pairs and check calibration state
                    is_calibrated = any(f'(calibrated {cam} {rov})' in state_set for cam, rov in capable_pairs)

                    if is_calibrated:
                        # Camera calibrated, need to navigate to image waypoint, take image, navigate to lander, communicate
                        # Estimated cost: navigate_to_img (1) + take_img (1) + navigate_to_lander (1) + communicate (1) = 4
                        # This ignores the fact that taking the image uncalibrates the camera.
                        total_cost += 4
                    else:
                        # Camera needs calibration, then take image, navigate, communicate
                        # Estimated cost: navigate_to_cal (1) + calibrate (1) + navigate_to_img (1) + take_img (1) + navigate_to_lander (1) + communicate (1) = 6
                        # This assumes a capable camera/rover pair exists if the problem is solvable.
                        total_cost += 6

        return total_cost
