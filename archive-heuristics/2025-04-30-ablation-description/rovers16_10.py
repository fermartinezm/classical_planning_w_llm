from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions used by the heuristic
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't match patterns longer than the fact
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class, assuming it inherits from a Heuristic base class
# If running this code standalone, you might need a dummy Heuristic class definition
# or ensure the base class is imported correctly in your environment.
# Example dummy definition if needed:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the remaining effort by summing up the costs for each unachieved goal.
    The cost for an unachieved goal is estimated based on the major steps required:
    1. Obtaining the required data (sampling soil/rock, taking image).
    2. Moving to a communication location.
    3. Communicating the data.
    For image goals, an additional cost is added if no suitable camera is calibrated.
    A cost for moving to a communication waypoint is added only once if any communication goal
    is unachieved and no rover is currently at a communication waypoint.

    # Assumptions
    - Actions have a unit cost (implicitly, as we count steps).
    - The heuristic is additive across different unachieved goals, but the move-to-comm cost is factored separately.
    - Resource constraints (like empty stores) and specific navigation paths are not explicitly modeled in the cost, only the need to reach a location type (sample location, image location, communication location).
    - Calibration is a prerequisite for taking an image.

    # Heuristic Initialization
    - Extracts static information: lander locations, visibility graph, communication waypoints,
      camera capabilities (on which rover, which modes supported). Identifies all rovers.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h = 0`.
    2. Identify current state facts: rover locations, `have_...` facts, `calibrated` facts.
    3. Check if any rover is currently at a communication waypoint (`rover_at_comm_waypoint`).
    4. Initialize a flag `needs_communication_move = False`.
    5. For each goal in the task:
       - If the goal fact is in the current state, continue.
       - If the goal fact is NOT in the current state:
         - Add 1 to `h` for the final communication action.
         - Set `needs_communication_move = True` (since this goal needs communication).

         - Check if the necessary prerequisite data (`have_soil_analysis`, `have_rock_analysis`, or `have_image`) is present in the state for *any* rover.
         - If the prerequisite data is NOT present:
           - Add 1 to `h` for the action to obtain the data (sample or take_image).
           - If the goal is an image goal (`communicated_image_data`) and the `have_image` fact is missing:
             - Check if *any* suitable camera (on board a rover, supporting the required mode) is currently calibrated.
             - If NO suitable camera is calibrated, add 1 to `h` for the calibration action.

    6. After checking all goals, if `needs_communication_move` is True AND `rover_at_comm_waypoint` is False:
       - Add 1 to `h` for the cost of moving a rover to a communication waypoint.

    7. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        # Call the base class constructor to store goals and static facts
        super().__init__(task)

        # Extract static information
        self.lander_locations = set()
        self.visible_graph = {} # waypoint -> set of visible waypoints
        self.communication_waypoints = set()
        self.camera_on_rover = {} # camera -> rover
        self.camera_supports_mode = {} # camera -> set of modes
        self.rovers = set() # Set of all rover names

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]

            if predicate == 'at_lander':
                lander, waypoint = parts[1], parts[2]
                self.lander_locations.add(waypoint)
            elif predicate == 'visible':
                w1, w2 = parts[1], parts[2]
                if w1 not in self.visible_graph:
                    self.visible_graph[w1] = set()
                self.visible_graph[w1].add(w2)
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_on_rover[camera] = rover
                self.rovers.add(rover) # Add rover to the set
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_supports_mode:
                    self.camera_supports_mode[camera] = set()
                self.camera_supports_mode[camera].add(mode)
            elif predicate.startswith('equipped_for_'): # Identify rovers from equipped facts
                 rover = parts[1]
                 self.rovers.add(rover)


        # Precompute communication waypoints
        # A waypoint X is a communication waypoint if it is visible from any lander location Y.
        # i.e., (visible X Y) is true for some Y in self.lander_locations
        for w1, visible_set in self.visible_graph.items():
             for lander_loc in self.lander_locations:
                 if lander_loc in visible_set:
                     self.communication_waypoints.add(w1)
                     # Optimization: if w1 can communicate with one lander, it's a comm waypoint
                     # break # No need to check other lander locations for this w1


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        h = 0

        # Extract relevant dynamic facts from the current state
        rover_locations = {} # rover -> waypoint
        have_soil_data_waypoints = set() # {waypoint, ...}
        have_rock_data_waypoints = set() # {waypoint, ...}
        have_image_data_obj_modes = set() # {(objective, mode), ...}
        calibrated_cameras_rovers = set() # {(camera, rover), ...}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'at':
                obj_name = parts[1]
                # Check if the object is one of the known rovers
                if obj_name in self.rovers:
                     rover, waypoint = obj_name, parts[2]
                     rover_locations[rover] = waypoint

            elif predicate == 'have_soil_analysis':
                 rover, waypoint = parts[1], parts[2]
                 have_soil_data_waypoints.add(waypoint) # We only care if data exists for the waypoint
            elif predicate == 'have_rock_analysis':
                 rover, waypoint = parts[1], parts[2]
                 have_rock_data_waypoints.add(waypoint) # We only care if data exists for the waypoint
            elif predicate == 'have_image':
                 rover, objective, mode = parts[1], parts[2], parts[3]
                 have_image_data_obj_modes.add((objective, mode)) # We only care if image exists for obj/mode
            elif predicate == 'calibrated':
                 camera, rover = parts[1], parts[2]
                 calibrated_cameras_rovers.add((camera, rover))

        # Check if any rover is currently at a communication waypoint
        rover_at_comm_waypoint = any(
            loc in self.communication_waypoints
            for loc in rover_locations.values()
        )

        needs_communication_move = False # Flag to add move cost only once if needed

        # Iterate through goals and sum up costs for unachieved ones
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            goal_type = parts[0]

            if goal_type == 'communicated_soil_data':
                waypoint = parts[1]
                h += 1 # Cost for communicate action
                needs_communication_move = True

                if waypoint not in have_soil_data_waypoints:
                    h += 1 # Cost for sample action

            elif goal_type == 'communicated_rock_data':
                waypoint = parts[1]
                h += 1 # Cost for communicate action
                needs_communication_move = True

                if waypoint not in have_rock_data_waypoints:
                    h += 1 # Cost for sample action

            elif goal_type == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                h += 1 # Cost for communicate action
                needs_communication_move = True

                if (objective, mode) not in have_image_data_obj_modes:
                    h += 1 # Cost for take_image action

                    # Check if calibration is needed for this specific image goal (objective, mode)
                    # Is there *any* camera `i` on *any* rover `r` such that:
                    # 1. `(on_board i r)` is true (static)
                    # 2. `(supports i mode)` is true (static)
                    # 3. `(calibrated i r)` is true (state)
                    calibration_needed = True
                    # Iterate through all cameras we know about from static facts that are on rovers
                    for cam, rover in self.camera_on_rover.items():
                        # Check if this camera supports the required mode
                        if mode in self.camera_supports_mode.get(cam, set()):
                            # Check if this camera on this rover is calibrated in the current state
                            if (cam, rover) in calibrated_cameras_rovers:
                                calibration_needed = False
                                break # Found a calibrated suitable camera, no need to check others

                    if calibration_needed:
                         h += 1 # Cost for calibrate action

        # Add cost for moving to communication waypoint only once if needed
        if needs_communication_move and not rover_at_comm_waypoint:
            h += 1 # Cost for moving to communication waypoint


        return h
