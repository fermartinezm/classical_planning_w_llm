from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string representation of a PDDL fact like "(predicate arg1 arg2)"
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
         return False # Mismatch in arity

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    conditions, focusing on the collection and communication of data (soil, rock, images).
    It counts the minimum required actions (sample/image, calibrate, communicate)
    and adds a fixed cost (1) for necessary navigation steps for each unachieved goal item.

    # Assumptions
    - Each goal fact (communicated_soil_data, communicated_rock_data, communicated_image_data)
      is treated independently.
    - Navigation cost to a required location (sample point, image point, calibration point,
      communication point) is estimated as a fixed cost (1 action) if the data is not
      yet collected or the data-holding rover is not at a communication point.
    - Resource constraints (empty store for samples, calibrated camera for images)
      and specific rover assignments are largely ignored, focusing on whether the
      *type* of data is collected by *any* capable rover.
    - Calibration is assumed to be needed before taking an image if the image data
      is not yet collected.

    # Heuristic Initialization
    - Extract goal conditions to identify required data communications.
    - Extract static facts:
        - Lander location.
        - Waypoints visible from the lander (communication points).
        - Which objectives are visible from which waypoints (image points).
        - Which calibration targets are for which cameras and visible from which waypoints (calibration points).
        - Which rovers are equipped (though this is less critical for this simplified heuristic).
        - Which cameras are on which rovers and which modes they support (less critical).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h` to 0.
    2. Identify all required soil sample waypoints, rock sample waypoints, and image (objective, mode) pairs from the goal state.
    3. Pre-calculate static information: lander location, communication waypoints, visibility facts for objectives and calibration targets.
    4. For each required soil sample waypoint `w`:
        - If `(communicated_soil_data w)` is not in the current state:
            - Check if `(have_soil_analysis r w)` is in the current state for *any* rover `r`.
            - If yes (data collected):
                - Add 1 to `h` (for the `communicate_soil_data` action).
                - Check if *any* rover `r` that has the data `(have_soil_analysis r w)` is currently at a communication waypoint.
                - If no such rover is at a communication waypoint: Add 1 to `h` (estimated navigation to a communication point).
            - If no (data not collected):
                - Add 1 to `h` (for the `sample_soil` action).
                - Add 1 to `h` (for the `communicate_soil_data` action).
                - Add 1 to `h` (estimated navigation to waypoint `w` for sampling).
                - Add 1 to `h` (estimated navigation to a communication point).
    5. For each required rock sample waypoint `w`:
        - If `(communicated_rock_data w)` is not in the current state:
            - Check if `(have_rock_analysis r w)` is in the current state for *any* rover `r`.
            - If yes (data collected):
                - Add 1 to `h` (for the `communicate_rock_data` action).
                - Check if *any* rover `r` that has the data `(have_rock_analysis r w)` is currently at a communication waypoint.
                - If no such rover is at a communication waypoint: Add 1 to `h` (estimated navigation to a communication point).
            - If no (data not collected):
                - Add 1 to `h` (for the `sample_rock` action).
                - Add 1 to `h` (for the `communicate_rock_data` action).
                - Add 1 to `h` (estimated navigation to waypoint `w` for sampling).
                - Add 1 to `h` (estimated navigation to a communication point).
    6. For each required image `(o, m)`:
        - If `(communicated_image_data o m)` is not in the current state:
            - Check if `(have_image r o m)` is in the current state for *any* rover `r`.
            - If yes (data collected):
                - Add 1 to `h` (for the `communicate_image_data` action).
                - Check if *any* rover `r` that has the data `(have_image r o m)` is currently at a communication waypoint.
                - If no such rover is at a communication waypoint: Add 1 to `h` (estimated navigation to a communication point).
            - If no (data not collected):
                - Add 1 to `h` (for the `calibrate` action).
                - Add 1 to `h` (for the `take_image` action).
                - Add 1 to `h` (for the `communicate_image_data` action).

                # Estimate navigation costs for image collection pipeline
                # Nav to calibration point: Need a waypoint visible from *any* relevant calibration target
                # Nav to image point: Need a waypoint visible from objective o.
                # Simplification: Assume nav to cal point and nav to image point are always needed if image not taken.
                h += 1 # nav to calibration point
                h += 1 # nav to image point (for objective o)
                h += 1 # nav to communication point
    7. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract required goals
        self.required_soil_data = set()
        self.required_rock_data = set()
        self.required_image_data = set() # Stores (objective, mode) tuples

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "communicated_soil_data":
                if len(parts) > 1:
                    self.required_soil_data.add(parts[1]) # waypoint
            elif parts[0] == "communicated_rock_data":
                 if len(parts) > 1:
                    self.required_rock_data.add(parts[1]) # waypoint
            elif parts[0] == "communicated_image_data":
                 if len(parts) > 2:
                    self.required_image_data.add((parts[1], parts[2])) # (objective, mode)

        # Extract static information
        self.lander_location = None
        self.communication_waypoints = set() # Waypoints visible from lander

        visible_facts = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "at_lander":
                 # Assuming only one lander and its location is static
                 if len(parts) > 2:
                    self.lander_location = parts[2]
             elif parts[0] == "visible":
                 visible_facts.add(fact)

        # Now find communication waypoints using the lander location
        if self.lander_location:
             for fact in visible_facts:
                 parts = get_parts(fact)
                 # A waypoint x is a communication point if (visible x lander_wp) or (visible lander_wp x)
                 # The domain defines (visible ?y ?z) where ?y is the rover's location
                 # We need waypoints ?x such that (visible ?x lander_location) is true in the static graph.
                 if len(parts) > 2 and parts[0] == "visible" and parts[2] == self.lander_location:
                     self.communication_waypoints.add(parts[1])
                 # Also consider the reverse if the graph is undirected (which is typical in rovers)
                 if len(parts) > 2 and parts[0] == "visible" and parts[1] == self.lander_location:
                      self.communication_waypoints.add(parts[2])

        # We don't strictly need other static facts like equipped status, camera details,
        # or visible_from mappings for this simplified heuristic, as it assumes
        # feasibility and adds fixed costs for navigation steps if data isn't collected.
        # Keeping the code minimal based on the heuristic logic.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Helper to check if any rover has a specific piece of data
        def has_soil_data(waypoint):
            for fact in state:
                if match(fact, "have_soil_analysis", "*", waypoint):
                    return True
            return False

        def has_rock_data(waypoint):
            for fact in state:
                if match(fact, "have_rock_analysis", "*", waypoint):
                    return True
            return False

        def has_image_data(objective, mode):
            for fact in state:
                if match(fact, "have_image", "*", objective, mode):
                    return True
            return False

        # Helper to check if a specific rover is at a communication point
        def is_at_comm_point(rover):
             for fact in state:
                  if match(fact, "at", rover, "*"):
                       current_wp = get_parts(fact)[2]
                       return current_wp in self.communication_waypoints
             return False # Rover location not found? Should not happen in valid state.


        # --- Estimate cost for Soil Data Goals ---
        for w in self.required_soil_data:
            if f"(communicated_soil_data {w})" not in state:
                if has_soil_data(w):
                    # Data collected, need to communicate
                    h += 1 # communicate action
                    # Check if any rover holding the data is at a comm point
                    found_rover_at_comm = False
                    for fact in state:
                         if match(fact, "have_soil_analysis", "*", w):
                              rover = get_parts(fact)[1]
                              if is_at_comm_point(rover):
                                   found_rover_at_comm = True
                                   break # Found one, no need to check others
                    if not found_rover_at_comm:
                         h += 1 # nav to comm point
                else:
                    # Data not collected, need to sample and communicate
                    h += 1 # sample action
                    h += 1 # communicate action
                    h += 1 # nav to sample point (w)
                    h += 1 # nav to comm point

        # --- Estimate cost for Rock Data Goals ---
        for w in self.required_rock_data:
            if f"(communicated_rock_data {w})" not in state:
                if has_rock_data(w):
                    # Data collected, need to communicate
                    h += 1 # communicate action
                    # Check if any rover holding the data is at a comm point
                    found_rover_at_comm = False
                    for fact in state:
                         if match(fact, "have_rock_analysis", "*", w):
                              rover = get_parts(fact)[1]
                              if is_at_comm_point(rover):
                                   found_rover_at_comm = True
                                   break # Found one, no need to check others
                    if not found_rover_at_comm:
                         h += 1 # nav to comm point
                else:
                    # Data not collected, need to sample and communicate
                    h += 1 # sample action
                    h += 1 # communicate action
                    h += 1 # nav to sample point (w)
                    h += 1 # nav to comm point

        # --- Estimate cost for Image Data Goals ---
        for o, m in self.required_image_data:
            if f"(communicated_image_data {o} {m})" not in state:
                if has_image_data(o, m):
                    # Data collected, need to communicate
                    h += 1 # communicate action
                    # Check if any rover holding the data is at a comm point
                    found_rover_at_comm = False
                    for fact in state:
                         if match(fact, "have_image", "*", o, m):
                              rover = get_parts(fact)[1]
                              if is_at_comm_point(rover):
                                   found_rover_at_comm = True
                                   break # Found one, no need to check others
                    if not found_rover_at_comm:
                         h += 1 # nav to comm point
                else:
                    # Data not collected, need to calibrate, take image, and communicate
                    h += 1 # calibrate action
                    h += 1 # take_image action
                    h += 1 # communicate action

                    # Estimate navigation costs for image collection pipeline
                    # Nav to calibration point: Need a waypoint visible from *any* relevant calibration target
                    # Nav to image point: Need a waypoint visible from objective o.
                    # Simplification: Assume nav to cal point and nav to image point are always needed if image not taken.
                    h += 1 # nav to calibration point
                    h += 1 # nav to image point (for objective o)
                    h += 1 # nav to communication point

        # The heuristic is 0 if and only if all goal facts are in the state.
        # This is guaranteed by the loop structure: h starts at 0 and only increases
        # if a goal fact is *not* in the state. If all goal facts are in the state,
        # the loops are skipped, and h remains 0.
        # The heuristic is finite for solvable states because the number of required
        # data items is finite, and the cost added per item is finite.

        return h
