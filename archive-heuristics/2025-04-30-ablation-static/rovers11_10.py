from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the number of arguments in the pattern.
    # This prevents matching "(at rover1 waypoint1 extra)" with ("at", "*", "*").
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal
    conditions, which involve communicating soil, rock, and image data. It
    breaks down the cost for each unachieved communication goal based on whether
    the intermediate data (sample/image) has already been collected.

    # Assumptions
    - Each unachieved goal is considered independently.
    - The cost of movement between relevant waypoints (sample location, image location,
      calibration location, communication location) is estimated as 1 action per move.
    - The cost of sampling, taking an image, calibrating, and communicating is estimated as 1 action each.
    - Dropping a sample is not explicitly modeled in the cost, assuming stores are managed efficiently or not a bottleneck for the heuristic estimate.
    - Equipment requirements and specific camera/mode/objective visibility constraints are simplified; the heuristic assumes a capable rover/camera exists and can reach the necessary locations.
    - If a sample/image is required but not yet collected, the heuristic adds a fixed cost representing the sequence of actions (move, collect, move, communicate for samples/rocks; move, calibrate, move, image, move, communicate for images).
    - If the sample/image is already collected (indicated by `have_...` facts), the heuristic adds a fixed cost representing the sequence of actions to communicate it (move to comm, communicate).

    # Heuristic Initialization
    - The heuristic stores the set of goal conditions from the task.
    - It also parses relevant static facts from the initial state to satisfy the requirement, although the primary heuristic logic relies on checking dynamic `have_...` facts in the state rather than using the parsed static data for cost calculation in this simplified version.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:

    1. Initialize total estimated cost to 0.
    2. Iterate through each goal fact specified in the problem.
    3. For each goal fact:
       a. If the goal fact is already true in the current state, add 0 cost for this goal and continue to the next goal.
       b. If the goal fact is `(communicated_soil_data ?w)`:
          - Check if any rover currently has the soil analysis for waypoint `?w` (i.e., `(have_soil_analysis ?r ?w)` is true for some rover `?r` in the state).
          - If yes: Add 2 to the total cost (estimated cost for moving to a communication point and communicating).
          - If no: Add 4 to the total cost (estimated cost for moving to the sample location, sampling, moving to a communication point, and communicating).
       c. If the goal fact is `(communicated_rock_data ?w)`:
          - Check if any rover currently has the rock analysis for waypoint `?w` (i.e., `(have_rock_analysis ?r ?w)` is true for some rover `?r` in the state).
          - If yes: Add 2 to the total cost (estimated cost for moving to a communication point and communicating).
          - If no: Add 4 to the total cost (estimated cost for moving to the sample location, sampling, moving to a communication point, and communicating).
       d. If the goal fact is `(communicated_image_data ?o ?m)`:
          - Check if any rover currently has the image for objective `?o` in mode `?m` (i.e., `(have_image ?r ?o ?m)` is true for some rover `?r` in the state).
          - If yes: Add 2 to the total cost (estimated cost for moving to a communication point and communicating).
          - If no: Add 6 to the total cost (estimated cost for moving to a calibration point, calibrating, moving to an imaging point, taking the image, moving to a communication point, and communicating).
    4. The total estimated cost accumulated across all unachieved goals is the heuristic value for the state.

    This heuristic is non-admissible as it sums costs for independent goals and simplifies action sequences, but it provides a quick estimate that differentiates states based on progress towards collecting and communicating data.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing goal conditions and parsing static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static # Store static facts

        # Parse some static facts as required, even if not heavily used in __call__
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        self.camera_details = {} # {camera: {'on_board': rover, 'supports': {mode}, 'calibration_target': objective}}
        self.objective_visible_from_waypoints = {} # {objective: {waypoint}}
        self.lander_at = None
        self.visible_waypoints = {} # {wp1: {wp2, wp3}, ...}
        self.can_traverse_rovers = {} # {rover: {(wp_from, wp_to)}}
        self.store_owner = {} # {store: rover}
        self.soil_samples_init = set() # Waypoints with soil samples initially
        self.rock_samples_init = set() # Waypoints with rock samples initially

        # Static facts are typically found in the initial state predicates
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts if any

             predicate = parts[0]
             if predicate == 'at_lander' and len(parts) == 3:
                 self.lander_at = parts[2] # Assuming only one lander
             elif predicate == 'equipped_for_soil_analysis' and len(parts) == 2:
                 self.equipped_for_soil.add(parts[1])
             elif predicate == 'equipped_for_rock_analysis' and len(parts) == 2:
                 self.equipped_for_rock.add(parts[1])
             elif predicate == 'equipped_for_imaging' and len(parts) == 2:
                 self.equipped_for_imaging.add(parts[1])
             elif predicate == 'on_board' and len(parts) == 3:
                 camera, rover = parts[1], parts[2]
                 self.camera_details.setdefault(camera, {})['on_board'] = rover
             elif predicate == 'supports' and len(parts) == 3:
                 camera, mode = parts[1], parts[2]
                 self.camera_details.setdefault(camera, {}).setdefault('supports', set()).add(mode)
             elif predicate == 'calibration_target' and len(parts) == 3:
                 camera, objective = parts[1], parts[2]
                 self.camera_details.setdefault(camera, {})['calibration_target'] = objective
             elif predicate == 'visible_from' and len(parts) == 3:
                 objective, waypoint = parts[1], parts[2]
                 self.objective_visible_from_waypoints.setdefault(objective, set()).add(waypoint)
             elif predicate == 'visible' and len(parts) == 3:
                 wp1, wp2 = parts[1], parts[2]
                 self.visible_waypoints.setdefault(wp1, set()).add(wp2)
             elif predicate == 'can_traverse' and len(parts) == 4:
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 self.can_traverse_rovers.setdefault(rover, set()).add((wp1, wp2))
             elif predicate == 'store_of' and len(parts) == 3:
                 store, rover = parts[1], parts[2]
                 self.store_owner[store] = rover
             elif predicate == 'at_soil_sample' and len(parts) == 2:
                 self.soil_samples_init.add(parts[1])
             elif predicate == 'at_rock_sample' and len(parts) == 2:
                 self.rock_samples_init.add(parts[1])

        # Identify communication waypoints (visible from lander location)
        self.comm_waypoints = set()
        if self.lander_at:
             # Waypoints visible *from* the lander location
             if self.lander_at in self.visible_waypoints:
                 self.comm_waypoints.update(self.visible_waypoints[self.lander_at])
             # Waypoints from which the lander location is visible
             # This requires iterating through all visible facts where the second waypoint is the lander location
             for wp1, visible_set in self.visible_waypoints.items():
                 if self.lander_at in visible_set:
                     self.comm_waypoints.add(wp1)
             # The lander location itself is also a communication point
             self.comm_waypoints.add(self.lander_at)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state
        total_cost = 0

        # Check each goal condition
        for goal in self.goals:
            # If the goal is already met, it contributes 0 to the heuristic
            if goal in state:
                continue

            # Parse the goal fact
            parts = get_parts(goal)
            if not parts: continue # Should not happen for valid goals

            predicate = parts[0]

            if predicate == 'communicated_soil_data' and len(parts) == 2:
                waypoint = parts[1]
                # Check if the soil analysis is already collected by any rover
                # We check for the dynamic fact (have_soil_analysis ?r ?w) in the current state
                have_analysis = any(match(fact, 'have_soil_analysis', '*', waypoint) for fact in state)
                if have_analysis:
                    # Need to move a rover with the sample to a comm point and communicate
                    # Estimated cost: 1 (move) + 1 (communicate) = 2
                    total_cost += 2
                else:
                    # Need to sample the soil, move to comm, and communicate
                    # Estimated cost: 1 (move to sample) + 1 (sample) + 1 (move to comm) + 1 (communicate) = 4
                    # This assumes the sample is available at the waypoint and a capable rover/store exists.
                    total_cost += 4

            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                waypoint = parts[1]
                # Check if the rock analysis is already collected by any rover
                # We check for the dynamic fact (have_rock_analysis ?r ?w) in the current state
                have_analysis = any(match(fact, 'have_rock_analysis', '*', waypoint) for fact in state)
                if have_analysis:
                    # Need to move a rover with the sample to a comm point and communicate
                    # Estimated cost: 1 (move) + 1 (communicate) = 2
                    total_cost += 2
                else:
                    # Need to sample the rock, move to comm, and communicate
                    # Estimated cost: 1 (move to sample) + 1 (sample) + 1 (move to comm) + 1 (communicate) = 4
                    # This assumes the sample is available at the waypoint and a capable rover/store exists.
                    total_cost += 4

            elif predicate == 'communicated_image_data' and len(parts) == 3:
                objective = parts[1]
                mode = parts[2]
                # Check if the image is already taken by any rover
                # We check for the dynamic fact (have_image ?r ?o ?m) in the current state
                have_image = any(match(fact, 'have_image', '*', objective, mode) for fact in state)
                if have_image:
                    # Need to move a rover with the image to a comm point and communicate
                    # Estimated cost: 1 (move) + 1 (communicate) = 2
                    total_cost += 2
                else:
                    # Need to calibrate, take image, move to comm, and communicate
                    # Estimated cost: 1 (move to calibrate) + 1 (calibrate) + 1 (move to image) + 1 (take image) + 1 (move to comm) + 1 (communicate) = 6
                    # This assumes a capable rover/camera/mode exists and the objective/calibration target are visible from some reachable waypoints.
                    total_cost += 6
            # Add other goal types if necessary, but the domain file and examples primarily show these three communicated types.

        return total_cost
