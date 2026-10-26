# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque
import math

# Helper functions to parse PDDL facts
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the Heuristic class
# class roversHeuristic(Heuristic): # Uncomment and inherit if Heuristic base is provided
class roversHeuristic:
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal
    conditions. It sums up the estimated costs for each unachieved goal fact.
    The cost for each goal fact is estimated based on the minimum number of
    actions needed to achieve it, including sampling/imaging, calibration (for images),
    and navigation to the required locations (sample/image location, calibration
    location, and communication location). Navigation costs are estimated using
    shortest path (BFS) on the waypoint graph for the relevant rover(s).

    # Assumptions
    - The heuristic assumes that for any unachieved soil/rock goal, if the
      corresponding sample is still present, it needs to be sampled. If the
      sample is gone, it assumes the sample data exists (sampled by some rover)
      and only communication steps are needed.
    - For image goals, it assumes a suitable imaging-equipped rover with a camera
      supporting the required mode exists and calculates the cost for one such pair.
    - It calculates navigation costs using BFS on the waypoint graph defined
      by `can_traverse` facts for the specific rover(s).
    - It assumes that if a store is full on a rover that needs to sample,
      one 'drop' action is sufficient to make a store empty.
    - It assumes solvable problems, so required static conditions (like
      `equipped_for`, `on_board`, `supports`, `calibration_target`,
      `visible_from`, `can_traverse`) are met for necessary tasks.
    - It assumes there is only one lander.

    # Heuristic Initialization
    The heuristic extracts and stores static information from the task:
    - Which rovers are equipped for soil, rock, and imaging.
    - Mapping of rovers to their stores (rover -> store).
    - Mapping of cameras to rovers they are on board (camera -> rover).
    - Mapping of cameras to modes they support (camera -> set of modes).
    - Mapping of cameras to their calibration targets (camera -> objective).
    - Mapping of objectives to waypoints they are visible from (objective -> set of waypoints).
    - The navigation graph (can_traverse) for each rover (rover -> {waypoint -> set of waypoints}).
    - The lander's location and the waypoints visible from it (communication points).
    - The initial locations of soil and rock samples.
    - A set of all rover names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Extract dynamic facts from the current state (rover locations, data held, calibration status, store status, sample presence).
    3. For each goal fact in `task.goals`:
        a. If the goal fact is already in the current state, continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            i. Calculate cost for this specific goal. Start with 1 for `communicate_soil_data`.
            ii. Check if any soil-equipped rover `r` has `(have_soil_analysis r w)`.
            iii. If a rover `r_data` has the data:
                - Add the minimum navigation cost for `r_data` to reach any communication waypoint.
            iv. If no rover has the data:
                - Add 1 for the `sample_soil` action.
                - Check if `(at_soil_sample w)` is in the current state.
                - If yes:
                    - Find the minimum navigation cost for any soil-equipped rover to reach waypoint `w`. Add this cost.
                    - Check if any soil-equipped rover has an empty store. If not, add 1 for a `drop` action.
                - If no (sample is gone): Assume sampling steps are not needed (data must exist elsewhere or was dropped). Add 0 cost for sampling steps.
                - Find the minimum navigation cost for any soil-equipped rover to reach any communication waypoint. Add this cost.
            v. Add the calculated cost for this goal to the total cost. Handle infinity propagation.
        c. If the goal is `(communicated_rock_data ?w)`:
            i. Analogous calculation to the soil data goal, using rock-specific facts and rovers.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Calculate cost for this specific goal. Start with 1 for `communicate_image_data`.
            ii. Find a suitable imaging-equipped rover `r` with a camera `i` supporting mode `m`. (Pick one).
            iii. Check if `(have_image r o m)` is in the current state for this rover/camera.
            iv. If no:
                - Add 1 for the `take_image` action.
                - Add 1 for the `calibrate` action (needed before `take_image`).
                - Find the calibration target `t` for camera `i`. Find waypoints `W_calib` visible from `t`. Find the minimum navigation cost for rover `r` to reach any waypoint in `W_calib`. Add this cost.
                - Find waypoints `W_image` visible from objective `o`. Find the minimum navigation cost for rover `r` to reach any waypoint in `W_image`. Add this cost.
            v. Find the minimum navigation cost for rover `r` to reach any communication waypoint. Add this cost.
            vi. Add the calculated cost for this goal to the total cost. Handle infinity propagation.
    4. Return the total calculated cost.

    Helper Function `min_nav_cost(state, rover_or_rovers, target_waypoint_or_waypoints)`:
    - Takes the current state, a single rover or a set of rovers, and a single target waypoint or a set of target waypoints.
    - Finds the current location(s) of the rover(s) from the state.
    - Performs a BFS starting from the current location(s) on the relevant rover's navigation graph(s) to find the minimum distance to any of the target waypoint(s).
    - Returns the minimum distance found, or math.inf if no path exists.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Extract Static Information ---
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_to_store = {} # rover -> store
        self.camera_to_rover = {} # camera -> rover
        self.camera_supports = {} # camera -> set of modes
        self.camera_calib_target = {} # camera -> objective
        self.objective_visible_from = {} # objective -> set of waypoints
        self.rover_can_traverse = {} # rover -> {waypoint -> set of waypoints} (adj list)
        self.lander_location = None
        self.comm_waypoints = set() # waypoints visible from lander
        self.initial_soil_samples = set() # waypoints with initial soil samples
        self.initial_rock_samples = set() # waypoints with initial rock samples
        self.all_rovers = set() # Keep track of all rover names

        # Need to parse initial state for initial sample locations
        initial_state_facts = task.initial_state

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
                self.all_rovers.add(parts[1])
            elif parts[0] == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])
                self.all_rovers.add(parts[1])
            elif parts[0] == "equipped_for_imaging":
                self.equipped_imaging.add(parts[1])
                self.all_rovers.add(parts[1])
            elif parts[0] == "store_of":
                self.rover_to_store[parts[2]] = parts[1] # rover -> store
            elif parts[0] == "on_board":
                self.camera_to_rover[parts[1]] = parts[2] # camera -> rover
            elif parts[0] == "supports":
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_supports:
                    self.camera_supports[camera] = set()
                self.camera_supports[camera].add(mode)
            elif parts[0] == "calibration_target":
                self.camera_calib_target[parts[1]] = parts[2] # camera -> objective
            elif parts[0] == "visible_from":
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visible_from:
                    self.objective_visible_from[objective] = set()
                self.objective_visible_from[objective].add(waypoint)
            elif parts[0] == "can_traverse":
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                if rover not in self.rover_can_traverse:
                    self.rover_can_traverse[rover] = {}
                if wp1 not in self.rover_can_traverse[rover]:
                    self.rover_can_traverse[rover][wp1] = set()
                self.rover_can_traverse[rover][wp1].add(wp2)
                self.all_rovers.add(rover) # Ensure all rovers are tracked

        # Get lander location from static facts (or initial state if not static)
        lander_facts = [fact for fact in task.static if match(fact, "at_lander", "*", "*")]
        if not lander_facts: # Check initial state if not in static
             lander_facts = [fact for fact in initial_state_facts if match(fact, "at_lander", "*", "*")]

        if lander_facts:
             self.lander_location = get_parts(lander_facts[0])[2] # Assuming only one lander

        # Get communication waypoints (visible from lander location)
        if self.lander_location:
             for fact in task.static:
                 if match(fact, "visible", "*", self.lander_location):
                     self.comm_waypoints.add(get_parts(fact)[1])
             # Also check if lander location is visible from itself (common)
             # This is implicitly handled if (visible lander_loc lander_loc) is in static facts.


        # Get initial sample locations from initial state
        for fact in initial_state_facts:
            if match(fact, "at_soil_sample", "*"):
                self.initial_soil_samples.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.initial_rock_samples.add(get_parts(fact)[1])


    def min_nav_cost(self, state, rovers, target_waypoint_or_waypoints):
        """
        Calculates the minimum navigation cost (BFS distance) from the current
        location of any of the given rovers to any of the target waypoints.

        Args:
            state: The current state (frozenset of facts).
            rovers: A single rover name (string) or a set of rover names.
            target_waypoints: A single waypoint name (string) or a set of waypoint names.

        Returns:
            The minimum BFS distance, or math.inf if no path exists.
        """
        if isinstance(rovers, str):
            rovers = {rovers}
        if isinstance(target_waypoint_or_waypoints, str):
            target_waypoints = {target_waypoint_or_waypoints}
        else:
            target_waypoints = set(target_waypoint_or_waypoints) # Ensure it's a set

        if not rovers or not target_waypoints:
             return math.inf # Cannot navigate if no rovers or no targets

        min_dist = math.inf

        # Find current locations of the given rovers
        current_rover_locations = {} # rover -> waypoint
        for fact in state:
            if match(fact, "at", "*", "*"):
                rover, wp = get_parts(fact)[1], get_parts(fact)[2]
                if rover in rovers:
                    current_rover_locations[rover] = wp

        if not current_rover_locations:
             # None of the specified rovers are currently located in the state.
             # This might happen if a rover object exists but isn't 'at' any waypoint.
             # Should not happen in valid states/problems.
             return math.inf

        # Perform BFS for each rover
        for rover, start_node in current_rover_locations.items():
            if start_node in target_waypoints:
                return 0 # Already at a target waypoint

            # Get the navigation graph for this rover
            graph = self.rover_can_traverse.get(rover, {})
            if not graph: continue # Rover cannot traverse anywhere

            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_node, dist = queue.popleft()

                if current_node in target_waypoints:
                    min_dist = min(min_dist, dist)
                    # Found a path for this rover to *a* target, can stop this specific BFS
                    break # Optimization: break inner while loop

                # Check neighbors
                for neighbor in graph.get(current_node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return min_dist # Return the minimum distance found across all rovers and targets

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Extract dynamic facts for quick lookup
        state_facts = set(state) # Convert frozenset to set for faster lookup

        rover_locations = {} # rover -> waypoint
        have_soil = {} # rover -> set of waypoints
        have_rock = {} # rover -> set of waypoints
        have_image = {} # rover -> set of (objective, mode)
        calibrated_cameras = set() # (camera, rover)
        empty_stores = set() # store
        full_stores = set() # store
        at_soil_sample = set() # waypoint
        at_rock_sample = set() # waypoint

        for fact in state_facts:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] in self.all_rovers: # Only track rover locations
                 rover_locations[parts[1]] = parts[2]
            elif parts[0] == "have_soil_analysis":
                 r, w = parts[1], parts[2]
                 if r not in have_soil: have_soil[r] = set()
                 have_soil[r].add(w)
            elif parts[0] == "have_rock_analysis":
                 r, w = parts[1], parts[2]
                 if r not in have_rock: have_rock[r] = set()
                 have_rock[r].add(w)
            elif parts[0] == "have_image":
                 r, o, m = parts[1], parts[2], parts[3]
                 if r not in have_image: have_image[r] = set()
                 have_image[r].add((o, m))
            elif parts[0] == "calibrated":
                 calibrated_cameras.add((parts[1], parts[2])) # (camera, rover)
            elif parts[0] == "empty":
                 empty_stores.add(parts[1])
            elif parts[0] == "full":
                 full_stores.add(parts[1])
            elif parts[0] == "at_soil_sample":
                 at_soil_sample.add(parts[1])
            elif parts[0] == "at_rock_sample":
                 at_rock_sample.add(parts[1])

        # Iterate through goal facts
        for goal in self.goals:
            if goal in state_facts:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                w = parts[1]
                cost_for_goal = 1 # communicate_soil_data

                # Check if data is already collected by any soil-equipped rover
                rover_with_data = None
                for r in self.equipped_soil:
                    if w in have_soil.get(r, set()):
                        rover_with_data = r
                        break

                if rover_with_data:
                    # Rover has data, just need to communicate
                    nav_cost = self.min_nav_cost(state, rover_with_data, self.comm_waypoints) # Use specific rover
                    if nav_cost == math.inf: return math.inf # Cannot reach comm point
                    cost_for_goal += nav_cost
                else:
                    # Need to sample
                    cost_for_goal += 1 # sample_soil
                    if w in at_soil_sample: # Sample is available
                        # Nav cost for *any* soil rover to w
                        nav_cost_to_sample = self.min_nav_cost(state, self.equipped_soil, w)
                        if nav_cost_to_sample == math.inf: return math.inf # Cannot reach sample
                        cost_for_goal += nav_cost_to_sample

                        # Check if any soil rover has an empty store
                        has_empty_store = False
                        for r in self.equipped_soil:
                            store = self.rover_to_store.get(r)
                            if store in empty_stores:
                                has_empty_store = True
                                break
                        if not has_empty_store:
                            cost_for_goal += 1 # drop (assuming one drop is enough)
                    # If sample not available, assume sampled, no cost for sampling steps.

                    # After sampling (or if sample was gone), need to communicate
                    # Calculate nav cost for *any* soil rover to comm point
                    nav_cost_to_comm = self.min_nav_cost(state, self.equipped_soil, self.comm_waypoints) # Use any soil rover
                    if nav_cost_to_comm == math.inf: return math.inf # Cannot reach comm point
                    cost_for_goal += nav_cost_to_comm

                total_cost += cost_for_goal


            elif predicate == "communicated_rock_data":
                w = parts[1]
                cost_for_goal = 1 # communicate_rock_data

                # Check if data is already collected by any rock-equipped rover
                rover_with_data = None
                for r in self.equipped_rock:
                    if w in have_rock.get(r, set()):
                        rover_with_data = r
                        break

                if rover_with_data:
                    # Rover has data, just need to communicate
                    nav_cost = self.min_nav_cost(state, rover_with_data, self.comm_waypoints) # Use specific rover
                    if nav_cost == math.inf: return math.inf # Cannot reach comm point
                    cost_for_goal += nav_cost
                else:
                    # Need to sample
                    cost_for_goal += 1 # sample_rock
                    if w in at_rock_sample: # Sample is available
                        # Nav cost for *any* rock rover to w
                        nav_cost_to_sample = self.min_nav_cost(state, self.equipped_rock, w)
                        if nav_cost_to_sample == math.inf: return math.inf # Cannot reach sample
                        cost_for_goal += nav_cost_to_sample

                        # Check if any rock rover has an empty store
                        has_empty_store = False
                        for r in self.equipped_rock:
                            store = self.rover_to_store.get(r)
                            if store in empty_stores:
                                has_empty_store = True
                                break
                        if not has_empty_store:
                            cost_for_goal += 1 # drop (assuming one drop is enough)
                    # If sample not available, assume sampled, no cost for sampling steps.

                    # After sampling (or if sample was gone), need to communicate
                    # Calculate nav cost for *any* rock rover to comm point
                    nav_cost_to_comm = self.min_nav_cost(state, self.equipped_rock, self.comm_waypoints) # Use any rock rover
                    if nav_cost_to_comm == math.inf: return math.inf # Cannot reach comm point
                    cost_for_goal += nav_cost_to_comm

                total_cost += cost_for_goal


            elif predicate == "communicated_image_data":
                o, m = parts[1], parts[2]
                cost_for_goal = 1 # communicate_image_data

                # Find a suitable imaging-equipped rover `r` with camera `i` supporting `m`.
                # Let's find *any* such pair (r, i) and calculate cost for it.
                suitable_rover_camera = None
                for r in self.equipped_imaging:
                    for i, modes in self.camera_supports.items():
                        if self.camera_to_rover.get(i) == r and m in modes:
                            suitable_rover_camera = (r, i)
                            break
                    if suitable_rover_camera:
                        break

                if suitable_rover_camera is None:
                    # Should not happen in solvable problems, but handle defensively
                    return math.inf # Cannot achieve this goal

                r, i = suitable_rover_camera

                # Check if image is already taken by this rover/camera
                image_taken = (o, m) in have_image.get(r, set())

                if not image_taken:
                    # Need to take image
                    cost_for_goal += 1 # take_image

                    # Always need to calibrate before taking a new image
                    cost_for_goal += 1 # calibrate
                    # Find calibration target t for i
                    t = self.camera_calib_target.get(i)
                    if t is None: return math.inf # Should not happen
                    # Find waypoint w visible from t
                    calib_waypoints = self.objective_visible_from.get(t, set())
                    if not calib_waypoints: return math.inf # Should not happen
                    nav_cost_to_calib = self.min_nav_cost(state, r, calib_waypoints) # Nav cost for r to *any* calib waypoint
                    if nav_cost_to_calib == math.inf: return math.inf # Cannot reach calib point
                    cost_for_goal += nav_cost_to_calib

                    # Need to be at image point
                    image_waypoints = self.objective_visible_from.get(o, set())
                    if not image_waypoints: return math.inf # Should not happen
                    nav_cost_to_image = self.min_nav_cost(state, r, image_waypoints) # Nav cost for r to *any* image waypoint
                    if nav_cost_to_image == math.inf: return math.inf # Cannot reach image point
                    cost_for_goal += nav_cost_to_image

                # After taking image (or if already taken), need to communicate
                nav_cost_to_comm = self.min_nav_cost(state, r, self.comm_waypoints)
                if nav_cost_to_comm == math.inf: return math.inf # Cannot reach comm point
                cost_for_goal += nav_cost_to_comm

                total_cost += cost_for_goal

            # Add costs for other potential goal types if they existed... (e.g. just having data, not communicating)
            # But the provided domain only has communicated_ data goals.

        return total_cost
