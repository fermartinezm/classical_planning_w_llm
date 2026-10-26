from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions from Logistics example
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
    # The zip function will stop when the shortest iterable is exhausted.
    # This means the pattern must match the beginning of the fact's parts.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions needed to achieve uncommunicated goals.
    It sums up estimated costs for each goal independently, ignoring negative
    interactions and resource contention beyond a basic check for empty stores
    for sampling. Navigation is assumed to cost 1 action per required move.

    Goal types:
    - communicated_soil_data(?w)
    - communicated_rock_data(?w)
    - communicated_image_data(?o ?m)

    Estimated costs per uncommunicated goal:
    - Soil/Rock at waypoint ?w:
        - If data already collected by a rover: 2 (move to lander + communicate)
        - If sample still at ?w: 4 (move to sample + sample + move to lander + communicate)
                                  + 1 if all equipped rovers have full stores (drop) = 4 or 5
    - Image for objective ?o, mode ?m:
        - If image already taken by a rover: 2 (move to lander + communicate)
        - If image not taken: 6 (move to cal target + calibrate + move to objective + take image + move to lander + communicate)
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals # Set of goal facts
        static_facts = task.static # Set of static facts

        # Data structures for static information
        self.lander_location = None
        self.lander_visible_waypoints = set()
        self.objective_visible_waypoints = {} # {objective: {waypoint, ...}}
        self.calibration_target_visible_waypoints = {} # {cal_target: {waypoint, ...}}
        self.rover_capabilities = {} # {rover: {capability, ...}} e.g., {'rover1': {'soil', 'imaging'}}
        self.rover_store = {} # {rover: store}
        self.rover_cameras = {} # {rover: {camera, ...}}
        self.camera_supports_mode = {} # {camera: {mode, ...}}
        self.camera_calibration_target = {} # {camera: objective}
        self.calibration_targets = set() # Set of objectives that are calibration targets

        # Parse static facts
        visible_pairs = set() # Store all (w1, w2) where (visible w1 w2) is true

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]

            if predicate == "at_lander":
                # (at_lander ?l ?y)
                if len(parts) > 2: # Basic check for expected structure
                    lander, waypoint = parts[1], parts[2]
                    self.lander_location = waypoint
            elif predicate == "visible":
                # (visible ?w1 ?w2)
                if len(parts) > 2:
                    w1, w2 = parts[1], parts[2]
                    visible_pairs.add((w1, w2))
            elif predicate == "visible_from":
                # (visible_from ?o ?w)
                if len(parts) > 2:
                    objective, waypoint = parts[1], parts[2]
                    self.objective_visible_waypoints.setdefault(objective, set()).add(waypoint)
            elif predicate == "equipped_for_soil_analysis":
                # (equipped_for_soil_analysis ?r)
                if len(parts) > 1:
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif predicate == "equipped_for_rock_analysis":
                # (equipped_for_rock_analysis ?r)
                if len(parts) > 1:
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif predicate == "equipped_for_imaging":
                # (equipped_for_imaging ?r)
                if len(parts) > 1:
                    rover = parts[1]
                    self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif predicate == "store_of":
                # (store_of ?s ?r)
                if len(parts) > 2:
                    store, rover = parts[1], parts[2]
                    self.rover_store[rover] = store
            elif predicate == "on_board":
                # (on_board ?i ?r)
                if len(parts) > 2:
                    camera, rover = parts[1], parts[2]
                    self.rover_cameras.setdefault(rover, set()).add(camera)
            elif predicate == "supports":
                # (supports ?i ?m)
                if len(parts) > 2:
                    camera, mode = parts[1], parts[2]
                    self.camera_supports_mode.setdefault(camera, set()).add(mode)
            elif predicate == "calibration_target":
                # (calibration_target ?i ?t)
                if len(parts) > 2:
                    camera, target = parts[1], parts[2]
                    self.camera_calibration_target[camera] = target
                    self.calibration_targets.add(target) # Keep track of which objectives are targets

        # Populate lander_visible_waypoints using the lander location
        if self.lander_location:
             for w1, w2 in visible_pairs:
                 if w2 == self.lander_location:
                     self.lander_visible_waypoints.add(w1)

        # Populate calibration_target_visible_waypoints
        for cal_target in self.calibration_targets:
            if cal_target in self.objective_visible_waypoints:
                 self.calibration_target_visible_waypoints[cal_target] = self.objective_visible_waypoints[cal_target]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state # Current world state (frozenset of fact strings)

        total_cost = 0

        # Check each goal fact
        for goal_fact in self.goals:
            if goal_fact in state:
                # Goal is already achieved
                continue

            # Goal is not achieved, estimate cost
            parts = get_parts(goal_fact)
            if not parts: # Should not happen with valid goals, but safety check
                continue
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                # Goal: (communicated_soil_data ?w)
                if len(parts) < 2: continue # Malformed goal
                waypoint = parts[1]

                # Check if any rover already has the soil analysis
                rover_has_data = any(match(fact, "have_soil_analysis", "*", waypoint) for fact in state)

                if rover_has_data:
                    # Cost to communicate: move to lander-visible + communicate
                    total_cost += 2
                else:
                    # Data needs to be sampled
                    soil_sample_at_waypoint = f"(at_soil_sample {waypoint})" in state

                    if soil_sample_at_waypoint:
                        # Cost to sample and communicate:
                        # move to sample + sample + move to lander + communicate
                        cost = 4

                        # Check if a drop is needed before sampling
                        needs_drop = True # Assume drop is needed unless an empty store is found
                        soil_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
                        if soil_rovers: # Ensure there's at least one equipped rover
                            for rover in soil_rovers:
                                store = self.rover_store.get(rover)
                                # Check if the rover exists and has a store fact associated
                                if store and f"(empty {store})" in state:
                                    needs_drop = False
                                    break # Found a rover with an empty store

                            if needs_drop:
                                cost += 1 # Add cost for drop action

                        total_cost += cost
                    # If sample is not at waypoint and no rover has it, ignore (assume unreachable)

            elif predicate == "communicated_rock_data":
                # Goal: (communicated_rock_data ?w)
                if len(parts) < 2: continue # Malformed goal
                waypoint = parts[1]

                # Check if any rover already has the rock analysis
                rover_has_data = any(match(fact, "have_rock_analysis", "*", waypoint) for fact in state)

                if rover_has_data:
                    # Cost to communicate: move to lander-visible + communicate
                    total_cost += 2
                else:
                    # Data needs to be sampled
                    rock_sample_at_waypoint = f"(at_rock_sample {waypoint})" in state

                    if rock_sample_at_waypoint:
                        # Cost to sample and communicate:
                        # move to sample + sample + move to lander + communicate
                        cost = 4

                        # Check if a drop is needed before sampling
                        needs_drop = True # Assume drop is needed unless an empty store is found
                        rock_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
                        if rock_rovers: # Ensure there's at least one equipped rover
                             for rover in rock_rovers:
                                store = self.rover_store.get(rover)
                                # Check if the rover exists and has a store fact associated
                                if store and f"(empty {store})" in state:
                                    needs_drop = False
                                    break # Found a rover with an empty store

                             if needs_drop:
                                cost += 1 # Add cost for drop action

                        total_cost += cost
                    # If sample is not at waypoint and no rover has it, ignore (assume unreachable)

            elif predicate == "communicated_image_data":
                # Goal: (communicated_image_data ?o ?m)
                if len(parts) < 3: continue # Malformed goal
                objective, mode = parts[1], parts[2]

                # Check if any rover already has the image
                rover_has_data = any(match(fact, "have_image", "*", objective, mode) for fact in state)

                if rover_has_data:
                    # Cost to communicate: move to lander-visible + communicate
                    total_cost += 2
                else:
                    # Image needs to be taken and communicated
                    # Cost: move to cal target + calibrate + move to objective + take image + move to lander + communicate
                    total_cost += 6

            # Add cost for other potential goal types if they existed (e.g., being at a specific location)
            # The provided domain only has communicated_... goals.

        # The heuristic is 0 iff all goals are in the state.
        # If total_cost is 0, it means the loop finished without finding any uncommunicated goals,
        # which implies all goals are in the state. So, h=0 iff goal is reached.

        return total_cost
