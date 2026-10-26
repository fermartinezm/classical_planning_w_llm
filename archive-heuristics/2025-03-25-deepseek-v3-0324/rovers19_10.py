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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all goals by:
    1. Identifying unsatisfied communication goals (soil, rock, image data)
    2. Calculating the steps required for each goal:
       - For soil/rock data: navigate to sample location, sample, navigate to lander, communicate
       - For image data: navigate to calibration target, calibrate, navigate to imaging location, take image, navigate to lander, communicate
    3. Considering parallel execution where possible (multiple goals achievable by same rover)

    # Assumptions:
    - Each unsatisfied goal requires a separate sequence of actions
    - Navigation between waypoints costs 1 action per hop (optimistic estimate)
    - Sampling, imaging, and communication actions each cost 1 action
    - Rovers can only work on one goal at a time (conservative estimate)

    # Heuristic Initialization
    - Extract goal conditions and static facts
    - Build mapping of waypoints to their connected neighbors (navigation graph)
    - Identify lander locations and sample locations from static facts
    - Identify camera capabilities and calibration targets

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unsatisfied communication goals from the goal state
    2. For each unsatisfied soil/rock data goal:
       - Find closest rover that can perform the task
       - Estimate navigation cost to sample location
       - Add sampling and communication actions
    3. For each unsatisfied image data goal:
       - Find rover with appropriate camera
       - Estimate calibration and imaging navigation costs
       - Add calibration, imaging, and communication actions
    4. Sum all estimated actions, considering some parallelism where possible
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract waypoint connectivity graph
        self.visible = {}
        self.can_traverse = {}
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                _, wp1, wp2 = get_parts(fact)
                self.visible.setdefault(wp1, set()).add(wp2)
            elif match(fact, "can_traverse", "*", "*", "*"):
                _, rover, wp1, wp2 = get_parts(fact)
                self.can_traverse.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
        
        # Extract lander positions
        self.lander_positions = {}
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                _, lander, wp = get_parts(fact)
                self.lander_positions[lander] = wp
        
        # Extract sample locations
        self.rock_samples = set()
        self.soil_samples = set()
        for fact in self.static:
            if match(fact, "at_rock_sample", "*"):
                self.rock_samples.add(get_parts(fact)[1])
            elif match(fact, "at_soil_sample", "*"):
                self.soil_samples.add(get_parts(fact)[1])
        
        # Extract camera capabilities
        self.camera_modes = {}
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.calibration_targets[camera] = objective
        
        # Extract rover equipment
        self.rover_equipment = {}
        for fact in self.static:
            if match(fact, "equipped_for_*", "*"):
                parts = get_parts(fact)
                capability = parts[0].split('_')[-1]  # soil_analysis, rock_analysis, imaging
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add(capability)
        
        # Extract objective visibility
        self.objective_visibility = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                _, objective, wp = get_parts(fact)
                self.objective_visibility.setdefault(objective, set()).add(wp)

    def __call__(self, node):
        """Estimate the number of actions required to achieve all goals."""
        state = node.state
        
        # Check if we've already reached the goal
        if self.goals <= state:
            return 0
        
        total_cost = 0
        
        # Track which rovers have been assigned tasks (simplistic approach)
        assigned_rovers = set()
        
        # Process soil data goals
        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                wp = get_parts(goal)[1]
                if goal in state:
                    continue
                
                # Find a rover that can do this task
                for rover, equipment in self.rover_equipment.items():
                    if "soil_analysis" not in equipment:
                        continue
                    if rover in assigned_rovers:
                        continue
                    
                    # Estimate navigation cost to sample location
                    rover_pos = None
                    for fact in state:
                        if match(fact, "at", rover, "*"):
                            rover_pos = get_parts(fact)[2]
                            break
                    
                    if not rover_pos:
                        continue  # Rover position unknown
                    
                    # Simple estimate: 1 action per waypoint hop (optimistic)
                    nav_cost = 1  # At least one navigate action
                    
                    # Check if we already have the sample
                    has_sample = any(
                        match(fact, "have_soil_analysis", rover, wp)
                        for fact in state
                    )
                    
                    if has_sample:
                        # Just need to navigate to lander and communicate
                        lander_pos = next(iter(self.lander_positions.values()))  # Assuming one lander
                        total_cost += nav_cost + 1  # navigate + communicate
                    else:
                        # Need to navigate to sample, sample, navigate to lander, communicate
                        total_cost += nav_cost + 1 + nav_cost + 1  # nav+sample+nav+communicate
                    
                    assigned_rovers.add(rover)
                    break
        
        # Process rock data goals
        for goal in self.goals:
            if match(goal, "communicated_rock_data", "*"):
                wp = get_parts(goal)[1]
                if goal in state:
                    continue
                
                # Find a rover that can do this task
                for rover, equipment in self.rover_equipment.items():
                    if "rock_analysis" not in equipment:
                        continue
                    if rover in assigned_rovers:
                        continue
                    
                    # Estimate navigation cost
                    rover_pos = None
                    for fact in state:
                        if match(fact, "at", rover, "*"):
                            rover_pos = get_parts(fact)[2]
                            break
                    
                    if not rover_pos:
                        continue
                    
                    nav_cost = 1  # At least one navigate action
                    
                    # Check if we already have the sample
                    has_sample = any(
                        match(fact, "have_rock_analysis", rover, wp)
                        for fact in state
                    )
                    
                    if has_sample:
                        # Just need to navigate to lander and communicate
                        lander_pos = next(iter(self.lander_positions.values()))
                        total_cost += nav_cost + 1  # navigate + communicate
                    else:
                        # Need to navigate to sample, sample, navigate to lander, communicate
                        total_cost += nav_cost + 1 + nav_cost + 1  # nav+sample+nav+communicate
                    
                    assigned_rovers.add(rover)
                    break
        
        # Process image data goals
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                objective, mode = get_parts(goal)[1], get_parts(goal)[2]
                if goal in state:
                    continue
                
                # Find a rover with a camera that supports this mode
                for rover, equipment in self.rover_equipment.items():
                    if "imaging" not in equipment:
                        continue
                    if rover in assigned_rovers:
                        continue
                    
                    # Check if rover has appropriate camera
                    has_camera = False
                    for fact in state:
                        if match(fact, "on_board", "*", rover):
                            camera = get_parts(fact)[1]
                            if mode in self.camera_modes.get(camera, set()):
                                has_camera = True
                                break
                    
                    if not has_camera:
                        continue
                    
                    # Estimate navigation costs
                    rover_pos = None
                    for fact in state:
                        if match(fact, "at", rover, "*"):
                            rover_pos = get_parts(fact)[2]
                            break
                    
                    if not rover_pos:
                        continue
                    
                    nav_cost = 1  # At least one navigate action
                    
                    # Check if we already have the image
                    has_image = any(
                        match(fact, "have_image", rover, objective, mode)
                        for fact in state
                    )
                    
                    if has_image:
                        # Just need to navigate to lander and communicate
                        lander_pos = next(iter(self.lander_positions.values()))
                        total_cost += nav_cost + 1  # navigate + communicate
                    else:
                        # Need to calibrate, take image, navigate to lander, communicate
                        # Calibration requires navigating to calibration target
                        # Imaging requires navigating to visible location
                        # This is complex, so we'll estimate 2 nav steps + calibration + image + communicate
                        total_cost += 2 * nav_cost + 1 + 1 + 1  # nav*2 + calibrate + image + communicate
                    
                    assigned_rovers.add(rover)
                    break
        
        return total_cost if total_cost > 0 else 1  # Ensure we never return 0 for non-goal states
