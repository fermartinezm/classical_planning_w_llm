from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all goals by:
    1. Counting remaining soil/rock samples to collect and communicate
    2. Counting remaining images to take and communicate
    3. Estimating navigation costs between waypoints
    4. Considering equipment requirements and calibration needs

    # Assumptions:
    - Each sample collection requires navigating to the waypoint if not already there
    - Each image requires calibration if not already done
    - Communication requires navigating to a waypoint visible from the lander
    - Rovers can perform tasks in parallel when possible

    # Heuristic Initialization
    - Extract goal conditions (communicated data)
    - Build mapping of waypoints to their visible neighbors
    - Identify lander location and calibration targets
    - Identify which rovers have which equipment

    # Step-By-Step Thinking for Computing Heuristic
    1. For each uncommunicated soil/rock data:
       - If not sampled yet: add cost for sampling (navigate + sample)
       - Add cost for communicating (navigate to visible waypoint + communicate)
    2. For each uncommunicated image:
       - If not taken yet: add cost for calibration (if needed) + imaging
       - Add cost for communicating the image
    3. For navigation steps, use minimum distance estimate (1 per waypoint)
    4. Sum all costs while trying to maximize parallel rover utilization
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract lander location
        self.lander_loc = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                _, lander, loc = get_parts(fact)
                self.lander_loc = loc
                break
                
        # Build visibility graph
        self.visible = {}
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                _, wp1, wp2 = get_parts(fact)
                self.visible.setdefault(wp1, set()).add(wp2)
                self.visible.setdefault(wp2, set()).add(wp1)
                
        # Extract rover capabilities
        self.rover_equipment = {}
        for fact in self.static:
            if match(fact, "equipped_for_*", "*"):
                _, equip, rover = get_parts(fact)
                self.rover_equipment.setdefault(rover, set()).add(equip)
                
        # Extract calibration targets
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                _, cam, obj = get_parts(fact)
                self.calibration_targets[cam] = obj
                
        # Extract objective visibility
        self.objective_visibility = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                _, obj, wp = get_parts(fact)
                self.objective_visibility.setdefault(obj, set()).add(wp)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the current state."""
        state = node.state
        cost = 0
        
        # Check which goals are already satisfied
        unsatisfied_goals = self.goals - state
        
        # If all goals are satisfied, return 0
        if not unsatisfied_goals:
            return 0
            
        # Initialize tracking of pending tasks
        pending_soil = set()
        pending_rock = set()
        pending_images = set()
        
        # Process unsatisfied goals to identify what needs to be done
        for goal in unsatisfied_goals:
            parts = get_parts(goal)
            if parts[0] == "communicated_soil_data":
                wp = parts[1]
                pending_soil.add(wp)
            elif parts[0] == "communicated_rock_data":
                wp = parts[1]
                pending_rock.add(wp)
            elif parts[0] == "communicated_image_data":
                obj, mode = parts[1], parts[2]
                pending_images.add((obj, mode))
        
        # Estimate costs for soil samples
        for wp in pending_soil:
            # Check if already sampled
            sampled = any(match(fact, "have_soil_analysis", "*", wp) for fact in state)
            
            if not sampled:
                # Need to navigate to waypoint and sample (2 actions)
                cost += 2
            # Need to communicate (navigate to visible waypoint + communicate)
            cost += 2
            
        # Estimate costs for rock samples
        for wp in pending_rock:
            # Check if already sampled
            sampled = any(match(fact, "have_rock_analysis", "*", wp) for fact in state)
            
            if not sampled:
                # Need to navigate to waypoint and sample (2 actions)
                cost += 2
            # Need to communicate (navigate to visible waypoint + communicate)
            cost += 2
            
        # Estimate costs for images
        for obj, mode in pending_images:
            # Check if image already taken
            taken = any(match(fact, "have_image", "*", obj, mode) for fact in state)
            
            if not taken:
                # Need to calibrate (if not already) and take image
                cost += 2  # 1 for calibration (if needed), 1 for imaging
            # Need to communicate the image
            cost += 1
            
        # Add base navigation cost (minimum 1 per rover that needs to move)
        # This is a rough estimate since we don't track rover positions precisely
        if pending_soil or pending_rock or pending_images:
            cost += 1
            
        return cost
