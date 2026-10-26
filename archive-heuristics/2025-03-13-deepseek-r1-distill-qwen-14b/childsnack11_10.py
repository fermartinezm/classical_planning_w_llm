from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches. It considers whether children are allergic to gluten, the availability of gluten-free ingredients, and the need to move trays between locations.

    # Assumptions:
    - Each child needs exactly one sandwich.
    - A sandwich can be made if the required bread and content portions are available.
    - A tray can hold multiple sandwiches but only one sandwich can be served at a time.
    - Moving a tray between locations costs 2 actions (except when moving from the kitchen to the child's location if the tray is already there).

    # Heuristic Initialization
    - Extracts static facts about children's allergies, available gluten-free ingredients, and tray locations.
    - Maps each child to their required sandwich type (gluten-free or regular).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served. If served, no actions are needed.
    2. For unserved children:
       a. Determine if they need a gluten-free sandwich based on their allergy status.
       b. Check if the required ingredients are available.
       c. Calculate the number of actions needed to make the sandwich (2 actions).
       d. Calculate the number of actions needed to put the sandwich on a tray (1 action).
       e. If the tray is not already at the child's location, calculate the actions needed to move the tray (2 actions).
       f. Calculate the action needed to serve the sandwich (1 action).
    3. Sum the actions needed for all unserved children to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        super().__init__(task)
        
        # Extract static facts
        self.allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.tray_locations = dict()
        
        for fact in task.static:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergic_children.add(child)
            elif match(fact, "no_gluten_bread", "*"):
                bread = get_parts(fact)[1]
                self.no_gluten_breads.add(bread)
            elif match(fact, "no_gluten_content", "*"):
                content = get_parts(fact)[1]
                self.no_gluten_contents.add(content)
            elif match(fact, "at", "*", "kitchen"):
                obj = get_parts(fact)[1]
                if obj.startswith("tray"):
                    tray = obj
                    self.tray_locations[tray] = "kitchen"
        
        # Map each child to their required sandwich type
        self.children = [fact[1] for fact in task.objects if fact[0] == "child"]
        self.child_type = {}
        for child in self.children:
            if child in self.allergic_children:
                self.child_type[child] = "gluten-free"
            else:
                self.child_type[child] = "regular"

    def __call__(self, node):
        """Estimate the minimum number of actions needed to serve all children."""
        state = node.state
        served_children = set()
        
        for fact in state:
            if match(fact, "served", "*"):
                child = get_parts(fact)[1]
                served_children.add(child)
        
        unserved_children = set(self.children) - served_children
        total_actions = 0
        
        for child in unserved_children:
            # Determine sandwich type
            if self.child_type[child] == "gluten-free":
                # Check if we have the required gluten-free ingredients
                # Assume we have enough resources as per problem constraints
                # Calculate actions for making a gluten-free sandwich
                total_actions += 2  # make_sandwich_no_gluten action
            else:
                # Calculate actions for making a regular sandwich
                total_actions += 2  # make_sandwich action
            
            # Action to put the sandwich on a tray
            total_actions += 1  # put_on_tray action
            
            # Check if tray is already at the child's location
            # We need to find which tray is being used
            # Assume only one tray is used per sandwich
            tray = None
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    if fact[1:-1].split()[2] == child:
                        tray = fact[1:-1].split()[1]
                        break
            
            if tray:
                # Check current location of the tray
                tray_location = None
                for fact in state:
                    if match(fact, "at", tray, "*"):
                        tray_location = get_parts(fact)[2]
                        break
                
                # If tray is not already at the child's place, need to move it
                if tray_location != get_parts(fact)[2] for fact in state if match(fact, "waiting", child, "*"):
                    # Move tray to child's location
                    total_actions += 2  # move_tray action
            
            # Action to serve the sandwich
            total_actions += 1  # serve_sandwich or serve_sandwich_no_gluten action
        
        return total_actions
