from motion import Drone, Rover, Scout


class Governor:

    def __init__(self, map_information, rover: Rover, scout: Scout, drone: Drone):
        self.map_information = map_information
        self.rover = rover
        self.scout = scout
        self.drone = drone

    def get_heading(self):
        """Return the heading of movement for each agent. This is a placeholder function and should be implemented with actual logic to determine the heading based on the current state of the agents and the environment."""
        if self.drone.x < 20 & self.drone.y<20 & self.rover.x < 20 & self.rover.y < 20 & self.scout.y < 20 & self.scout.x < 20:
            drone = 50  # drone = none if t has to recharge
            scout = 20
            rover = 10
        else:
            drone = None
            scout = None
            rover = None
        return rover, scout, drone