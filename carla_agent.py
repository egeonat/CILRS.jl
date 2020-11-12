import carla
import julia.Main
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.autoagents.autonomous_agent import Track


def get_entry_point():
    return "cil_agent"

class cil_agent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS

        self.jl = julia.Main
        self.jl.include("carla_agent.jl")


    def sensors(self):
        sensors = [{
                    "type": "sensor.camera.rgb",
                    "x": 0.7,
                    "y": 0.0,
                    "z": 1.60,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "width": 128,
                    "height": 96,
                    "fov": 100,
                    "id": "rgb_center"},
                   {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                   }]
        return sensors


    def run_step(self, input_data, timestamp):
        img = input_data["rbg_center"][1]
        speed = input_data["speed"][1]["speed"]

        self.jl.run_step(img, speed)

        control = carla.VehicleControl()
        control.throttle = 1.0
        return control

if __name__ == "__main__":
    julia_interface = julia.Main
    julia_interface.include("carla_agent.jl")
    img = [[[0, 0, 0]]]
    speed = 10
    result = julia_interface.run_step(img, speed)
    print(result)
