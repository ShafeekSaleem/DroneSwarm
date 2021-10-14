from gym.spaces import Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import from_parallel
from gym import spaces
import numpy as np
import airsim
import time

class MultiDroneEnv(ParallelEnv):
    metadata = {'render.modes':  ["rgb_array"]}

    def __init__(self, step_length, image_shape, no_of_agents):

        self.step_length = step_length
        self.image_shape = image_shape

        self.possible_agents = ["Drone" + str(i) for i in range(no_of_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.state_space = {agent: {"position": np.zeros(3), "collision": False, "prev_position": np.zeros(3)} for agent in self.possible_agents}
        self.drone = airsim.MultirotorClient()
        self.action_spaces = {agent: Discrete(7) for agent in self.possible_agents}
        self.observation_spaces = {agent: spaces.Box(0, 255, shape=image_shape, dtype=np.uint8) for agent in self.possible_agents}

    def _setup_flight(self):
        self.drone.reset()

        self.agents = self.possible_agents[:]

        for agent in self.agents:
            self.drone.enableApiControl(True, agent)
            self.drone.armDisarm(True, agent)

            # Set start position and velocity
            self.drone.moveToPositionAsync(0.0, 0.0, -15 + -2*self.agent_name_mapping[agent], 5, vehicle_name=agent).join()
            self.drone_state = self.drone.getMultirotorState(vehicle_name=agent)
            self.state_space[agent]["position"] = self.drone_state.kinematics_estimated.position

        self.image_request = airsim.ImageRequest(
            1, airsim.ImageType.Scene, False, False
        )
        time.sleep(3)

    def transform_obs(self, responses):
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width,3))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_temp = image.resize((self.image_shape[0], self.image_shape[1]))
        im_temp = np.mean(np.array(im_temp), axis=2).astype(np.uint8)
        im_final = np.reshape(im_temp, (self.image_shape[0],self.image_shape[1],1))
        return im_final

    def _get_obs(self):

        observations = {agent: 'NONE' for agent in self.agents}

        for agent in self.agents:
            responses = self.drone.simGetImages([self.image_request], vehicle_name=agent)
            image = self.transform_obs(responses)
            self.drone_state = self.drone.getMultirotorState(vehicle_name=agent)

            self.state_space[agent]["prev_position"] = self.state_space[agent]["position"]
            self.state_space[agent]["position"] = self.drone_state.kinematics_estimated.position
            self.state_space[agent]["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

            collision = self.drone.simGetCollisionInfo(vehicle_name=agent).has_collided
            self.state_space[agent]["collision"] = collision
            observations[agent] = image

        return observations

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def _do_action(self, action, agent):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState(vehicle_name=agent).kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
            vehicle_name=agent
        ).join()

    def step(self, actions):
        
        for agent in self.agents:
            action = actions[self.agent_name_mapping[agent]]
            self._do_action(action, agent)

        obs = self._get_obs()
        rewards, dones = self._compute_reward()

        return obs, rewards, dones, self.state_space

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset

    def _compute_reward(self):
        pass


    """
    def _compute_reward(self):
        thresh_dist = 7
        beta = 1

        z = -15.0 # Set z axis

        target_dest = np.array([-45.0, -95.0, z])

        pts = [
            np.array([0.0, 0.0, z]),
            np.array([-20.0, 0.0, z]),
            np.array([-45.0, 0.0, z]),
            np.array([-45.0, -50.0, z]),
            np.array([-45.0, -95.0, z]),
        ]

        quad_prev_pt = np.array(
            list(
                (
                    self.state["prev_position"].x_val,
                    self.state["prev_position"].y_val,
                    self.state["prev_position"].z_val,
                )
            )
        )

        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        if self.state["collision"]:
            print("collided")
            reward = -100
        else:
            dist = 10000000
            for i in range(0, len(pts) - 1):
                dist = min(
                    dist,
                    np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
                    / np.linalg.norm(pts[i] - pts[i + 1]),
                )

            if dist > thresh_dist:
                print("Far from track")
                reward = -10
            else:
                reward_dist = math.exp(-beta * dist) - 0.25
                reward_target = 0.01
                reward_speed = (
                    np.linalg.norm(
                        [
                            self.state["velocity"].x_val,
                            self.state["velocity"].y_val,
                            self.state["velocity"].z_val,
                        ]
                    )
                    - 0.25
                )
                if np.linalg.norm(target_dest-quad_pt) < np.linalg.norm(target_dest-quad_prev_pt):
                    print("reward for distance keeping: "+str(reward_dist))
                    print("reward for speed: "+str(reward_speed))
                    print("reward for target: "+str(reward_target))
                    
                    reward = reward_dist + reward_speed + reward_target
                    print("Total reward for the step: ", str(reward))
                else:
                    print("reward for distance keeping: "+str(reward_dist))
                    print("reward for speed: "+str(reward_speed))
                    print("reward for target: "+str(-1*reward_target))
                    reward = reward_dist + reward_speed - reward_target
                    print("Total reward for the step: ", str(reward))

        done = 0
        if reward <= -10:
            done = 1
        return reward, done
    """