import torch
import utils

class View:
    def __init__(self, position):
        self.position = position

class Light:
    def __init__(self, position, color):
        self.position   = position
        self.color = color

class Scene:
    def __init__(self, view, light):
        self.view = view
        self.light  = light

def generate_random_scenes(batch_size):
    """
    Generate random scenes with random view and light positions.
    
        Args:
            batch_size: Number of scenes to generate
        Returns:
            List of Scene objects
    """

    view_position   = utils.generate_normalized_random_direction(batch_size, 0.001, 0.1) # shape = [batch_size, 3]
    light_positions = utils.generate_normalized_random_direction(batch_size, 0.001, 0.1) # shape = [batch_size, 3]

    scenes = []
    for i in range(batch_size): 
        scenes.append(Scene( view = View(view_position[i]), light  = Light(light_positions[i], [20.0, 20.0, 20.0])))

    return scenes

def generate_specular_scenes(batch_size):
    """
    Generate scenes with specular highlights in a perfect mirror configuration.

        Args:
            batch_size: Number of scenes to generate
        Returns:
            List of Scene objects
    """
    
    view_position   = utils.generate_normalized_random_direction(batch_size, 0.001, 0.1) # shape = [batch_size, 3]
    light_positions = view_position * torch.Tensor([-1.0, -1.0, 1.0]).unsqueeze(0)

    shift = torch.cat([ torch.Tensor(batch_size, 2).uniform_(-1.0, 1.0), torch.zeros((batch_size, 1)) + 0.0001], dim=-1)

    distance_view  = torch.exp(torch.Tensor(batch_size, 1).normal_(mean=0.5, std=0.75)) 
    distance_light = torch.exp(torch.Tensor(batch_size, 1).normal_(mean=0.5, std=0.75))

    
    view_position   = view_position   * distance_view  + shift
    light_positions = light_positions * distance_light + shift

    scenes = []
    for i in range(batch_size):
        scenes.append(Scene(view= View(view_position[i]), light= Light(light_positions[i], [50.0, 50.0, 50.0])))

    return scenes