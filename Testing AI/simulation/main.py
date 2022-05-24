import sys
import pygame
from penguin import Penguin
import neat
from dotenv import load_dotenv
import os
import pandas as pd
import openpyxl

# Constant variables
SCREEN_HEIGHT = 1500
SCREEN_WIDTH = 800
PENGUIN_HEIGHT = 192
CPENGUIN_WIDTH = 112
GENERATION = 0

load_dotenv()

# Window display settings
pygame.display.set_caption('Penguin Simulation')
icon = pygame.image.load('penguin.png')
pygame.display.set_icon(icon)

# Map to be tested
env_map = os.getenv('MAP')

if env_map == '1':
    game_map = pygame.image.load('Opv3.png')
elif env_map == '2':
    game_map = pygame.image.load('Opv2.png')
elif env_map == '3':
    game_map = pygame.image.load('Opv1.png')

env_run = os.getenv('SINGULAR')

#Global data
max_reward_per_gen = []
reward_per_entity = []

def run_penguin(genomes, config):

    # Init NEAT
    nets = []
    penguins = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        # Init my cars
        penguins.append(Penguin(game_map))

    # Init my game
    pygame.init()
    screen = pygame.display.set_mode((
        SCREEN_HEIGHT, SCREEN_WIDTH
    ))

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 70)
    font = pygame.font.SysFont("Arial", 30)
    #map = pygame.image.load('map.png')

    # Main loop
    global GENERATION
    GENERATION += 1
    maximum_reward = 0
    while True:
        screen.blit(game_map, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # Input my data and get result from network
        for index, pen in enumerate(penguins):
            output = nets[index].activate(pen.get_data())
            i = output.index(max(output))
            if i == 0:
                pen.angle += 10
            else:
                pen.angle -= 10

        # Update car and fitness
        remain_penguins = 0
        errors_detected = 0
        for i, pen in enumerate(penguins):
            if not(pen.get_collided()):
                remain_penguins += 1
                pen.update()
                genomes[i][1].fitness += pen.get_reward()
                if remain_penguins != 1:
                    maximum_reward = pen.get_reward()
            if (pen.get_error()):
                errors_detected += 1

        # check
        if remain_penguins == 0:
            print(maximum_reward)
            max_reward_per_gen.append(maximum_reward)
            break

        # Drawing
        screen.blit(game_map, (0, 0))
        for pen in penguins:
            if not(pen.get_collided()):
                pen.draw(screen)

        text = font.render(
            "Generación : " + str(GENERATION), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH + 530, 700)
        screen.blit(text, text_rect)

        text = font.render("Pinguinos restantes : " +
                           str(remain_penguins), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH + 530, 730)
        screen.blit(text, text_rect)

        text = font.render("Errores detectados : " +
                           str(errors_detected), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (SCREEN_WIDTH + 530, 760)
        screen.blit(text, text_rect)
        

        pygame.display.flip()
        clock.tick(0)

    for i, pen in enumerate(penguins):
        reward_per_entity.append(pen.get_reward())  # Se agrega toda la info
    #reward_per_entity.append(errors_detected)  #Esto se aprego, pero se puede hcar para otros de cada especie mejor
    max_reward_per_gen.append(errors_detected)


if __name__ == "__main__":

    # Get Run Type
    if env_run == '1':
        # Sigular run
        # Set configuration file
        config_path = "./config-feedforward.txt"
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        # Create core evolution algorithm class
        p = neat.Population(config)

        # Add reporter for fancy statistical result
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Run NEAT
        p.run(run_penguin, 1000)

        # Send data to Excel individualy
        df1 = pd.DataFrame(max_reward_per_gen,
                           columns=['Distancia'])
        df1.to_excel("mejores.xlsx")

        df2 = pd.DataFrame(reward_per_entity,
                           columns=['Distancia'])
        df2.to_excel("distancias.xlsx")

    elif env_run == '0':
        # Multiple runs
        multipleGenerationsData = []

        for i in range(50):
            # Set configuration file
            config_path = "./config-feedforward.txt"
            config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

            # Create core evolution algorithm class
            p = neat.Population(config)

            # Add reporter for fancy statistical result
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)

            # Run NEAT
            p.run(run_penguin, 100)
            multipleGenerationsData.append(max_reward_per_gen)
            max_reward_per_gen = []

        # Send data to Excel Multiple
        df1 = pd.DataFrame(multipleGenerationsData)
        df1.to_excel("DataCorridas.xlsx")