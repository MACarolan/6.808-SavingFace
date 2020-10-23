# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:54:30 2020

@author: micha
"""

from __future__ import print_function
import os
import neat
import pickle
import random
# import visualize


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 200
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(data_train, labels_train):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def run_first(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # Display the winning genome.
    # print('\nFound Best Genome:\n{!s}'.format(winner))
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    
    # return winner_net
    #return population
    return p, winner_net

def run(config_file, p):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    #Use the same population

    # Add a stdout reporter to show progress in the terminal.
    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # Display the winning genome.
    # print('\nFound Best Genome:\n{!s}'.format(winner))
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    
    return p, winner_net

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    
    #load data
    file = open("all_data_processed_15_large_3.pkl", "rb")
    data, labels = pickle.load(file)
    file.close()
    
    #get training subset for this epoch
    indices = list(range(len(data)))
    random.shuffle(indices)
    data_train = [0]*len(data)
    labels_train = [0]*len(data)
    for i in range(len(data)):
        data_train[indices[i]] = data[i]
        labels_train[indices[i]] = labels[i]
    data_train = data_train[:200]
    labels_train = labels_train[:200]
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    p = run_first(config_path)[0]
    
    #sets of sets of epochs to run
    # for i in range(1):
    j = 0
    while True:
        #get training subset for this epoch
        indices = list(range(len(data)))
        random.shuffle(indices)
        data_train = [0]*len(data)
        labels_train = [0]*len(data)
        for i in range(len(data)):
            data_train[indices[i]] = data[i]
            labels_train[indices[i]] = labels[i]
        for i in range(400, len(data), 200):
            data_train = data_train[i-200:i]
            labels_train = labels_train[i-200:i]
            p, winner = run(config_path, p)
        j += 1
        if j%100 == 0:
            file = open('NN_NEWin200s_' + str(i) + 'it50g_rssisize15_3.pkl', 'wb')
            pickle.dump(winner, file)
            file.close()