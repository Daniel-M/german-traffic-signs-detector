import os
import sys
import click

import skit_model
import tfsoftmax_model 
import lenet_model 

from conf import Config
from helpers.paths import create_path 
from helpers.downloader import download_dataset


# Read configurations from file
path_list = Config.path_list
images_path = Config.images_path
images_url = Config.images_url 
images_zip_file = Config.images_zip_file
temp_path = Config.temp_path


@click.group()
def cli():
    pass

@cli.command("download", help="Downloads dataset for German Traffic Signs Detector")
def main_handler():

    global images_url
    global images_zip_file
    global path_list

    click.echo("Hey there!")
    click.echo("I'm about to download the dataset for German Traffic Signs Detector")

    zipimage_path = create_path(images_path)

    # If the image path couldn't be created
    file_name = os.path.join(zipimage_path, images_zip_file)

    download_dataset(images_url, file_name)

@cli.command("train", help="Train a model")
@click.option("-m")
@click.option("-d", type=click.Path(exists=True))
def train(m, d):
    if d == None:
        d = "images/FullIJCNN2013"

    if m.lower() == "model1" or m.lower() == "scikit":
        skit_model.train_model(path=d)

    if m.lower() == "model2" or m.lower() == "softmax":
        tfsoftmax_model.train_model(path=d)

    if m.lower() == "model3" or m.lower() == "lenet":
        lenet_model.train_model(path=d)

@cli.command("test", help="Test a model")
@click.option("-m")
@click.option("-d", type=click.Path(exists=True))
def test(m, d):
    if d == None:
        d = "images/FullIJCNN2013"

    if m.lower() == "model1" or m.lower() == "scikit":
        skit_model.test_model(path=d)

    if m.lower() == "model2" or m.lower() == "softmax":
        tfsoftmax_model.test_model(path=d)

    if m.lower() == "model3" or m.lower() == "lenet":
        lenet_model.test_model(path=d)

@cli.command("infer", help="Not implemented")
@click.option("-m")
@click.option("-d", type=click.Path(exists=True))
def infer(m, d):
    print("Not implemented")


if __name__=="__main__":
    cli()
