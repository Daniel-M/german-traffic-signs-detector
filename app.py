import os
import sys
import click
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

@cli.command("download", help="downloads dataset for German Traffic Signs Detector")
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

@cli.command("train", help="Trains a model")
@cli.option("m", help="Model", default=1)
def train_handler():

    click.echo("Hey there!")
    click.echo("I'm not implemented yet")




if __name__=="__main__":
    cli()
