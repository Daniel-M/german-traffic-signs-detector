import os
import sys
import click
from conf import Config
from helpers.paths import create_path 
from helpers.downloader import download_dataset

images_path = Config.images_path
images_url = Config.images_url 
images_zip_file = Config.images_zip_file 


@click.group()
def cli():
    pass

@cli.command("download", help="downloads dataset for German Traffic Signs Detector")
def download_handler():

    global images_url
    global images_zip_file

    click.echo("Hey there!")
    click.echo("I'm about to download the dataset for German Traffic Signs Detector")

    img_path = create_path(images_path)

    file_name = os.path.join(img_path, images_zip_file)

    download_dataset(images_url, file_name)


if __name__=="__main__":
    cli()
