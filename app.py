import click

@click.group()
def cli():
    pass

@cli.command("download")
def download():
    click.echo("Hey there!")

if __name__=="__main__":
    cli()
