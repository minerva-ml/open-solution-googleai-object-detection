import click
from src.pipeline_manager import PipelineManager

pipeline_manager = PipelineManager()


@click.group()
def main():
    pass


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train(pipeline_name, dev_mode):
    pipeline_manager.train(pipeline_name, dev_mode)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--image_dir', help='path to image folder with test files', required=False)
@click.option('-s', '--single_image', help='predict single image and visualize', required=False)
@click.option('-n', '--n_files', help='number of files to visualize', type=int, default=16)
@click.option('-sp', '--show_popups', help="if showing images in a popup window")
def visualize(pipeline_name, image_dir=None, single_image=None, n_files=16, show_popups=False):
    """

    Makes predictions on test_images (or images in image_dir) draw bounding boxes on them and sends to neptune.
    If show pop_ups it will display the predictions locally in a window popup
    """
    pipeline_manager.visualize(pipeline_name, image_dir, single_image, n_files, show_popups)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run evaluation on', type=int, default=None,
              required=False)
def evaluate(pipeline_name, dev_mode, chunk_size):
    pipeline_manager.evaluate(pipeline_name, dev_mode, chunk_size)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def predict(pipeline_name, dev_mode, submit_predictions, chunk_size):
    pipeline_manager.predict(pipeline_name, dev_mode, submit_predictions, chunk_size)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run evaluation and prediction on', type=int,
              default=None, required=False)
def train_evaluate_predict(pipeline_name, submit_predictions, dev_mode, chunk_size):
    pipeline_manager.train(pipeline_name, dev_mode)
    pipeline_manager.evaluate(pipeline_name, dev_mode, chunk_size)
    pipeline_manager.predict(pipeline_name, dev_mode, submit_predictions, chunk_size)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run evaluation and prediction on', type=int,
              default=None, required=False)
def train_evaluate(pipeline_name, dev_mode, chunk_size):
    pipeline_manager.train(pipeline_name, dev_mode)
    pipeline_manager.evaluate(pipeline_name, dev_mode, chunk_size)


@main.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-s', '--submit_predictions', help='submit predictions if true', is_flag=True, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def evaluate_predict(pipeline_name, submit_predictions, dev_mode, chunk_size):
    pipeline_manager.evaluate(pipeline_name, dev_mode, chunk_size)
    pipeline_manager.predict(pipeline_name, dev_mode, submit_predictions, chunk_size)


@main.command()
@click.option('-f', '--submission_filepath', help='filepath to json submission file', required=True)
def submit_predictions(submission_filepath):
    pipeline_manager.make_submission(submission_filepath)


if __name__ == "__main__":
    main()
