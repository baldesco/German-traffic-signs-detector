import click
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from support_functions import *

@click.command()
@click.option('-m', default=None, help='Model used to classify new images.')
@click.option('-d', default=None, help='Directory where the images are located.')
@click.option('--download/--no-download', default=False, help='Command to download images for the challenge and place them into folders.')
@click.option('--infer/--no-infer', default=False, help='Command to classify new images placed in a determined folder.')
@click.option('--train/--no-train', default=False, help='Command to train a classification model.')
@click.option('--test/--no-test', default=False, help='Command to calculate performance of a classification model on a test set.')

def main(m, d, download, infer, test, train):
    """Program for downloading and classifying traffic signs images."""
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    if infer:
        if m is not None:
            if d is None:
                print('You must provide a folder in which there are images to be classified.')
            else:
                if m == 'model1':
                    # Load model
                    model_location = os.sep.join([script_dir,'models','model1','model1.pkl'])
                    with open(model_location, 'rb') as fp:
                        model = pickle.load(fp)
                    size = model['best_size']
                    model = model['best_model']
                    logreg_classify(model,d,size)
                elif m == 'model2':
                    # Load model
                    model_location = os.sep.join([script_dir,'models','model2'])
                    softmax_classify(model_location,d)
                else:
                    print('Please provide a valid model: model1 (sklearn logistic regression) or model2 (tensorflow softmax regression)')
    if train:
        if m is not None:
            if d is None:
                print('You must provide a folder with images to train the model.')
            else:
                if m == 'model1':
                    train_logreg(d)
                elif m == 'model2':
                    train_softmax(d)
                else:
                    print('Please provide a valid model: model1 (sklearn logistic regression) or model2 (tensorflow softmax regression)')
    if test:
        if m is not None:
            if d is None:
                print('You must provide a folder with images to test the model.')
            else:
                if m == 'model1':
                    # Load model
                    model_location = os.sep.join([script_dir,'models','model1','model1.pkl'])
                    with open(model_location, 'rb') as fp:
                        model = pickle.load(fp)
                    size = model['best_size']
                    model = model['best_model']
                    accuracy = test_logreg(model,d,size)
                    print('Test set accuracy for logistic regression model: {0}.'.format(accuracy))
                elif m == 'model2':
                    # Load model
                    model_location = os.sep.join([script_dir,'models','model2'])
                    accuracy = test_softmax(model_location,d)
                    print('Test set accuracy for softmax regression model: {0}.'.format(accuracy))
                else:
                    print('Please provide a valid model: model1 (sklearn logistic regression) or model2 (tensorflow softmax regression)')
    if download:
        download_imgs()

if __name__ == '__main__':
    main()
