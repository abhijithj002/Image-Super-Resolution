import sys
#print(os.getcwd())
sys.path.insert(0, '/home/ubuntu/Documents/image-super-resolution-master2/src/utils/')
#import utils
from utils import get_parser, load_model, load_configuration
import metrics
sys.path.insert(0, '/home/ubuntu/Documents/image-super-resolution-master2/src/models/')
#import utils
import rdn
#sys.path.insert(0, '/home/ubuntu/Documents/image-super-resolution-master/src/')
sys.path.insert(0, '/home/ubuntu/Documents/image-super-resolution-master2/src/predict/')
import predict

#sys.path.insert(0, '/home/deepquanty/projects/image-super-resolution-master/src/')
#from predict import predict





def main(arguments):
    model_parameters = {
        'scale': arguments['scale'],
        'learning_rate': 1e-5,
        'D': arguments['D'],
        'C': arguments['C'],
        'G': arguments['G'],
        'kernel_size': 3,
        'c_dim': 3,
        'G0': arguments['G0'],
    }

    model = load_model(model_parameters, arguments['vgg'], verbose=arguments['verbose'])

    if arguments['summary'] is True:
        model.rdn.summary()

    if arguments['train'] is True:
        from trainer.train import Trainer

        trainer = Trainer(train_arguments=arguments)
        trainer.train_model(model)

    if arguments['test'] is True:
        from predict import Predictor

        predictor = Predictor(test_arguments=arguments)
        predictor.get_predictions(model)


if __name__ == '__main__':
    parser = get_parser()
    cl_args = parser.parse_args()
    cl_args = vars(cl_args)
    load_configuration(cl_args)
    main(cl_args)
