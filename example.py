
import json

from pyceptron import SingleLayerPerceptron

def main():
    dataset_file = 'datasets/and.json'
    print('Loading dataset', dataset_file)
    raw_dataset = json.load(open(dataset_file))
    print('Expected outputs:')
    for sample in raw_dataset:
        print('f({0}, {1})'.format(*sample['input']), '=', sample['output'])


    print('Training perceptron...')
    perceptron = SingleLayerPerceptron()
    perceptron.train(raw_dataset)
    print('Perceptron trained in', perceptron.num_iterations, 'iterations')

    print('Perceptron outputs:')
    for i in range(0, 2):
        for j in range(0, 2):
            output = perceptron.get_output((i, j))
            print('f({0}, {1})'.format(i, j), '=', output)
            assert output == (i and j)

    print('All tests passed!')


if __name__ == '__main__':
    main()
