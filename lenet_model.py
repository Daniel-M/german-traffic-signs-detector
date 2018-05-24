import os
import cv2
import imageio
import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
from  tensorflow.contrib.layers import flatten

# Set parameters
learning_rate = 0.0035
training_iteration = 100
batch_size = 121
display_step = 10
test_size = 0.2

input_image_shape = (32,32)



def annotation_parser(annotation, sep=";"):
    """
    anotation_parser extracts information of the ROI for a single annotation
    
    Input
    -----
    - annotation(string): The annotation for a single image
    - sep(string): The separator for the annotations. (default ";")
    
    Returns
    -------
    - filename(string): The name of the file corresponding to the annotation
    - left_column(int): Coordinates for left column of the ROI
    - top_row(int): Coordinates of the top row of the ROI
    - right_column(int): Coordinates of the left column of the ROI
    - bottom_row(int): Coordinates of the bottom row of the ROI
    - category(int): The numerical category the image in the ROI belongs to
    
    """
    filename, left_column, top_row, right_column, bottom_row, category = annotation.split(sep)
    return filename, int(left_column), int(top_row), int(right_column), int(bottom_row), int(category)


def get_annotated_ROI(annotation, path="images/FullIJCNN2013", as_gray=True, sep=";"):
    """
    get_annotated_ROI returns the ROI(an array) provided an annotation (string)
    
    Input
    -----
    - annotation(string): The annotation for a single image
    - path(string): Path containing the image files
    - as_gray(bool): If the image is loaded as RGB or Grayscale
    - sep(string): The separator for the annotations. (default ";")
    
    Returns
    -------
    - img(numpy.array): The ROI
    - category(int): The numerical category the image in the ROI belongs to
    """
    image_file, left_column, top_row, right_column, bottom_row, cat = annotation_parser(annotation, sep)
    image_file = os.path.join(path, image_file)
    img = imageio.imread(image_file, as_gray=as_gray)
    return img[top_row:top_row+(right_column-left_column),left_column:left_column+(bottom_row-top_row)].copy(), cat


def create_data_set_from_annotations(annotations_file, path="images/FullIJCNN2013", as_gray=True, target_shape = input_image_shape, sep=";"):
    """
    create_data_set_from_annotations Generates a shuffled dataset of ROIs (X) and their labels (y)
    provided an annotations_file corresponding to the german traffic sings dataset
    
    Input
    -----
    - annotation_file(string): The name of annotations file
    - path(string): Path containing the annotations file
    - as_gray(bool): If the image is loaded as RGB or Grayscale
    - target_shape(tuple): Tuple with the expected size of each of the output ROIs'
    - sep(string): The separator for the annotations. (default ";")
    
    Returns
    -------
    - X(numpy.array): Array of ROIs, one for each index
    - y(numpy.array): Array of labels corresponding to each of the ROIs on X
    """
    X = list()
    y = list()
    the_file = os.path.join(path, annotations_file)
    with open(the_file, "r") as annf:
        for line in annf:
            img, cat = get_annotated_ROI(line, path, as_gray, sep)
            img = cv2.resize(img, target_shape)
            # It is needed to add one dimension to have a result 32x32x1
            X.append(np.expand_dims(img, axis=2)) 
            y.append(cat)
            
    X, y = np.array(X), np.array(y)
    X, y = shuffle(X, y)
    return X, y


def split_data(Xo, yo, test_size=test_size):
    """
    split_data splits the data set X, y in two sets train_data, test_data and, train_labels, test_labels
    where the amount of elements of the second set correspond to the `test_size`
    percent of the amount of X, y
    
    Input
    -----
    - X(numpy.array): Array of features
    - y(numpy.array): Array of labels
    - test_size(float): Percentage of split(defalt 0.2)
    
    Returns
    -------
    - train_data(numpy.array) features to train
    - test_data(numpy.array) features to test
    - train_labels(numpy.array) labels of train_data
    - test_labels(numpy.array) labels of test_data
    
    """
    data_size = len(Xo)
    X, y = shuffle(Xo, yo)
    if (len(y) == data_size and test_size*data_size <= data_size) and (test_size <= 1.0):
        train_sz = int((1-test_size)*data_size)
        test_sz = int(test_size*data_size)
        
        train_data = np.zeros((train_sz,) + X.shape[1:])
        train_labels = np.zeros(train_sz)
        test_data = np.zeros((test_sz,) + X.shape[1:])
        test_labels = np.zeros(test_sz)
        
        for e in range(train_sz):
            index = np.random.randint(data_size)
            train_data[e] = X[index]
            train_labels[e] = y[index]
            
        for e in range(test_sz):
            index = np.random.randint(data_size)
            test_data[e] = X[index]
            test_labels[e] = y[index]
            
        return train_data, test_data, train_labels, test_labels
    else:
        print("Size miss_match")
        return Xo, yo, Xo, yo


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    
    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # Activation.
    fc2 = tf.nn.relu(fc2)
    
    # Layer 5: Fully Connected. Input = 84. Output = 43(number_of_classes).
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84, number_of_classes), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(number_of_classes))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    
    return logits



def train_model(path="images/FullIJCNN2013", annotation_file="gt.txt"):
    annotations = os.path.join(path, annotation_file) 

    X, y = create_data_set_from_annotations(annotations)


    unique_labels = np.array(list(set(y)))


    data_shape = X[0].shape

    number_of_classes = len(set(y))

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    X_test, X_validation, y_test, y_validation = split_data(X_test, y_test, test_size=0.33)

    n_train_samples = len(X_train)

    x = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1],1])
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, number_of_classes)

    # The model instance for the TF Graph
    logits = LeNet(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss_operation)


    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        
        print("Training...")
        print()
        for i in range(training_iteration):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                
            validation_accuracy = evaluate(X_validation, y_validation)
            if i % display_step == 0:
                print("EPOCH {} of {} ...".format(i+1, training_iteration))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()

        validation_accuracy = evaluate(X_validation, y_validation)
        print("Validation Accuracy at the end of the training= {:.3f}".format(validation_accuracy))
        print()
        saver.save(sess, "./models/model3/saved")
        print("Model saved to './models/model3/saved'")


def test_model(path="images/FullIJCNN2013", annotation_file="gt.txt"):
    print("Ready to load model from './models/model3/saved'")
    annotations = os.path.join(path, annotation_file) 

    X, y = create_data_set_from_annotations(annotations)

    unique_labels = np.array(list(set(y)))

    data_shape = X[0].shape[0]

    number_of_clases = len(set(y))


    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    n_train_samples = len(X_train)

    X_test, y_test, _ = generate_data_batch(len(X_test), X_test, y_test, unique_labels)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        saver.restore(sess, save_path="./models/model3/saved")
        
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))  
