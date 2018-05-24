import os
import cv2
import imageio
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

# Set parameters
learning_rate = 0.005
training_iteration = 100
batch_size = 121#244 #122
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
            img = img.flatten() # To get flat arrays of the image
            X.append(img) 
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

def split_data(X, y, test_size=0.2):
    data_size = len(X)
    if len(y) == data_size and test_size*data_size <= data_size:
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
        return X, y, X, y


def generate_data_batch(batch_size, features, labels, unique_labels):
    features_qty = len(features)
        
    if batch_size <= features_qty:
        
        labels_mapping = dict()
        clases_qty = len(unique_labels)
        ft_shape = features[0].shape
        
        # Generate a dictionary with the mapping of each unique_labels
        for cls_idx in range(clases_qty):
            labels_mapping[unique_labels[cls_idx]] = cls_idx
            
        
        the_batch_features = np.zeros((batch_size,) + ft_shape)
        the_batch_labels = np.zeros((batch_size, clases_qty))
        
        for i in range(batch_size):
            index = np.random.randint(features_qty)
            
            the_batch_features[i] = features[index]
            
            the_label = np.zeros(clases_qty)
            the_label[labels_mapping[labels[index]]] = 1.0
            
            the_batch_labels[i] = the_label
            
            # Shuffe the array a bit
            the_batch_features, the_batch_labels = shuffle(the_batch_features, the_batch_labels)
        
        return the_batch_features, the_batch_labels, labels_mapping

    else:
        return features, labels, None


def train_model(path="images/FullIJCNN2013", annotation_file="gt.txt"):
    annotations = os.path.join(path, annotation_file) 

    X, y = create_data_set_from_annotations(annotations)

    unique_labels = np.array(list(set(y)))

    data_shape = X[0].shape[0]

    number_of_clases = len(set(y))


    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    n_train_samples = len(X_train)

    X_test, y_test, _ = generate_data_batch(len(X_test), X_test, y_test, unique_labels)


    # TF graph input (the flattened images)
    x = tf.placeholder(tf.float32, [None, data_shape])


    # Create a model

    # Set model weights
    W = tf.Variable(tf.zeros([data_shape, number_of_clases]))
    b = tf.Variable(tf.zeros([number_of_clases]))

    # To implement cross_entropy, here we store the correct answers
    y_ = tf.placeholder(tf.float32, [None, number_of_clases])

    # Construct a linear y `y = x*W + b`
    y = tf.matmul(x, W) + b # Softmax

    # Numerically more stable softmax 
    cost_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    batch_xs, batch_ys, _ = generate_data_batch(5, X_train, y_train, unique_labels)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        
        # Training loop
        for iteration in range(training_iteration):
            avg_result = 0.
            total_batch = int(n_train_samples/batch_size)
            #print("Iteration {}/{} with a total batch of {}".format(iteration, training_iteration, total_batch))
            for i in range(total_batch):
                batch_xs, batch_ys, labelings = generate_data_batch(batch_size, X_train, y_train, unique_labels)
                assert batch_xs.shape[0] == batch_ys.shape[0]

                
                sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
                
                avg_result += sess.run(cost_function, feed_dict={x: batch_xs, y_: batch_ys})/total_batch
                #print("Accumulated cost{}",avg_cost)
            # Display logs per eiteration step
            if iteration % display_step == 0:
                print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_result))
        
        print("Training complete!")
       
        saver.save(sess, "./models/model2/saved")

        print("Let's see the score:")
        #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        correct_prediction = tf.equal(tf.argmax(y,0), tf.argmax(y_,0))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        error_ = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})
        print("The measured error is {}".format(error_))
        print("The measured accuracy is {}".format(1-error_))
       

def test_model(path="images/FullIJCNN2013", annotation_file="gt.txt"):
    print("Ready to load model from './models/model2/saved'")
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
        saver.restore(sess, save_path="./models/model2/saved")
        
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
