#%% Import libraries
import errno, inspect, math, os, random, zipfile
import requests
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sklearn
import tensorflow as tf
from datetime import datetime as dt
from shutil import copyfile, rmtree
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def download_imgs():
    #%% location of the actual directory
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ruta_descarga = os.sep.join([script_dir,'descarga.zip'])

    # Download images from website
    print(r'Images download has started.')
    t_inicial = dt.now()
    url = 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'
    r = requests.get(url)
    t = (dt.now() - t_inicial).total_seconds()
    if r.status_code == 200:
        with open(ruta_descarga, 'wb') as f:
            f.write(r.content)
        print('Download completed. Elapsed time: {0} seconds.'.format(t))
    else:
        print('Download failed (status {0}). Please try again.'.format(r.status_code))

    #%% Create new folders for images
    print(r'Relocating images.')
    t_inicial = dt.now()
    def create_folder(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    folders = ['images','images/train','images/test','images/user']
    for folder in folders:
        route = os.sep.join([script_dir,folder])
        create_folder(route)

    zip_ref = zipfile.ZipFile(ruta_descarga, 'r')
    zip_ref.extractall(script_dir)
    zip_ref.close()

    #%% Change names of images
    ruta_imagenes = os.sep.join([script_dir,'FullIJCNN2013'])

    directorios = []
    for directorio in os.walk(ruta_imagenes):
        directorios.append(directorio)

    for d in directorios[1:]:
        label = d[0].split(os.sep)[-1]
        n_train = int(np.floor(0.8*len(d[2])))

        train_imgs = random.sample(d[2],n_train)
        test_imgs = [i for i in d[2] if i not in train_imgs]

        for img in train_imgs:
            source = d[0] + os.sep + img
            dest = os.sep.join([script_dir,'images','train',label + '_' + img])
            copyfile(source, dest)

        for img in test_imgs:
            source = d[0] + os.sep + img
            dest = os.sep.join([script_dir,'images','test',label + '_' + img])
            copyfile(source, dest)

    #%% Delete downloaded files
    rmtree(ruta_imagenes)
    os.remove(ruta_descarga)
    t = (dt.now() - t_inicial).total_seconds()
    print('Relocation completed. Elapsed time: {0} seconds.'.format(t))

def process_img(img,size):
    out = img.resize(size, Image.ANTIALIAS)
    out = np.asarray(out)
    out = out.reshape(1, -1)
    return out

def formar_set(images,im_size):
    X = [process_img(img, im_size) for img in images]
    X = np.vstack(X)

    return X

# List with the traffic signs corresponding to each label
signs = ['speed limit 20','speed limit 30','speed limit 50','speed limit 60','speed limit 70',
'speed limit 80','restriction ends 80','speed limit 100','speed limit 120','no overtaking',
'no overtaking (trucks)','priority at next intersection','priority road','give way','stop',
'no traffic both ways','no trucks','no entry','danger','bend left','bend right','bend',
'uneven road','slippery road','road narrows','construction','traffic signal',
'pedestrian crossing','school crossing','cycles crossing','snow','animals','restriction ends',
'go right','go left','go straight','go right or straight','go left or straight','keep right',
'keep left','roundabout','restriction ends (overtaking)','restriction ends (overtaking (trucks))']

def show_imgs_and_label(images,predictions):
    for i in range(len(predictions)):
        plt.figure(figsize=(4,2))
        sign = signs[predictions[i]]
        plt.imshow(images[i])
        plt.title('Prediction: {a}, Sign: {b}'.format(a=predictions[i],b=sign), fontsize = 8)
        plt.show()

def logreg_classify(model,directory,size):
    images = []
    for filename in glob.iglob(directory + os.sep + '*.ppm'):
        images.append(Image.open(filename))

    X = formar_set(images,size)
    predictions = model.predict(X)
    show_imgs_and_label(images,predictions)

def softmax_classify(model_loc,directory):
    images = []
    for filename in glob.iglob(directory + os.sep + '*.ppm'):
        images.append(Image.open(filename))

    X = formar_set(images,(35,35))
    X = X.astype('float32')/255
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(os.sep.join([model_loc,'model2-1000.meta']))
        new_saver.restore(sess, tf.train.latest_checkpoint(model_loc))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        W = graph.get_tensor_by_name("W:0")
        b = graph.get_tensor_by_name("b:0")
        model = graph.get_tensor_by_name("model:0")
        pred = tf.argmax(model,1)
        predictions = sess.run(pred,feed_dict={x:X})

    show_imgs_and_label(images,predictions)

def test_logreg(model,directory,size):
    images = []
    labels = []
    for filename in glob.iglob(directory + os.sep + '*.ppm'):
        images.append(Image.open(filename))
        label = int(filename.split(os.sep)[-1].split('_')[0])
        labels.append(label)

    X = formar_set(images,size)
    predictions = model.predict(X)
    accuracy = accuracy_score(labels,predictions)
    return accuracy

def train_logreg(directory):
    # NOTE: Here we are going to assume there is a 'test' folder, cointaining the test set images.
    d_test = os.sep.join([directory,'..','test'])
    images = []
    labels = []
    for filename in glob.iglob(directory + os.sep + '*.ppm'):
        images.append(Image.open(filename))
        label = int(filename.split(os.sep)[-1].split('_')[0])
        labels.append(label)
    # Definition of the model's hyperparameters (solver and image size), and initialization of the expected outcomes
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    sizes = [(17,17),(25,25),(35,35),(45,45),(55,55)]
    best_solver = None
    best_size = (0,0)
    best_accuracy = 0
    best_model = None
    # Loop through parameters to find the best model
    print('Model training started.')
    for solver in solvers:
        for size in sizes:
            X = formar_set(images,size)
            lr = LogisticRegression(solver=solver)
            lr.fit(X,labels)
            accuracy = test_logreg(lr,d_test,size)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_solver = solver
                best_size = size
                best_model = lr
    # Print training accuracy
    print('Model training has ended.')
    train_accuracy = test_logreg(best_model,directory,best_size)
    print('Training set accuracy for logistic regression model: {0}.'.format(train_accuracy))
    # Save the best model
    model_location = os.sep.join([directory,'..','..','models','model1','model1.pkl'])
    model = {'best_solver': best_solver, 'best_size': best_size, 'best_model': best_model}
    with open(model_location, 'wb') as fid:
        pickle.dump(model, fid)
    print('Model has been saved at: {0}.'.format(model_location))

def test_softmax(model_loc,directory):
    images = []
    labels = []
    for filename in glob.iglob(directory + os.sep + '*.ppm'):
        images.append(Image.open(filename))
        label = int(filename.split(os.sep)[-1].split('_')[0])
        labels.append(label)
    X = formar_set(images,(35,35))
    X = X.astype('float32')/255
    labels = np.array(pd.get_dummies(pd.Series(labels)))

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(os.sep.join([model_loc,'model2-1000.meta']))
        new_saver.restore(sess, tf.train.latest_checkpoint(model_loc))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        W = graph.get_tensor_by_name("W:0")
        b = graph.get_tensor_by_name("b:0")
        model = graph.get_tensor_by_name("model:0")

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(model,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        acc = sess.run(accuracy, feed_dict={x: X, y: labels})
        return acc

def train_softmax(directory):
    # NOTE: Here we are going to assume there is a 'test' folder, cointaining the test set images.
    d_test = os.sep.join([directory,'..','test'])
    train_images = []
    test_images = []
    y_train = []
    y_test = []
    for filename in glob.iglob(directory + os.sep + '*.ppm'):
        label = int(filename.split(os.sep)[-1].split('_')[0])
        y_train.append(label)
        train_images.append(Image.open(filename))
    for filename in glob.iglob(d_test + os.sep + '*.ppm'):
        label = int(filename.split(os.sep)[-1].split('_')[0])
        y_test.append(label)
        test_images.append(Image.open(filename))
    # Set parameters
    im_size = (35,35)
    learning_rate = 0.0002
    batch_size = 500
    training_iteration = 10000 # max number of iterations
    termination_margin = 0.001/100  # If the cost function varies less than this between iterations, the training process stops.
    # Create and normalize training and test feature sets
    X_train = formar_set(train_images,im_size)
    X_test = formar_set(test_images,im_size)
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    # Convert class vectors (y_train and y_test) to binary class matrices
    Y_train = np.array(pd.get_dummies(pd.Series(y_train)))
    Y_test = np.array(pd.get_dummies(pd.Series(y_test)))
    ## Create tf model
    size_x = X_train.shape[1]
    size_y = Y_train.shape[1]
    # TF graph input
    x = tf.placeholder("float", [None, size_x], name='x')
    y = tf.placeholder("float", [None, size_y], name='y')
    # Set model weights
    W = tf.Variable(tf.zeros([size_x, size_y]), name='W')
    b = tf.Variable(tf.zeros([size_y]), name='b')
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b, name='model') # Softmax
    # Define cross_entropy as the cost function
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Define Gradient Descent as the model optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    print('Model training started.')
    with tf.Session() as sess:
        # Initialize variables and calculate initial cost
        sess.run(init)
        initial_cost = sess.run(cost_function,{x:X_test,y:Y_test})
        cond = True
        overfit = False
        cont = 0
        # Training cycle
        while cond:
            cont += 1
            rand_ind = random.sample(range(len(X_train)),batch_size)
            batch_xs, batch_ys = X_train[rand_ind,:],Y_train[rand_ind,:]
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            cost_epoch = sess.run(cost_function,{x:X_test,y:Y_test})
            # Progress is printed every 200 iterations
            if cont % 200 == 0:
                print("Iteration:", '%04d' % (cont), "cost=", "{:.9f}".format(cost_epoch))
            # Progress is checked every 50 iterations
            if cont % 50 == 0:
                overfit = (cost_epoch > initial_cost) and (abs(cost_epoch - initial_cost)/initial_cost < termination_margin)
            # Termination condition
            if (cont >= training_iteration) or overfit:
                print("Iteration:", '%04d' % (cont), "cost=", "{:.9f}".format(cost_epoch))
                cond = False
            else:
                initial_cost = cost_epoch
        print("Tuning completed!")
        # Measuring performance
        sess.run(cost_function,{x:X_test,y:Y_test})
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(model,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        print('Training set accuracy for softmax regression model: {0}'.format(acc))
        #Create a saver object which will save all the variables
        model_location = os.sep.join([directory,'..','..','models','model2','model2'])
        saver = tf.train.Saver()
        saver.save(sess, model_location,global_step=1000)
        print('Model has been saved at: {0}.'.format(model_location))
