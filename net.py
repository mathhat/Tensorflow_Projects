import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
#np.random.seed(2) #makes sure random numbers are the same each runtime so results next time you run the code

cifar10 = tf.keras.datasets.cifar10.load_data() #the entire dataset

(X_train, y_train), (X_test, y_test) = cifar10  #training and test set, labels in y_

permutation = np.random.permutation(np.arange(50000,dtype=np.int32))

X_train = np.asarray(X_train[permutation], dtype=np.float32).reshape((X_train.shape[0],-1))
y_train = np.asarray(y_train[permutation], dtype=np.int32).flatten()


X_val = np.asarray(X_train[49000:], dtype=np.float32)
y_val = np.asarray(y_train[49000:], dtype=np.int32).flatten()

X_train = np.asarray(X_train[:10000], dtype=np.float32)
y_train = np.asarray(y_train[:10000], dtype=np.int32).flatten()


X_test = np.asarray(X_test, dtype=np.float32).reshape((X_test.shape[0],-1))
y_test = np.asarray(y_test, dtype=np.int32).flatten()
# Normalize the data: subtract the mean pixel and divide by std
'''
mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
X_train = (X_train - mean_pixel) / std_pixel
X_val = (X_val - mean_pixel) / std_pixel
X_test = (X_test - mean_pixel) / std_pixel
'''
names = ["airplane","automobile","bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_labels = len(names) 
y_train = (np.arange(num_labels) == y_train[:,None]).astype(np.float32)
#names = np.asarray(names) #for test and validation set plots

print X_train[0].flatten().shape[0]

'''

plt.figure()
for i in range(4):
    plt.subplot(221+i)
    plt.imshow(X_val[i])
    print(names[y_val[i]])
    #print(names[np.argmax(y_train[i])])
plt.show()
'''

#Images of size 32x32x3, filters of size HxW
image_size = 32
channels = 3
H = W = 5
depth = 8
padding = "SAME"#filtering with zero-padding (conserves image-size)
batch_size = 64*2 #using sgd, we'll use one batch of images at a time
device = '/device:GPU:0' #yeah baby 
epochs = 1001
hidden_size1 = X_train[0].flatten().shape[0]
hidden_size2 = 512*4
hidden_size3 = 512*2
beta0 = 0.001
beta1 = 0.001 #regulates weights
beta2 = 0.001

def kaiming_normal(shape):
    if len(shape) == 2:
        fan_in = shape[0]
    elif len(shape) == 4:
        fan_in = np.prod(shape[:3])
    return tf.random_normal(shape,stddev=0.05) * np.sqrt(2.0 / fan_in)

def forward(x,w1,w2,w3,b1,b2,b3):
    N = x.shape[0]
    layer = tf.nn.relu(tf.matmul(x, w1)+b1)
    layer = tf.nn.relu(tf.matmul(layer, w2)+b2)
    layer = tf.nn.softmax(tf.matmul(layer, w3)+b3)
    return layer


def accuracy(pred,labels):
    if len(labels.shape) == 1:
        return np.mean(np.argmax(pred, axis=1) == labels)
                
    else:
        return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, 1))
               
#graph = tf.Graph()
#tf.reset_default_graph()


#with graph.as_default():
tf_train_data = tf.placeholder(
    tf.float32, shape=(batch_size,hidden_size1))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,num_labels))
tf_test_data = tf.constant(X_test)
tf_val_data = tf.constant(X_val)


fcw = tf.Variable(kaiming_normal((hidden_size1,hidden_size2)))

fcb = tf.Variable(tf.zeros((hidden_size2)))

fcw2 = tf.Variable(kaiming_normal((hidden_size2, hidden_size3)))

fcb2 = tf.Variable(tf.zeros((hidden_size3)))

fcw3 = tf.Variable(kaiming_normal((hidden_size3, num_labels)))

fcb3 = tf.Variable(tf.zeros((num_labels)))

soft_scores = forward(tf_train_data,fcw,fcw2,fcw3,fcb,fcb2,fcb3)


val_pred = forward(tf_val_data,fcw,fcw2,fcw3,fcb,fcb2,fcb3)

test_pred = forward(tf_test_data,fcw,fcw2,fcw3,fcb,fcb2,fcb3)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf_train_labels, logits=soft_scores)) + beta1*tf.nn.l2_loss(fcw) +  beta1*tf.nn.l2_loss(fcw2) + beta1*tf.nn.l2_loss(fcw3) 

learning_rate = 0.01

#optimizer = tf.train.GradientDescentOptimizer(0.003).minimize(loss)

global_step = tf.Variable(0)
learn = tf.train.exponential_decay(
learning_rate = learning_rate,
global_step=global_step,
decay_steps=10,
decay_rate=0.99,
staircase=1,
name="hope_this_works"
)
optimizer = tf.train.GradientDescentOptimizer(learn).minimize(loss,global_step=global_step)
#optimizer = tf.train.Adam
saver = tf.train.Saver()
print_every = 50
val = []
train = []
losss = []

'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in xrange(epochs):
        offset = i*batch_size % (X_train.shape[0]-batch_size)
        idx = np.random.randint(0,X_train.shape[0],batch_size)
        #idx = range(offset,offset+batch_size)
        x_batch = X_train[idx]
        y_batch = y_train[idx]
        feed = {tf_train_data : x_batch, tf_train_labels : y_batch}
        _,l,scores = sess.run([optimizer,loss,soft_scores],feed_dict=feed)
        if i % print_every==0:
            val.append(accuracy(val_pred.eval(),y_val))
            train.append(accuracy(scores,y_batch))
            losss.append(l/3.)
            print("loss",l)
            print("validation",val[-1])
            print("train", train[-1])
            print "at global step: ",sess.run(global_step),"lr = ", sess.run(learn)
    #print("test",accuracy(test_pred.eval(),y_test))
    save_path = saver.save(sess, './shit')
    print("Model saved in file: %s" % save_path)
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './shit')
    if sess.run(global_step) > 7000:
        global_step = tf.Variable(0)
    for i in xrange(epochs):
        offset = i*batch_size % (X_train.shape[0]-batch_size)
        idx = np.random.randint(0,X_train.shape[0],batch_size)
        #idx = range(offset,offset+batch_size)
        x_batch = X_train[idx]
        y_batch = y_train[idx]
        feed = {tf_train_data : x_batch, tf_train_labels : y_batch}
        _,l,scores = sess.run([optimizer,loss,soft_scores],feed_dict=feed)
        if i % print_every==0:
            val.append(accuracy(val_pred.eval(),y_val))
            train.append(accuracy(scores,y_batch))
            losss.append( l/3.)
            print("loss",l)
            print train[i/print_every]
            print("validation",val[-1])
            print("train", train[-1])

            print "at global step: ",sess.run(global_step),"lr = ", sess.run(learn)
            print sess.run(global_step)
    save_path = saver.save(sess, './shit')
    print("test",accuracy(test_pred.eval(),y_test))
plt.plot(np.linspace(0,epochs,len(val)),val,label="val")
plt.plot(np.linspace(0,epochs,len(train)),train,label="test")
#plt.plot(range(0,epochs,print_every),losss,label="loss")
plt.legend()
plt.show()