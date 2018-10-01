import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.python.tools import inspect_checkpoint as chkp

names = ["airplane","automobile","bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_labels = len(names) 

cifar10 = tf.keras.datasets.cifar10.load_data() #the entire dataset
X_train, y_train = cifar10[0]
X_test, y_test = cifar10[1]
cifar10 = 0

perm = np.random.permutation(50000)

X_train = np.asarray(X_train[perm], dtype=np.float32)/255.
y_train = np.asarray(y_train[perm], dtype=np.int32)
y_train = (np.arange(num_labels) == y_train)


X_val = np.asarray(X_train[49000:], dtype=np.float32)
y_val = np.asarray(y_train[49000:], dtype=np.int32)

X_train = X_train[:49000]
y_train = y_train[:49000]


X_test = np.asarray(X_test, dtype=np.float32)/255. 
y_test = np.asarray(y_test, dtype=np.int32)
y_test = (np.arange(num_labels) == y_test)

# Normalize the data: subtract the mean pixel and divide by std
mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
X_train = (X_train - mean_pixel) / std_pixel
X_val = (X_val - mean_pixel) / std_pixel
X_test = (X_test - mean_pixel) / std_pixel



def acc(pred,labels):
    return np.mean(np.argmax(pred,axis=1) == np.argmax(labels,axis=1))

def fc_forward(x,w,b): #regular forward pass, for both fc to fc and conv layer to fc layer
    print x
    if len(x.shape)==2:
        return tf.matmul(x,w)+b
    else:
        N = x.shape[0]
        x = tf.reshape(x,(N,-1))
        return tf.matmul(x,w)+b

def conv_forward(x,w,b,s,s2):
    x = tf.nn.relu(tf.nn.conv2d(x,w,s2,"SAME"))
    x =tf.nn.max_pool(
    value=x,
    ksize=s2,
    strides=s,
    padding="SAME",
    data_format='NHWC',
    name="pooling_is_pronounced_the_same_way_as_fucking_in_Norwegian"
    )
    return x


def init(shape):
    return tf.truncated_normal(shape,stddev=0.04)

batch_size = 64
filter1 = 5
filter2 = 3
depth1 = 32
depth2 = 64
num_channels = 3
epochs =10
noskip =  [1,1,1,1]
skip = [1,2,2,1]
noskip_foo = 32**2
skip_foo = 2**2
hidden_layer = 128*4
def forward(x,conv1,conv2,conv3,conv4,fc,fc2,b1,b2,b3,b4,b5,b6):#forward pass without dropout
    x = conv_forward(x,conv1,b1,skip,noskip)                    #for validation checks
    x = conv_forward(x,conv2,b2,skip,noskip)
    x = conv_forward(x,conv3,b5,skip,noskip)
    x = conv_forward(x,conv4,b6,skip,noskip)
    x = fc_forward(x,fc,b3)
    x = fc_forward(x,fc2,b4)
    return tf.nn.softmax(x)

def forward2(x,conv1,conv2,conv3,conv4,fc,fc2,b1,b2,b3,b4,b5,b6):
    x = conv_forward(x,conv1,b1,skip,noskip)
    x = conv_forward(x,conv2,b2,skip,noskip)
    x = tf.nn.dropout(x,0.75)
    x = conv_forward(x,conv3,b5,skip,noskip)
    x = conv_forward(x,conv4,b6,skip,noskip)
    x = tf.nn.dropout(x,0.75)
    x = fc_forward(x,fc,b3)
    x = fc_forward(x,fc2,b4)
    return tf.nn.softmax(x)

#graph = tf.Graph()
#tf.reset_default_graph()
#with graph.as_default():
conv1 = tf.Variable(init((filter1,filter1,num_channels,depth1)))
b1 = tf.Variable(tf.zeros(depth1))
conv2 = tf.Variable(init((filter2,filter2,depth1,depth2)))
b2 = tf.Variable(tf.zeros(depth2))
conv3 = tf.Variable(init((filter2,filter2,depth2,depth2*2)))
b5 = tf.Variable(tf.zeros(depth2*2))
conv4 = tf.Variable(init((filter2,filter2,depth2*2,depth2*4)))
b6 = tf.Variable(tf.zeros(depth2*4))
fc = tf.Variable(init((depth2*4*skip_foo,hidden_layer)))
b3 = tf.Variable(tf.zeros(hidden_layer))
fc2 = tf.Variable(init((hidden_layer,num_labels)))
b4 = tf.Variable(tf.zeros(num_labels))

train_data = tf.placeholder(tf.float32,shape=(batch_size,32,32,3))
train_labl = tf.placeholder(tf.float32,shape=(batch_size,num_labels))
test_data  = tf.constant(X_test)
val_data  = tf.constant(X_val)

train_pred = forward2(train_data,conv1,conv2,conv3,conv4,fc,fc2,b1,b2,b3,b4,b5,b6)
test_pred = forward(test_data,conv1,conv2,conv3,conv4,fc,fc2,b1,b2,b3,b4,b5,b6)
val_pred = forward(val_data,conv1,conv2,conv3,conv4,fc,fc2,b1,b2,b3,b4,b5,b6)
tf.add_to_collection("logits", train_pred)


loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = train_pred,
                                                    labels = train_labl) + 0.001*( \
                                                    tf.nn.l2_loss(conv1) +
                                                    tf.nn.l2_loss(conv2) +
                                                    tf.nn.l2_loss(fc)
                                                    )
loss = tf.reduce_mean(loss)
learning_rate = 0.1
global_step = tf.Variable(0)
learn = tf.train.exponential_decay(
learning_rate = learning_rate,
global_step=global_step,
decay_steps=int(765),
decay_rate=0.91,
staircase=1,
name="hope_this_works"
)
optimizer = tf.train.GradientDescentOptimizer(learn).minimize(loss,global_step=global_step)
#optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()
print_every = 50
val = []
train = []
losss = []
load = False
save = False
Test = 1

def Train(load,save,train,val,losss,Test,optimizer): 
    with tf.Session() as session:
        if load:
            saver.restore(session, './goodshit')
            if session.run(global_step)>20000:
                optimizer = tf.train.GradientDescentOptimizer(learn).minimize(loss,global_step=tf.Variable(10000))
                session.run(tf.global_variables_initializer())
                #return session.run((conv1))
        else:
            session.run(tf.global_variables_initializer())
        for j in range(epochs):
            for i in xrange(0,X_train.shape[0]-batch_size,batch_size):
                idx = range(i,i+batch_size)
                x_batch = X_train[idx]
                y_batch = y_train[idx]
                feed = {train_data:x_batch,train_labl:y_batch}
                _,l,pred = session.run([optimizer,loss,train_pred],feed_dict=feed)
                step = session.run(global_step)
                if i%2000 == 0:
                    train.append(acc(pred,y_batch))
                    losss.append(l)
                    if i%20000==0:
                        val.append(acc(val_pred.eval(),y_val))
            print session.run(global_step)
            t = acc(pred,y_batch)
            v = acc(val_pred.eval(),y_val)
            train.append(t)
            val.append(v)
            print "batch_accuracy = ", t
            print "val_accuracy = ", v
            print "at global step: ",session.run(global_step),"lr = ", session.run(learn)
        if save:
            save_path = saver.save(session, './goodshit')
            print "shit's saved in", save_path
        if Test:
            print "test_accuracy = ", acc(test_pred.eval(),y_test)
        return train, val, losss
#train,val,losss = Train(load,save,train,val,losss,Test)
image = Train(load,save,train,val,losss,Test,optimizer) #last weight layer

'''
j = 0
fig, ax = plt.subplots(depth1/4, 4)

for i in range(depth1/4):
    ax[i,0].imshow(image[:,:,:,i])
    ax[i,1].imshow(image[:,:,:,i+depth1/4])
    ax[i,2].imshow(image[:,:,:,i+depth1/2])
    ax[i,3].imshow(image[:,:,:,i+depth1/4*3])
plt.show()


exit()
'''
try :
    plt.plot(np.linspace(0,epochs,len(train)),train)
    plt.title("SGD, %d epochs"%epochs)
    plt.plot(np.linspace(0,epochs,len(val)),val)  
    plt.plot(np.linspace(0,epochs,len(losss)),losss)
    plt.show()
except:
    print "don't you wanna train, boi?"

