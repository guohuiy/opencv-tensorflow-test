import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import mnist_reader

from keras.datasets import mnist
from keras import backend as Keras
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy


def plot_loss_accuracy_graph(fit_record):
    #蓝线绘制错误历史记录，黑线绘制测试错误
    plt.plot(fit_record.history['loss'],"-D",color="blue",label="train_loss",linewidth=2)
    plt.plot(fit_record.history['val_loss'],"-D",color="black",label="val_loss",linewidth=2)
    plt.title("LOSS")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()

    #绿线绘制精度历史记录，黑线绘制测试精度
    plt.plot(fit_record.history['accuracy'],"-o",color="green",label="train_accuracy",linewidth=2)
    plt.plot(fit_record.history['val_accuracy'],"-o",color="black",label="val_accuracy",linewidth=2)
    plt.title('ACCURACY')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.show()

def plot_image(data_location,prediction_array,real_teacher_labels,dataset):
    prediction_array,real_teacher_labels,img=prediction_array[data_location],real_teacher_labels[data_location],dataset[data_location]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label=np.argmax(prediction_array)

    if predicted_label == real_teacher_labels:
        color = 'green'
    else:
        color='red'
    
    plt.xlabel("{}{:2.0f}%({})".format(handwritten_number_names[predicted_label],100*np.max(prediction_array),handwritten_number_names[real_teacher_labels]),color=color)
    plt.show()

def plot_teacher_labels_graph(data_location,prediction_array,real_teacher_labels):
    prediction_array,real_teacher_labels=prediction_array[data_location],real_teacher_labels[data_location]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot=plt.bar(range(10),prediction_array,color='#666666')

    plt.ylim([0,1])
    predicted_label=np.argmax(prediction_array)
    thisplot[predicted_label].set_color('red')
    thisplot[real_teacher_labels].set_color('green')

def convertOneHotVector2Integers(ont_hot_vector):
    return [np.where(r==1)[0][0] for r in ont_hot_vector]

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 10
IMG_ROWS, IMG_COLS = 28, 28

handwritten_number_names = ['0','1','2','3','4','5','6','7','8','9']

train_data, train_teacher_labels = mnist_reader.load_mnist('./data', kind='train')
test_data, test_teacher_labels = mnist_reader.load_mnist('./data', kind='t10k')
# (train_data, train_teacher_labels), (test_data, test_teacher_labels) = mnist.load_data()
print('train_data shape: ',train_data.shape)
print('test_data shape:',test_data.shape)

if Keras.image_data_format() == 'channels_first':
    train_data=train_data.reshape(train_data.shape[0],1,IMG_ROWS,IMG_COLS)
    test_data=test_data.reshape(test_data.shape[0],1,IMG_ROWS,IMG_COLS)
    
    input_shape=(1,IMG_ROWS,IMG_COLS)
else:
    train_data=train_data.reshape(train_data.shape[0],IMG_ROWS,IMG_COLS,1)
    test_data=test_data.reshape(test_data.shape[0],IMG_ROWS,IMG_COLS,1)
    
    input_shape=(IMG_ROWS,IMG_COLS,1)
print('train_data shape: ',train_data.shape)
print('test_data shape:',test_data.shape)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /= 255
test_data /= 255

#教师标签转换为one-hot向量
train_teacher_labels=utils.to_categorical(train_teacher_labels,NUM_CLASSES)
test_teacher_labels=utils.to_categorical(test_teacher_labels)

#创建序列模型
model = Sequential()

#添加卷集层，Conv2d是2维卷基层，32个神经元，卷及区域宽度和高度是3,激活函数ReLU
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
#添加卷积层
model.add(Conv2D(64,(3,3),activation='relu'))
#添加池化层
model.add(MaxPooling2D(pool_size=(2,2)))
#添加Dropout层
model.add(Dropout(0.25))
model.add(Flatten())        #保证平滑输入
#全连接层
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dropout(0.5))     #防止过渡学习
#输出层
model.add(Dense(NUM_CLASSES,activation='softmax'))

model.summary()

#优化算法，损失函数，评估函数列表
model.compile(optimizer=Adadelta(),loss=categorical_crossentropy,metrics=['accuracy'])

print('反复学习次数：', EPOCHS)
fit_record=model.fit(train_data,train_teacher_labels,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1,validation_data=(test_data,test_teacher_labels))

# print(fit_record.history)
# plot_loss_accuracy_graph(fit_record)

result_score=model.evaluate(test_data,test_teacher_labels,verbose=0)
print('测试误差：',result_score[0])
print('测试正确率：',result_score[1])

#预测
prediction_array=model.predict(test_data)

test_data = test_data.reshape(test_data.shape[0],IMG_ROWS,IMG_COLS)

data_location=77
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(data_location,prediction_array,convertOneHotVector2Integers(test_teacher_labels),test_data)

plt.subplot(1,2,2)
plot_teacher_labels_graph(data_location,prediction_array,convertOneHotVector2Integers(test_teacher_labels))
_=plt.xticks(range(10),handwritten_number_names, rotation=45)
plt.show()

#保存训练模型
model.save('keras-mnist-model.h5')

#预测一张手写数字的图像
def img_predict(img):
    print(img.shape)
    plt.imshow(img)
    plt.show()

    img=(np.expand_dims(img,0))
    img=img.reshape(1,IMG_ROWS,IMG_COLS,1)
    print(img.shape)

    predictions_result_array=model.predict(img)
    print(predictions_result_array)

    num = np.argmax(predictions_result_array[0])
    print('预测结果： ', handwritten_number_names[num])

img_predict(test_data[data_location])