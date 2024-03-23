import mnist_reader
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Flatten         #Dense全连接层类，Flatten平滑输入
from keras.optimizers import Adam               #Adam优化算法

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

train_data, train_teacher_labels = mnist_reader.load_mnist('./data/fashion', kind='train')
test_data, test_teacher_labels = mnist_reader.load_mnist('./data/fashion', kind='t10k')

print("train_data shape: ", train_data.shape)
print("train teacher labels: ", len(train_teacher_labels))
print("test_data shape: ", test_data.shape)
print("test teacher labels: ", len(test_teacher_labels))

fashion_names=('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# plt.figure()
# plt.imshow(train_data[3], cmap='inferno')
# plt.colorbar()
# plt.grid(False)
# plt.show()

# plt.figure(figsize=(12,12))
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_data[i],cmap='inferno')
#     plt.xlabel(fashion_names[train_teacher_labels[i]]) 
# plt.show()

#批量次数，类型数量，学习次数，图像大小
BATCH_SIZE = 128    #中间层128，也可以64,600,2000,寻找合适的
NUM_CLASSES = 10    #输出层10，fashion_names十种
EPOCH = 20
IMG_ROWS, IMG_COLS = 28, 28     #输入层28*28=784

train_data=train_data.astype('float32')
test_data=test_data.astype('float32')

train_data /= 25
test_data /= 25

# print('train_data shape: ',train_data.shape)
# print(train_data.shape[0])
# print('test_data shape: ',test_data.shape)
# print(test_data.shape[0])

model = Sequential()

#输入层
model.add(Flatten(input_shape=(IMG_ROWS,IMG_COLS)))
#中间层
model.add(Dense(128,activation=tf.nn.relu))
#输出层
model.add(Dense(10,activation=tf.nn.softmax))

model.summary()

model.compile(optimizer=Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print('反复学习次数：', EPOCH)
fit_record=model.fit(train_data,train_teacher_labels,batch_size=BATCH_SIZE,epochs=EPOCH,verbose=1,validation_data=(test_data,test_teacher_labels))

# print(fit_record.history)
# plot_loss_accuracy_graph(fit_record)

result_score=model.evaluate(test_data,test_teacher_labels)
print('测试误差：',result_score[0])
print('测试正确率：',result_score[1])

#显示测试数据
data_location=4
img=test_data[data_location]
print(img.shape)
img=(np.expand_dims(img, 0))
print(img.shape)

prediction_result_array=model.predict(img)
print(prediction_result_array)      #fashion_names中十个元素的最大可能性
num=np.argmax(prediction_result_array)
print('预测结果：',fashion_names[num])

plt.figure()
plt.imshow(test_data[data_location], cmap='inferno')
plt.colorbar()
plt.grid(False)
plt.show()

#保存训练模型
model.save('keras-fashion-mnist-model.h5')
