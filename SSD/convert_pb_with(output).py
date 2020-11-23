import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
from nets import ssd_vgg_300 # network是你们自己定义的模型结构（代码结构）
# egs:
# def network(input)：
# return tf.layers.softmax(input)
 
model_path = "train_model/model.ckpt-5000" #设置model的路径，因新版tensorflow会生成三个文件，只需写到数字前
 
def main():
 tf.reset_default_graph()
 # 设置输入网络的数据维度，根据训练时的模型输入数据的维度自行修改
 input_node = tf.placeholder(tf.float32, shape=(4,300, 300,3)) 
 output_node = ssd_vgg_300.ssd_net(input_node) # 神经网络的输出
 # 设置输出数据类型（特别注意，这里必须要跟输出网络参数的数据格式保持一致，不然会导致模型预测  精度或者预测能力的丢失）以及重新定义输出节点的名字（这样在后面保存pb文件以及之后使用pb文件时直接使用重新定义的节点名字即可）
 flow = tf.cast(output_node , tf.float16, 'the_outputs') 
 saver = tf.train.Saver()
 with tf.Session() as sess:
    saver.restore(sess, model_path)
    #保存模型图（结构），为一个json文件
    tf.train.write_graph(sess.graph_def, 'train_model/pb/', 'frozen_model.pb')
    #将模型参数与模型图结合，并保存为pb文件
    freeze_graph.freeze_graph('train_model/pb/frozen_model.pb', '', False, model_path, 'the_outputs','save/restore_all', 'save/Const:0', 'train_model/pb/frozen_model.pb', False, "")
    print("done")
if __name__ == '__main__':
 main()