
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

from keras.models import load_model  #下面报错  没有相关的提示信息？？？？？？

model = SqueezeNet()


#保存model部分  # save as JSON  保存模型的结构，而不包含其权重或配置信息
model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# del_json = model.to_json()

#     json_file.write(model_json)

#只保存权重
model.save_weights("model.h5")


#保存model和权重
model.save('my_model.h5')           # creates a HDF5 file 'my_model.h5'
del model                            # deletes the existing model
model = load_model('my_model.h5')   # returns a compiled model   identical to the previous one


#img = image.load_img(img_path, target_size=(224, 224))  # 加载图像，归一化大小
img = image.load_img('images/cat.jpeg', target_size=(227, 227))
x = image.img_to_array(img)                      # 序列化
x = np.expand_dims(x, axis=0)                    # 展开
x = preprocess_input(x)                          # 预处理到0～1

preds = model.predict(x)                            # 预测结果，1000维的向量
print('Predicted:', decode_predictions(preds))    # decode_predictions 输出5个最高概率：(类名, 语义概念, 预测概率)