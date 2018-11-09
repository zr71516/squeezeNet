import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# decode_predictions 输出5个最高概率：(类名, 语义概念, 预测概率) decode_predictions(y_pred)
from keras.preprocessing import image
import keras
import unittest



class SqueezeNetTests(unittest.TestCase):

    def testModelInit(self):
        model = SqueezeNet()
        self.assertIsNotNone(model)
        #assertIsNotNone(x，[msg='测试失败时打印的信息'])： 断言x是否None，不是None则测试用例通过。
    def testTFwPrediction(self):
        keras.backend.set_image_dim_ordering('tf')   #通道顺序和配置的通道顺序
        model = SqueezeNet()
        img = image.load_img('images/cat.jpeg', target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds)
        #print('Predicted:', decoded_preds)
        self.assertIn(decoded_preds[0][0][1], 'tabby')
        #self.assertAlmostEqual(decode_predictions(preds)[0][0][2], 0.82134342)
    def testTHPrediction(self):
        keras.backend.set_image_dim_ordering('th')
        model = SqueezeNet()
        img = image.load_img('images/cat.jpeg', target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds)
        #print('Predicted:', decoded_preds)
        self.assertIn(decoded_preds[0][0][1], 'tabby')
        #self.assertAlmostEqual(decode_predictions(preds)[0][0][2], 0.82134342)

#表示当前程序是直接调用的，而不是从其它程序中导入作为模块使用
if __name__ == '__main__':
    unittest.main()

