from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

#别人写的应该只有这个部分！！！！！！！！！！
WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Modular function for Fire Node    //fire module部分
#建模
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'   #第几个fire模块

    if K.image_data_format() == 'channels_first':    # Keras默认的数据组织形式 使用（样本数，通道数，行或称为高，列或称为宽）通道在前的方式
        channel_axis = 1
    else:
        channel_axis = 3
    #一个squeeze层  两个expand层
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)  #过滤器个数  过滤器行数 列数 不填充  x什么？？？
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')  #数组拼接函数
    return x


# Original SqueezeNet from paper.   整个squeezenet
#weights： 定义为‘imagenet’，表示加载在imagenet数据库上训练的预训练权重，定义为None则不加载权重，参数随机初始化
 #include_top 是否包含最后的3个全连接层
 #input_shape就是指输入张量的shape。例如，input_dim=784，说明输入是一个784维的向量，这相当于一个一阶的张量，它的shape就是(784,)。因此，input_shape=(784,)。
def SqueezeNet(include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None,
               classes=1000):
    """Instantiates the SqueezeNet architecture.
    """
    # 检查weight与分类设置是否正确 如果权重不再其中报错
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and classes != 1000:     #
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # 设置图像尺寸，类似caffe中的transform
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=48,
                                         # 模型所能接受的最小长宽
                                      data_format=K.image_data_format(),
                                      # 数据的使用格式
                                      require_flatten=include_top)
                                        # 是否通过一个Flatten层再连接到分类器
    # 数据简单处理，resize
    if input_tensor is None:
        img_input = Input(shape=input_shape)  #转为矩阵数据
        # 这里的Input是keras的格式，可以用于转换
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # 如果是tensor的数据格式，需要两步走： 先判断是否是keras指定的数据类型，is_keras_tensor
    # 然后get_source_inputs(input_tensor)

    # 编写网络结构，prototxt
    # Block 1   卷积 激活函数  池化
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)  #合理张量
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    # Block 2  fire的
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
    # Block 3
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    # Block 4
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    
    if include_top:
        # It's not obvious where to cut the network... 
        # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    
        x = Dropout(0.5, name='drop9')(x)
        #dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃
        #好像dropout=0.8是每次随机80%的神经元失效

        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)   #卷积 classes过滤器个数
        x = Activation('relu', name='relu_conv10')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='loss')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)
        elif pooling==None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)

    # 调整数据
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
        # get_source_inputs 返回计算需要的数据列表，List of input tensors.
        # 如果是tensor的数据格式，需要两步走：
        # 先判断是否是keras指定的数据类型，is_keras_tensor
        # 然后get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # 创建模型
    # Create model.
    model = Model(inputs, x, name='squeezenet')

    # 加载权重
    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        model.load_weights(weights_path)
        # backend() 返回当前后端 Theano是一个Python库,专门用于定义、优化、求值数学表达
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


