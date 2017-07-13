from resnet50 import ResNet50

model = ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000, weight_decay = 0)

from vgg19 import VGG19

model = VGG19(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000, weight_decay = 0)