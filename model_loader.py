import keras

from resnet50 import ResNet50
from vgg19 import VGG19

def load_model(model_name, no_cats, final_activation = 'softmax', weight_decay = 0, random_weights = False):
	"""
	Loads a model with a randomly initialized last layer.
		model_name: ResNet50, VGG19
		no_cats: Number of outputs
		final_activation: activation of the final layer (None, softmax, sigmoid)
		weight decay: L2 weight decay for all layers
		random_weights: Random weights or ImageNet pre-training
	"""

	if random_weights:
		weights = None
	else:
		weights = 'imagenet'

	# load the model without the final layer
	if model_name == 'ResNet50':
		model = ResNet50(weights = weights,
			include_top = False,
			input_tensor = keras.layers.Input(shape = (224, 224, 3)),
			weight_decay = weight_decay)
	elif model_name == 'VGG19':
		model = VGG19(weights = weights,
			include_top = False,
			input_tensor = keras.layers.Input(shape = (224, 224, 3)),
			weight_decay = weight_decay)
	else:
		raise ValueError('Invalid model_name')

	# add the final layer
	x = keras.layers.Flatten()(model.output)
	x = keras.layers.Dense(no_cats, activation = final_activation,
		W_regularizer = keras.regularizers.l2(weight_decay), name = 'fc_final')(x)
	model = keras.models.Model(model.input, x)

	return model

def set_no_trainable_layers(model, no_trainable_layers):
	"""
	Sets only the final no_trainable_layers layers as trainable.
	"""
	for layer in model.layers:
		layer.trainable = False

	for layer in model.layers[-no_trainable_layers:]:
		layer.trainable = True

	return model