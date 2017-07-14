import model_loader
from keras.utils import plot_model

source_model = model_loader.load_full_model('ResNet50', random_weights=False, no_cats=2, weight_decay=0, activation='softmax')
partial_model = model_loader.load_tail_model('ResNet50', source_model, no_cats=2, weight_decay=0, activation='softmax', stage='5')

plot_model(source_model, to_file='model1.png')
plot_model(partial_model, to_file='model2.png')

