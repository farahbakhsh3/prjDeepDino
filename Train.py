import numpy as np
import tensorflow as tf
from Prepare import model_def
import matplotlib.pyplot as plt


def create_model_Conv2D(INPUT_SHAPE, OUT_SHAPE, dropout_prob=0.3):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE))

    model.add(tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    #     model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_prob))

    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    #     model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_prob))

    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_prob))

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    #     model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_prob))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_prob))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_prob))
    model.add(tf.keras.layers.Dense(OUT_SHAPE, activation='softmax'))
    print(model.summary())

    return model


if __name__ == '__main__':
    _model_def = model_def()
    np.random.seed(_model_def.random_seed)

    x_data = np.load("X.npy")
    y_data = np.load("Y.npy")
    y_data = tf.keras.utils.to_categorical(y_data, num_classes=_model_def.OUT_SHAPE)

    epochs = 75
    batch_size = 32

    print('---------------------------')
    print('X Shape: ', x_data.shape)
    print('Y Shape: ', y_data.shape)
    print('---------------------------')

    model = create_model_Conv2D(INPUT_SHAPE=_model_def.INPUT_SHAPE,
                                OUT_SHAPE=_model_def.OUT_SHAPE,
                                dropout_prob=0.3)

    # if input('Train from zero: <z>  ,  Retrain by load prev weights: <r>  ::   ') == 'r':
    #     model.load_weights(_model_def.weights_file)
    #     print('Model weights loaded : ', _model_def.weights_file)
    # else:
    #     print('Train from zero.')
    # print('---------------------------')

    chekpoint = tf.keras.callbacks.ModelCheckpoint(filepath=_model_def.weights_file,
                                                   monitor='val_acc',
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   verbose=True)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])

    history = model.fit(x_data, y_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        callbacks=[chekpoint])

    history_dict = history.history
    acc_value = history_dict['acc']
    loss_value = history_dict['loss']
    val_acc_value = history_dict['val_acc']
    val_loss_value = history_dict['val_loss']
    epoches = range(1, len(acc_value) + 1)

    plt.plot(epoches, val_acc_value, 'r', label='Validation accuracy')
    plt.plot(epoches, acc_value, 'b', label='Train accuracy')
    plt.title('accuracy')
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epoches, val_loss_value, 'r', label='Valdation loss')
    plt.plot(epoches, loss_value, 'b', label='Train loss')
    plt.title('loss')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print()
    print('---------------------------')
    print('Best weights saved. ', _model_def.weights_file)
    print('---------------------------')

