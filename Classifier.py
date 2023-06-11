import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.callbacks as callbacks


class BasicClassifier:
    def __init__(self, num_classes, image_width, image_height, num_channels):
        '''
        A basic implementation of a Convolutional Neural Network (CNN) for the purpose of classification of photos.
        
        '''
        self.num_classes = num_classes
        self.image_width = image_width
        self.image_height = image_height
        self.num_channels = num_channels

        

        #--------Define The Model's Architecture----------------------------------------------------#
        self.layers = [layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, num_channels)),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(64, (3, 3), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(128, (3, 3), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.Flatten(),
                        layers.Dense(128, activation='relu'),
                        layers.Dense(num_classes, activation='softmax')]
        



        #--------Create The Model----------------------------------------------------#
        self.model = tf.keras.Sequential(self.layers)




        #--------Compile The Model----------------------------------------------------#
        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])
        



    #--------Load In Data For Model----------------------------------------------------#
    def get_data(self, train_folder_path, test_folder_path, batch_size):
        self.batch_size = batch_size



        #--------Create Train and Test Generators----------------------------------------------------#
        self.train_data_folder_loader = ImageDataGenerator(rescale=1./255)
        self.test_data_folder_loader = ImageDataGenerator(rescale=1./255)




        #--------Load and Augment Train and Test Images----------------------------------------------------#
        self.train_data_folder_loader = self.train_data_folder_loader.flow_from_directory(
            train_folder_path,
            target_size=(self.image_width, self.image_height),
            batch_size=batch_size,
            class_mode='sparse'
        )

        self.test_data_folder_loader = self.test_data_folder_loader.flow_from_directory(
            test_folder_path,
            target_size=(self.image_width, self.image_height),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False
        )



        #-------- Verify the class names----------------------------------------------------#
        self.class_names = list(self.train_data_folder_loader.class_indices.keys())
        print("Class Names:", self.class_names)



    #-------- Train Model----------------------------------------------------#
    def train(self, n_epochs):

        self.callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=2),
                tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
                tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None),
                tf.keras.callbacks.TensorBoard(log_dir='./logs')]
        
        self.model.fit(self.train_data_folder_loader, epochs=n_epochs, validation_data=self.test_data_folder_loader, callbacks=self.callbacks)

        



    #--------Use Test Data to Check Model's Accuracy----------------------------------------------------#
    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.test_data_folder_loader)
        print(f'Model Accuracy: {test_acc * 100}%')
        return test_loss, test_acc

        

#--------Used For Testing Model----------------------------------------------------#
if __name__ == "__main__":
    micro_express = BasicClassifier(7, 80, 80, 3)
    micro_express.get_data("train", "test", 32)
    micro_express.train(1)
    micro_express.evaluate()
