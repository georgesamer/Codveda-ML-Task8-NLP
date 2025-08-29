import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
tf.random.set_seed(42)

class MNISTClassifier:
    def __init__(self, model_name="mnist_classifier"):
        self.model_name = model_name
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST"""
        print("Loading MNIST")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        # Reshape layers
        x_train = x_train.reshape(-1, 28*28)
        x_test = x_test.reshape(-1, 28*28)
        
        print(f"Data loaded - Train: {x_train.shape}, Test: {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)
    
    def build_model(self, architecture="standard"):
        """Building neural network"""
        if architecture == "standard":
            model = keras.Sequential([
                layers.Dense(128, activation="relu", input_shape=(784,)),
                layers.Dropout(0.2),  # Add dropout for regularization
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(10, activation="softmax")
            ], name=self.model_name)
            
        elif architecture == "deep":
            model = keras.Sequential([
                layers.Dense(256, activation="relu", input_shape=(784,)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(10, activation="softmax")
            ], name=self.model_name)
            
        else:  # light weight
            model = keras.Sequential([
                layers.Dense(64, activation="relu", input_shape=(784,)),
                layers.Dropout(0.1),
                layers.Dense(32, activation="relu"),
                layers.Dense(10, activation="softmax")
            ], name=self.model_name)
        
        self.model = model
        print(f"Model built with '{architecture}' architecture")
        return model

    def get_callbacks(self, patience=5, monitor='val_loss'):
        """Setup training callbacks"""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            # Model checkpoint
            callbacks.ModelCheckpoint(
                f"best_{self.model_name}.h5",
                monitor=monitor,
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks_list
    
    def train(self, x_train, y_train, epochs=20, batch_size=32, validation_split=0.2):
        """Train the model"""
        print(f"Start training {epochs} epochs")
        
        # Callbacks
        training_callbacks = self.get_callbacks()
        
        # Train model
        self.history = self.model.fit(
            x_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=training_callbacks,
            verbose=1
        )
        
        print("Training completed")
        return self.history

    def evaluate(self, x_test, y_test, show_detailed_metrics=True):
        """model evaluation"""
        print("Evaluating model")
        
        # Basic evaluation
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        if show_detailed_metrics:
            # Get predictions
            y_pred = self.model.predict(x_test, verbose=0)
            y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
            
            # Calculate per-class
            print("\nPer-Class:")
            print(f"{'Class':<5} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print("-" * 40)
            
            for class_id in range(10):
                tp = tf.reduce_sum(tf.cast((y_test == class_id) & (y_pred_classes == class_id), tf.int32)).numpy()
                fp = tf.reduce_sum(tf.cast((y_test != class_id) & (y_pred_classes == class_id), tf.int32)).numpy()
                fn = tf.reduce_sum(tf.cast((y_test == class_id) & (y_pred_classes != class_id), tf.int32)).numpy()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                print(f"{class_id:<5} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")
            
            return test_loss, test_acc, y_pred_classes
        
        return test_loss, test_acc

    def plot_training_history(self, figsize=(12, 4)):
        """Plot training history with improved visualization"""
        if self.history is None:
            print("No training history")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        ax1.plot(self.history.history["loss"], label="Train Loss", linewidth=2)
        ax1.plot(self.history.history["val_loss"], label="Validation Loss", linewidth=2)
        ax1.set_title("Model Loss", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.history.history["accuracy"], label="Train Accuracy", linewidth=2)
        ax2.plot(self.history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
        ax2.set_title("Model Accuracy", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(8, 6)):
        """Plot confusion matrix"""
        # Calculate confusion matrix manually
        cm = tf.zeros((10, 10), dtype=tf.int32)
        for true_label in range(10):
            for pred_label in range(10):
                count = tf.reduce_sum(tf.cast((y_true == true_label) & (y_pred == pred_label), tf.int32))
                cm = tf.tensor_scatter_nd_update(cm, [[true_label, pred_label]], [count])
        
        cm = cm.numpy()
        
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
        plt.colorbar()
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), 
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.tight_layout()
        plt.show()
    
    def predict_sample(self, x_test, y_test, num_samples=5):
        """Predict and visualize sample images"""
        indices = tf.random.uniform([num_samples], 0, len(x_test), dtype=tf.int32)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        for i, idx in enumerate(indices.numpy()):
            # Reshape for visualization
            image = x_test[idx].reshape(28, 28)
            
            # Make prediction
            pred = self.model.predict(x_test[idx:idx+1], verbose=0)
            pred_class = tf.argmax(pred, axis=1).numpy()[0]
            confidence = tf.reduce_max(pred).numpy()
            
            # Plot
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f"True: {y_test[idx]}\nPred: {pred_class}\nConf: {confidence:.2f}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    # Initialize classifier
    classifier = MNISTClassifier("improved_mnist_model")
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = classifier.load_and_preprocess_data()
    
    # Build model (try 'standard', 'deep', or 'lightweight')
    classifier.build_model(architecture="standard")

    # 4- Compile
    classifier.model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    
    # Print model summary
    classifier.model.summary()
    
    # Train model
    history = classifier.train(x_train, y_train, epochs=20, batch_size=32)
    
    # Evaluate model
    test_loss, test_acc, y_pred = classifier.evaluate(x_test, y_test)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(y_test, y_pred)
    
    # Show sample predictions
    classifier.predict_sample(x_test, y_test, num_samples=8)
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()