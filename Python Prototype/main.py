import Data_Handling
import Brains
import training

all_data = Data_Handling.load_all_data("/Users/tyloftin/Downloads/MIT Data")
scaled_data = Data_Handling.data_scaling(all_data)
segments = Data_Handling.data_segmentation(scaled_data, 100)

train_data = segments[:int(len(segments) * 0.8)]
test_data = segments[int(len(segments) * 0.8):]

weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = training.train_model(train_data)

accuracy = training.evaluate_model(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, test_data)
print(f"Test Accuracy: {accuracy}")
