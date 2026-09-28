# Q1) write a python program to simulate single artificial neuron

import tensorflow as tf
import matplotlib.pyplot as plt

inputs = tf.constant([2,3], dtype=tf.float32)
weights = tf.constant([0.4,0.6], dtype=tf.float32)
bias = tf.constant([0.5], dtype=tf.float32)


weighted_sum = tf.reduce_sum(inputs * weights) + bias

print("inputs :", inputs.numpy())
print("weights :", weights.numpy())
print("bias :", bias.numpy())

print("weighted sum :", weighted_sum.numpy())

output = tf.sigmoid(weighted_sum)
print("output :", output.numpy())

# Q2) write a python program to demonstrate different activation functions like sigmoid, relu and tanh 
# accept input values from -10 to 10
# plot all activation function using matplotlib
# explian use of each activation function in deep learning

input_values = tf.constant([-10, -5, 0, 5, 10], dtype=tf.float32)

sigmoid_output = tf.sigmoid(input_values)
relu_output = tf.nn.relu(input_values)
tanh_output = tf.tanh(input_values)

plt.figure(figsize=(10, 6))
plt.plot(input_values.numpy(), sigmoid_output.numpy(), label='Sigmoid')
plt.plot(input_values.numpy(), relu_output.numpy(), label='ReLU')
plt.plot(input_values.numpy(), tanh_output.numpy(), label='Tanh')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()

# Q3) write a python program to calculate the loss manually
# implement mean squared error and binary cross entropy loss functions
# take actual and predicted values 
# display calcullated loss
# explain which loss function is used for regression and classification problems

# Mean Squared Error (MSE) for regression
actual = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
predicted = tf.constant([1.1, 1.9, 3.2, 3.8, 5.1], dtype=tf.float32)
mse_loss = tf.reduce_mean(tf.square(actual - predicted))
print("Mean Squared Error Loss:", mse_loss.numpy())

# Binary Cross Entropy for classification
actual_binary = tf.constant([0, 1, 1, 0, 1], dtype=tf.float32)
predicted_binary = tf.constant([0.1, 0.9, 0.8, 0.2, 0.95], dtype=tf.float32)
bce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=actual_binary, logits=predicted_binary))
print("Binary Cross Entropy Loss:", bce_loss.numpy())

calculated_loss = mse_loss.numpy()  # or bce_loss.numpy() depending on the problem type
print("Calculated Loss:", calculated_loss)


# Q4) write a python program to show how weights are updated in ANN
# take input values, weights, bias and learning rate, target output
# calculate prediction 
# calculate error
# update weights using gradient descent
# display old weights, new weights

old_weights = tf.constant([0.4, 0.6], dtype=tf.float32)
old_bias = tf.constant([0.5], dtype=tf.float32)
learning_rate = 0.1
target_output = 1.0

# calculate prediction
inputs = tf.constant([2, 3], dtype=tf.float32)
weighted_sum = tf.reduce_sum(inputs * old_weights) + old_bias
prediction = tf.sigmoid(weighted_sum)

# calculate error
error = target_output - prediction

# update weights using gradient descent
new_weights = old_weights + learning_rate * error * inputs
new_bias = old_bias + learning_rate * error

print("Old Weights:", old_weights.numpy())
print("New Weights:", new_weights.numpy())
print("Old Bias:", old_bias.numpy())
print("New Bias:", new_bias.numpy())

