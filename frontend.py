from modelBackend import model
import os
import cv2


content_path  = "./content images/giza.jpg"
style_path = "./style images/the-great-wave.jpg"
alpha = 7*1000000000
beta = 1
lambdas = [0.25, 0.25, 0.25, 0.25]
num_iterations = 2000
learning_rate = 0.22
learning_rate_decay = 0.002

# for content_path in os.listdir("./content images"):
#     for style_path in os.listdir("./style images"):
#         model("./content images/" + content_path, "./style images/" + style_path, alpha, beta, lambdas, num_iterations, learning_rate, learning_rate_decay)

model(content_path, style_path, alpha, beta, lambdas, num_iterations, learning_rate, learning_rate_decay)
