from backend import model
import os

# content path --------> the path to the content image
# style path ----------> the path to the style image
# alpha ---------------> the higher the alpha relative to beta, the more the final image will resemble the content image
# beta ----------------> the higher the beta relative to alpha, the more the final image will resemble the style image
# lambdas -------------> a list of values specifying how much weight to attribute to each layer activation
# num_iterations ------> how many training steps the program should run
# learning_rate -------> how big the adjustments on every training step should be
# learning rate_decay -> adjusts how much the learning rate decreases over the course of training

content_path  = "./content images/giza.jpg"
style_path = "./style images/erin_hanson.png"
alpha = 7*1000000000
beta = 1
lambdas = [0.25, 0.25, 0.25, 0.25]
num_iterations = 1500
learning_rate = 0.27
learning_rate_decay = 0.009

# for content_path in os.listdir("./content images"):
#     for style_path in os.listdir("./style images"):
#         model("./content images/" + content_path, "./style images/" + style_path, alpha, beta, lambdas, num_iterations, learning_rate, learning_rate_decay)

model(content_path, style_path, alpha, beta, lambdas, num_iterations, learning_rate, learning_rate_decay)
